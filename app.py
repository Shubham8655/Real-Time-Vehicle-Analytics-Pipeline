"""HTTP dashboard and PostgreSQL access for the vehicle analytics pipeline."""

from __future__ import annotations

import os
from collections import Counter
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Integer, String, create_engine, desc, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

load_dotenv()
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://vehicle:vehicle@localhost:5432/vehicle_analytics"
)


class Base(DeclarativeBase):
    """Base class for the small analytics schema."""


class DetectionEvent(Base):
    """One immutable event, recorded only when a tracked vehicle crosses the line."""

    __tablename__ = "detection_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vehicle_id: Mapped[str] = mapped_column(String(64), index=True)
    vehicle_type: Mapped[str] = mapped_column(String(32))
    color: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[str] = mapped_column(String(24), index=True)
    confidence: Mapped[int] = mapped_column(Integer)  # Stored as percentage for simple reporting.
    crossed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class EventInput(BaseModel):
    """Validated event shape used by the pipeline and the demo endpoint."""

    vehicle_id: str = Field(min_length=1, max_length=64)
    vehicle_type: str = Field(min_length=1, max_length=32)
    color: str = Field(min_length=1, max_length=32)
    direction: str = Field(min_length=1, max_length=24)
    confidence: int = Field(ge=0, le=100)
    crossed_at: datetime | None = None


def initialize_database() -> None:
    """Create the table on startup; migrations are unnecessary for this single-table exercise."""
    Base.metadata.create_all(engine)


def insert_event(event: EventInput) -> int:
    """Persist an event synchronously; the pipeline calls this from a bounded worker thread."""
    timestamp = event.crossed_at or datetime.now(timezone.utc)
    with SessionLocal.begin() as session:
        row = DetectionEvent(**event.model_dump(exclude={"crossed_at"}), crossed_at=timestamp)
        session.add(row)
        session.flush()
        return row.id


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(title="Vehicle Analytics", lifespan=lifespan)
STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def dashboard() -> FileResponse:
    """Serve the dependency-free dashboard."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    """Small readiness endpoint for Compose and manual checks."""
    with engine.connect():
        pass
    return {"status": "ok"}


@app.get("/api/events")
def recent_events(limit: int = 30) -> list[dict[str, object]]:
    """Return the latest crossings, newest first."""
    limit = max(1, min(limit, 200))
    with SessionLocal() as session:
        rows = session.scalars(select(DetectionEvent).order_by(desc(DetectionEvent.crossed_at)).limit(limit))
        return [_serialize(row) for row in rows]


@app.get("/api/metrics")
def metrics() -> dict[str, object]:
    """Aggregate totals directly from PostgreSQL so restarts do not lose counts."""
    with SessionLocal() as session:
        total = session.scalar(select(func.count()).select_from(DetectionEvent)) or 0
        colors = session.execute(
            select(DetectionEvent.color, func.count()).group_by(DetectionEvent.color)
        ).all()
    return {"total": total, "by_color": dict(colors)}


@app.post("/api/events", status_code=201)
def create_event(event: EventInput) -> dict[str, int]:
    """Allow the independent inference process to post events or seed a demo."""
    try:
        return {"id": insert_event(event)}
    except Exception as exc:  # Keep an operational database failure readable to the caller.
        raise HTTPException(status_code=503, detail="Database is unavailable") from exc


def _serialize(row: DetectionEvent) -> dict[str, object]:
    return {
        "id": row.id,
        "vehicle_id": row.vehicle_id,
        "vehicle_type": row.vehicle_type,
        "color": row.color,
        "direction": row.direction,
        "confidence": row.confidence,
        "crossed_at": row.crossed_at.isoformat(),
    }
