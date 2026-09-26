"""Streamlit dashboard and database access for vehicle crossing events."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import DateTime, Integer, String, create_engine, desc, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from streamlit_autorefresh import st_autorefresh

load_dotenv()
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql+psycopg://vehicle:vehicle@localhost:5432/vehicle_analytics"
)


class Base(DeclarativeBase):
    """SQLAlchemy metadata for the analytics schema."""


class DetectionEvent(Base):
    """A vehicle recorded once when its tracked ID crosses the counting line."""

    __tablename__ = "detection_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vehicle_id: Mapped[str] = mapped_column(String(96), index=True)
    vehicle_type: Mapped[str] = mapped_column(String(32))
    color: Mapped[str] = mapped_column(String(24), index=True)
    direction: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column()
    crossed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=5)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


def initialize_database() -> None:
    """Create the single-table schema if it does not exist yet."""
    Base.metadata.create_all(engine)


def insert_event(
    *, vehicle_id: str, vehicle_type: str, color: str, direction: str,
    confidence: float, crossed_at: datetime | None = None,
) -> int:
    """Persist one crossing event and return its database ID."""
    event_time = crossed_at or datetime.now(timezone.utc)
    with SessionLocal.begin() as session:
        row = DetectionEvent(
            vehicle_id=vehicle_id,
            vehicle_type=vehicle_type,
            color=color,
            direction=direction,
            confidence=confidence,
            crossed_at=event_time,
        )
        session.add(row)
        session.flush()
        return row.id


def get_dashboard_data(limit: int = 50) -> tuple[int, dict[str, int], list[dict[str, object]]]:
    """Fetch durable totals, color groups, and recent events in short-lived sessions."""
    with SessionLocal() as session:
        total = session.scalar(select(func.count()).select_from(DetectionEvent)) or 0
        colors = session.execute(
            select(DetectionEvent.color, func.count())
            .group_by(DetectionEvent.color)
            .order_by(func.count().desc())
        ).all()
        rows = session.scalars(
            select(DetectionEvent).order_by(desc(DetectionEvent.crossed_at)).limit(limit)
        ).all()
    recent = [
        {
            "Time (UTC)": row.crossed_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "Vehicle": row.vehicle_type,
            "Color": row.color,
            "Direction": row.direction.replace("_", " "),
            "Confidence": f"{row.confidence:.0%}",
            "Track ID": row.vehicle_id,
        }
        for row in rows
    ]
    return total, dict(colors), recent


def main() -> None:
    """Render a periodically refreshed dashboard without side effects on import."""
    st.set_page_config(page_title="Vehicle Analytics", page_icon="🚦", layout="wide")
    st_autorefresh(interval=2000, key="dashboard_refresh")
    st.title("🚦 Real-Time Vehicle Analytics")
    st.caption("Live crossing counts and recent detections · refreshes every 2 seconds")

    try:
        initialize_database()
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        total, colors, recent = get_dashboard_data()
    except Exception as exc:
        st.error(f"Database unavailable: {exc}")
        st.info("Start PostgreSQL with `docker compose up -d postgres`, then refresh this page.")
        st.stop()

    st.subheader("Crossing totals")
    metric_columns = st.columns(5)
    metric_columns[0].metric("All vehicles", total)
    for column, color in zip(metric_columns[1:], ("white", "black", "gray", "other")):
        count = sum(value for name, value in colors.items() if name == color)
        if color == "other":
            count = sum(value for name, value in colors.items() if name not in {"white", "black", "gray"})
        column.metric(color.title(), count)

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Color breakdown")
        if colors:
            st.bar_chart(pd.DataFrame({"Vehicles": colors}))
        else:
            st.caption("No vehicles have crossed the line yet.")
    with right:
        st.subheader("Recent crossings")
        if recent:
            st.dataframe(pd.DataFrame(recent), use_container_width=True, hide_index=True)
        else:
            st.caption("Events appear here after the inference pipeline records a line crossing.")

    with st.sidebar:
        st.header("Pipeline status")
        st.success("PostgreSQL connected")
        st.write(f"**Stored crossings:** {total}")
        st.write(f"**Configured stream:** `{os.getenv('RTSP_URL', 'rtsp://localhost:8554/traffic')}`")
        st.caption("Run `python pipeline.py` in a terminal to start inference.")


if __name__ == "__main__":
    main()
