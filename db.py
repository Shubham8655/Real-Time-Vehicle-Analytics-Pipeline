"""Database schema and repository methods for counted vehicle crossings."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, UniqueConstraint, create_engine, desc, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


class DetectionEvent(Base):
    """One durable line-crossing event for one tracker ID during one pipeline run."""

    __tablename__ = "detection_events"
    __table_args__ = (UniqueConstraint("run_id", "tracker_id", name="uq_detection_run_track"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(String(32), nullable=False)
    tracker_id: Mapped[int] = mapped_column(Integer, nullable=False)
    vehicle_type: Mapped[str] = mapped_column(String(32), nullable=False)
    color: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    direction: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    crossed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)

    @property
    def vehicle_id(self) -> str:
        """A human-readable identifier combining pipeline run and tracker ID."""
        return f"{self.run_id}:{self.tracker_id}"


class EventRepository:
    """Short-lived transactional access to the analytics database."""

    def __init__(self, database_url: str) -> None:
        engine_options: dict[str, object] = {"pool_pre_ping": True}
        if not database_url.startswith("sqlite"):
            engine_options.update(pool_size=5, max_overflow=5)
        self.engine = create_engine(database_url, **engine_options)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def initialize(self) -> None:
        """Create the schema when the service starts for the first time."""
        Base.metadata.create_all(self.engine)

    def insert_crossing(
        self,
        *,
        run_id: str,
        tracker_id: int,
        vehicle_type: str,
        color: str,
        direction: str,
        confidence: float,
        crossed_at: datetime | None = None,
    ) -> int | None:
        """Store a crossing and return its ID; duplicate tracker events return ``None``."""
        event = DetectionEvent(
            run_id=run_id,
            tracker_id=tracker_id,
            vehicle_type=vehicle_type,
            color=color,
            direction=direction,
            confidence=confidence,
            crossed_at=crossed_at or datetime.now(timezone.utc),
        )
        try:
            with self.sessions.begin() as session:
                session.add(event)
                session.flush()
                return event.id
        except IntegrityError:
            return None

    def dashboard_data(self, limit: int = 50) -> tuple[int, dict[str, int], list[dict[str, object]]]:
        """Return aggregate totals and recent events for the dashboard."""
        with self.sessions() as session:
            total = session.scalar(select(func.count()).select_from(DetectionEvent)) or 0
            color_counts = session.execute(
                select(DetectionEvent.color, func.count())
                .group_by(DetectionEvent.color)
                .order_by(func.count().desc())
            ).all()
            events = session.scalars(
                select(DetectionEvent).order_by(desc(DetectionEvent.crossed_at)).limit(limit)
            ).all()

        recent = []
        for event in events:
            timestamp = event.crossed_at
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            recent.append(
                {
                    "Time (UTC)": timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "Vehicle": event.vehicle_type,
                    "Color": event.color,
                    "Direction": event.direction.replace("_", " "),
                    "Confidence": f"{event.confidence:.0%}",
                    "Track ID": event.vehicle_id,
                }
            )
        return total, dict(color_counts), recent
