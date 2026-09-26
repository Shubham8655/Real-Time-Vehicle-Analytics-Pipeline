"""Runtime configuration for the vehicle analytics services."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Configuration read once from the environment (and optional ``.env`` file)."""

    database_url: str
    rtsp_url: str
    yolo_model: str
    line_ratio: float


def load_settings() -> Settings:
    """Load and validate environment-based application settings."""
    load_dotenv()
    line_ratio = float(os.getenv("LINE_RATIO", "0.55"))
    if not 0 < line_ratio < 1:
        raise ValueError("LINE_RATIO must be between 0 and 1")
    return Settings(
        database_url=os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://vehicle:vehicle@localhost:5432/vehicle_analytics",
        ),
        rtsp_url=os.getenv("RTSP_URL", "rtsp://localhost:8554/traffic"),
        yolo_model=os.getenv("YOLO_MODEL", "yolo11n.pt"),
        line_ratio=line_ratio,
    )
