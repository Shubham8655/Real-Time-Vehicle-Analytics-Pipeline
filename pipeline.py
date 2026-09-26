"""Python-only vehicle detector, tracker, colour classifier, and line counter."""

from __future__ import annotations

import argparse
import logging
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

from app import EventInput, initialize_database, insert_event

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}  # COCO labels.


@dataclass
class TrackState:
    """The last observed centre and a one-shot crossing flag for an object ID."""

    center_y: float
    counted: bool = False


def classify_color(crop: np.ndarray) -> str:
    """Classify the central vehicle crop by HSV; avoids road/background pixels at the edges."""
    if crop.size == 0:
        return "unknown"
    height, width = crop.shape[:2]
    crop = crop[height // 5 : height * 4 // 5, width // 5 : width * 4 // 5]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    # Low saturation colours are best distinguished using brightness.
    if float(np.median(saturation)) < 45:
        if float(np.median(value)) < 55:
            return "black"
        if float(np.median(value)) > 185:
            return "white"
        return "gray"
    hue = int(np.median(hsv[:, :, 0]))
    if hue < 10 or hue >= 170:
        return "red"
    if hue < 25:
        return "orange"
    if hue < 38:
        return "yellow"
    if hue < 85:
        return "green"
    if hue < 135:
        return "blue"
    if hue < 170:
        return "purple"
    return "unknown"


def crossed_line(previous_y: float, current_y: float, line_y: int) -> str | None:
    """Return the travel direction only when the bounding-box centre changes sides."""
    if previous_y < line_y <= current_y:
        return "north_to_south"
    if previous_y > line_y >= current_y:
        return "south_to_north"
    return None


def run(source: str, line_ratio: float, show: bool) -> None:
    """Track each vehicle from an RTSP/file source and asynchronously persist crossings."""
    initialize_database()
    model = YOLO(os.getenv("YOLO_MODEL", "yolo11n.pt"))
    tracks: dict[int, TrackState] = {}
    recent_ids: deque[int] = deque(maxlen=2_000)
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="db-writer")

    # Ultralytics' ByteTrack runs in Python and provides persistent per-object IDs.
    results = model.track(
        source=source,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        classes=list(VEHICLE_CLASSES),
        conf=0.35,
        verbose=False,
    )
    try:
        for result in results:
            frame = result.orig_img
            line_y = int(frame.shape[0] * line_ratio)
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 255, 255), 2)
            if result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                ids = result.boxes.id.int().cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()
                confidences = result.boxes.conf.cpu().tolist()
                for (x1, y1, x2, y2), track_id, class_id, confidence in zip(boxes, ids, classes, confidences):
                    center_y = (y1 + y2) / 2
                    state = tracks.get(track_id)
                    direction = crossed_line(state.center_y, center_y, line_y) if state else None
                    tracks[track_id] = TrackState(center_y=center_y, counted=state.counted if state else False)
                    if direction and not tracks[track_id].counted:
                        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                        event = EventInput(
                            vehicle_id=str(track_id),
                            vehicle_type=VEHICLE_CLASSES[class_id],
                            color=classify_color(crop),
                            direction=direction,
                            confidence=round(confidence * 100),
                            crossed_at=datetime.now(timezone.utc),
                        )
                        executor.submit(insert_event, event)
                        tracks[track_id].counted = True
                        logging.info("counted vehicle=%s color=%s direction=%s", track_id, event.color, direction)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                    cv2.putText(frame, f"{VEHICLE_CLASSES[class_id]} #{track_id}", (x1, max(20, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
            if show:
                cv2.imshow("Vehicle Analytics", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        executor.shutdown(wait=True)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the Python vehicle analytics pipeline.")
    parser.add_argument("--source", default=os.getenv("RTSP_URL", "rtsp://localhost:8554/traffic"))
    parser.add_argument("--line-ratio", type=float, default=0.55, help="Horizontal line as fraction of frame height")
    parser.add_argument("--show", action="store_true", help="Show annotated frames; press q to exit")
    args = parser.parse_args()
    if not 0 < args.line_ratio < 1:
        parser.error("--line-ratio must be between 0 and 1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.source, args.line_ratio, args.show)
