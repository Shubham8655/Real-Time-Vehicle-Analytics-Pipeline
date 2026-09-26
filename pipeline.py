"""Detect, track, classify, and count vehicles from a video or RTSP stream."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone

# OpenCV's FFmpeg backend reads RTSP options from the process environment at startup.
# Re-exec once so a normal `python pipeline.py` command inherits TCP before OpenCV loads.
if __name__ == "__main__" and "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
    process_env = os.environ.copy()
    process_env["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    child = subprocess.run([sys.executable, *sys.argv], env=process_env, check=False)
    raise SystemExit(child.returncode)

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

from app import initialize_database, insert_event

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
LOG = logging.getLogger("vehicle_pipeline")


@dataclass
class TrackState:
    """Small amount of state retained for each current tracker ID."""

    center_y: float
    last_seen: float
    counted: bool = False


def classify_color(crop: np.ndarray) -> str:
    """Estimate a vehicle's dominant visible color from its central crop area."""
    if crop.size == 0:
        return "unknown"
    height, width = crop.shape[:2]
    central = crop[height // 5 : max(height // 5 + 1, height * 4 // 5),
                   width // 5 : max(width // 5 + 1, width * 4 // 5)]
    hsv = cv2.cvtColor(central, cv2.COLOR_BGR2HSV)
    saturation = float(np.median(hsv[:, :, 1]))
    value = float(np.median(hsv[:, :, 2]))
    if saturation < 45:
        if value < 55:
            return "black"
        if value > 185:
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
    return "purple"


def crossed_line(previous_y: float, current_y: float, line_y: int) -> str | None:
    """Return the direction when a tracked center changes sides of a horizontal line."""
    if previous_y < line_y <= current_y:
        return "north_to_south"
    if previous_y > line_y >= current_y:
        return "south_to_north"
    return None


def _persist_event(future: Future[int], vehicle_id: str) -> None:
    """Surface asynchronous database errors in the inference process log."""
    try:
        future.result()
    except Exception:
        LOG.exception("Could not persist crossing event for track %s", vehicle_id)


def run(
    source: str,
    line_ratio: float,
    show: bool,
    model_path: str,
    max_frames: int | None = None,
) -> None:
    """Run a persistent ByteTrack tracker and asynchronously write crossing events."""
    initialize_database()
    model = YOLO(model_path)
    stream_id = uuid.uuid4().hex[:8]
    tracks: dict[int, TrackState] = {}
    executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="event-writer")
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
        for frame_number, result in enumerate(results, start=1):
            frame = result.orig_img
            now = time.monotonic()
            line_y = int(frame.shape[0] * line_ratio)
            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 220, 255), 2)
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                coordinates = boxes.xyxy.cpu().numpy().astype(int)
                identifiers = boxes.id.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                for (x1, y1, x2, y2), track_id, class_id, confidence in zip(
                    coordinates, identifiers, class_ids, confidences
                ):
                    previous = tracks.get(track_id)
                    center_y = (y1 + y2) / 2
                    direction = crossed_line(previous.center_y, center_y, line_y) if previous else None
                    already_counted = previous.counted if previous else False
                    tracks[track_id] = TrackState(center_y, now, already_counted)
                    x1, x2 = max(0, x1), min(frame.shape[1], x2)
                    y1, y2 = max(0, y1), min(frame.shape[0], y2)
                    label = VEHICLE_CLASSES[class_id]
                    if direction and not already_counted:
                        vehicle_id = f"{stream_id}:{track_id}"
                        event = {
                            "vehicle_id": vehicle_id,
                            "vehicle_type": label,
                            "color": classify_color(frame[y1:y2, x1:x2]),
                            "direction": direction,
                            "confidence": float(confidence),
                            "crossed_at": datetime.now(timezone.utc),
                        }
                        future = executor.submit(insert_event, **event)
                        future.add_done_callback(lambda completed, key=vehicle_id: _persist_event(completed, key))
                        tracks[track_id].counted = True
                        LOG.info("counted track=%s type=%s color=%s direction=%s", vehicle_id, label,
                                 event["color"], direction)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 210, 70), 2)
                    cv2.putText(frame, f"{label} #{track_id}", (x1, max(20, y1 - 7)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 210, 70), 2)
                # Drop IDs that ByteTrack has not observed recently; avoids unbounded growth.
                stale = [key for key, state in tracks.items() if now - state.last_seen > 60]
                for key in stale:
                    del tracks[key]
            if show:
                cv2.imshow("Vehicle Analytics", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if max_frames is not None and frame_number >= max_frames:
                break
    finally:
        executor.shutdown(wait=True)
        if show:
            cv2.destroyAllWindows()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run Python vehicle detection and line counting.")
    parser.add_argument("--source", default=os.getenv("RTSP_URL", "rtsp://localhost:8554/traffic"))
    parser.add_argument("--line-ratio", type=float, default=float(os.getenv("LINE_RATIO", "0.55")))
    parser.add_argument("--model", default=os.getenv("YOLO_MODEL", "yolo11n.pt"))
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames (useful for short checks)")
    parser.add_argument("--show", action="store_true", help="Display annotated frames; press q to stop")
    args = parser.parse_args()
    if not 0 < args.line_ratio < 1:
        parser.error("--line-ratio must be between 0 and 1")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be a positive integer")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.source, args.line_ratio, args.show, args.model, args.max_frames)


if __name__ == "__main__":
    main()
