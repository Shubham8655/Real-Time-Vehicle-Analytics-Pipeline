"""Python-only vehicle detection, tracking, color classification, and line counting."""

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

# OpenCV's FFmpeg backend reads RTSP options at process start. Re-exec the CLI
# once so RTSP streams use TCP before OpenCV is imported by Ultralytics.
if __name__ == "__main__" and "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
    child_environment = os.environ.copy()
    child_environment["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    completed = subprocess.run([sys.executable, *sys.argv], env=child_environment, check=False)
    raise SystemExit(completed.returncode)

import cv2
import numpy as np

from config import load_settings
from db import EventRepository

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
LOG = logging.getLogger("vehicle_pipeline")


@dataclass
class TrackState:
    """The state needed to count each persistent tracker ID at most once."""

    center_y: float
    last_seen: float
    counted: bool = False


def classify_color(crop: np.ndarray) -> str:
    """Estimate a vehicle's visible color from the central part of its crop.

    HSV separates brightness and saturation from hue, making neutral vehicles
    distinguishable from colored ones without another heavyweight model.
    """
    if crop.size == 0:
        return "unknown"
    height, width = crop.shape[:2]
    central = crop[
        height // 5 : max(height // 5 + 1, height * 4 // 5),
        width // 5 : max(width // 5 + 1, width * 4 // 5),
    ]
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
    """Return direction if a center point moved through a horizontal line."""
    if previous_y < line_y <= current_y:
        return "north_to_south"
    if previous_y > line_y >= current_y:
        return "south_to_north"
    return None


def update_track(
    tracks: dict[int, TrackState], track_id: int, center_y: float, line_y: int, seen_at: float
) -> str | None:
    """Update a track and return its first crossing direction, if any."""
    previous = tracks.get(track_id)
    direction = crossed_line(previous.center_y, center_y, line_y) if previous else None
    counted = previous.counted if previous else False
    tracks[track_id] = TrackState(center_y=center_y, last_seen=seen_at, counted=counted)
    if direction and not counted:
        tracks[track_id].counted = True
        return direction
    return None


def _log_write_result(future: Future[int | None], vehicle_id: str) -> None:
    """Report asynchronous database outcomes without interrupting inference."""
    try:
        inserted_id = future.result()
        if inserted_id is None:
            LOG.warning("duplicate event prevented for track %s", vehicle_id)
    except Exception:
        LOG.exception("could not persist crossing event for track %s", vehicle_id)


def run(
    *,
    source: str,
    line_ratio: float,
    show: bool,
    model_path: str,
    repository: EventRepository,
    max_frames: int | None = None,
) -> None:
    """Run YOLO + ByteTrack against a file or RTSP stream and persist crossings."""
    if not 0 < line_ratio < 1:
        raise ValueError("line_ratio must be between 0 and 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be a positive integer")

    # Delayed import keeps pure geometry/color tests fast and independent of Torch.
    from ultralytics import YOLO

    repository.initialize()
    model = YOLO(model_path)
    run_id = uuid.uuid4().hex[:12]
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
    LOG.info("pipeline run %s started from %s", run_id, source)
    try:
        for frame_number, result in enumerate(results, start=1):
            frame = result.orig_img
            now = time.monotonic()
            line_y = int(frame.shape[0] * line_ratio)
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                coordinates = boxes.xyxy.cpu().numpy().astype(int)
                identifiers = boxes.id.int().cpu().tolist()
                class_ids = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                for (x1, y1, x2, y2), tracker_id, class_id, confidence in zip(
                    coordinates, identifiers, class_ids, confidences
                ):
                    center_y = (y1 + y2) / 2
                    direction = update_track(tracks, tracker_id, center_y, line_y, now)
                    x1, x2 = max(0, x1), min(frame.shape[1], x2)
                    y1, y2 = max(0, y1), min(frame.shape[0], y2)
                    vehicle_type = VEHICLE_CLASSES.get(class_id, "vehicle")
                    if direction:
                        color = classify_color(frame[y1:y2, x1:x2])
                        future = executor.submit(
                            repository.insert_crossing,
                            run_id=run_id,
                            tracker_id=tracker_id,
                            vehicle_type=vehicle_type,
                            color=color,
                            direction=direction,
                            confidence=float(confidence),
                            crossed_at=datetime.now(timezone.utc),
                        )
                        vehicle_id = f"{run_id}:{tracker_id}"
                        future.add_done_callback(
                            lambda completed, key=vehicle_id: _log_write_result(completed, key)
                        )
                        LOG.info("counted %s: %s %s moving %s", vehicle_id, color, vehicle_type, direction)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 210, 70), 2)
                    cv2.putText(
                        frame,
                        f"{vehicle_type} #{tracker_id}",
                        (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (30, 210, 70),
                        2,
                    )

            cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 220, 255), 2)
            stale_ids = [track_id for track_id, state in tracks.items() if now - state.last_seen > 60]
            for stale_id in stale_ids:
                del tracks[stale_id]
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
    """Parse command-line options and start the inference pipeline."""
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Detect, track, classify, and count vehicles.")
    parser.add_argument("--source", default=settings.rtsp_url, help="RTSP URL, video path, or camera input")
    parser.add_argument("--line-ratio", type=float, default=settings.line_ratio)
    parser.add_argument("--model", default=settings.yolo_model, help="Ultralytics model path or name")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    parser.add_argument("--show", action="store_true", help="Show annotated video; press q to stop")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        run(
            source=args.source,
            line_ratio=args.line_ratio,
            show=args.show,
            model_path=args.model,
            repository=EventRepository(settings.database_url),
            max_frames=args.max_frames,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
