from datetime import datetime, timezone

import numpy as np

from db import EventRepository
from pipeline import classify_color, crossed_line, run, update_track


def test_crossing_direction_and_boundary_behavior():
    assert crossed_line(10, 30, 20) == "north_to_south"
    assert crossed_line(30, 10, 20) == "south_to_north"
    assert crossed_line(10, 15, 20) is None
    assert crossed_line(20, 25, 20) is None


def test_track_is_counted_only_once_even_if_it_recrosses():
    tracks = {}
    assert update_track(tracks, 4, 10, 20, 1.0) is None
    assert update_track(tracks, 4, 30, 20, 2.0) == "north_to_south"
    assert update_track(tracks, 4, 10, 20, 3.0) is None
    assert tracks[4].counted is True


def test_color_classifier_identifies_neutral_and_colored_crops():
    assert classify_color(np.full((30, 30, 3), 240, dtype=np.uint8)) == "white"
    assert classify_color(np.full((30, 30, 3), 15, dtype=np.uint8)) == "black"
    assert classify_color(np.full((30, 30, 3), 110, dtype=np.uint8)) == "gray"
    assert classify_color(np.full((30, 30, 3), (0, 0, 255), dtype=np.uint8)) == "red"
    assert classify_color(np.empty((0, 0, 3), dtype=np.uint8)) == "unknown"


def test_repository_prevents_duplicate_track_events_and_returns_dashboard_data(tmp_path):
    repository = EventRepository(f"sqlite+pysqlite:///{tmp_path / 'analytics.db'}")
    repository.initialize()
    first_id = repository.insert_crossing(
        run_id="run-a",
        tracker_id=42,
        vehicle_type="car",
        color="blue",
        direction="north_to_south",
        confidence=0.91,
        crossed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    duplicate_id = repository.insert_crossing(
        run_id="run-a",
        tracker_id=42,
        vehicle_type="car",
        color="blue",
        direction="north_to_south",
        confidence=0.91,
    )
    total, colors, recent = repository.dashboard_data()

    assert first_id == 1
    assert duplicate_id is None
    assert total == 1
    assert colors == {"blue": 1}
    assert recent[0]["Track ID"] == "run-a:42"


class _Tensor:
    """Small stand-in for the Torch tensor methods used by the pipeline."""

    def __init__(self, value):
        self.value = np.asarray(value)

    def cpu(self):
        return self

    def numpy(self):
        return self.value

    def int(self):
        return self

    def tolist(self):
        return self.value.tolist()


class _Boxes:
    def __init__(self, y1, y2):
        self.xyxy = _Tensor([[10, y1, 30, y2]])
        self.id = _Tensor([7])
        self.cls = _Tensor([2])
        self.conf = _Tensor([0.9])


class _Result:
    def __init__(self, y1, y2):
        self.orig_img = np.full((60, 80, 3), 120, dtype=np.uint8)
        self.boxes = _Boxes(y1, y2)


class _FakeModel:
    def track(self, **_kwargs):
        return iter([_Result(5, 15), _Result(25, 35)])


class _RecordingRepository:
    def __init__(self):
        self.initialized = False
        self.events = []

    def initialize(self):
        self.initialized = True

    def insert_crossing(self, **event):
        self.events.append(event)
        return len(self.events)


def test_run_persists_one_event_for_a_mocked_tracked_crossing(monkeypatch):
    import ultralytics

    monkeypatch.setattr(ultralytics, "YOLO", lambda _path: _FakeModel())
    repository = _RecordingRepository()

    run(
        source="synthetic",
        line_ratio=1 / 3,
        show=False,
        model_path="mock.pt",
        repository=repository,
        max_frames=2,
    )

    assert repository.initialized is True
    assert len(repository.events) == 1
    assert repository.events[0]["tracker_id"] == 7
    assert repository.events[0]["direction"] == "north_to_south"
