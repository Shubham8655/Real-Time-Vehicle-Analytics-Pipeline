import numpy as np

from pipeline import classify_color, crossed_line


def test_crossing_direction_and_boundary_behavior():
    assert crossed_line(10, 30, 20) == "north_to_south"
    assert crossed_line(30, 10, 20) == "south_to_north"
    assert crossed_line(10, 15, 20) is None
    assert crossed_line(20, 25, 20) is None


def test_color_classifier_identifies_neutral_colors():
    assert classify_color(np.full((30, 30, 3), 240, dtype=np.uint8)) == "white"
    assert classify_color(np.full((30, 30, 3), 15, dtype=np.uint8)) == "black"
    assert classify_color(np.full((30, 30, 3), 110, dtype=np.uint8)) == "gray"


def test_color_classifier_handles_empty_crop():
    assert classify_color(np.empty((0, 0, 3), dtype=np.uint8)) == "unknown"
