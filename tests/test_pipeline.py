import numpy as np

from pipeline import classify_color, crossed_line


def test_crossed_line_reports_both_directions():
    assert crossed_line(10, 30, 20) == "north_to_south"
    assert crossed_line(30, 10, 20) == "south_to_north"
    assert crossed_line(10, 15, 20) is None


def test_color_classifier_handles_neutral_colours():
    assert classify_color(np.full((30, 30, 3), 240, dtype=np.uint8)) == "white"
    assert classify_color(np.full((30, 30, 3), 15, dtype=np.uint8)) == "black"
