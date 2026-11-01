# -*- coding: utf-8 -*-
"""Small generated pictures, so that the tests are fast and do not depend on the examples"""

import cv2
import matplotlib
import numpy as np
import pytest

from paint_by_numbers import Canvas

matplotlib.use("Agg")


@pytest.fixture(scope="session")
def picture_path(tmp_path_factory):
    """A 600x400 landscape with a large background and spots of 5 colors, from tiny to large"""
    colors = [(40, 40, 200), (40, 200, 40), (200, 40, 40), (40, 200, 200)]
    rng = np.random.default_rng(0)
    picture = np.full((400, 600, 3), 230, dtype=np.uint8)
    for _ in range(150):
        center = (int(rng.integers(600)), int(rng.integers(400)))
        cv2.circle(picture, center, int(rng.integers(2, 30)), colors[rng.integers(len(colors))], -1)
    # A large plain area, far from the spots
    picture[:150] = 230
    path = tmp_path_factory.mktemp("pictures") / "sample.png"
    cv2.imwrite(str(path), picture)
    return path


@pytest.fixture(scope="session")
def generated(picture_path):
    """A generated canvas, shared by the tests which only read it"""
    canvas = Canvas(picture_path, 5, pixel_size=600)
    canvas.generate()
    return canvas
