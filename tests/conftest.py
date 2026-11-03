# -*- coding: utf-8 -*-
"""Small generated pictures, so that the tests are fast and do not depend on the examples"""

import os

import cv2
import matplotlib
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from paint_by_numbers import Canvas

matplotlib.use("Agg")

# With pytest-xdist, each worker gets its share of the cores: OpenCV and the K-mean would otherwise each use all of
# them in every worker, and the slowed down previews would miss the timeouts of the browser tests
if "PYTEST_XDIST_WORKER_COUNT" in os.environ:
    _THREADS = max(1, (os.cpu_count() or 1) // int(os.environ["PYTEST_XDIST_WORKER_COUNT"]))
    cv2.setNumThreads(_THREADS)
    threadpool_limits(_THREADS)


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
