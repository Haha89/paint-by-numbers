# -*- coding: utf-8 -*-

import re
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest

from paint_by_numbers import Canvas
from paint_by_numbers.export import MAX_LABEL_SIZE, MIN_LABEL_SIZE, colormap_picture, label_size, printed_size

ROOT = Path(__file__).parent.parent
PNGS = {"sample-result.png", "sample-canvas.png", "sample-colormap.png"}


def names(folder):
    return {path.name for path in folder.iterdir()}


def run_cli(*args):
    return subprocess.run([sys.executable, str(ROOT / "run.py"), *map(str, args)], capture_output=True, text=True,
                          cwd=ROOT, check=False)


def test_save_writes_the_pictures_and_the_pdf(generated, tmp_path):
    generated.save(tmp_path / "new" / "folder")
    assert names(tmp_path / "new" / "folder") == PNGS | {"sample.pdf"}
    assert cv2.imread(str(tmp_path / "new" / "folder" / "sample-canvas.png")).shape[:2] == (400, 600)


def test_save_without_pdf(generated, tmp_path):
    generated.save(tmp_path, pdf=False)
    assert names(tmp_path) == PNGS


@pytest.mark.parametrize("paper", sorted(Canvas.paper_sizes))
def test_pdf_has_two_pages(generated, tmp_path, paper):
    generated.save(tmp_path, title="A title", subtitle="A subtitle", paper=paper)
    content = (tmp_path / "sample.pdf").read_bytes()
    assert content.startswith(b"%PDF")
    assert content.count(b"/Type /Page\n") == 2 or content.count(b"/Type /Page ") == 2


def page_boxes(path):
    """The (width, height) in points of the pages of a PDF, in the order of their size"""
    boxes = re.findall(rb"/MediaBox \[ *0 +0 +([\d.]+) +([\d.]+) *\]", path.read_bytes())
    return sorted((float(width), float(height)) for width, height in boxes)


@pytest.mark.parametrize("radius, expected", [(0, MIN_LABEL_SIZE), (1, MIN_LABEL_SIZE), (1000, MAX_LABEL_SIZE)])
def test_label_size_stays_within_its_bounds(radius, expected):
    assert label_size(radius, 0.005) == expected


def test_label_size_grows_with_the_region_between_its_bounds():
    assert MIN_LABEL_SIZE < label_size(20, 0.005) < MAX_LABEL_SIZE


def test_print_size_only_sets_the_canvas_page(generated, tmp_path):
    generated.save(tmp_path, paper="letter", print_size="16x20")
    # Letter for the palette, then the landscape picture on a landscape 20x16 page
    assert page_boxes(tmp_path / "sample.pdf") == [(8.5 * 72, 11 * 72), (20 * 72, 16 * 72)]


def test_printed_size_fits_the_canvas_in_its_orientation():
    assert printed_size((400, 600), "16x20") == pytest.approx((20, 40 / 3))
    assert printed_size((600, 400), "16x20") == pytest.approx((40 / 3, 20))
    assert printed_size((750, 600), "16x20") == pytest.approx((16, 20))


def test_printed_size_on_the_paper_leaves_the_margins_and_the_header():
    # A landscape picture on a landscape Letter page: 11 - 2 x 0.6 inches wide
    assert printed_size((400, 600), paper="letter") == pytest.approx((9.8, 9.8 * 2 / 3))


def test_cli_print_size(picture_path, tmp_path):
    result = run_cli(picture_path, 5, "--pixel-size", 600, "--print-size", "8x10", "--output-dir", tmp_path)
    assert result.returncode == 0, result.stderr
    assert (10 * 72, 8 * 72) in page_boxes(tmp_path / "sample.pdf")


def test_cli_rejects_an_invalid_print_size(picture_path):
    assert run_cli(picture_path, 5, "--print-size", "1x1").returncode != 0


def test_colormap_picture_is_rgb_and_grows_with_the_palette():
    small = colormap_picture([[10, 20, 30]] * 2)
    large = colormap_picture([[10, 20, 30]] * 6)
    assert small.ndim == 3 and small.shape[2] == 3
    assert large.shape[0] > small.shape[0]


def test_plot_shows_three_figures(generated, monkeypatch):
    shown = []
    monkeypatch.setattr(Canvas, "plot_figure", staticmethod(lambda image, title: shown.append(title)))
    generated.plot()
    assert shown == ["Expected Result", "Canvas", "Colormap"]


def test_plot_figure_shows_the_picture(generated, monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(plt.gca().get_title()))
    Canvas.plot_figure(generated.canvas, "Canvas")
    assert shown == ["Canvas"]


def test_cli_saves_the_results_with_the_options(picture_path, tmp_path):
    result = run_cli(picture_path, 5, "--pixel-size", 600, "--detail", "easy", "--output-dir", tmp_path)
    assert result.returncode == 0, result.stderr
    assert names(tmp_path) == PNGS | {"sample.pdf"}
    expected = Canvas(picture_path, 5, pixel_size=600, detail="easy").generate()
    assert np.array_equal(cv2.imread(str(tmp_path / "sample-canvas.png"), cv2.IMREAD_GRAYSCALE), expected)


def test_cli_no_pdf(picture_path, tmp_path):
    result = run_cli(picture_path, 5, "--pixel-size", 600, "--no-pdf", "--output-dir", tmp_path)
    assert result.returncode == 0, result.stderr
    assert names(tmp_path) == PNGS


def test_cli_rejects_an_invalid_detail(picture_path):
    result = run_cli(picture_path, 5, "--detail", "hard")
    assert result.returncode != 0
    assert "--detail" in result.stderr
