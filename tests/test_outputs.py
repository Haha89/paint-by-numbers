# -*- coding: utf-8 -*-

import re
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from paint_by_numbers import Canvas
from paint_by_numbers.export import (MAX_LABEL_SIZE, MIN_LABEL_SIZE, PAPER_SIZES, colormap_picture,
                                     crowded_label_share, fill_crop, label_size, paint_volumes, palette_page,
                                     printed_size)

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


@pytest.mark.parametrize("print_size, paper", [("24x36", "a4"), ("8x10", "a4"), (None, "a3")])
def test_saved_pictures_open_at_their_printed_size(generated, tmp_path, print_size, paper):
    generated.save(tmp_path, pdf=False, paper=paper, print_size=print_size)
    expected = printed_size(generated.canvas.shape, print_size, paper)
    for suffix in ("canvas", "result"):
        with Image.open(tmp_path / f"sample-{suffix}.png") as picture:
            width_dpi, height_dpi = picture.info["dpi"]
            assert picture.size[0] / width_dpi == pytest.approx(expected[0], rel=1e-3)
            assert picture.size[1] / height_dpi == pytest.approx(expected[1], rel=1e-3)
    with Image.open(tmp_path / "sample-colormap.png") as picture:
        assert picture.info["dpi"] == pytest.approx((200, 200), rel=1e-3)


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


@pytest.mark.parametrize("width, height, expected", [(600, 400, (1 / 12, 0, 11 / 12, 1)), (400, 600, (0, 1 / 12, 1, 11 / 12)),
                                                     (600, 300, (0.1875, 0, 0.8125, 1)),
                                                     (500, 400, (0, 0, 1, 1))])
def test_fill_crop_gives_the_shape_of_the_page_in_its_orientation(width, height, expected):
    assert fill_crop(width, height, "16x20") == pytest.approx(expected)


def test_crop_to_print_makes_the_picture_fill_the_page(picture_path):
    canvas = Canvas(picture_path, 5, pixel_size=600)
    canvas.crop_to_print("16x20")
    canvas.generate()
    width, height = printed_size(canvas.borders.shape, "16x20")
    assert (width, height) == pytest.approx((20, 16), abs=0.05)


def test_cli_crop_to_fill(picture_path, tmp_path):
    result = run_cli(picture_path, 5, "--pixel-size", 600, "--print-size", "16x20", "--crop-to-fill", "--output-dir",
                     tmp_path)
    assert result.returncode == 0, result.stderr
    assert cv2.imread(str(tmp_path / "sample-canvas.png")).shape[:2] == (480, 600)


def test_cli_crop_to_fill_needs_a_print_size(picture_path):
    result = run_cli(picture_path, 5, "--crop-to-fill")
    assert result.returncode != 0
    assert "--print-size" in result.stderr


def test_coverage_describes_each_color(generated):
    assert len(generated.coverage) == len(generated.colormap)
    assert sum(generated.coverage) == pytest.approx(1)
    assert min(generated.coverage) > 0


def test_paint_volumes_follow_the_area_and_the_share():
    # 10 x 10 in is 645 cm2, so about 13 ml at 50 cm2 per ml
    assert paint_volumes([0.5, 0.5], (10, 10)) == pytest.approx([6.4516, 6.4516])
    assert paint_volumes([1], (20, 20))[0] == pytest.approx(4 * paint_volumes([1], (10, 10))[0])


def texts_of(figure):
    return [text.get_text() for text in figure.axes[0].texts]


def test_palette_page_shows_the_paint_for_the_printed_size(generated):
    small = texts_of(palette_page(generated, PAPER_SIZES["letter"], "A title", "", "8x10"))
    large = texts_of(palette_page(generated, PAPER_SIZES["letter"], "A title", "", "24x36"))
    share = f"{generated.coverage[0]:.1%}"
    assert any(text.startswith(f"{share}, ") and text.endswith(" ml") for text in small)
    assert sum(text.endswith(" ml") for text in small) == len(generated.colormap)
    total = [text for text in small if text.startswith("Estimated paint")]
    assert len(total) == 1
    assert total != [text for text in large if text.startswith("Estimated paint")]


def test_crowded_label_share_depends_on_the_printed_size():
    regions = [{"label": 12, "spots": [{"x": 0, "y": 0, "radius": 8.0}]}]
    assert crowded_label_share(regions, (2000, 3000), print_size="8x10") == 1
    assert crowded_label_share(regions, (2000, 3000), print_size="24x36") == 0
    assert crowded_label_share([], (2000, 3000)) == 0


def test_crowded_label_share_inside_a_mask():
    regions = [{"label": 12, "spots": [{"x": 10, "y": 10, "radius": 8.0}]},
               {"label": 3, "spots": [{"x": 2900, "y": 1900, "radius": 500.0}]}]
    mask = np.zeros((2000, 3000), dtype=bool)
    assert crowded_label_share(regions, mask.shape, print_size="8x10") == 0.5
    mask[:100, :100] = True
    assert crowded_label_share(regions, mask.shape, print_size="8x10", mask=mask) == 1
    assert crowded_label_share(regions, mask.shape, print_size="8x10", mask=~mask) == 0
