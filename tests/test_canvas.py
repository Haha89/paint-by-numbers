# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pytest

from paint_by_numbers import Canvas


def nb_regions(picture_path, detail):
    canvas = Canvas(picture_path, 5, pixel_size=600, detail=detail)
    canvas.generate()
    return len(canvas.regions)


def test_outputs_match_the_target_size(generated):
    assert generated.canvas.shape == (400, 600)
    assert generated.quantified_image.shape == (400, 600, 3)
    assert generated.borders.shape == (400, 600)


def test_result_only_uses_the_palette(generated):
    used = np.unique(generated.quantified_image.reshape(-1, 3), axis=0)
    assert 1 < len(used) <= generated.nb_color
    assert sorted(map(tuple, used.tolist())) == sorted(map(tuple, generated.colormap))


def test_every_region_has_a_label_on_its_color(generated):
    assert generated.regions
    for region in generated.regions:
        assert region["spots"]
        color = generated.colormap[region["label"] - 1]
        for spot in region["spots"]:
            assert generated.quantified_image[spot["y"], spot["x"]].tolist() == color


def test_regions_cover_the_picture(generated):
    # The borders are drawn on the pixels of the regions, which keep their area
    assert sum(region["area"] for region in generated.regions) == 400 * 600


def test_repeated_labels_respect_the_spacing(generated):
    spacing = generated.label_spacing_ratio * 600
    repeated = [region for region in generated.regions if len(region["spots"]) > 1]
    assert repeated
    for region in repeated:
        points = np.array([(spot["x"], spot["y"]) for spot in region["spots"]])
        gaps = np.linalg.norm(points[:, None] - points[None], axis=2)
        np.fill_diagonal(gaps, np.inf)
        assert gaps.min() >= spacing


def test_repeats_can_be_disabled(picture_path):
    canvas = Canvas(picture_path, 5, pixel_size=600, label_spacing_ratio=0)
    canvas.generate()
    assert all(len(region["spots"]) == 1 for region in canvas.regions)


def test_detail_levels_order_the_number_of_regions(picture_path):
    assert (nb_regions(picture_path, "easy") < nb_regions(picture_path, "medium")
            <= nb_regions(picture_path, "faithful"))


@pytest.mark.parametrize("ratio", ["min_region_ratio", "min_width_ratio"])
def test_detail_levels_order_the_thresholds(ratio):
    levels = Canvas.detail_levels
    assert levels["easy"][ratio] > levels["medium"][ratio] > levels["faithful"][ratio]


def test_medium_is_the_default_detail(picture_path):
    canvas = Canvas(picture_path, 5)
    preset = Canvas.detail_levels["medium"]
    assert canvas.min_region_ratio == preset["min_region_ratio"]
    assert canvas.min_width_ratio == preset["min_width_ratio"]
    assert canvas.smoothing_ratio == preset["smoothing_ratio"]


def test_explicit_ratio_overrides_the_detail(picture_path):
    canvas = Canvas(picture_path, 5, detail="easy", min_region_ratio=0.5)
    assert canvas.min_region_ratio == 0.5
    assert canvas.min_width_ratio == Canvas.detail_levels["easy"]["min_width_ratio"]


def test_generation_is_reproducible(picture_path, generated):
    canvas = Canvas(picture_path, 5, pixel_size=600)
    canvas.generate()
    assert np.array_equal(canvas.canvas, generated.canvas)


def test_portrait_picture_keeps_its_orientation(tmp_path, picture_path):
    path = tmp_path / "portrait.png"
    cv2.imwrite(str(path), cv2.rotate(cv2.imread(str(picture_path)), cv2.ROTATE_90_CLOCKWISE))
    canvas = Canvas(path, 5, pixel_size=600)
    canvas.generate()
    assert canvas.canvas.shape == (600, 400)


@pytest.mark.parametrize("nb_color", [0, 257])
def test_invalid_nb_color_is_rejected(picture_path, nb_color):
    with pytest.raises(ValueError, match="nb_color"):
        Canvas(picture_path, nb_color)


def test_invalid_detail_is_rejected(picture_path):
    with pytest.raises(ValueError, match="detail"):
        Canvas(picture_path, 5, detail="hard")


def test_missing_picture_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        Canvas(tmp_path / "missing.png", 5)
