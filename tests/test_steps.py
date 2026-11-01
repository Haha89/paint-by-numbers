# -*- coding: utf-8 -*-

import numpy as np

from paint_by_numbers.colors import merge_close_colors, rgb_to_lab, sort_palette
from paint_by_numbers.preprocessing import resize, target_dim
from paint_by_numbers.regions import (distance_to_borders, find_borders, find_regions, label_regions,
                                      merge_small_regions)


def test_target_dim_respects_the_ratio():
    assert target_dim((400, 600), 300) == (300, 200)
    assert target_dim((600, 400), 300) == (200, 300)


def test_resize_matches_the_largest_dimension():
    assert resize(np.zeros((400, 600, 3), dtype=np.uint8), 1200).shape == (800, 1200, 3)


def test_rgb_to_lab_extremes():
    black, white = rgb_to_lab(np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8))
    assert abs(black[0]) < 0.01
    assert abs(white[0] - 100) < 0.01


def test_merge_close_colors_averages_by_number_of_pixels():
    centers = np.array([[50, 0, 0], [52, 0, 0], [90, 0, 0]], dtype=np.float32)
    lut, merged = merge_close_colors(centers, np.array([300, 100, 50]), min_color_distance=6)
    assert lut.tolist() == [0, 0, 1]
    assert np.allclose(merged, [[50.5, 0, 0], [90, 0, 0]])


def test_merge_close_colors_keeps_distant_colors():
    centers = np.array([[20, 0, 0], [50, 0, 0], [80, 0, 0]], dtype=np.float32)
    lut, merged = merge_close_colors(centers, np.array([1, 1, 1]), min_color_distance=6)
    assert lut.tolist() == [0, 1, 2]
    assert len(merged) == 3


def test_sort_palette_removes_unused_colors_and_keeps_the_pixels_colors():
    colors = np.array([[200, 30, 30], [10, 10, 10], [250, 250, 250], [30, 200, 30]], dtype=np.uint8)
    labels = np.array([[0, 1], [2, 2]], dtype=np.uint8)
    new_labels, new_colors, region_colors = sort_palette(labels, colors, np.array([0, 1, 2]))
    assert len(new_colors) == 3
    assert np.array_equal(new_colors[new_labels], colors[labels])
    assert np.array_equal(new_colors[region_colors], colors[[0, 1, 2]])
    # Greys first, from light to dark, then the colors
    assert new_colors.tolist() == [[250, 250, 250], [10, 10, 10], [200, 30, 30]]


def test_find_borders_includes_the_frame_and_the_color_changes():
    labels = np.zeros((5, 6), dtype=np.uint8)
    labels[:, 3:] = 1
    borders = find_borders(labels)
    assert borders[0].all() and borders[-1].all()
    assert borders[:, 0].all() and borders[:, -1].all()
    assert borders[2].tolist() == [True, False, True, False, False, True]


def test_label_regions_splits_disconnected_areas_of_a_color():
    labels = np.array([[0, 1, 0],
                       [0, 1, 0]], dtype=np.uint8)
    ids, region_colors, stats = label_regions(labels, 2)
    assert region_colors.tolist() == [0, 0, 1]
    assert ids[0, 0] != ids[0, 2]
    assert ids.min() == 0 and ids.max() == 2
    assert stats[:, 4].tolist() == [2, 2, 2]


def test_label_regions_does_not_connect_diagonals():
    labels = np.array([[0, 1],
                       [1, 0]], dtype=np.uint8)
    _, region_colors, _ = label_regions(labels, 2)
    assert len(region_colors) == 4


def test_merge_small_regions_gives_a_speck_its_closest_neighbour_color():
    # A speck of color 1 on the frontier of a color 2 close to it, and of a color 0 far from it
    colors = np.array([[0, 0, 0], [200, 200, 200], [230, 230, 230]], dtype=np.uint8)
    labels = np.zeros((100, 100), dtype=np.uint8)
    labels[:, 50:] = 2
    labels[48:52, 48:52] = 1
    merged, *_ = merge_small_regions(labels, colors, min_region_ratio=0.01, min_width_ratio=0)
    assert merged[50, 49] == 2
    assert set(np.unique(merged)) == {0, 2}


def test_merge_small_regions_keeps_large_regions():
    colors = np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
    labels = np.zeros((100, 100), dtype=np.uint8)
    labels[:, 50:] = 1
    merged, *_ = merge_small_regions(labels, colors, min_region_ratio=0.01, min_width_ratio=0.01)
    assert np.array_equal(merged, labels)


def regions_of(labels, spacing=0, min_radius=0):
    ids, region_colors, stats = label_regions(labels, labels.max() + 1)
    return find_regions(ids, region_colors, stats, distance_to_borders(find_borders(labels)), spacing, min_radius)


def test_find_regions_puts_the_label_at_the_center():
    region, = regions_of(np.zeros((101, 101), dtype=np.uint8))
    assert region["label"] == 1
    assert region["area"] == 101 * 101
    spot, = region["spots"]
    assert (spot["x"], spot["y"]) == (50, 50)


def test_find_regions_repeats_the_label_along_a_long_region():
    region, = regions_of(np.zeros((21, 401), dtype=np.uint8), spacing=100, min_radius=5)
    assert len(region["spots"]) >= 3
    assert all(region["spots"][0]["y"] - 3 <= spot["y"] <= region["spots"][0]["y"] + 3 for spot in region["spots"])


def test_find_regions_does_not_repeat_where_a_label_does_not_fit():
    region, = regions_of(np.zeros((5, 401), dtype=np.uint8), spacing=100, min_radius=5)
    assert len(region["spots"]) == 1
