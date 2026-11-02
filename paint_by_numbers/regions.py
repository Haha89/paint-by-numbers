# -*- coding: utf-8 -*-
"""Smoothing of the color borders, and detection and merging of the regions"""

import cv2
import numpy as np

from .colors import rgb_to_lab


def vote(labels, nb_color, score_function):
    """Each pixel takes the color with the highest score, score_function maps a color mask to its scores"""
    best, voted = None, None
    for ind in range(nb_color):
        score = score_function(cv2.compare(labels, ind, cv2.CMP_EQ))
        if best is None:
            best, voted = score, np.zeros(score.shape, dtype=np.uint8)
            continue
        better = score > best
        np.copyto(best, score, where=better)
        np.copyto(voted, ind, where=better)
    return voted


def smooth_labels(labels, nb_color, smoothing_ratio):
    """Smooth the borders: each pixel takes the color most present around it, weighted by a gaussian"""
    sigma = smoothing_ratio * max(labels.shape)
    if sigma <= 0:
        return labels
    return vote(labels, nb_color, lambda mask: cv2.GaussianBlur(mask, (0, 0), sigma))


def merge_small_regions(labels, colors, min_region_ratio, min_width_ratio, max_iter=3, protected=None,
                        protected_scale=1):
    """Give the regions too small or too thin to be painted the color of their closest large neighbour

    The minimum area and width are multiplied by protected_scale for the regions mostly inside the protected mask, to
    keep small details which matter, like the eyes of a face.

    Returns the merged labels, their borders, the distance to the borders and the regions (see label_regions)
    """
    lab = rgb_to_lab(colors)
    min_area = min_region_ratio * labels.size
    min_radius = min_width_ratio * max(labels.shape) / 2
    for iteration in range(max_iter + 1):
        borders = find_borders(labels)
        dist = distance_to_borders(borders)
        regions = label_regions(labels, len(colors))
        if iteration == max_iter:
            break
        ids, region_colors, stats = regions
        # A region is wide enough if one of its pixels is farther than min_radius from any border
        is_wide = np.zeros(len(stats), dtype=bool)
        is_wide[ids[dist >= min_radius]] = True
        small_regions = (stats[:, cv2.CC_STAT_AREA] < min_area) | ~is_wide
        if protected is not None:
            inside = np.bincount(ids[protected], minlength=len(stats)) > stats[:, cv2.CC_STAT_AREA] / 2
            is_wide_enough = np.zeros(len(stats), dtype=bool)
            is_wide_enough[ids[dist >= protected_scale * min_radius]] = True
            small_inside = (stats[:, cv2.CC_STAT_AREA] < protected_scale * min_area) | ~is_wide_enough
            small_regions = np.where(inside, small_inside, small_regions)
        small = small_regions[ids]
        if not small.any() or small.all():
            break
        labels = labels.copy()
        new_colors = closest_color_neighbours(ids, region_colors, small_regions, lab)[ids]
        assigned = small & (new_colors >= 0)
        labels[assigned] = new_colors[assigned]
        # Small regions with no large neighbour: each pixel gets the index (in raster order, from 1)
        # of the closest pixel of another region
        remaining = small & ~assigned
        if remaining.any():
            _, closest = cv2.distanceTransformWithLabels(remaining.view(np.uint8), cv2.DIST_L2, 5,
                                                         labelType=cv2.DIST_LABEL_PIXEL)
            labels[remaining] = labels[~remaining][closest[remaining] - 1]
    return labels, borders, dist, regions


def closest_color_neighbours(ids, region_colors, small_regions, lab):
    """For each small region, the color of its large neighbour with the closest color (-1 if none)"""
    nb_regions = len(region_colors)
    firsts, seconds = [], []
    for first, second in ((ids[:, :-1], ids[:, 1:]), (ids[:-1], ids[1:])):
        touching = first != second
        firsts += [first[touching], second[touching]]
        seconds += [second[touching], first[touching]]
    small, neighbour = np.concatenate(firsts), np.concatenate(seconds)
    keep = small_regions[small] & ~small_regions[neighbour]
    # Unique pairs of (small region, large neighbour)
    pairs = np.unique(small[keep].astype(np.int64) * nb_regions + neighbour[keep])
    small, neighbour = pairs // nb_regions, pairs % nb_regions
    distance = np.linalg.norm(lab[region_colors[small]] - lab[region_colors[neighbour]], axis=1)
    # Sort by small region, then by color distance: the first pair of each small region is the closest
    order = np.lexsort((distance, small))
    small, neighbour = small[order], neighbour[order]
    _, first_pairs = np.unique(small, return_index=True)
    new_colors = np.full(nb_regions, -1, dtype=np.int16)
    new_colors[small[first_pairs]] = region_colors[neighbour[first_pairs]]
    return new_colors


def label_regions(labels, nb_color):
    """Split the picture in connected regions of a single color

    Returns the region id of each pixel, and the color index and stats (x, y, width, height, area) of each region
    """
    ids = np.zeros(labels.shape, dtype=np.int32)
    offsets = np.zeros(nb_color, dtype=np.int32)
    region_colors, region_stats = [], []
    for ind in range(nb_color):
        nb_comp, components, stats, _ = cv2.connectedComponentsWithStats(cv2.compare(labels, ind, cv2.CMP_EQ),
                                                                         connectivity=4)
        # Each pixel has a single color, so it only receives the id of its own component
        ids += components
        offsets[ind] = len(region_colors) - 1
        region_colors += [ind] * (nb_comp - 1)
        region_stats.append(stats[1:])
    ids += offsets[labels]
    return ids, np.array(region_colors), np.concatenate(region_stats)


def distance_to_borders(borders):
    """Distance of each pixel to the closest border"""
    return cv2.distanceTransform((~borders).view(np.uint8), cv2.DIST_L2, 5)


def find_borders(labels):
    """Pixels where the color changes, the picture frame included"""
    borders = np.zeros(labels.shape, dtype=bool)
    borders[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    borders[:-1] |= labels[:-1] != labels[1:]
    borders[[0, -1]] = True
    borders[:, [0, -1]] = True
    return borders


def find_regions(ids, region_colors, stats, dist, label_spacing=0, min_repeat_radius=0):
    """List the regions, with the spots where to put their label

    The first spot is the point farthest from the borders. Then, while label_spacing is positive, the next farthest
    point at least label_spacing away from the other spots, if its radius is at least min_repeat_radius
    """
    regions = []
    for region_id, (ind, (x, y, width, height, area)) in enumerate(zip(region_colors, stats)):
        local = np.where(ids[y:y + height, x:x + width] == region_id, dist[y:y + height, x:x + width], 0)
        spots = []
        while True:
            py, px = np.unravel_index(np.argmax(local), local.shape)
            radius = float(local[py, px])
            if spots and radius < max(min_repeat_radius, np.finfo(np.float32).tiny):
                break
            spots.append({"x": int(x + px), "y": int(y + py), "radius": radius})
            if label_spacing <= 0:
                break
            cv2.circle(local, (int(px), int(py)), int(np.ceil(label_spacing)), 0, -1)
        regions.append({"label": int(ind) + 1, "area": int(area), "spots": spots})
    return regions
