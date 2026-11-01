# -*- coding: utf-8 -*-
"""Search of the colors summarizing a picture, and sorting of the palette"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


def rgb_to_lab(colors):
    """Convert a list of uint8 RGB colors to CIELAB"""
    return cv2.cvtColor(colors.reshape(1, -1, 3).astype(np.float32) / 255, cv2.COLOR_RGB2Lab)[0]


def merge_close_colors(centers, counts, min_color_distance):
    """Merge the CIELAB colors closer than min_color_distance, averaged by their number of pixels

    Returns the lookup table from the old to the new color indices, and the new colors
    """
    centers, counts = centers.astype(np.float64), counts.astype(np.float64)
    lut, active = np.arange(len(centers)), list(range(len(centers)))
    while len(active) > 1:
        distance = np.linalg.norm(centers[active][:, None] - centers[active][None], axis=2)
        np.fill_diagonal(distance, np.inf)
        i, j = np.unravel_index(np.argmin(distance), distance.shape)
        if distance[i, j] >= min_color_distance:
            break
        keep, drop = active[i], active[j]
        total = counts[keep] + counts[drop]
        if total > 0:
            centers[keep] = (centers[keep] * counts[keep] + centers[drop] * counts[drop]) / total
        counts[keep] = total
        lut[lut == drop] = keep
        active.remove(drop)
    compact = np.zeros(len(centers), dtype=np.int64)
    compact[active] = np.arange(len(active))
    return compact[lut].astype(np.uint8), centers[active]


def quantification(picture, nb_color, nb_samples, min_color_distance):
    """Returns the color index of each pixel and the K-mean colors

    The K-mean runs in the CIELAB color space, where distances match the perceived color differences
    """
    height, width, dimension = picture.shape
    lab = cv2.cvtColor(picture.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    image_array = lab.reshape(height * width, dimension)
    rng = np.random.default_rng(0)
    sample = image_array[rng.choice(len(image_array), min(nb_samples, len(image_array)), replace=False)]
    kmeans = KMeans(n_clusters=nb_color, random_state=42).fit(sample)
    # uint8 labels are lighter to process, and enough for up to 256 colors
    labels = kmeans.predict(image_array).reshape(height, width).astype(np.uint8)
    lut, centers = merge_close_colors(kmeans.cluster_centers_, np.bincount(labels.ravel(), minlength=nb_color),
                                      min_color_distance)
    labels = lut[labels]
    centers = centers.astype(np.float32).reshape(1, -1, dimension)
    colors = cv2.cvtColor(centers, cv2.COLOR_Lab2RGB).reshape(-1, dimension)
    colors = np.clip(np.rint(colors * 255), 0, 255).astype(np.uint8)
    return labels, colors


def sort_palette(labels, colors, region_colors):
    """Remove the unused colors and sort the others: greys, then colors by hue, each group from light to dark"""
    used = np.unique(region_colors)
    lab = rgb_to_lab(colors[used])
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    # Groups of 30° of hue, starting from red, through orange, yellow, green, blue and purple
    hue_group = ((np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) - 10) % 360) // 30
    group = np.where(chroma < 15, -1, hue_group)
    order = used[np.lexsort((-lab[:, 0], group))]
    lut = np.zeros(len(colors), dtype=np.uint8)
    lut[order] = np.arange(len(order))
    return lut[labels], colors[order], lut[region_colors]
