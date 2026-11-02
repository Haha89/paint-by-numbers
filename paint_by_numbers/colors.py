# -*- coding: utf-8 -*-
"""Search of the colors summarizing a picture, and sorting of the palette"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

# Where the subject of a picture usually is, and how far it spreads, as fractions of the width and height
SUBJECT_CENTER = (0.5, 0.4)
SUBJECT_SPREAD = 0.2
# How far the weight spreads around a face, as a share of its size
FACE_SPREAD = 0.8


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


def subject_weights(indices, height, width, subject_weight, faces=()):
    """The weight of pixels given by their raster indices: subject_weight on the subject, fading to 1 away from it

    The subject is the faces, given as (left, top, width, height) fractions, or else the usual place of a subject
    """
    y, x = np.divmod(indices, width)
    y, x = y / height, x / width
    spots = [(left + box_width / 2, top + box_height / 2, FACE_SPREAD * box_width, FACE_SPREAD * box_height)
             for left, top, box_width, box_height in faces] or [(*SUBJECT_CENTER, SUBJECT_SPREAD, SUBJECT_SPREAD)]
    closeness = np.max([np.exp(-((x - cx) ** 2 / (2 * sx ** 2) + (y - cy) ** 2 / (2 * sy ** 2)))
                        for cx, cy, sx, sy in spots], axis=0)
    return 1 + (subject_weight - 1) * closeness


def quantification(picture, nb_color, nb_samples, min_color_distance, subject_weight=1, faces=()):
    """Returns the color index of each pixel and the K-mean colors

    The K-mean runs in the CIELAB color space, where distances match the perceived color differences. With a
    subject_weight above 1, the subject counts more, so that more colors go to its shades, see subject_weights
    """
    height, width, dimension = picture.shape
    lab = cv2.cvtColor(picture.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    image_array = lab.reshape(height * width, dimension)
    rng = np.random.default_rng(0)
    indices = rng.choice(len(image_array), min(nb_samples, len(image_array)), replace=False)
    weights = subject_weights(indices, height, width, subject_weight, faces)
    kmeans = KMeans(n_clusters=nb_color, random_state=42).fit(image_array[indices], sample_weight=weights)
    # uint8 labels are lighter to process, and enough for up to 256 colors
    labels = kmeans.predict(image_array).reshape(height, width).astype(np.uint8)
    lut, centers = merge_close_colors(kmeans.cluster_centers_, np.bincount(labels.ravel(), minlength=nb_color),
                                      min_color_distance)
    labels = lut[labels]
    centers = centers.astype(np.float32).reshape(1, -1, dimension)
    colors = cv2.cvtColor(centers, cv2.COLOR_Lab2RGB).reshape(-1, dimension)
    colors = np.clip(np.rint(colors * 255), 0, 255).astype(np.uint8)
    return labels, colors


def merge_rare_colors(labels, colors, min_share):
    """Lookup table giving the colors covering less than min_share of the picture their closest kept color

    The most used color is always kept. Returns None when no color is rare
    """
    counts = np.bincount(labels.ravel(), minlength=len(colors))
    rare = (counts > 0) & (counts < min_share * labels.size)
    rare[np.argmax(counts)] = False
    if not rare.any():
        return None
    kept = np.flatnonzero((counts > 0) & ~rare)
    lab = rgb_to_lab(colors)
    lut = np.arange(len(colors), dtype=np.uint8)
    closest = np.argmin(np.linalg.norm(lab[rare][:, None] - lab[kept][None], axis=2), axis=1)
    lut[rare] = kept[closest]
    return lut


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
