# -*- coding: utf-8 -*-
"""Resizing and denoising of the pictures"""

import cv2


def target_dim(shape, size):
    """Returns the (width, height) whose largest dimension matches size, respecting the picture ratio"""
    (height, width) = shape[:2]
    # At least 1 pxl, which a very thin picture would round down to 0
    if height > width:  # portrait mode
        return max(1, int(width * size / float(height))), size
    return size, max(1, int(height * size / float(width)))


def resize(picture, size):
    """Resize the picture so that its largest dimension matches size, respecting the picture ratio"""
    interpolation = cv2.INTER_AREA if size < max(picture.shape[:2]) else cv2.INTER_CUBIC
    return cv2.resize(picture, target_dim(picture.shape, size), interpolation=interpolation)


def denoise(picture, iterations=2):
    """Reduction of noize with a bilateral filter, which flattens the textures but keeps the edges sharp"""
    for _ in range(iterations):
        picture = cv2.bilateralFilter(picture, 9, 40, 7)
    return picture
