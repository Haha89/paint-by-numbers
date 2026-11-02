# -*- coding: utf-8 -*-
"""Search of the faces of a picture, the subject which deserves the most colors"""

from pathlib import Path

import cv2
import numpy as np

from .preprocessing import resize

# YuNet, from the OpenCV model zoo, under the MIT license
MODEL_PATH = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"
# Largest dimension searched: enough for a face, and fast
DETECTION_SIZE = 640
# Low, for the painted and drawn faces which score less than photographs
MIN_SCORE = 0.6


def face_mask(shape, faces):
    """The pixels of a picture of this (height, width) inside the given faces, or None without any face"""
    if not len(faces):
        return None
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=bool)
    for left, top, box_width, box_height in faces:
        mask[max(0, round(top * height)):round((top + box_height) * height),
             max(0, round(left * width)):round((left + box_width) * width)] = True
    return mask


def detect_faces(picture):
    """Returns the faces of an RGB picture as (left, top, width, height) fractions of the picture, largest first

    No face is found if the model can't run: the faces only improve the colors, they are not worth stopping for
    """
    small = resize(picture, min(DETECTION_SIZE, max(picture.shape[:2])))
    height, width = small.shape[:2]
    # OpenCV 5 logs a harmless warning for each model it loads
    log_level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    try:
        detector = cv2.FaceDetectorYN.create(str(MODEL_PATH), "", (width, height), MIN_SCORE)
        _, faces = detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    except cv2.error:
        return []
    finally:
        cv2.utils.logging.setLogLevel(log_level)
    if faces is None:
        return []
    boxes = faces[:, :4] / np.array([width, height, width, height])
    return sorted(map(tuple, boxes.tolist()), key=lambda box: -box[2] * box[3])
