# -*- coding: utf-8 -*-
"""Generation of a canvas out of a picture"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .colors import merge_rare_colors, quantification, sort_palette
from .faces import detect_faces, face_mask
from .export import PAPER_SIZES, PRINT_SIZES, colormap_picture, fill_crop, save_pdf
from .preprocessing import denoise, resize
from .regions import find_regions, merge_small_regions, smooth_labels


class Canvas:
    """
    Definition of the canvas object

    Parameters
    ----------
    path_pic : str or Path
        the source picture you want to transform in a canvas
    nb_color : int
        number of colors you want to keep
    pixel_size : int, optional, default 4000
        size in pxl of the largest dimension of the output canvas
    denoise_size : int, optional, default 1500
        size in pxl of the largest dimension used for denoising
    detail : str, optional, default "medium"
        balance between easy to paint and faithful to the picture, one of detail_levels. It sets the values below,
        unless they are given
    min_region_ratio : float, optional
        regions smaller than this fraction of the picture area are merged into their neighbours
    min_width_ratio : float, optional
        regions thinner than this fraction of the largest dimension are merged into their neighbours
    smoothing_ratio : float, optional
        strength of the borders smoothing, as a fraction of the largest dimension (0 to disable)
    min_color_share : float, optional
        colors covering less than this fraction of the picture are replaced by their closest color (0 to disable)
    min_color_distance : float, optional, default 6
        colors closer than this CIELAB distance (delta E) are merged, so the canvas may use less than nb_color
    subject_weight : float, optional, default 10
        how much more the center of the picture counts when choosing the colors, so that the subject gets more shades
        and fewer large flat shapes (1 to disable)
    face_weight : float, optional, default 30
        the same for the faces found in the picture, which replace the center. It can be higher, as they surely are a
        subject and cover less of the picture (1 to disable the search of faces)
    face_detail : float, optional
        the minimum area and width of the regions are multiplied by this inside the faces, to keep the eyes and the
        mouth (1 to disable). It comes from the detail level, unless it is given
    nb_samples : int, optional, default 10000
        number of pixels sampled to fit the K-mean
    label_spacing_ratio : float, optional, default 1/6
        the label is repeated in large regions, this far apart as a fraction of the largest dimension (0 to disable)
    """

    border_color = 150
    label_color = 90
    # Labels are only repeated where the region is this wide, as a fraction of the largest dimension
    min_repeat_width_ratio = 1 / 100
    paper_sizes = PAPER_SIZES
    print_sizes = PRINT_SIZES
    detail_levels = {
        "easy": {"min_region_ratio": 4e-4, "min_width_ratio": 1 / 120, "smoothing_ratio": 1 / 600,
                 "min_color_share": 0.01, "face_detail": 0.35},
        "medium": {"min_region_ratio": 1e-4, "min_width_ratio": 1 / 250, "smoothing_ratio": 1 / 1000,
                   "min_color_share": 0.002, "face_detail": 0.5},
        "detailed": {"min_region_ratio": 5e-5, "min_width_ratio": 1 / 400, "smoothing_ratio": 1 / 1000,
                     "min_color_share": 0, "face_detail": 0.25},
    }

    def __init__(self, path_pic, nb_color, pixel_size=4000, denoise_size=1500, detail="medium", min_region_ratio=None,
                 min_width_ratio=None, smoothing_ratio=None, min_color_share=None, min_color_distance=6,
                 nb_samples=10000, label_spacing_ratio=1 / 6, subject_weight=10, face_weight=30,
                 face_detail=None):
        if not 1 <= nb_color <= 256:
            raise ValueError("nb_color must be between 1 and 256")
        if detail not in self.detail_levels:
            raise ValueError(f"detail must be one of {', '.join(self.detail_levels)}")
        path_pic = Path(path_pic)
        src = cv2.imread(str(path_pic))
        if src is None:
            raise FileNotFoundError(f"Cannot read picture: {path_pic}")
        self.namefile = path_pic.stem
        self.src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        self.nb_color = nb_color
        self.tar_width = pixel_size
        self.denoise_size = denoise_size
        preset = self.detail_levels[detail]
        self.min_region_ratio = preset["min_region_ratio"] if min_region_ratio is None else min_region_ratio
        self.min_width_ratio = preset["min_width_ratio"] if min_width_ratio is None else min_width_ratio
        self.smoothing_ratio = preset["smoothing_ratio"] if smoothing_ratio is None else smoothing_ratio
        self.min_color_share = preset["min_color_share"] if min_color_share is None else min_color_share
        self.min_color_distance = min_color_distance
        self.nb_samples = nb_samples
        self.subject_weight = subject_weight
        self.face_weight = face_weight
        self.face_detail = preset["face_detail"] if face_detail is None else face_detail
        self.faces = []
        self.label_spacing_ratio = label_spacing_ratio
        self.quantified_image = None
        self.canvas = None
        self.borders = None
        self.regions = []
        self.colormap = []
        self.coverage = []

    def crop_to_print(self, print_size):
        """Crop the source around its center to the shape of a standard canvas size, to call before generate"""
        height, width = self.src.shape[:2]
        left, top, right, bottom = fill_crop(width, height, print_size)
        self.src = self.src[round(top * height):round(bottom * height), round(left * width):round(right * width)]

    def generate(self):
        """Main function, computes the expected result, the canvas and the colormap"""
        # Denoising is done before upscaling to the target size, as its cost grows with the number of pixels
        denoise_width = min(self.denoise_size, self.tar_width, max(self.src.shape[:2]))
        denoised = denoise(resize(self.src, denoise_width))
        clean_img = resize(denoised, self.tar_width)
        self.faces = detect_faces(denoised) if self.face_weight > 1 else []
        labels, colors = quantification(clean_img, self.nb_color, self.nb_samples, self.min_color_distance,
                                        self.face_weight if self.faces else self.subject_weight, self.faces)
        labels = smooth_labels(labels, len(colors), self.smoothing_ratio)
        protected = face_mask(labels.shape, self.faces)
        # Merging the regions changes the coverage of the colors, and replacing a color changes the regions
        while True:
            labels, borders, dist, (ids, region_colors, stats) = merge_small_regions(labels, colors,
                                                                                     self.min_region_ratio,
                                                                                     self.min_width_ratio,
                                                                                     protected=protected,
                                                                                     protected_scale=self.face_detail)
            lut = merge_rare_colors(labels, colors, self.min_color_share)
            if lut is None:
                break
            labels = lut[labels]
        labels, colors, region_colors = sort_palette(labels, colors, region_colors)
        self.colormap = colors.tolist()
        self.coverage = (np.bincount(labels.ravel(), minlength=len(colors)) / labels.size).tolist()
        self.quantified_image = colors[labels]
        self.borders = borders
        largest = max(labels.shape)
        self.regions = find_regions(ids, region_colors, stats, dist, self.label_spacing_ratio * largest,
                                    self.min_repeat_width_ratio * largest / 2)

        canvas = np.full(labels.shape, 255, dtype=np.uint8)
        canvas[borders] = self.border_color
        for region in self.regions:
            for spot in region["spots"]:
                self.draw_label(canvas, region["label"], spot)
        self.canvas = canvas
        return canvas

    def draw_label(self, canvas, label, spot):
        """Draw a label at a spot of its region, centered and sized to fit in the region"""
        text = str(label)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, 1, 1)
        # Scale so that the text fits in the largest inscribed circle, within readable bounds
        min_scale, max_scale = self.tar_width / 5500, self.tar_width / 4500
        scale = float(np.clip(1.4 * spot["radius"] / np.hypot(text_w, text_h), min_scale, max_scale))
        thickness = max(1, round(scale * 1.5))
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        origin = (int(spot["x"] - text_w / 2), int(spot["y"] + text_h / 2))
        cv2.putText(canvas, text, origin, font, scale, self.label_color, thickness, cv2.LINE_AA)

    def plot(self):
        """Display the expected result, the canvas and the colormap"""
        self.plot_figure(self.quantified_image, "Expected Result")
        self.plot_figure(self.canvas, "Canvas")
        self.plot_figure(colormap_picture(self.colormap), "Colormap")

    def save(self, output_dir="./outputs", pdf=True, title=None, subtitle="", paper="a4", print_size=None):
        """Save the expected result, the canvas and the colormap in output_dir, and the printable PDF"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {"result": cv2.cvtColor(self.quantified_image, cv2.COLOR_RGB2BGR),
                   "canvas": self.canvas,
                   "colormap": cv2.cvtColor(colormap_picture(self.colormap), cv2.COLOR_RGB2BGR)}
        for suffix, picture in outputs.items():
            path = output_dir / f"{self.namefile}-{suffix}.png"
            if not cv2.imwrite(str(path), picture):
                raise OSError(f"Cannot write picture: {path}")
        if pdf:
            save_pdf(self, output_dir / f"{self.namefile}.pdf", title, subtitle, paper, print_size)

    @staticmethod
    def plot_figure(image, title):
        """Function that display an image with a title and hides some stuff"""
        plt.figure()
        plt.axis("off")
        plt.title(title)
        plt.imshow(image, cmap="gray" if image.ndim == 2 else None)
        plt.show()
