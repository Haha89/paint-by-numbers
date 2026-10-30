#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script to generate a canvas out of a picture"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


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
        size in pxl of the largest dimension used for denoising (the costliest step)
    max_contours : int, optional, default 500
        number of regions kept, largest first
    min_region_size : int, optional, default 10
        minimal width and height in pxl of a region bounding box
    min_region_area : int, optional, default 100
        minimal area in pxl of a region
    nb_samples : int, optional, default 1000
        number of pixels sampled to fit the K-mean
    """

    def __init__(self, path_pic, nb_color, pixel_size=4000, denoise_size=1500, max_contours=500,
                 min_region_size=10, min_region_area=100, nb_samples=1000):
        path_pic = Path(path_pic)
        src = cv2.imread(str(path_pic))
        if src is None:
            raise FileNotFoundError(f"Cannot read picture: {path_pic}")
        self.namefile = path_pic.stem
        self.src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        self.nb_color = nb_color
        self.tar_width = pixel_size
        self.denoise_size = denoise_size
        self.max_contours = max_contours
        self.min_region_size = min_region_size
        self.min_region_area = min_region_area
        self.nb_samples = nb_samples
        self.quantified_image = None
        self.canvas = None
        self.colormap = []

    def generate(self):
        """Main function, computes the expected result, the canvas and the colormap"""
        # Denoising is done before upscaling to the target size, as its cost grows with the number of pixels
        denoise_width = min(self.denoise_size, self.tar_width, max(self.src.shape[:2]))
        denoised = self.denoise(self.resize(self.src, denoise_width))
        clean_img = self.morphology(self.resize(denoised, self.tar_width))
        labels, colors = self.quantification(clean_img)
        self.colormap = colors.tolist()
        self.quantified_image = colors[labels]

        contours = []
        for ind in range(len(colors)):
            mask = (labels == ind).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for contour in cnts:
                _, _, width_ctr, height_ctr = cv2.boundingRect(contour)
                if width_ctr <= self.min_region_size or height_ctr <= self.min_region_size:
                    continue
                # Oriented area is negative for outer contours
                area = -cv2.contourArea(contour, True)
                if area > self.min_region_area:
                    contours.append({"contour": contour, "area": area, "x": contour[0][0][0],
                                     "y": contour[0][0][1] + 15, "label": str(ind + 1)})

        # Keep the largest contours only
        sorted_contours = sorted(contours, key=lambda d: d["area"], reverse=True)[:self.max_contours]
        canvas = np.zeros(labels.shape, dtype=np.uint8)
        cv2.drawContours(canvas, [c["contour"] for c in sorted_contours], -1, 255, 1)
        canvas = 255 - self.thin_contours(canvas)
        for cont in sorted_contours:
            cv2.putText(canvas, cont["label"], (cont["x"], cont["y"]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
        self.canvas = canvas
        return canvas

    @staticmethod
    def resize(picture, size):
        """Resize the picture so that its largest dimension matches size, respecting the picture ratio"""
        (height, width) = picture.shape[:2]
        if height > width:  # portrait mode
            dim = (int(width * size / float(height)), size)
        else:
            dim = (size, int(height * size / float(width)))
        interpolation = cv2.INTER_AREA if size < max(height, width) else cv2.INTER_CUBIC
        return cv2.resize(picture, dim, interpolation=interpolation)

    @staticmethod
    def denoise(picture):
        """Reduction of noize"""
        blurred = cv2.blur(picture, (4, 4))
        return cv2.fastNlMeansDenoisingColored(blurred, None, 10, 10, 7, 21)

    @staticmethod
    def morphology(picture):
        """Morphomat operations, closing then opening"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        clean_pic = cv2.morphologyEx(picture, cv2.MORPH_CLOSE, kernel, borderType=cv2.BORDER_REPLICATE)
        return cv2.morphologyEx(clean_pic, cv2.MORPH_OPEN, kernel, borderType=cv2.BORDER_REPLICATE)

    @staticmethod
    def thin_contours(picture):
        """Morphological gradient, keeps a thin line for each contour"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.subtract(picture, cv2.erode(picture, kernel, iterations=1))

    def quantification(self, picture):
        """Returns the color index of each pixel and the K-mean colors"""
        height, width, dimension = picture.shape
        image_array = picture.reshape(height * width, dimension).astype(np.float32)
        rng = np.random.default_rng(0)
        sample = image_array[rng.choice(len(image_array), min(self.nb_samples, len(image_array)), replace=False)]
        kmeans = KMeans(n_clusters=self.nb_color, random_state=42).fit(sample)
        labels = kmeans.predict(image_array).reshape(height, width)
        colors = np.clip(np.rint(kmeans.cluster_centers_), 0, 255).astype(np.uint8)
        return labels, colors

    def colormap_picture(self):
        """Returns the colormap as a picture for the user"""
        picture = np.full((len(self.colormap) * 30 + 20, 300, 3), 255, dtype=np.uint8)
        for ind, col in enumerate(self.colormap):
            cv2.putText(picture, f"{ind + 1:d}", (15, 30 * ind + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
            cv2.rectangle(picture, (45, 30 * ind + 5), (85, 30 * ind + 20), col, thickness=-1)
            cv2.putText(picture, str(col), (100, 30 * ind + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
        return picture

    def plot(self):
        """Display the expected result, the canvas and the colormap"""
        self.plot_figure(self.quantified_image, "Expected Result")
        self.plot_figure(self.canvas, "Canvas")
        self.plot_figure(self.colormap_picture(), "Colormap")

    def save(self, output_dir="./outputs"):
        """Save the expected result, the canvas and the colormap in output_dir"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {"result": cv2.cvtColor(self.quantified_image, cv2.COLOR_RGB2BGR),
                   "canvas": self.canvas,
                   "colormap": cv2.cvtColor(self.colormap_picture(), cv2.COLOR_RGB2BGR)}
        for suffix, picture in outputs.items():
            path = output_dir / f"{self.namefile}-{suffix}.png"
            if not cv2.imwrite(str(path), picture):
                raise OSError(f"Cannot write picture: {path}")

    @staticmethod
    def plot_figure(image, title):
        """Function that display an image with a title and hides some stuff"""
        plt.figure()
        plt.axis("off")
        plt.title(title)
        plt.imshow(image, cmap="gray" if image.ndim == 2 else None)
        plt.show()


if __name__ == "__main__":
    main_canvas = Canvas("./inputs/chirico.jpg", 25)
    main_canvas.generate()
    main_canvas.plot()
