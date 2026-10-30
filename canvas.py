#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script to generate a canvas out of a picture"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch
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
    min_region_ratio : float, optional, default 1e-4
        regions smaller than this fraction of the picture area are merged into their neighbours
    min_width_ratio : float, optional, default 1/250
        regions thinner than this fraction of the largest dimension are merged into their neighbours
    nb_samples : int, optional, default 1000
        number of pixels sampled to fit the K-mean
    """

    border_color = 150
    label_color = 90

    def __init__(self, path_pic, nb_color, pixel_size=4000, denoise_size=1500, min_region_ratio=1e-4,
                 min_width_ratio=1 / 250, nb_samples=1000):
        path_pic = Path(path_pic)
        src = cv2.imread(str(path_pic))
        if src is None:
            raise FileNotFoundError(f"Cannot read picture: {path_pic}")
        self.namefile = path_pic.stem
        self.src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        self.nb_color = nb_color
        self.tar_width = pixel_size
        self.denoise_size = denoise_size
        self.min_region_ratio = min_region_ratio
        self.min_width_ratio = min_width_ratio
        self.nb_samples = nb_samples
        self.quantified_image = None
        self.canvas = None
        self.regions = []
        self.colormap = []

    def generate(self):
        """Main function, computes the expected result, the canvas and the colormap"""
        # Denoising is done before upscaling to the target size, as its cost grows with the number of pixels
        denoise_width = min(self.denoise_size, self.tar_width, max(self.src.shape[:2]))
        denoised = self.denoise(self.resize(self.src, denoise_width))
        clean_img = self.morphology(self.resize(denoised, self.tar_width))
        labels, colors = self.quantification(clean_img)
        labels, borders, dist, regions = self.merge_small_regions(labels, len(colors))
        self.colormap = colors.tolist()
        self.quantified_image = colors[labels]
        self.regions = self.find_regions(*regions, dist)

        canvas = np.full(labels.shape, 255, dtype=np.uint8)
        canvas[borders] = self.border_color
        for region in self.regions:
            self.draw_label(canvas, region)
        self.canvas = canvas
        return canvas

    def merge_small_regions(self, labels, nb_color, max_iter=3):
        """Give the regions too small or too thin to be painted the color of the closest large region

        Returns the merged labels, their borders, the distance to the borders and the regions (see label_regions)
        """
        min_area = self.min_region_ratio * labels.size
        min_radius = self.min_width_ratio * max(labels.shape) / 2
        for iteration in range(max_iter + 1):
            borders = self.borders(labels)
            dist = self.distance_to_borders(borders)
            regions = self.label_regions(labels, nb_color)
            if iteration == max_iter:
                break
            ids, _, stats = regions
            # A region is wide enough if one of its pixels is farther than min_radius from any border
            is_wide = np.zeros(len(stats), dtype=bool)
            is_wide[ids[dist >= min_radius]] = True
            small = ((stats[:, cv2.CC_STAT_AREA] < min_area) | ~is_wide)[ids]
            if not small.any() or small.all():
                break
            # Each small pixel gets the index (in raster order, from 1) of the closest pixel of a large region
            _, closest = cv2.distanceTransformWithLabels(small.view(np.uint8), cv2.DIST_L2, 5,
                                                         labelType=cv2.DIST_LABEL_PIXEL)
            labels = labels.copy()
            labels[small] = labels[~small][closest[small] - 1]
        return labels, borders, dist, regions

    @staticmethod
    def label_regions(labels, nb_color):
        """Split the picture in connected regions of a single color

        Returns the region id of each pixel, and the color index and stats (x, y, width, height, area) of each region
        """
        ids = np.zeros(labels.shape, dtype=np.int32)
        offsets = np.zeros(nb_color, dtype=np.int32)
        region_colors, region_stats = [], []
        for ind in range(nb_color):
            nb_comp, components, stats, _ = cv2.connectedComponentsWithStats((labels == ind).view(np.uint8),
                                                                             connectivity=4)
            # Each pixel has a single color, so it only receives the id of its own component
            ids += components
            offsets[ind] = len(region_colors) - 1
            region_colors += [ind] * (nb_comp - 1)
            region_stats.append(stats[1:])
        ids += offsets[labels]
        return ids, np.array(region_colors), np.concatenate(region_stats)

    @staticmethod
    def distance_to_borders(borders):
        """Distance of each pixel to the closest border"""
        return cv2.distanceTransform((~borders).view(np.uint8), cv2.DIST_L2, 5)

    @staticmethod
    def borders(labels):
        """Pixels where the color changes, the picture frame included"""
        borders = np.zeros(labels.shape, dtype=bool)
        borders[:, :-1] |= labels[:, :-1] != labels[:, 1:]
        borders[:-1] |= labels[:-1] != labels[1:]
        borders[[0, -1]] = True
        borders[:, [0, -1]] = True
        return borders

    @staticmethod
    def find_regions(ids, region_colors, stats, dist):
        """List the regions, with the point farthest from their borders where to put the label"""
        regions = []
        for region_id, (ind, (x, y, width, height, area)) in enumerate(zip(region_colors, stats)):
            local = np.where(ids[y:y + height, x:x + width] == region_id, dist[y:y + height, x:x + width], 0)
            py, px = np.unravel_index(np.argmax(local), local.shape)
            regions.append({"label": int(ind) + 1, "area": int(area), "x": int(x + px), "y": int(y + py),
                            "radius": float(local[py, px])})
        return regions

    def draw_label(self, canvas, region):
        """Draw the label of a region, centered and sized to fit in the region"""
        text = str(region["label"])
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), _ = cv2.getTextSize(text, font, 1, 1)
        # Scale so that the text fits in the largest inscribed circle, within readable bounds
        min_scale, max_scale = self.tar_width / 8000, self.tar_width / 4500
        scale = float(np.clip(1.4 * region["radius"] / np.hypot(text_w, text_h), min_scale, max_scale))
        thickness = max(1, round(scale * 1.5))
        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        origin = (int(region["x"] - text_w / 2), int(region["y"] + text_h / 2))
        cv2.putText(canvas, text, origin, font, scale, self.label_color, thickness, cv2.LINE_AA)

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

    def quantification(self, picture):
        """Returns the color index of each pixel and the K-mean colors

        The K-mean runs in the CIELAB color space, where distances match the perceived color differences
        """
        height, width, dimension = picture.shape
        lab = cv2.cvtColor(picture.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
        image_array = lab.reshape(height * width, dimension)
        rng = np.random.default_rng(0)
        sample = image_array[rng.choice(len(image_array), min(self.nb_samples, len(image_array)), replace=False)]
        kmeans = KMeans(n_clusters=self.nb_color, random_state=42).fit(sample)
        labels = kmeans.predict(image_array).reshape(height, width)
        centers = kmeans.cluster_centers_.astype(np.float32).reshape(1, -1, dimension)
        colors = cv2.cvtColor(centers, cv2.COLOR_Lab2RGB).reshape(-1, dimension)
        colors = np.clip(np.rint(colors * 255), 0, 255).astype(np.uint8)
        return labels, colors

    def colormap_picture(self, dpi=200):
        """Returns the colormap as a picture for the user: numbered swatches with their hex and RGB codes"""
        nb_cols = 2 if len(self.colormap) > 8 else 1
        nb_rows = -(-len(self.colormap) // nb_cols)
        # Layout in inches
        margin, col_width, row_height, swatch_width = 0.3, 2.4, 0.5, 0.7
        width, height = 2 * margin + nb_cols * col_width, 2 * margin + nb_rows * row_height
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor="white")
        axes = fig.add_axes((0, 0, 1, 1))
        axes.set_xlim(0, width)
        axes.set_ylim(height, 0)
        axes.axis("off")
        for ind, col in enumerate(self.colormap):
            column, row = divmod(ind, nb_rows)
            x, y = margin + column * col_width, margin + row * row_height
            rgb = np.array(col) / 255
            axes.add_patch(FancyBboxPatch((x, y + 0.07), swatch_width, row_height - 0.14,
                                          boxstyle="round,pad=0,rounding_size=0.07", fc=rgb, ec="#00000030", lw=0.8))
            luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            axes.text(x + swatch_width / 2, y + row_height / 2, str(ind + 1), ha="center", va="center", fontsize=11,
                      weight="bold", color="white" if luminance < 0.55 else "#222222")
            axes.text(x + swatch_width + 0.15, y + row_height * 0.4, "#{:02X}{:02X}{:02X}".format(*col),
                      va="center", fontsize=10, family="monospace", color="#222222")
            axes.text(x + swatch_width + 0.15, y + row_height * 0.72, "RGB {} {} {}".format(*col), va="center",
                      fontsize=7, color="#888888")
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()

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
