#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Script to generate a canvas out of a picture"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection
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
        size in pxl of the largest dimension used for denoising
    min_region_ratio : float, optional, default 1e-4
        regions smaller than this fraction of the picture area are merged into their neighbours
    min_width_ratio : float, optional, default 1/250
        regions thinner than this fraction of the largest dimension are merged into their neighbours
    smoothing_ratio : float, optional, default 1/1000
        strength of the borders smoothing, as a fraction of the largest dimension (0 to disable)
    min_color_distance : float, optional, default 6
        colors closer than this CIELAB distance (delta E) are merged, so the canvas may use less than nb_color
    nb_samples : int, optional, default 1000
        number of pixels sampled to fit the K-mean
    """

    border_color = 150
    label_color = 90
    # Paper sizes in inches, portrait
    paper_sizes = {"a4": (8.27, 11.69), "a3": (11.69, 16.54), "letter": (8.5, 11)}

    def __init__(self, path_pic, nb_color, pixel_size=4000, denoise_size=1500, min_region_ratio=1e-4,
                 min_width_ratio=1 / 250, smoothing_ratio=1 / 1000, min_color_distance=6, nb_samples=1000):
        if not 1 <= nb_color <= 256:
            raise ValueError("nb_color must be between 1 and 256")
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
        self.smoothing_ratio = smoothing_ratio
        self.min_color_distance = min_color_distance
        self.nb_samples = nb_samples
        self.quantified_image = None
        self.canvas = None
        self.borders = None
        self.regions = []
        self.colormap = []

    def generate(self):
        """Main function, computes the expected result, the canvas and the colormap"""
        # Denoising is done before upscaling to the target size, as its cost grows with the number of pixels
        denoise_width = min(self.denoise_size, self.tar_width, max(self.src.shape[:2]))
        denoised = self.denoise(self.resize(self.src, denoise_width))
        clean_img = self.resize(denoised, self.tar_width)
        labels, colors = self.quantification(clean_img)
        labels = self.smooth_labels(labels, len(colors))
        labels, borders, dist, (ids, region_colors, stats) = self.merge_small_regions(labels, colors)
        labels, colors, region_colors = self.sort_palette(labels, colors, region_colors)
        self.colormap = colors.tolist()
        self.quantified_image = colors[labels]
        self.borders = borders
        self.regions = self.find_regions(ids, region_colors, stats, dist)

        canvas = np.full(labels.shape, 255, dtype=np.uint8)
        canvas[borders] = self.border_color
        for region in self.regions:
            self.draw_label(canvas, region)
        self.canvas = canvas
        return canvas

    @staticmethod
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

    def smooth_labels(self, labels, nb_color):
        """Smooth the borders: each pixel takes the color most present around it, weighted by a gaussian"""
        sigma = self.smoothing_ratio * max(labels.shape)
        if sigma <= 0:
            return labels
        return self.vote(labels, nb_color, lambda mask: cv2.GaussianBlur(mask, (0, 0), sigma))

    def merge_small_regions(self, labels, colors, max_iter=3):
        """Give the regions too small or too thin to be painted the color of their closest large neighbour

        Returns the merged labels, their borders, the distance to the borders and the regions (see label_regions)
        """
        lab = self.rgb_to_lab(colors)
        min_area = self.min_region_ratio * labels.size
        min_radius = self.min_width_ratio * max(labels.shape) / 2
        for iteration in range(max_iter + 1):
            borders = self.find_borders(labels)
            dist = self.distance_to_borders(borders)
            regions = self.label_regions(labels, len(colors))
            if iteration == max_iter:
                break
            ids, region_colors, stats = regions
            # A region is wide enough if one of its pixels is farther than min_radius from any border
            is_wide = np.zeros(len(stats), dtype=bool)
            is_wide[ids[dist >= min_radius]] = True
            small_regions = (stats[:, cv2.CC_STAT_AREA] < min_area) | ~is_wide
            small = small_regions[ids]
            if not small.any() or small.all():
                break
            labels = labels.copy()
            new_colors = self.closest_color_neighbours(ids, region_colors, small_regions, lab)[ids]
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

    @staticmethod
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

    def sort_palette(self, labels, colors, region_colors):
        """Remove the unused colors and sort the others: greys, then colors by hue, each group from light to dark"""
        used = np.unique(region_colors)
        lab = self.rgb_to_lab(colors[used])
        chroma = np.hypot(lab[:, 1], lab[:, 2])
        # Groups of 30° of hue, starting from red, through orange, yellow, green, blue and purple
        hue_group = ((np.degrees(np.arctan2(lab[:, 2], lab[:, 1])) - 10) % 360) // 30
        group = np.where(chroma < 15, -1, hue_group)
        order = used[np.lexsort((-lab[:, 0], group))]
        lut = np.zeros(len(colors), dtype=np.uint8)
        lut[order] = np.arange(len(order))
        return lut[labels], colors[order], lut[region_colors]

    @staticmethod
    def rgb_to_lab(colors):
        """Convert a list of uint8 RGB colors to CIELAB"""
        return cv2.cvtColor(colors.reshape(1, -1, 3).astype(np.float32) / 255, cv2.COLOR_RGB2Lab)[0]

    def merge_close_colors(self, centers, counts):
        """Merge the CIELAB colors closer than min_color_distance, averaged by their number of pixels

        Returns the lookup table from the old to the new color indices, and the new colors
        """
        centers, counts = centers.astype(np.float64), counts.astype(np.float64)
        lut, active = np.arange(len(centers)), list(range(len(centers)))
        while len(active) > 1:
            distance = np.linalg.norm(centers[active][:, None] - centers[active][None], axis=2)
            np.fill_diagonal(distance, np.inf)
            i, j = np.unravel_index(np.argmin(distance), distance.shape)
            if distance[i, j] >= self.min_color_distance:
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

    @staticmethod
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

    @staticmethod
    def distance_to_borders(borders):
        """Distance of each pixel to the closest border"""
        return cv2.distanceTransform((~borders).view(np.uint8), cv2.DIST_L2, 5)

    @staticmethod
    def find_borders(labels):
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
    def target_dim(shape, size):
        """Returns the (width, height) whose largest dimension matches size, respecting the picture ratio"""
        (height, width) = shape[:2]
        if height > width:  # portrait mode
            return int(width * size / float(height)), size
        return size, int(height * size / float(width))

    def resize(self, picture, size):
        """Resize the picture so that its largest dimension matches size, respecting the picture ratio"""
        interpolation = cv2.INTER_AREA if size < max(picture.shape[:2]) else cv2.INTER_CUBIC
        return cv2.resize(picture, self.target_dim(picture.shape, size), interpolation=interpolation)

    @staticmethod
    def denoise(picture, iterations=2):
        """Reduction of noize with a bilateral filter, which flattens the textures but keeps the edges sharp"""
        for _ in range(iterations):
            picture = cv2.bilateralFilter(picture, 9, 40, 7)
        return picture

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
        # uint8 labels are lighter to process, and enough for up to 256 colors
        labels = kmeans.predict(image_array).reshape(height, width).astype(np.uint8)
        lut, centers = self.merge_close_colors(kmeans.cluster_centers_, np.bincount(labels.ravel(),
                                                                                    minlength=self.nb_color))
        labels = lut[labels]
        centers = centers.astype(np.float32).reshape(1, -1, dimension)
        colors = cv2.cvtColor(centers, cv2.COLOR_Lab2RGB).reshape(-1, dimension)
        colors = np.clip(np.rint(colors * 255), 0, 255).astype(np.uint8)
        return labels, colors

    @staticmethod
    def inch_figure(width, height, dpi=100):
        """Returns a figure and axes covering it, with coordinates in inches from the top left corner"""
        fig = Figure(figsize=(width, height), dpi=dpi, facecolor="white")
        axes = fig.add_axes((0, 0, 1, 1))
        axes.set_xlim(0, width)
        axes.set_ylim(height, 0)
        axes.axis("off")
        return fig, axes

    def draw_palette(self, axes, x_0, y_0, nb_cols, col_width, row_height, swatch_width=0.7):
        """Draw the numbered swatches with their hex and RGB codes, returns the height used"""
        nb_rows = -(-len(self.colormap) // nb_cols)
        for ind, col in enumerate(self.colormap):
            column, row = divmod(ind, nb_rows)
            x, y = x_0 + column * col_width, y_0 + row * row_height
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
        return nb_rows * row_height

    def colormap_picture(self, dpi=200):
        """Returns the colormap as a picture for the user: numbered swatches with their hex and RGB codes"""
        nb_cols = 2 if len(self.colormap) > 8 else 1
        nb_rows = -(-len(self.colormap) // nb_cols)
        margin, col_width, row_height = 0.3, 2.4, 0.5
        fig, axes = self.inch_figure(2 * margin + nb_cols * col_width, 2 * margin + nb_rows * row_height, dpi)
        self.draw_palette(axes, margin, margin, nb_cols, col_width, row_height)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())[..., :3].copy()

    def palette_page(self, paper, title, subtitle):
        """PDF page with the palette, the source picture and the expected result"""
        width, height = paper
        margin = 0.6
        fig, axes = self.inch_figure(width, height)
        axes.text(margin, margin, "Palette", fontsize=20, weight="bold", va="top", color="#222222")
        axes.text(margin, margin + 0.45, " - ".join(filter(None, [title, subtitle])), fontsize=9, va="top",
                  color="#777777")
        nb_cols = 2 if len(self.colormap) > 8 else 1
        y = margin + 0.9
        y += self.draw_palette(axes, margin, y, nb_cols, (width - 2 * margin) / 2, 0.46) + 0.5

        # Source picture and expected result side by side, in the remaining space
        gap, caption = 0.3, 0.25
        box_width, box_height = (width - 2 * margin - gap) / 2, height - margin - y - caption
        pictures = [(self.src, "Source picture"), (self.quantified_image, "Expected result")]
        if box_height > 1:
            for ind, (picture, name) in enumerate(pictures):
                scale = min(box_width / picture.shape[1], box_height / picture.shape[0])
                pic_width, pic_height = picture.shape[1] * scale, picture.shape[0] * scale
                left = margin + ind * (box_width + gap) + (box_width - pic_width) / 2
                axes.text(left + pic_width / 2, y, name, ha="center", va="top", fontsize=9, color="#555555")
                # Embedded at 300 dpi, enough for printing and much lighter than the full resolution
                preview = self.resize(picture, min(max(picture.shape[:2]), int(300 * max(pic_width, pic_height))))
                axes.imshow(preview, extent=(left, left + pic_width, y + caption + pic_height, y + caption),
                            aspect="auto", interpolation="none")
        axes.set_xlim(0, width)
        axes.set_ylim(height, 0)
        return fig

    def canvas_page(self, paper, title, subtitle):
        """PDF page with the canvas, borders and labels as vectors"""
        height_px, width_px = self.borders.shape
        # Landscape page for landscape pictures
        width, height = (max(paper), min(paper)) if width_px > height_px else (min(paper), max(paper))
        margin, header = 0.6, 0.65
        fig, axes = self.inch_figure(width, height)
        axes.text(margin, margin, title, fontsize=15, weight="bold", va="top", color="#222222")
        axes.text(margin, margin + 0.3, subtitle, fontsize=9, va="top", color="#777777")
        axes.text(width - margin, margin, f"{len(self.colormap)} colors", fontsize=9, ha="right", va="top",
                  color="#777777")

        scale = min((width - 2 * margin) / width_px, (height - 2 * margin - header) / height_px)
        offset = np.array([(width - width_px * scale) / 2, margin + header])
        # The borders are 1 pxl wide, so both sides of a border are traced on the same pixels: no double lines
        contours, _ = cv2.findContours(self.borders.view(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        polygons = [(cv2.approxPolyDP(contour, 0.8, True)[:, 0, :] + 0.5) * scale + offset for contour in contours]
        axes.add_collection(PolyCollection(polygons, closed=True, facecolors="none", edgecolors="#9a9a9a",
                                           linewidths=0.3))
        for region in self.regions:
            size = float(np.clip(region["radius"] * scale * 72 * 0.9, 3.5, 7))
            x, y = (np.array([region["x"], region["y"]]) + 0.5) * scale + offset
            axes.text(x, y, str(region["label"]), ha="center", va="center", fontsize=size, color="#6b6b6b")
        return fig

    def save_pdf(self, path, title=None, subtitle="", paper="a4"):
        """Save a printable PDF: the palette page, then the canvas page"""
        title = title or self.namefile
        paper = self.paper_sizes[paper]
        with plt.rc_context({"pdf.fonttype": 42}), PdfPages(path, metadata={"Title": title}) as pdf:
            pdf.savefig(self.palette_page(paper, title, subtitle))
            pdf.savefig(self.canvas_page(paper, title, subtitle))

    def plot(self):
        """Display the expected result, the canvas and the colormap"""
        self.plot_figure(self.quantified_image, "Expected Result")
        self.plot_figure(self.canvas, "Canvas")
        self.plot_figure(self.colormap_picture(), "Colormap")

    def save(self, output_dir="./outputs", pdf=True, title=None, subtitle="", paper="a4"):
        """Save the expected result, the canvas and the colormap in output_dir, and the printable PDF"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {"result": cv2.cvtColor(self.quantified_image, cv2.COLOR_RGB2BGR),
                   "canvas": self.canvas,
                   "colormap": cv2.cvtColor(self.colormap_picture(), cv2.COLOR_RGB2BGR)}
        for suffix, picture in outputs.items():
            path = output_dir / f"{self.namefile}-{suffix}.png"
            if not cv2.imwrite(str(path), picture):
                raise OSError(f"Cannot write picture: {path}")
        if pdf:
            self.save_pdf(output_dir / f"{self.namefile}.pdf", title, subtitle, paper)

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
