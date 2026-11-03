# -*- coding: utf-8 -*-
"""Colormap picture and printable PDF of a generated canvas"""

import struct
import zlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

from .preprocessing import resize

# Paper sizes in inches, portrait
PAPER_SIZES = {"a4": (8.27, 11.69), "a3": (11.69, 16.54), "letter": (8.5, 11)}
# Standard canvas sizes in inches, portrait
PRINT_SIZES = {"8x10": (8, 10), "9x12": (9, 12), "11x14": (11, 14), "12x16": (12, 16), "16x20": (16, 20),
               "18x24": (18, 24), "24x36": (24, 36)}
# Margins and header of a canvas page fitting the paper, in inches
PAGE_MARGIN, PAGE_HEADER = 0.6, 0.65
# Font sizes of the labels in points: readable once printed, even when the number overflows a tiny region
MIN_LABEL_SIZE, MAX_LABEL_SIZE = 5, 7
# Area painted by a milliliter of acrylic paint in one covering coat, a rough and rather generous figure
CM2_PER_ML = 50
CM2_PER_SQUARE_INCH = 2.54 ** 2
INCHES_PER_METER = 100 / 2.54
# Resolution of the colormap picture
COLORMAP_DPI = 200


def label_size(radius, scale):
    """The font size of a label, for a region of this radius in pixels and a page scale in inches per pixel"""
    return float(np.clip(radius * scale * 72 * 0.9, MIN_LABEL_SIZE, MAX_LABEL_SIZE))


def oriented(size, width_px, height_px):
    """The (width, height) of a page, in landscape for landscape pictures"""
    return (max(size), min(size)) if width_px > height_px else (min(size), max(size))


def page_layout(shape, paper, print_size=None):
    """Returns the (width, height) of the canvas page, its inches per pixel and the position of the canvas

    With a print_size, the page has exactly this size and only holds the canvas, centered. Otherwise, the canvas fits
    in the margins of the paper, under a header
    """
    height_px, width_px = shape[:2]
    width, height = oriented(PRINT_SIZES[print_size] if print_size else paper, width_px, height_px)
    margin, header = (0, 0) if print_size else (PAGE_MARGIN, PAGE_HEADER)
    scale = min((width - 2 * margin) / width_px, (height - 2 * margin - header) / height_px)
    top = (height - height_px * scale) / 2 if print_size else margin + header
    return (width, height), scale, np.array([(width - width_px * scale) / 2, top])


def printed_size(shape, print_size=None, paper="a4"):
    """The (width, height) in inches of a picture once printed"""
    return printed_on_page(shape, PAPER_SIZES[paper], print_size)


def printed_on_page(shape, paper, print_size=None):
    """The (width, height) in inches of a picture on a page, for a paper given by its dimensions"""
    _, scale, _ = page_layout(shape, paper, print_size)
    return shape[1] * scale, shape[0] * scale


def fill_crop(width, height, print_size):
    """The centered crop (left, top, right, bottom as fractions) giving a picture the shape of a standard canvas size"""
    page_width, page_height = oriented(PRINT_SIZES[print_size], width, height)
    kept_width = min(1.0, height * page_width / page_height / width)
    kept_height = min(1.0, width * page_height / page_width / height)
    return (1 - kept_width) / 2, (1 - kept_height) / 2, (1 + kept_width) / 2, (1 + kept_height) / 2


def paint_volumes(coverage, printed):
    """The estimated milliliters of each color, from its share of the canvas and the printed (width, height) in inches"""
    return [share * printed[0] * printed[1] * CM2_PER_SQUARE_INCH / CM2_PER_ML for share in coverage]


def crowded_label_share(regions, shape, print_size=None, paper="a4", mask=None):
    """The share of the regions whose label touches their borders once printed: it doesn't fit in the region

    With a mask of the picture, only the regions whose first label is inside it count, such as those of the faces
    """
    if mask is not None:
        regions = [region for region in regions if mask[region["spots"][0]["y"], region["spots"][0]["x"]]]
    if not regions:
        return 0.0
    _, scale, _ = page_layout(shape, PAPER_SIZES[paper], print_size)
    crowded = 0
    for region in regions:
        radius = region["spots"][0]["radius"]
        size = label_size(radius, scale)
        # Half of the diagonal of the number, in points, for the proportions of a usual font
        half_diagonal = np.hypot(len(str(region["label"])) * 0.55 * size, 0.72 * size) / 2
        crowded += radius * scale * 72 < half_diagonal
    return crowded / len(regions)


def print_dpi(shape, print_size=None, paper="a4"):
    """The dots per inch of a picture of this shape once printed, as in the PDF canvas page"""
    _, scale, _ = page_layout(shape, PAPER_SIZES[paper], print_size)
    return 1 / scale


def encode_png(picture, dpi=None):
    """The PNG file of a picture, with its print resolution in a pHYs chunk so that other software opens it at the
    right size"""
    ok, png = cv2.imencode(".png", picture)
    if not ok:
        raise OSError("Cannot encode the picture as PNG")
    png = png.tobytes()
    if not dpi:
        return png
    pixels_per_meter = round(dpi * INCHES_PER_METER)
    data = b"pHYs" + struct.pack(">IIB", pixels_per_meter, pixels_per_meter, 1)
    chunk = struct.pack(">I", 9) + data + struct.pack(">I", zlib.crc32(data))
    # The chunk goes right after IHDR: the 8 bytes signature, then IHDR's length, type, 13 bytes of data and CRC
    end_of_header = 8 + 4 + 4 + 13 + 4
    return png[:end_of_header] + chunk + png[end_of_header:]


def inch_figure(width, height, dpi=100):
    """Returns a figure and axes covering it, with coordinates in inches from the top left corner"""
    fig = Figure(figsize=(width, height), dpi=dpi, facecolor="white")
    axes = fig.add_axes((0, 0, 1, 1))
    axes.set_xlim(0, width)
    axes.set_ylim(height, 0)
    axes.axis("off")
    return fig, axes


def draw_palette(axes, colormap, x_0, y_0, nb_cols, col_width, row_height, swatch_width=0.7,
                 quantities=None):
    """Draw the numbered swatches with their hex and RGB codes and optional quantities, returns the height used"""
    nb_rows = -(-len(colormap) // nb_cols)
    for ind, col in enumerate(colormap):
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
        if quantities:
            axes.text(x + col_width - 0.2, y + row_height / 2, quantities[ind], ha="right", va="center", fontsize=8,
                      color="#444444")
    return nb_rows * row_height


def colormap_picture(colormap, dpi=COLORMAP_DPI):
    """Returns the colormap as a picture for the user: numbered swatches with their hex and RGB codes"""
    nb_cols = 2 if len(colormap) > 8 else 1
    nb_rows = -(-len(colormap) // nb_cols)
    margin, col_width, row_height = 0.3, 2.4, 0.5
    fig, axes = inch_figure(2 * margin + nb_cols * col_width, 2 * margin + nb_rows * row_height, dpi)
    draw_palette(axes, colormap, margin, margin, nb_cols, col_width, row_height)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())[..., :3].copy()


def palette_page(canvas, paper, title, subtitle, print_size=None):
    """PDF page with the palette, the source picture and the expected result"""
    width, height = paper
    margin = 0.6
    fig, axes = inch_figure(width, height)
    axes.text(margin, margin, "Palette", fontsize=20, weight="bold", va="top", color="#222222")
    axes.text(margin, margin + 0.45, " - ".join(filter(None, [title, subtitle])), fontsize=9, va="top",
              color="#777777")
    nb_cols = 2 if len(canvas.colormap) > 8 else 1
    y = margin + 0.9
    volumes = paint_volumes(canvas.coverage, printed_on_page(canvas.borders.shape, paper, print_size))
    quantities = [f"{share:.1%}, {'< 1' if volume < 1 else round(volume)} ml"
                  for share, volume in zip(canvas.coverage, volumes)]
    y += draw_palette(axes, canvas.colormap, margin, y, nb_cols, (width - 2 * margin) / 2, 0.46, quantities=quantities)
    axes.text(margin, y + 0.05, f"Estimated paint for one covering coat: about {max(1, round(sum(volumes)))} ml in "
              "total. Prepare some extra.", fontsize=8, va="top", color="#777777")
    y += 0.5

    # Source picture and expected result side by side, in the remaining space
    gap, caption = 0.3, 0.25
    box_width, box_height = (width - 2 * margin - gap) / 2, height - margin - y - caption
    pictures = [(canvas.src, "Source picture"), (canvas.quantified_image, "Expected result")]
    if box_height > 1:
        for ind, (picture, name) in enumerate(pictures):
            scale = min(box_width / picture.shape[1], box_height / picture.shape[0])
            pic_width, pic_height = picture.shape[1] * scale, picture.shape[0] * scale
            left = margin + ind * (box_width + gap) + (box_width - pic_width) / 2
            axes.text(left + pic_width / 2, y, name, ha="center", va="top", fontsize=9, color="#555555")
            # Embedded at 300 dpi, enough for printing and much lighter than the full resolution
            preview = resize(picture, min(max(picture.shape[:2]), int(300 * max(pic_width, pic_height))))
            axes.imshow(preview, extent=(left, left + pic_width, y + caption + pic_height, y + caption),
                        aspect="auto", interpolation="none")
    axes.set_xlim(0, width)
    axes.set_ylim(height, 0)
    return fig


def canvas_page(canvas, paper, title, subtitle, print_size=None):
    """PDF page with the canvas, borders and labels as vectors, see page_layout"""
    (width, height), scale, offset = page_layout(canvas.borders.shape, paper, print_size)
    fig, axes = inch_figure(width, height)
    if not print_size:
        axes.text(PAGE_MARGIN, PAGE_MARGIN, title, fontsize=15, weight="bold", va="top", color="#222222")
        axes.text(PAGE_MARGIN, PAGE_MARGIN + 0.3, subtitle, fontsize=9, va="top", color="#777777")
        axes.text(width - PAGE_MARGIN, PAGE_MARGIN, f"{len(canvas.colormap)} colors", fontsize=9, ha="right", va="top",
                  color="#777777")
    # The borders are 1 pxl wide, so both sides of a border are traced on the same pixels: no double lines
    contours, _ = cv2.findContours(canvas.borders.view(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [(cv2.approxPolyDP(contour, 0.8, True)[:, 0, :] + 0.5) * scale + offset for contour in contours]
    axes.add_collection(PolyCollection(polygons, closed=True, facecolors="none", edgecolors="#9a9a9a",
                                       linewidths=0.3))
    for region in canvas.regions:
        for spot in region["spots"]:
            size = label_size(spot["radius"], scale)
            x, y = (np.array([spot["x"], spot["y"]]) + 0.5) * scale + offset
            axes.text(x, y, str(region["label"]), ha="center", va="center", fontsize=size, color="#6b6b6b")
    return fig


def save_pdf(canvas, path, title=None, subtitle="", paper="a4", print_size=None):
    """Save a printable PDF: the palette page on the paper, then the canvas page, on a standard canvas size if given"""
    title = title or canvas.namefile
    paper = PAPER_SIZES[paper]
    with plt.rc_context({"pdf.fonttype": 42}), PdfPages(path, metadata={"Title": title}) as pdf:
        pdf.savefig(palette_page(canvas, paper, title, subtitle, print_size))
        pdf.savefig(canvas_page(canvas, paper, title, subtitle, print_size))
