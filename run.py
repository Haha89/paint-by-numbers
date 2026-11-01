# -*- coding: utf-8 -*-

import argparse

from paint_by_numbers import Canvas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform a picture in a paint-by-number canvas")
    parser.add_argument("path", help="path of the source picture")
    parser.add_argument("nb_color", type=int, help="number of colors you want in the canvas (10 to 20)")
    parser.add_argument("--plot", action="store_true", help="show the results in plots")
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="do not save the results in the output folder")
    parser.add_argument("--pixel-size", type=int, default=4000,
                        help="size in pixel of the largest dimension of the output canvas (default 4000)")
    parser.add_argument("--detail", choices=list(Canvas.detail_levels), default="medium",
                        help="balance between easy to paint and faithful to the picture (default medium)")
    parser.add_argument("--output-dir", default="./outputs", help="folder where results are saved (default ./outputs)")
    parser.add_argument("--no-pdf", dest="pdf", action="store_false", help="do not save the printable PDF")
    parser.add_argument("--title", help="title printed in the PDF (default: name of the picture)")
    parser.add_argument("--subtitle", default="", help="subtitle printed in the PDF, e.g. the artist and year")
    parser.add_argument("--paper", choices=sorted(Canvas.paper_sizes), default="a4", help="PDF paper size (default a4)")
    parser.add_argument("--print-size", choices=list(Canvas.print_sizes),
                        help="standard canvas size in inches: the PDF canvas page gets exactly this size, without "
                             "margins (default: fit the paper)")
    args = parser.parse_args()

    CANVAS = Canvas(args.path, args.nb_color, pixel_size=args.pixel_size, detail=args.detail)
    CANVAS.generate()
    if args.save:
        CANVAS.save(args.output_dir, args.pdf, args.title, args.subtitle, args.paper, args.print_size)
    if args.plot:
        CANVAS.plot()
    print("Canvas generated")
