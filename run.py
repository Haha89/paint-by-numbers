# -*- coding: utf-8 -*-

import argparse

from canvas import Canvas

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform a picture in a paint-by-number canvas")
    parser.add_argument("path", help="path of the source picture")
    parser.add_argument("nb_color", type=int, help="number of colors you want in the canvas (10 to 20)")
    parser.add_argument("--plot", action="store_true", help="show the results in plots")
    parser.add_argument("--no-save", dest="save", action="store_false",
                        help="do not save the results in the output folder")
    parser.add_argument("--pixel-size", type=int, default=4000,
                        help="size in pixel of the largest dimension of the output canvas (default 4000)")
    parser.add_argument("--output-dir", default="./outputs", help="folder where results are saved (default ./outputs)")
    args = parser.parse_args()

    CANVAS = Canvas(args.path, args.nb_color, pixel_size=args.pixel_size)
    CANVAS.generate()
    if args.save:
        CANVAS.save(args.output_dir)
    if args.plot:
        CANVAS.plot()
    print("Canvas generated")
