# -*- coding: utf-8 -*-

import os
import sys

from canvas import Canvas

if __name__ == "__main__":

    if len(sys.argv) < 3:
        raise SyntaxError("Insufficient arguments. Please provide :\n"
                          "  -path: path of the source picture \n"
                          "  -nb_color : number of colors you want in the canvas (10 to 20)\n"
                          "  -plot: optional, boolean to set to True if you want to see some plots. Default False\n"
                          "  -save: optional, boolean to set to True if you want to save results in the ./outputs folder. Default True\n"
                          "  -pixel_size: optional, interger, size in pixel of the largest dimension of the output canvas (default 4000)\n")

    if len(sys.argv) != 3:
        CANVAS = Canvas(os.path.abspath(sys.argv[1]), int(sys.argv[2]), *sys.argv[3:])
    else:

        CANVAS = Canvas(os.path.abspath(sys.argv[1]), int(sys.argv[2]))

    CANVAS.generate()
    CANVAS.display_colormap()
    print("Canvas generated")
