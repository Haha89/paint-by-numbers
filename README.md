# paint-by-numbers

[![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue)](LICENSE)

A quick Python tool to transform any picture in a paint-by-number canvas, using opencv

# How it works
1) Picture is resized, denoised and cleaned using morphomat  
2) K-mean algorithm, in the CIELAB color space, to detect the k-colors summarizing the best the picture  
3) Transformation of the picture to match these colors, with smoothed borders  
4) Regions too small or too thin to be painted are merged into their neighbours  
5) Drawing of the borders and of the label of each region on a new canvas  
6) Creation of the colormap for the user, and of a printable PDF (palette page, then vector canvas page)

# This project includes
A set of PNG pictures in `./inputs` folder, associated canvas and colormaps in the `./outputs` folder
The definition of the class Canvas in the python script `canvas.py`

### Requirements:
- Numpy
- Matplotlib
- OpenCV
- scikit-learn

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install them with `uv sync`.

### Execution:
run `uv run run.py path nb_color [--plot] [--no-save] [--pixel-size N] [--output-dir DIR] [--no-pdf] [--title T] [--subtitle S] [--paper P]` with:

  **path**: path of the source picture  
  **nb_color** : number of colors you want in the canvas (10 to 20)  
  **--plot**: optional, show the results in plots  
  **--no-save**: optional, do not save the results in the output folder  
  **--pixel-size**: optional, integer, size in pixel of the largest dimension of the output canvas (default 4000)  
  **--output-dir**: optional, folder where results are saved (default `./outputs`)  
  **--no-pdf**: optional, do not save the printable PDF  
  **--title**: optional, title printed in the PDF (default: name of the picture)  
  **--subtitle**: optional, subtitle printed in the PDF, e.g. the artist and year  
  **--paper**: optional, PDF paper size: `a4`, `a3` or `letter` (default `a4`)

Example: `uv run run.py inputs/chirico.jpg 20 --title "Place d'Italie" --subtitle "Giorgio De Chirico, 1922"`

### Result:

#### De Chirico, Place d'Italie, 1922
![alt text](https://github.com/Haha89/paint-by-number/blob/master/inputs/chirico.jpg "De Chirico, Place d'Italie, 1922")
#### Expected result (20 colors)
![alt text](https://github.com/Haha89/paint-by-number/blob/master/outputs/chirico-result.png "Expected result")
#### Canvas
![alt text](https://github.com/Haha89/paint-by-number/blob/master/outputs/chirico-canvas.png "Canvas")
#### Colormap
![alt text](https://github.com/Haha89/paint-by-number/blob/master/outputs/chirico-colormap.png "Colormap")

### Next Steps:
- Better positioning of the label in each region
- Better contour filtering to remove small regions or regions almost fully included in another one
