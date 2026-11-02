# Roadmap

## To do

### Painting
- **Real paint matching**: map each palette color to the closest paint of a brand, or give a mixing recipe from a few base tubes. Needs a source for paint color data

### Printing
- **Multi-page posters**: split the canvas across several sheets with overlap marks
- **Labels scaled to the printed size**: their size is capped at 7 pt, which may be small on the largest canvases
- **PNG print resolution**: save the print size inside the PNG, so that it opens at the right size in other software
- **Line and label strength**: choose how dark they are, as the grey can show through light paint or be too faint on some printers

## Done

- **Subject weighting**: the faces of the picture, found with the YuNet model, or else its center, count 10 times more when choosing the colors, so that a face gets more shades and fewer large flat shapes
- **Framed 3D preview**: the painting at its real size on a wall with Three.js, with wood and gold frames
- **More K-mean samples**: 10000 pixels rather than 1000, for lower color errors on both the subject and the background. Capping the weight of the big uniform areas was measured too, and left out: little gain on the subject, for a clear loss on the background
- **Merge rarely used colors**: colors under a share of the canvas are replaced by their closest color, 1% for `easy`, 0.2% for `medium`, none for `faithful`
- **Paint quantities**: the PDF palette page shows the share of the canvas and the estimated paint of each color for the print size
- **Crop to the printed size**: a button in the web interface and `--crop-to-fill`, to avoid white bands
- **Warning for small prints**: the web interface says when many labels will touch the borders at the chosen size and detail level
- **ZIP download**: a single button for the PDF and the pictures
- **Live print estimate**: the printed size updates with the picture, the crop and the chosen size, before building
- **Browser tests**: load the real page in Chrome and go through the upload, link, crop, slider, build, download and phone layout
- **Waitress server**: replaces the development server of Dash, which sometimes reset the connection while sending plotly.min.js
- **Printed sizes**: standard canvas sizes in inches, from 8x10 to 24x36, with `--print-size` or in the web interface
- **Web interface with Dash**: upload a picture or give a link, crop it, compare the detail levels, preview and download the results and the PDF, with `uv run app.py`
- **Tests**: fast checks on small generated pictures, run with `uv run pytest`
- **Repeated labels**: large regions get several labels, spaced by `label_spacing_ratio` (1/6 of the largest dimension by default, 0 to disable)
- **Detail level**: `--detail easy|medium|faithful`, a balance between easy to paint and faithful to the picture
- **Package**: `canvas.py` split into the `paint_by_numbers` package
