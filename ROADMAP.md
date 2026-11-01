# Roadmap

## To do

### Painting
- **Real paint matching**: map each palette color to the closest paint of a brand, or give a mixing recipe from a few base tubes. Needs a source for paint color data
- **Paint quantities**: show the share of the canvas of each color and estimate the paint needed for a print size, on the PDF palette page
- **Merge rarely used colors**: a minimum coverage threshold, to avoid a pot of paint for a color barely used. The `easy` detail level could use it too

### Printing
- **Multi-page posters**: split the canvas across several sheets with overlap marks
- **Crop to the printed size**: crop the picture to the shape of the chosen canvas, to avoid white bands
- **Labels scaled to the printed size**: their size is capped at 7 pt, which may be small on the largest canvases

## Done

- **Printed sizes**: standard canvas sizes in inches, from 8x10 to 24x36, with `--print-size` or in the web interface
- **Web interface with Dash**: upload a picture or give a link, crop it, compare the detail levels, preview and download the results and the PDF, with `uv run app.py`
- **Tests**: fast checks on small generated pictures, run with `uv run pytest`
- **Repeated labels**: large regions get several labels, spaced by `label_spacing_ratio` (1/6 of the largest dimension by default, 0 to disable)
- **Detail level**: `--detail easy|medium|faithful`, a balance between easy to paint and faithful to the picture
- **Package**: `canvas.py` split into the `paint_by_numbers` package
