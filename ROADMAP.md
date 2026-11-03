# Roadmap

## To do

### Painting
- **Real paint matching**: map each palette color to the closest paint of a brand, or give a mixing recipe from a few base tubes. Needs a source for paint color data
- **Calibrate the paint estimate**: 1 ml per 50 cm² is a rough guess, to check against a real canvas or a manufacturer's coverage figure
- **Colors for each detail level**: a default of about 15 colors for `easy` and 30 for the others, still changeable with the slider. `easy` has large regions, but can need 29 paints on the Joconde

### Faces
- **Crowded numbers on faces**: numbers can touch or overlap around the eyes and the mouth (half of those in the Joconde's face with `detailed` at 16x20). Options: a higher `face_detail`, or a legend or leader lines for the tiniest regions

### Printing
- **Multi-page posters**: split the canvas across several sheets with overlap marks
- **Labels scaled to the printed size**: their size is capped at 7 pt, which may be small on the largest canvases
- **Line and label strength**: choose how dark they are, as the grey can show through light paint or be too faint on some printers

### Web interface
- **Standard or custom frame in the 3D preview**: the frame fits the picture, while the PDF page keeps white bands when the shapes differ. Say so in the preview, or offer a frame of the standard size

### Tests
- **Quality checks on real pictures**: save reference images for the detail thresholds and the face weights, as the current tests can't see a loss of quality such as a face losing its eyes

## Done

- **Saved projects**: the web interface keeps projects in the localStorage of the browser: the picture (a JPEG of at most 2400 pxl, enough as the canvas is denoised at 1500 pxl), the crop and every setting, detail level and frame included. They are listed in step 1 to open or delete, and the page says when the storage, about 5 MB, is full
- **PNG print resolution**: the canvas and the expected result hold the resolution of the PDF canvas page, so they open at their printed size in other software
- **Crowding warning on faces**: the web interface also judges the faces alone, as the whole picture can stay under the 25% limit while a face is crowded (17% overall but 52% on the face for the Joconde with `detailed` at 16x20)
- **Scrolling over the 3D preview**: the wheel scrolls the page and Ctrl + wheel zooms. On a touch screen, a vertical swipe scrolls, a horizontal one turns and pinching zooms (not tested on a real phone yet)
- **Face details**: the minimum size of the regions is lower inside the faces, to keep the eyes and the mouth: about 3 times lower for `easy`, 2 times for `medium` and 4 times for `detailed`, which still leaves the 3 levels in that order of region size
- **30 colors by default**: in the web interface, for softer shading. More adds little, as the closest colors get merged
- **Photographed wood**: the wood frames of the 3D preview use CC0 textures from ambientCG
- **Subject weighting**: the faces of the picture, found with the YuNet model, or else its center, count 10 times more when choosing the colors, so that a face gets more shades and fewer large flat shapes
- **Framed 3D preview**: the painting at its real size on a wall with Three.js, with wood and gold frames
- **More K-mean samples**: 10000 pixels rather than 1000, for lower color errors on both the subject and the background. Capping the weight of the big uniform areas was measured too, and left out: little gain on the subject, for a clear loss on the background
- **Merge rarely used colors**: colors under a share of the canvas are replaced by their closest color, 1% for `easy`, 0.2% for `medium`, none for `detailed`
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
- **Detail level**: `--detail easy|medium|detailed`, a balance between easy to paint and faithful to the picture
- **Package**: `canvas.py` split into the `paint_by_numbers` package
