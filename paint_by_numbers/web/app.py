# -*- coding: utf-8 -*-
"""Web interface: upload a picture, compare the detail levels, download the canvas

Run it with `uv run app.py`, then open http://127.0.0.1:8051
"""

import argparse
import time
import uuid
from pathlib import Path

import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import ALL, ClientsideFunction, Dash, Input, Output, Patch, State, ctx, dcc, html, no_update
from flask import abort, send_file
from waitress import serve

from ..canvas import Canvas
from ..export import fill_crop, printed_size
from . import service

COLOR_MARKS = [5, 10, 15, 20, 25, 30, 35, 40]
CM_PER_INCH = 2.54
FIT_PAPER = "fit"
CROP_SHAPE_STYLE = {"line": {"color": "#4c6ef5", "width": 3}, "fillcolor": "rgba(76, 110, 245, 0.12)"}
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
THEME = {"primaryColor": "indigo", "defaultRadius": "md", "fontFamily": FONT, "headings": {"fontFamily": FONT}}
HIDDEN = {"display": "none"}
# Only frame3d.js loads the Three.js modules, Dash would break on them by adding them to the page as plain scripts
THREE_FILES = r"^(three\.module|three\.core|OrbitControls|RoomEnvironment)\.js$"
# The Three.js modules import each other by this name
IMPORT_MAP = '<script type="importmap">{"imports": {"three": "/assets/vendor/three/three.module.js"}}</script>'
# Share of the labels touching their borders from which the page warns about it
CROWDED_WARNING = 0.25
# Longest title or subtitle kept from a saved project
MAX_TEXT_LENGTH = 200


def step(number, title, subtitle, *children):
    """A numbered card of the page"""
    header = dmc.Group([dmc.ThemeIcon(str(number), radius="xl", size="lg", variant="filled"),
                        dmc.Stack([dmc.Title(title, order=3), dmc.Text(subtitle, c="dimmed", size="sm")], gap=0)],
                       gap="md", mb="lg", wrap="nowrap")
    return dmc.Paper([header, *children], withBorder=True, shadow="sm", radius="lg", p="xl", mb="lg")


def error(message):
    """An error message for the page"""
    return dmc.Alert(message, color="red", variant="light", radius="md")


def choices(pairs):
    """The data of a segmented control or a select, from (label, value) pairs"""
    return [{"label": label, "value": value} for label, value in pairs]


# The frames of the 3D preview, see assets/frame3d.js
FRAMES = [{"group": "No frame", "items": [{"label": "Stretched canvas", "value": "none"}]},
          {"group": "Wood", "items": choices([("Oak", "oak"), ("Pine", "pine"), ("Walnut", "walnut"),
                                               ("Mahogany", "mahogany"), ("Black ebony", "ebony"),
                                               ("Whitewashed", "whitewashed")])},
          {"group": "Gold", "items": choices([("Old gold", "old-gold"), ("Baroque gold", "baroque-gold")])}]


FRAME_NAMES = {item["value"] for group in FRAMES for item in group["items"]}


def print_size_choices():
    """The standard canvas sizes, in inches and centimeters, and the choice of fitting the paper"""
    sizes = [(f"{width} x {height} in ({width * CM_PER_INCH:.0f} x {height * CM_PER_INCH:.0f} cm)", name)
             for name, (width, height) in Canvas.print_sizes.items()]
    return choices([*sizes, ("Fit the paper, with a title", FIT_PAPER)])


def cropped_size(source, crop):
    """The (width, height) of the displayed picture once cropped"""
    width, height = source["width"], source["height"]
    if crop:
        left, top, right, bottom = crop
        width, height = width * (right - left), height * (bottom - top)
    return width, height


def frame_size(source, crop, print_size, paper):
    """What the 3D preview needs, in inches: the (width, height) of the page in the frame and of the picture on it

    The frame is made to measure around the picture, without the white bands of a page of another shape
    """
    if not source:
        return None
    width, height = cropped_size(source, crop)
    picture = list(printed_size((height, width), None if print_size == FIT_PAPER else print_size, paper))
    return {"page": picture, "picture": picture}


def describe_print(source, crop, print_size, paper):
    """The size the picture will have once printed, for the current crop and choices"""
    if not source:
        return "Choose a picture to see its printed size."
    width, height = cropped_size(source, crop)
    standard = None if print_size == FIT_PAPER else print_size
    printed_width, printed_height = printed_size((height, width), standard, paper)
    text = (f"Printed size: {printed_width:.1f} x {printed_height:.1f} in "
            f"({printed_width * CM_PER_INCH:.0f} x {printed_height * CM_PER_INCH:.0f} cm)")
    if not standard:
        return f"{text}, on a {paper.upper() if paper != 'letter' else 'Letter'} sheet."
    page_width, page_height = Canvas.print_sizes[standard]
    filled = printed_width * printed_height / (page_width * page_height)
    if filled > 0.99:
        return f"{text}, filling the page."
    return f"{text}, {filled:.0%} of the {page_width} x {page_height} in page: crop the picture to fill it."


def describe_crowding(crowded, detail, print_size, paper):
    """A warning when many labels of the chosen detail level will touch the borders at the chosen size, over the whole
    picture or only on its faces"""
    shares = (crowded or {}).get(detail) or {}
    size = paper if print_size == FIT_PAPER else print_size
    share = (shares.get("all") or {}).get(size, 0)
    face_share = (shares.get("faces") or {}).get(size, 0)
    where = "the numbers"
    if face_share >= CROWDED_WARNING > share:
        where, share = "the numbers on the faces", face_share
    if share < CROWDED_WARNING:
        return None
    return dmc.Alert(f"About {share:.0%} of {where} will touch the borders of their region at this size. A larger "
                     "size or a lower detail level would make them easier to read.", color="yellow", variant="light",
                     radius="md")


def crop_shape(box):
    """The editable crop rectangle of the graph, for a (left, top, right, bottom) box in displayed pixels"""
    left, top, right, bottom = box
    return {"type": "rect", "editable": True, "x0": left - 0.5, "y0": top - 0.5, "x1": right - 0.5, "y1": bottom - 0.5,
            **CROP_SHAPE_STYLE}


def saved_project(project_id, source, crop, nb_color, detail, print_size, paper, title, subtitle, frame):
    """A project to keep in the browser: the picture, the crop and every setting of the page"""
    picture, thumbnail = service.project_pictures(source["session"])
    return {"id": project_id or uuid.uuid4().hex, "name": title or Path(source["name"]).stem,
            "saved_at": time.strftime("%Y-%m-%d %H:%M"), "picture_name": source["name"], "picture": picture,
            "thumbnail": thumbnail, "crop": crop, "nb_color": nb_color, "detail": detail, "print_size": print_size,
            "paper": paper, "title": title or "", "subtitle": subtitle or "", "frame": frame}


def project_settings(project):
    """The settings of a saved project, checked as they come from the browser storage: an unknown value keeps the
    current one of the page"""

    def choice(key, allowed):
        value = project.get(key)
        return value if isinstance(value, str) and value in allowed else no_update

    def text(key):
        value = project.get(key)
        return value[:MAX_TEXT_LENGTH] if isinstance(value, str) else ""

    nb_color, crop, detail = project.get("nb_color"), project.get("crop"), project.get("detail")
    valid_crop = isinstance(crop, list) and len(crop) == 4 and all(
        isinstance(value, (int, float)) and not isinstance(value, bool) for value in crop)
    valid_colors = isinstance(nb_color, int) and not isinstance(nb_color, bool)
    return {"crop": service.validate_crop(crop) if valid_crop else None,
            "nb_color": min(COLOR_MARKS[-1], max(COLOR_MARKS[0], nb_color)) if valid_colors else no_update,
            "detail": detail if isinstance(detail, str) and detail in Canvas.detail_levels else None,
            "print_size": choice("print_size", {*Canvas.print_sizes, FIT_PAPER}),
            "paper": choice("paper", Canvas.paper_sizes), "frame": choice("frame", FRAME_NAMES),
            "title": text("title"), "subtitle": text("subtitle")}


def saved_projects(index):
    """The projects saved in the browser, to open or delete"""
    index = [project for project in index or [] if isinstance(project, dict) and isinstance(project.get("id"), str)]
    if not index:
        return None
    items = []
    for project in index:
        thumbnail = project.get("thumbnail")
        items.append(dmc.Paper(dmc.Group([
            html.Img(src=thumbnail if isinstance(thumbnail, str) and thumbnail.startswith("data:image/") else None,
                     className="project-thumbnail"),
            dmc.Stack([dmc.Text(str(project.get("name") or "Project"), fw=600, size="sm", truncate="end"),
                       dmc.Text(f"Saved {project.get('saved_at', '')}", c="dimmed", size="xs")],
                      gap=0, style={"flex": 1, "minWidth": 0}),
            dmc.Button("Open", id={"type": "open-project", "index": project["id"]}, size="xs", variant="light"),
            dmc.Button("Delete", id={"type": "delete-project", "index": project["id"]}, size="xs", variant="subtle",
                       color="red")], gap="sm", wrap="nowrap"),
            withBorder=True, radius="md", p="xs", className="saved-project"))
    return dmc.Stack([dmc.Divider(my="lg"), dmc.Text("Or open a project saved in this browser", fw=600, size="sm"),
                      dmc.SimpleGrid(items, cols={"base": 1, "sm": 2}, spacing="sm")], gap="sm")


def fill_crop_box(source, crop, print_size):
    """The crop box in displayed pixels giving the current crop the shape of a standard size, centered in it"""
    left, top, right, bottom = crop or (0, 0, 1, 1)
    left, right = left * source["width"], right * source["width"]
    top, bottom = top * source["height"], bottom * source["height"]
    width, height = right - left, bottom - top
    fill_left, fill_top, fill_right, fill_bottom = fill_crop(width, height, print_size)
    return left + fill_left * width, top + fill_top * height, left + fill_right * width, top + fill_bottom * height


def layout():
    """The whole page"""
    hero = dmc.Stack([dmc.Title("Paint by numbers", order=1, c="indigo", fz=44, ta="center"),
                      dmc.Text("Turn any picture into a canvas to paint, with its palette and a printable PDF.",
                               c="dimmed", ta="center", size="lg")], gap=4, mb="xl")

    source = step(
        1, "Choose a picture", f"PNG, JPEG or WebP, up to {service.MAX_IMAGE_BYTES // 2 ** 20} MB.",
        dmc.SimpleGrid([
            # No accept filter: the upload would silently ignore other files, the server explains why it refuses them
            dcc.Upload(id="upload", className="dropzone", children=dmc.Stack(
                [dmc.Text("Drop your picture here", fw=600),
                 dmc.Text("or click to browse your files", c="dimmed", size="sm")], gap=2, align="center")),
            dmc.Stack([dmc.TextInput(id="url", label="Or paste a link to a picture", placeholder="https://..."),
                       dmc.Button("Load the link", id="load-url", variant="light")], justify="center", gap="sm")],
            cols={"base": 1, "sm": 2}, spacing="lg"),
        html.Div(id="saved-projects"),
        html.Div(id="source-status", className="status"),
        html.Div([
            dmc.Divider(my="lg"),
            dmc.Group([
                dmc.Stack([dmc.Text("Crop the picture (optional)", fw=600, size="sm"),
                           dmc.Text("Draw a rectangle on the picture to only keep this area. You can then move or "
                                    "resize it.", c="dimmed", size="xs")], gap=0),
                dmc.Button("Reset", id="crop-reset", variant="subtle", size="xs")],
                justify="space-between", align="flex-end", mb="sm"),
            dcc.Graph(id="crop-graph", style={"height": "420px"},
                      config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False}),
            dmc.Text(id="crop-status", c="dimmed", size="xs", ta="center", mt=4)],
            id="crop-area", style=HIDDEN))

    settings = step(
        2, "Set the colors", "Each color is a paint to prepare. Colors too close to tell apart are merged, so the "
                             "canvas may use a few less.",
        dmc.Stack([
            dmc.Stack([dmc.Text("Number of colors", fw=600, size="sm"),
                       dmc.Slider(id="nb-color", min=COLOR_MARKS[0], max=COLOR_MARKS[-1], step=1, value=30, mb="md",
                                  updatemode="drag",  # Keep each value, rather than only the one ending a gesture
                                  marks=[{"value": value, "label": str(value)} for value in COLOR_MARKS])], gap=6),
            dmc.Group(dmc.Button("Update the previews", id="generate", size="md", disabled=True), mt="sm")],
            gap="lg"))

    details = step(3, "Choose a detail level", "Click the preview you like. Fewer regions are easier to paint, more "
                                               "regions are closer to the picture.", html.Div(id="previews"))

    download = step(
        4, "Get your canvas", "The PDF has the palette with the paint to prepare on a sheet of paper, then the canvas at "
                              "its printed size. A picture with another shape than the canvas leaves white bands: crop "
                              "it to fill it.",
        dmc.Stack([
            dmc.SimpleGrid([
                dmc.TextInput(id="title", label="PDF title", placeholder="Name of the picture"),
                dmc.TextInput(id="subtitle", label="PDF subtitle", placeholder="e.g. the artist and year")],
                cols={"base": 1, "sm": 2}),
            dmc.Group([
                dmc.Select(id="print-size", label="Printed size", value="16x20", data=print_size_choices(), w=300,
                           allowDeselect=False),
                dmc.Stack([dmc.Text("Paper for the palette", fw=600, size="sm"),
                           dmc.SegmentedControl(id="paper", value="letter", radius="xl", data=choices(
                               (name.upper() if name != "letter" else "Letter", name)
                               for name in sorted(Canvas.paper_sizes)))], gap=6)],
                gap="xl", align="flex-start"),
            dmc.Group([dmc.Text(id="print-estimate", fw=500, size="sm"),
                       dmc.Button("Crop to fill the page", id="crop-fill", variant="light", size="xs", disabled=True)],
                      gap="md"),
            html.Div(id="print-warning"),
            html.Div([
                dmc.Group([
                    dmc.Stack([dmc.Text("See it framed", fw=600, size="sm"),
                               dmc.Text("Your painting at its real size, once finished. Drag to turn around it, Ctrl + "
                                        "scroll or pinch to get closer. The frame is only a preview: it is not in the files.",
                                        c="dimmed", size="xs")], gap=0, style={"flex": "1 1 280px"}),
                    dmc.Select(id="frame", label="Frame", value="old-gold", data=FRAMES, allowDeselect=False, w=220)],
                    justify="space-between", align="flex-end", mb="sm"),
                html.Div(id="frame-view", className="frame-view"),
                dmc.Text(id="frame-message", c="red", size="sm", mt=4)],
                id="frame-area", style=HIDDEN),
            dmc.Group([dmc.Button("Build the canvas", id="build", size="md", disabled=True),
                       dmc.Button("Save the project", id="save-project", size="md", variant="light", disabled=True),
                       dmc.Text(id="project-status", size="sm")]),
            html.Div(id="build-result")],
            gap="lg"))

    return dmc.MantineProvider(
        dmc.Container([hero, source, settings, details, download,
                       dcc.Store(id="source"),  # {"session", "name", "width", "height"}
                       dcc.Store(id="crop"),  # [left, top, right, bottom] as fractions, or None
                       dcc.Store(id="settings"),  # The settings the current previews were made with
                       dcc.Store(id="selected-detail"),
                       dcc.Store(id="frame-size"),  # {"page": [width, height], "picture": [width, height]} in inches
                       # {detail: {"all" or "faces": {print size or paper: share of crowded labels}}}
                       dcc.Store(id="crowded"),
                       dcc.Store(id="project-index"),  # The saved projects, without their pictures
                       dcc.Store(id="saved-project"),  # The last project to write in the browser storage
                       dcc.Store(id="project-to-load"),  # The project read from the browser storage, to open
                       dcc.Store(id="project-id"),  # The saved project the page shows, saved again under its id
                       dcc.Store(id="pending-detail")],  # The detail level of an opened project, for its previews
                      size="lg", py=40),
        theme=THEME, forceColorScheme="light")


def detail_card(preview):
    """A card comparing a detail level, picked by clicking anywhere on it"""
    return html.Div(
        dmc.Card([
            dmc.CardSection(html.Img(id=f"preview-image-{preview['detail']}", src=preview["image"],
                                     className="preview-image")),
            dmc.Group([dmc.Text(preview["detail"].capitalize(), fw=700, size="lg"),
                       dmc.Badge(f"{preview['colors']} colors", variant="light")], justify="space-between", mt="md"),
            dmc.Text(preview["description"], c="dimmed", size="sm"),
            dmc.Text(f"About {preview['regions']} regions", fw=600, size="sm", mt="xs")],
            withBorder=True, radius="lg", padding="md", shadow="xs"),
        id={"type": "detail-card", "index": preview["detail"]}, className="detail-card", n_clicks=0)


def build_result(built, session_id):
    """The final pictures in tabs, and the button to download them with the PDF"""
    # The timestamp keeps the browser from serving a previous build of the same session
    button = html.A(dmc.Button("Download everything (ZIP)", id="download"), download="",
                    href=f"/files/{session_id}?v={time.time_ns()}")
    tabs = [("Canvas", "canvas"), ("Expected result", "result"), ("Colormap", "colormap")]
    return dmc.Stack([
        dmc.Text(f"Canvas ready: {built['regions']} regions, {built['colors']} colors.", c="green", size="sm", fw=500),
        dmc.Stack([dmc.Group(button),
                   dmc.Text("The printable PDF, the canvas, the expected result and the colormap.", c="dimmed",
                            size="xs")], gap=4),
        dmc.Tabs([dmc.TabsList([dmc.TabsTab(label, value=kind) for label, kind in tabs]),
                  *[dmc.TabsPanel(html.Img(src=built[kind], className="result-image"), value=kind)
                    for _, kind in tabs]],
                 value="canvas")],
        gap="md")


def crop_figure(image_uri, width, height):
    """The source picture as a Plotly figure where dragging draws the crop box

    Pixel (0, 0) is centered on (0, 0), so the picture spans -0.5 to width - 0.5
    """
    figure = go.Figure(go.Image(source=image_uri, hoverinfo="skip"))
    figure.update_layout(margin={"l": 0, "r": 0, "t": 0, "b": 0}, dragmode="drawrect", newshape=CROP_SHAPE_STYLE,
                         xaxis={"visible": False, "range": [-0.5, width - 0.5]},
                         yaxis={"visible": False, "range": [height - 0.5, -0.5], "scaleanchor": "x"},
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", shapes=[])
    return figure


def next_crop_box(relayout, current, width, height):
    """The crop box after an interaction with the graph, in pixels of the displayed picture

    Returns None for no crop, or False when the interaction doesn't touch the crop. relayout is the relayoutData of the
    graph: a list of shapes when one is drawn or erased, shapes[0].x0-style keys when one is moved. current is the crop
    shape on the graph, if any.
    """
    relayout = relayout or {}
    if "shapes" in relayout:
        if not relayout["shapes"]:
            return None
        current = relayout["shapes"][-1]  # Drawing again replaces the previous box
    elif current and any(key.startswith("shapes[") for key in relayout):
        current = dict(current)
        for key, value in relayout.items():
            if key.startswith("shapes[") and "." in key:
                current[key.rsplit(".", 1)[1]] = value
    else:
        return False  # An autosize or similar
    left, right = sorted((float(current["x0"]) + 0.5, float(current["x1"]) + 0.5))
    top, bottom = sorted((float(current["y0"]) + 0.5, float(current["y1"]) + 0.5))
    left, right = max(0.0, left), min(float(width), right)
    top, bottom = max(0.0, top), min(float(height), bottom)
    if right - left < 1 or bottom - top < 1:
        return None
    return left, top, right, bottom


def create_app():
    """The Dash application"""
    app = Dash(__name__, title="Paint by numbers", assets_ignore=THREE_FILES)
    app.index_string = app.index_string.replace("{%metas%}", "{%metas%}" + IMPORT_MAP)
    app.layout = layout()

    @app.callback(
        Output("source", "data"),
        Output("crop-graph", "figure"),
        Output("crop-area", "style"),
        Output("crop", "data"),
        Output("source-status", "children"),
        Output("project-id", "data"),
        Output("pending-detail", "data"),
        Output("nb-color", "value"),
        Output("print-size", "value"),
        Output("paper", "value"),
        Output("frame", "value"),
        Output("title", "value"),
        Output("subtitle", "value"),
        Input("upload", "contents"),
        Input("load-url", "n_clicks"),
        Input("url", "n_submit"),
        Input("project-to-load", "data"),
        State("upload", "filename"),
        State("url", "value"),
        running=[(Output("load-url", "loading"), True, False)],
        prevent_initial_call=True)
    def load_source(contents, _clicks, _submit, project, filename, url):
        unchanged = (no_update,) * 13
        kind = {"upload": "upload", "project-to-load": "project"}.get(ctx.triggered_id, "url")
        if not {"upload": contents, "project": project, "url": url}[kind]:
            return unchanged
        try:
            if kind == "upload":
                name, data = filename or "picture", service.image_bytes_from_upload(contents)
            elif kind == "url":
                name, data = service.name_from_url(url), service.image_bytes_from_url(url)
            else:
                name, picture = project.get("picture_name"), project.get("picture")
                if not isinstance(picture, str):
                    raise service.ImageError("This project has no picture.")
                name, data = name if isinstance(name, str) else "picture", service.image_bytes_from_upload(picture)
            session = service.save_source(data, name)
            image_uri, width, height = service.source_preview(session)
        except service.ImageError as exc:
            return *unchanged[:4], error(str(exc)), *unchanged[5:]
        source = {"session": session, "name": name, "width": width, "height": height}
        figure = crop_figure(image_uri, width, height)
        if kind != "project":
            status = dmc.Text(f"Picture loaded: {name}", c="green", size="sm", fw=500)
            # A new picture is a new project
            return source, figure, {}, None, status, None, None, *unchanged[7:]
        settings = project_settings(project)
        crop = settings["crop"]
        if crop:
            left, top, right, bottom = crop
            figure.update_layout(shapes=[crop_shape((left * width, top * height, right * width, bottom * height))])
        status = dmc.Text(f"Project opened: {project.get('name') or name}", c="green", size="sm", fw=500)
        project_id = project.get("id") if isinstance(project.get("id"), str) else None
        return (source, figure, {}, None if crop is None else list(crop), status, project_id, settings["detail"],
                settings["nb_color"], settings["print_size"], settings["paper"], settings["frame"], settings["title"],
                settings["subtitle"])

    @app.callback(
        Output("crop", "data", allow_duplicate=True),
        Output("crop-graph", "figure", allow_duplicate=True),
        Input("crop-graph", "relayoutData"),
        Input("crop-reset", "n_clicks"),
        Input("crop-fill", "n_clicks"),
        State("crop-graph", "figure"),
        State("source", "data"),
        State("crop", "data"),
        State("print-size", "value"),
        prevent_initial_call=True)
    def update_crop(relayout, _reset, _fill, figure, source, current_crop, print_size):
        if not source:
            return no_update, no_update
        shapes = (figure or {}).get("layout", {}).get("shapes") or []
        width, height = source["width"], source["height"]
        box = None
        if ctx.triggered_id == "crop-graph":
            box = next_crop_box(relayout, shapes[0] if shapes else None, width, height)
        elif ctx.triggered_id == "crop-fill":
            box = False if print_size == FIT_PAPER else fill_crop_box(source, current_crop, print_size)
        if box is False:
            return no_update, no_update
        # Patch only the shapes, rather than sending the whole picture back
        patch = Patch()
        crop = None
        if box is None:
            patch["layout"]["shapes"] = []
        else:
            left, top, right, bottom = box
            patch["layout"]["shapes"] = [crop_shape(box)]
            crop = service.validate_crop([left / width, top / height, right / width, bottom / height])
        return (None if crop is None else [round(value, 4) for value in crop]), patch

    @app.callback(Output("crop-status", "children"), Input("crop", "data"))
    def describe_crop(crop):
        if not crop:
            return "Whole picture."
        left, top, right, bottom = crop
        return f"Area kept: {(right - left) * 100:.0f}% of the width x {(bottom - top) * 100:.0f}% of the height."

    app.callback(Output("print-estimate", "children"), Input("source", "data"), Input("crop", "data"),
                 Input("print-size", "value"), Input("paper", "value"))(describe_print)

    app.callback(Output("frame-size", "data"), Input("source", "data"), Input("crop", "data"),
                 Input("print-size", "value"), Input("paper", "value"))(frame_size)
    app.clientside_callback(ClientsideFunction("frame3d", "update"), Output("frame-area", "style"),
                            Input("selected-detail", "data"), Input("frame", "value"), Input("frame-size", "data"))

    @app.callback(Output("crop-fill", "disabled"), Input("source", "data"), Input("print-size", "value"))
    def enable_crop_fill(source, print_size):
        return not source or print_size == FIT_PAPER

    app.callback(Output("print-warning", "children"), Input("crowded", "data"), Input("selected-detail", "data"),
                 Input("print-size", "value"), Input("paper", "value"))(describe_crowding)

    @app.callback(Output("generate", "disabled"), Input("source", "data"))
    def enable_generate(source):
        return not source

    @app.callback(
        Output("previews", "children"),
        Output("settings", "data"),
        Output("selected-detail", "data"),
        Output("crowded", "data"),
        Output("pending-detail", "data", allow_duplicate=True),
        Input("generate", "n_clicks"),
        Input("source", "data"),
        Input("crop", "data"),
        State("nb-color", "value"),
        State("pending-detail", "data"),
        running=[(Output("generate", "loading"), True, False)],
        prevent_initial_call=True)
    def generate_previews(_clicks, source, crop, nb_color, pending_detail):
        if not source:
            return no_update, no_update, no_update, no_update, no_update
        try:
            previews = service.render_previews(source["session"], nb_color, crop)
        except ValueError as exc:
            return error(str(exc)), None, None, None, None
        cards = dmc.SimpleGrid([detail_card(preview) for preview in previews], cols={"base": 1, "sm": 3}, spacing="lg")
        crowded = {preview["detail"]: preview["crowded"] for preview in previews}
        # An opened project gets its detail level back, once
        return cards, {"nb_color": nb_color, "crop": crop}, pending_detail, crowded, None

    @app.callback(
        Output("selected-detail", "data", allow_duplicate=True),
        Input({"type": "detail-card", "index": ALL}, "n_clicks"),
        prevent_initial_call=True)
    def select_detail(clicks):
        # Cards being (re)drawn also fires this, with no clicks
        if not ctx.triggered_id or not any(clicks):
            return no_update
        return ctx.triggered_id["index"]

    @app.callback(
        Output({"type": "detail-card", "index": ALL}, "className"),
        Output("build", "disabled"),
        Output("build-result", "children"),
        Input("selected-detail", "data"),
        State({"type": "detail-card", "index": ALL}, "id"))
    def highlight_selection(selected, card_ids):
        classes = ["detail-card selected" if card["index"] == selected else "detail-card" for card in card_ids]
        return classes, selected is None, None

    @app.callback(
        Output("build-result", "children", allow_duplicate=True),
        Input("build", "n_clicks"),
        State("source", "data"),
        State("settings", "data"),
        State("selected-detail", "data"),
        State("print-size", "value"),
        State("title", "value"),
        State("subtitle", "value"),
        State("paper", "value"),
        running=[(Output("build", "loading"), True, False)],
        prevent_initial_call=True)
    def build_canvas(_clicks, source, settings, detail, print_size, title, subtitle, paper):
        if not (source and settings and detail):
            return no_update
        try:
            built = service.build_files(source["session"], detail, settings["nb_color"], settings["crop"],
                                        title=title or Path(source["name"]).stem, subtitle=subtitle or "", paper=paper,
                                        print_size=None if print_size == FIT_PAPER else print_size)
        except (ValueError, OSError) as exc:
            return error(str(exc))
        return build_result(built, source["session"])

    # The browser storage is only touched in the page, see assets/projects.js
    app.clientside_callback(ClientsideFunction("projects", "save"), Output("project-index", "data"),
                            Output("project-status", "children"), Output("project-status", "c"),
                            Input("saved-project", "data"))
    app.clientside_callback(ClientsideFunction("projects", "open"), Output("project-to-load", "data"),
                            Output("project-status", "children", allow_duplicate=True),
                            Output("project-status", "c", allow_duplicate=True),
                            Input({"type": "open-project", "index": ALL}, "n_clicks"), prevent_initial_call=True)
    app.clientside_callback(ClientsideFunction("projects", "remove"),
                            Output("project-index", "data", allow_duplicate=True),
                            Input({"type": "delete-project", "index": ALL}, "n_clicks"), prevent_initial_call=True)
    app.callback(Output("saved-projects", "children"), Input("project-index", "data"))(saved_projects)

    @app.callback(Output("save-project", "disabled"), Input("source", "data"))
    def enable_save(source):
        return not source

    @app.callback(
        Output("saved-project", "data"),
        Output("project-id", "data", allow_duplicate=True),
        Input("save-project", "n_clicks"),
        State("project-id", "data"),
        State("source", "data"),
        State("crop", "data"),
        State("nb-color", "value"),
        State("selected-detail", "data"),
        State("print-size", "value"),
        State("paper", "value"),
        State("title", "value"),
        State("subtitle", "value"),
        State("frame", "value"),
        running=[(Output("save-project", "loading"), True, False)],
        prevent_initial_call=True)
    def save_project(_clicks, project_id, source, *settings):
        if not source:
            return no_update, no_update
        try:
            project = saved_project(project_id, source, *settings)
        except service.ImageError:
            return no_update, no_update
        return project, project["id"]

    @app.server.route("/files/<session_id>")
    def serve_zip(session_id):
        # The ZIP goes over plain HTTP rather than base64 inside a callback response, as it can be large
        try:
            path = service.built_zip(session_id)
        except service.ImageError:
            abort(404)
        return send_file(path, as_attachment=True, download_name=path.name)

    return app


def main(argv=None):
    """Run the web interface"""
    parser = argparse.ArgumentParser(description="Run the paint-by-numbers web interface")
    parser.add_argument("--host", default="127.0.0.1", help="interface to listen on (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8051, help="port to listen on (default 8051)")
    parser.add_argument("--debug", action="store_true", help="enable the Dash hot reload and debug panel")
    args = parser.parse_args(argv)
    app = create_app()
    if args.debug:
        app.run(host=args.host, port=args.port, debug=True)
        return
    # Not the development server of Dash, which sometimes resets the connection while sending plotly.min.js
    print(f"Paint by numbers is running on http://{args.host}:{args.port}/", flush=True)
    serve(app.server, host=args.host, port=args.port, threads=8)
