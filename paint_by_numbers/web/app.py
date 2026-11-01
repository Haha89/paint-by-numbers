# -*- coding: utf-8 -*-
"""Web interface: upload a picture, compare the detail levels, download the canvas

Run it with `uv run app.py`, then open http://127.0.0.1:8051
"""

import argparse
import time
from pathlib import Path

import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, Patch, State, ctx, dcc, html, no_update
from flask import abort, send_file
from waitress import serve

from ..canvas import Canvas
from ..export import printed_size
from . import service

COLOR_MARKS = [5, 10, 15, 20, 25, 30, 35, 40]
CM_PER_INCH = 2.54
FIT_PAPER = "fit"
CROP_SHAPE_STYLE = {"line": {"color": "#4c6ef5", "width": 3}, "fillcolor": "rgba(76, 110, 245, 0.12)"}
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
THEME = {"primaryColor": "indigo", "defaultRadius": "md", "fontFamily": FONT, "headings": {"fontFamily": FONT}}
HIDDEN = {"display": "none"}


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


def print_size_choices():
    """The standard canvas sizes, in inches and centimeters, and the choice of fitting the paper"""
    sizes = [(f"{width} x {height} in ({width * CM_PER_INCH:.0f} x {height * CM_PER_INCH:.0f} cm)", name)
             for name, (width, height) in Canvas.print_sizes.items()]
    return choices([*sizes, ("Fit the paper, with a title", FIT_PAPER)])


def describe_print(source, crop, print_size, paper):
    """The size the picture will have once printed, for the current crop and choices"""
    if not source:
        return "Choose a picture to see its printed size."
    width, height = source["width"], source["height"]
    if crop:
        left, top, right, bottom = crop
        width, height = width * (right - left), height * (bottom - top)
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
                       dmc.Slider(id="nb-color", min=COLOR_MARKS[0], max=COLOR_MARKS[-1], step=1, value=20, mb="md",
                                  updatemode="drag",  # Keep each value, rather than only the one ending a gesture
                                  marks=[{"value": value, "label": str(value)} for value in COLOR_MARKS])], gap=6),
            dmc.Group(dmc.Button("Update the previews", id="generate", size="md", disabled=True), mt="sm")],
            gap="lg"))

    details = step(3, "Choose a detail level", "Click the preview you like. Fewer regions are easier to paint, more "
                                               "regions are closer to the picture.", html.Div(id="previews"))

    download = step(
        4, "Get your canvas", "The PDF has the palette on a sheet of paper, then the canvas at its printed size. A "
                              "picture with another shape than the canvas leaves white bands: crop it to fill it.",
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
            dmc.Text(id="print-estimate", fw=500, size="sm"),
            dmc.Group(dmc.Button("Build the canvas", id="build", size="md", disabled=True)),
            html.Div(id="build-result")],
            gap="lg"))

    return dmc.MantineProvider(
        dmc.Container([hero, source, settings, details, download,
                       dcc.Store(id="source"),  # {"session", "name", "width", "height"}
                       dcc.Store(id="crop"),  # [left, top, right, bottom] as fractions, or None
                       dcc.Store(id="settings"),  # The settings the current previews were made with
                       dcc.Store(id="selected-detail")],
                      size="lg", py=40),
        theme=THEME, forceColorScheme="light")


def detail_card(preview):
    """A card comparing a detail level, picked by clicking anywhere on it"""
    return html.Div(
        dmc.Card([
            dmc.CardSection(html.Img(src=preview["image"], className="preview-image")),
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
    app = Dash(__name__, title="Paint by numbers")
    app.layout = layout()

    @app.callback(
        Output("source", "data"),
        Output("crop-graph", "figure"),
        Output("crop-area", "style"),
        Output("crop", "data"),
        Output("source-status", "children"),
        Input("upload", "contents"),
        Input("load-url", "n_clicks"),
        Input("url", "n_submit"),
        State("upload", "filename"),
        State("url", "value"),
        running=[(Output("load-url", "loading"), True, False)],
        prevent_initial_call=True)
    def load_source(contents, _clicks, _submit, filename, url):
        from_upload = ctx.triggered_id == "upload"
        if not (contents if from_upload else url):
            return no_update, no_update, no_update, no_update, no_update
        try:
            if from_upload:
                name, data = filename or "picture", service.image_bytes_from_upload(contents)
            else:
                name, data = service.name_from_url(url), service.image_bytes_from_url(url)
            session = service.save_source(data, name)
            image_uri, width, height = service.source_preview(session)
        except service.ImageError as exc:
            return no_update, no_update, no_update, no_update, error(str(exc))
        source = {"session": session, "name": name, "width": width, "height": height}
        status = dmc.Text(f"Picture loaded: {name}", c="green", size="sm", fw=500)
        return source, crop_figure(image_uri, width, height), {}, None, status

    @app.callback(
        Output("crop", "data", allow_duplicate=True),
        Output("crop-graph", "figure", allow_duplicate=True),
        Input("crop-graph", "relayoutData"),
        Input("crop-reset", "n_clicks"),
        State("crop-graph", "figure"),
        State("source", "data"),
        prevent_initial_call=True)
    def update_crop(relayout, _reset, figure, source):
        if not source:
            return no_update, no_update
        shapes = (figure or {}).get("layout", {}).get("shapes") or []
        width, height = source["width"], source["height"]
        box = None
        if ctx.triggered_id == "crop-graph":
            box = next_crop_box(relayout, shapes[0] if shapes else None, width, height)
        if box is False:
            return no_update, no_update
        # Patch only the shapes, rather than sending the whole picture back
        patch = Patch()
        crop = None
        if box is None:
            patch["layout"]["shapes"] = []
        else:
            left, top, right, bottom = box
            patch["layout"]["shapes"] = [{"type": "rect", "editable": True, "x0": left - 0.5, "y0": top - 0.5,
                                          "x1": right - 0.5, "y1": bottom - 0.5, **CROP_SHAPE_STYLE}]
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

    @app.callback(Output("generate", "disabled"), Input("source", "data"))
    def enable_generate(source):
        return not source

    @app.callback(
        Output("previews", "children"),
        Output("settings", "data"),
        Output("selected-detail", "data"),
        Input("generate", "n_clicks"),
        Input("source", "data"),
        Input("crop", "data"),
        State("nb-color", "value"),
        running=[(Output("generate", "loading"), True, False)],
        prevent_initial_call=True)
    def generate_previews(_clicks, source, crop, nb_color):
        if not source:
            return no_update, no_update, no_update
        try:
            previews = service.render_previews(source["session"], nb_color, crop)
        except ValueError as exc:
            return error(str(exc)), None, None
        cards = dmc.SimpleGrid([detail_card(preview) for preview in previews], cols={"base": 1, "sm": 3}, spacing="lg")
        return cards, {"nb_color": nb_color, "crop": crop}, None

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
