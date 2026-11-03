# -*- coding: utf-8 -*-
"""The real page in the installed Chrome, as the callbacks and the components only run together in a browser"""

import functools
import http.server
import re
import threading
import time
import zipfile

import cv2
import numpy as np
import pytest
from playwright.sync_api import Error, expect, sync_playwright
from waitress.server import create_server

from paint_by_numbers.web import service
from paint_by_numbers.web.app import create_app

TIMEOUT = 30000
# The final canvases are built smaller than in the app, to keep the tests fast
BUILD_SIZE = 800


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as playwright:
        try:
            chrome = playwright.chromium.launch(channel="chrome")
        except Error as exc:
            pytest.skip(f"Chrome is not available: {exc}")
        yield chrome
        chrome.close()


@pytest.fixture(scope="module")
def server_calls():
    """What the page asked the server: the last arguments of the previews and of the build"""
    return {}


def recorded(name, function, calls):
    """A service function which keeps its last arguments"""

    def wrapper(*args, **kwargs):
        calls[name] = (args, kwargs)
        return function(*args, **kwargs)

    return wrapper


@pytest.fixture(scope="module")
def app_url(tmp_path_factory, server_calls):
    originals = service.SESSION_ROOT, service.render_previews, service.build_files
    service.SESSION_ROOT = tmp_path_factory.mktemp("sessions")
    service.render_previews = recorded("previews", service.render_previews, server_calls)
    service.build_files = recorded("build", functools.partial(service.build_files, pixel_size=BUILD_SIZE), server_calls)
    server = create_server(create_app().server, host="127.0.0.1", port=0, threads=8)
    threading.Thread(target=server.run, daemon=True).start()
    yield f"http://127.0.0.1:{server.effective_port}/"
    server.close()
    service.SESSION_ROOT, service.render_previews, service.build_files = originals


@pytest.fixture
def page(browser, app_url):
    """The loaded page, failing the test if it logs an error"""
    context = browser.new_context(viewport={"width": 1280, "height": 900}, accept_downloads=True)
    page = context.new_page()
    page.set_default_timeout(TIMEOUT)
    expect.set_options(timeout=TIMEOUT)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)
    page.on("requestfailed", lambda request: errors.append(f"{request.url}: {request.failure}"))
    page.goto(app_url)
    yield page
    context.close()
    assert not errors


def upload(page, path):
    page.locator("#upload input[type=file]").set_input_files(path)


def cards(page):
    return page.locator(".detail-card")


def choose_print_size(page, name):
    page.locator("#print-size").click()
    page.get_by_role("option", name=name).click()


def upload_and_select(page, path, card=0):
    upload(page, path)
    expect(cards(page)).to_have_count(3)
    cards(page).nth(card).click()
    expect(cards(page).nth(card)).to_have_class("detail-card selected")


def build(page):
    """Build the canvas, waiting for this build and not for the result of a previous one"""
    link = page.locator("#build-result a")
    previous = link.get_attribute("href") if link.count() else None
    page.locator("#build").click()
    expect(page.locator("#build-result")).to_contain_text("Canvas ready")
    if previous:
        expect(link).not_to_have_attribute("href", previous)


def download_zip(page, target):
    """Download the ZIP in the target path and return it"""
    with page.expect_download() as download:
        page.locator("#download").click()
    download.value.save_as(target)
    return zipfile.ZipFile(target)


def pdf_pages(archive, name="sample.pdf"):
    """The (width, height) in inches of the pages of the PDF of a ZIP"""
    boxes = re.findall(rb"/MediaBox \[ *0 +0 +([\d.]+) +([\d.]+) *\]", archive.read(name))
    return sorted((round(float(width) / 72, 2), round(float(height) / 72, 2)) for width, height in boxes)


def draw_crop(page, left, top, right, bottom):
    """Drag a crop rectangle, its sides being fractions of the picture"""
    page.locator("#crop-graph").scroll_into_view_if_needed()
    area = page.locator("#crop-graph .nsewdrag").first
    expect(area).to_be_visible()
    box = area.bounding_box()
    page.mouse.move(box["x"] + box["width"] * left, box["y"] + box["height"] * top)
    page.mouse.down()
    page.mouse.move(box["x"] + box["width"] * right, box["y"] + box["height"] * bottom, steps=5)
    page.mouse.up()
    expect(page.locator("#crop-status")).to_contain_text("Area kept")


def test_page_renders(page):
    expect(page.get_by_role("heading", name="Paint by numbers")).to_be_visible()
    expect(page.get_by_text("Drop your picture here")).to_be_visible()
    expect(page.locator("#generate")).to_be_disabled()
    expect(page.locator("#print-estimate")).to_have_text("Choose a picture to see its printed size.")


def test_from_upload_to_zip(page, picture_path):
    upload(page, picture_path)
    expect(page.locator("#source-status")).to_contain_text("Picture loaded: sample.png")
    expect(cards(page)).to_have_count(3)
    expect(page.locator("#build")).to_be_disabled()

    cards(page).first.click()
    expect(cards(page).first).to_have_class("detail-card selected")
    page.locator("#title").fill("A title")
    page.locator("#build").click()
    expect(page.locator("#build-result")).to_contain_text("Canvas ready")

    with page.expect_download() as download:
        page.locator("#download").click()
    assert download.value.suggested_filename == "sample.zip"
    with zipfile.ZipFile(download.value.path()) as archive:
        assert sorted(archive.namelist()) == ["sample-canvas.png", "sample-colormap.png", "sample-result.png",
                                              "sample.pdf"]
        assert archive.testzip() is None
        assert b"A title" in archive.read("sample.pdf")


def test_result_tabs_show_each_picture(page, picture_path):
    upload(page, picture_path)
    expect(cards(page)).to_have_count(3)
    cards(page).first.click()
    page.locator("#build").click()
    for tab in ("Canvas", "Expected result", "Colormap"):
        page.get_by_role("tab", name=tab).click()
        image = page.locator("#build-result [role=tabpanel] img:visible")
        expect(image).to_have_count(1)
        assert image.evaluate("image => image.naturalWidth") > 0


def click_first_color_mark(page):
    slider = page.get_by_role("slider")
    slider.scroll_into_view_if_needed()
    track = page.locator(".mantine-Slider-track").bounding_box()
    # The track goes from 5 to 40 colors
    page.mouse.click(track["x"], track["y"] + track["height"] / 2)
    expect(slider).to_have_attribute("aria-valuenow", "5")


def wait_until(condition, timeout=TIMEOUT / 1000):
    """Wait for a condition on the server side, which the page can't show"""
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "The condition was not met in time"
        time.sleep(0.05)


def test_click_on_the_slider_sets_the_number_of_colors(page, picture_path, server_calls):
    upload(page, picture_path)
    expect(cards(page)).to_have_count(3)
    click_first_color_mark(page)
    page.locator("#generate").click()
    # The cards can't tell, as this picture only has about 5 colors: check what the server received
    wait_until(lambda: server_calls["previews"][0][1] == 5)


def test_dragging_the_slider_sets_the_number_of_colors(page, picture_path, server_calls):
    upload(page, picture_path)
    expect(cards(page)).to_have_count(3)
    slider = page.get_by_role("slider")
    slider.scroll_into_view_if_needed()
    track = page.locator(".mantine-Slider-track").bounding_box()
    page.mouse.move(track["x"] + track["width"] * 15 / 35, track["y"] + track["height"] / 2)
    page.mouse.down()
    page.mouse.move(track["x"] + track["width"], track["y"] + track["height"] / 2, steps=5)
    page.mouse.up()
    expect(slider).to_have_attribute("aria-valuenow", "40")
    page.locator("#generate").click()
    wait_until(lambda: server_calls["previews"][0][1] == 40)


def test_keyboard_sets_the_number_of_colors(page, picture_path, server_calls):
    upload(page, picture_path)
    expect(cards(page)).to_have_count(3)
    slider = page.get_by_role("slider")
    slider.focus()
    slider.press("ArrowRight")
    slider.press("ArrowRight")
    expect(slider).to_have_attribute("aria-valuenow", "32")
    page.locator("#generate").click()
    wait_until(lambda: server_calls["previews"][0][1] == 32)


def test_print_estimate_follows_the_choices_and_the_crop(page, picture_path):
    estimate = page.locator("#print-estimate")
    upload(page, picture_path)
    expect(estimate).to_contain_text("20.0 x 13.3 in")
    expect(estimate).to_contain_text("83% of the 16 x 20 in page")

    page.locator("#print-size").click()
    page.get_by_role("option", name="8 x 10 in").click()
    expect(estimate).to_contain_text("10.0 x 6.7 in")

    page.locator("#print-size").click()
    page.get_by_role("option", name="Fit the paper").click()
    expect(estimate).to_contain_text("on a Letter sheet")
    page.get_by_text("A4", exact=True).click()
    expect(estimate).to_contain_text("on a A4 sheet")

    uncropped = estimate.inner_text()
    draw_crop(page, 0.3, 0.1, 0.6, 0.9)
    expect(estimate).not_to_have_text(uncropped)
    cropped = estimate.inner_text()
    page.locator("#crop-reset").click()
    expect(page.locator("#crop-status")).to_have_text("Whole picture.")
    expect(estimate).not_to_have_text(cropped)


def test_invalid_sources_show_an_error(page, tmp_path):
    fake = tmp_path / "notes.txt"
    fake.write_text("not a picture")
    upload(page, fake)
    expect(page.locator("#source-status")).to_contain_text("not a supported picture")

    page.locator("#url").fill("http://localhost/picture.png")
    page.locator("#load-url").click()
    expect(page.locator("#source-status")).to_contain_text("can't be reached")
    expect(page.locator("#generate")).to_be_disabled()


def test_only_one_card_is_selected_and_it_chooses_the_detail(page, picture_path, server_calls, tmp_path):
    upload_and_select(page, picture_path, card=0)
    cards(page).nth(2).click()
    expect(cards(page).nth(2)).to_have_class("detail-card selected")
    expect(cards(page).nth(0)).to_have_class("detail-card")
    expect(page.locator(".detail-card.selected")).to_have_count(1)
    expect(cards(page).nth(2)).to_contain_text("Detailed")
    build(page)
    assert server_calls["build"][0][1] == "detailed"


def test_build_needs_a_selected_card(page, picture_path):
    upload(page, picture_path)
    expect(cards(page)).to_have_count(3)
    expect(page.locator("#build")).to_be_disabled()
    expect(page.locator("#download")).to_have_count(0)


def test_build_uses_the_colors_of_the_previews_not_the_moved_slider(page, picture_path, server_calls):
    upload_and_select(page, picture_path)
    click_first_color_mark(page)
    build(page)
    assert server_calls["build"][0][2] == 30


def test_new_previews_clear_the_selection_and_the_build(page, picture_path):
    upload_and_select(page, picture_path)
    build(page)
    page.locator("#generate").click()
    expect(page.locator(".detail-card.selected")).to_have_count(0)
    expect(page.locator("#download")).to_have_count(0)
    expect(page.locator("#build")).to_be_disabled()


def test_crop_clears_the_build_and_applies_to_the_next_one(page, picture_path, tmp_path, server_calls):
    upload_and_select(page, picture_path)
    build(page)
    draw_crop(page, 0.3, 0.1, 0.6, 0.9)
    expect(page.locator("#download")).to_have_count(0)
    expect(page.locator(".detail-card.selected")).to_have_count(0)

    cards(page).first.click()
    build(page)
    left, top, right, bottom = server_calls["build"][0][3]
    assert (right - left, bottom - top) == pytest.approx((0.3, 0.8), abs=0.05)
    with download_zip(page, tmp_path / "cropped.zip") as archive:
        canvas = cv2.imdecode(np.frombuffer(archive.read("sample-canvas.png"), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    expected = (right - left) * 600 / ((bottom - top) * 400)
    assert canvas.shape[1] / canvas.shape[0] == pytest.approx(expected, abs=0.02)
    assert max(canvas.shape) == BUILD_SIZE


def test_crop_reset_without_a_crop_changes_nothing(page, picture_path):
    upload(page, picture_path)
    expect(cards(page)).to_have_count(3)
    page.locator("#crop-reset").click()
    expect(page.locator("#crop-status")).to_have_text("Whole picture.")
    expect(cards(page)).to_have_count(3)


def test_second_picture_replaces_the_first(page, picture_path, tmp_path):
    upload_and_select(page, picture_path)
    build(page)
    portrait = tmp_path / "portrait.png"
    cv2.imwrite(str(portrait), cv2.rotate(cv2.imread(str(picture_path)), cv2.ROTATE_90_CLOCKWISE))
    upload(page, portrait)
    expect(page.locator("#source-status")).to_contain_text("Picture loaded: portrait.png")
    expect(page.locator("#print-estimate")).to_contain_text("13.3 x 20.0 in")
    expect(page.locator("#download")).to_have_count(0)
    expect(page.locator(".detail-card.selected")).to_have_count(0)


def test_error_keeps_the_loaded_picture(page, picture_path, tmp_path):
    upload(page, picture_path)
    expect(cards(page)).to_have_count(3)
    fake = tmp_path / "fake.png"
    fake.write_text("not a picture")
    upload(page, fake)
    expect(page.locator("#source-status")).to_contain_text("not a supported picture")
    expect(cards(page)).to_have_count(3)
    expect(page.locator("#generate")).to_be_enabled()


@pytest.fixture
def picture_link(picture_path, monkeypatch):
    """A link to the picture on a local server seen as public, and the missing page next to it"""
    content = picture_path.read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            found = self.path == "/from%20link.png"
            self.send_response(200 if found else 404)
            self.end_headers()
            if found:
                self.wfile.write(content)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(service, "is_public_host", lambda host: True)
    yield f"http://127.0.0.1:{server.server_port}/from%20link.png"
    server.shutdown()


@pytest.mark.parametrize("submit", ["button", "enter"])
def test_picture_is_loaded_from_a_link(page, picture_link, submit):
    page.locator("#url").fill(picture_link)
    if submit == "button":
        page.locator("#load-url").click()
    else:
        page.locator("#url").press("Enter")
    expect(page.locator("#source-status")).to_contain_text("Picture loaded: from link.png")
    expect(cards(page)).to_have_count(3)
    expect(page.locator("#print-estimate")).to_contain_text("20.0 x 13.3 in")


def test_broken_link_shows_an_error(page, picture_link):
    page.locator("#url").fill(picture_link.replace("from%20link", "missing"))
    page.locator("#load-url").click()
    expect(page.locator("#source-status")).to_contain_text("can't be downloaded")
    expect(page.locator("#generate")).to_be_disabled()


def test_empty_link_does_nothing(page):
    page.locator("#load-url").click()
    page.locator("#url").press("Enter")
    expect(page.locator("#load-url")).to_be_enabled()
    expect(page.locator("#source-status")).to_be_empty()


def test_every_print_size_is_offered(page):
    page.locator("#print-size").click()
    options = page.get_by_role("option")
    expect(options).to_have_count(8)
    for size in ("8 x 10 in", "9 x 12 in", "11 x 14 in", "12 x 16 in", "16 x 20 in", "18 x 24 in", "24 x 36 in",
                 "Fit the paper"):
        expect(page.get_by_role("option", name=size)).to_be_visible()


def test_pdf_has_the_chosen_sizes(page, picture_path, tmp_path):
    upload_and_select(page, picture_path)
    build(page)
    with download_zip(page, tmp_path / "default.zip") as archive:
        assert pdf_pages(archive) == [(8.5, 11), (20, 16)]

    choose_print_size(page, "8 x 10 in")
    page.get_by_text("A4", exact=True).click()
    build(page)
    with download_zip(page, tmp_path / "chosen.zip") as archive:
        assert pdf_pages(archive) == [(8.27, 11.69), (10, 8)]


def test_pdf_can_fit_the_paper(page, picture_path, tmp_path):
    upload_and_select(page, picture_path)
    choose_print_size(page, "Fit the paper")
    page.get_by_text("A3", exact=True).click()
    build(page)
    with download_zip(page, tmp_path / "fit.zip") as archive:
        assert pdf_pages(archive) == [(11.69, 16.54), (16.54, 11.69)]


def test_pdf_title_defaults_to_the_picture_name_then_follows_the_field(page, picture_path, tmp_path):
    upload_and_select(page, picture_path)
    build(page)
    with download_zip(page, tmp_path / "default.zip") as archive:
        assert b"/Title (sample)" in archive.read("sample.pdf")
    first_link = page.locator("#build-result a").get_attribute("href")

    page.locator("#title").fill("Another title")
    build(page)
    # The new build has a new link, which the browser can't answer from its cache
    expect(page.locator("#build-result a")).not_to_have_attribute("href", first_link)
    with download_zip(page, tmp_path / "titled.zip") as archive:
        assert b"/Title (Another title)" in archive.read("sample.pdf")


def test_estimate_of_a_picture_filling_the_page(page, tmp_path):
    picture = tmp_path / "shaped.png"
    cv2.imwrite(str(picture), np.full((500, 400, 3), 128, dtype=np.uint8))
    upload(page, picture)
    expect(page.locator("#print-estimate")).to_contain_text("16.0 x 20.0 in (41 x 51 cm), filling the page")


def test_plain_picture_does_not_break_the_page(page, tmp_path, server_calls):
    picture = tmp_path / "plain.png"
    cv2.imwrite(str(picture), np.full((400, 600, 3), 128, dtype=np.uint8))
    upload(page, picture)
    expect(cards(page)).to_have_count(3)
    expect(cards(page).first).to_contain_text("1 colors", ignore_case=True)
    cards(page).first.click()
    build(page)


def test_narrow_screen_stacks_the_page_without_overflow(page, picture_path):
    page.set_viewport_size({"width": 375, "height": 800})
    upload(page, picture_path)
    expect(cards(page)).to_have_count(3)
    assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
    # A control can also be clipped by its card, without the page overflowing
    for control in ("#upload", "#url", "#load-url", "#crop-graph", "#generate", "#title", "#subtitle", "#print-size",
                    "#build"):
        box = page.locator(control).bounding_box()
        assert box["x"] >= 0 and box["x"] + box["width"] <= 375, control
    first, second = cards(page).nth(0).bounding_box(), cards(page).nth(1).bounding_box()
    assert first["x"] == second["x"]
    assert second["y"] >= first["y"] + first["height"]


def test_reload_starts_over(page, picture_path):
    upload(page, picture_path)
    expect(cards(page)).to_have_count(3)
    page.reload()
    expect(page.get_by_text("Drop your picture here")).to_be_visible()
    expect(cards(page)).to_have_count(0)
    expect(page.locator("#generate")).to_be_disabled()


def test_crop_to_fill_the_page(page, picture_path, server_calls):
    estimate = page.locator("#print-estimate")
    expect(page.locator("#crop-fill")).to_be_disabled()
    upload(page, picture_path)
    expect(cards(page)).to_have_count(3)
    expect(estimate).to_contain_text("83% of the 16 x 20 in page")
    page.locator("#crop-fill").click()
    expect(estimate).to_contain_text("20.0 x 16.0 in (51 x 41 cm), filling the page")
    expect(page.locator("#crop-status")).to_contain_text("83% of the width x 100% of the height")
    expect(page.locator("#crop-graph .shapelayer path").first).to_be_attached()
    # The previews follow the crop
    wait_until(lambda: server_calls["previews"][0][2] is not None)
    expect(page.locator("#generate")).not_to_have_attribute("data-loading", "true")
    expect(cards(page)).to_have_count(3)
    cards(page).first.click()
    build(page)

    choose_print_size(page, "Fit the paper")
    expect(page.locator("#crop-fill")).to_be_disabled()


def test_warning_for_labels_too_large_for_the_print(page, tmp_path):
    rng = np.random.default_rng(0)
    blocks = rng.integers(0, 256, (100, 150, 3), dtype=np.uint8)
    picture = tmp_path / "busy.png"
    cv2.imwrite(str(picture), cv2.resize(blocks, (600, 400), interpolation=cv2.INTER_NEAREST))
    upload_and_select(page, picture, card=2)
    warning = page.locator("#print-warning")
    choose_print_size(page, "8 x 10 in")
    expect(warning).to_contain_text("will touch the borders")
    choose_print_size(page, "24 x 36 in")
    expect(warning).to_be_empty()


def test_framed_preview_in_3d(page, picture_path):
    area, view = page.locator("#frame-area"), page.locator("#frame-view")
    expect(area).to_be_hidden()
    upload_and_select(page, picture_path)
    expect(area).to_be_visible()
    expect(view).to_have_attribute("data-rendered", "old-gold")
    assert page.evaluate("Boolean(document.querySelector('#frame-view canvas').getContext('webgl2'))")
    gold = view.screenshot()

    page.locator("#frame").click()
    page.get_by_role("option", name="Walnut").click()
    expect(view).to_have_attribute("data-rendered", "walnut")
    assert view.screenshot() != gold
    # A new crop makes new previews, so the choice of a level starts over and the viewer goes away with it
    page.locator("#crop-fill").click()
    expect(area).to_be_hidden()


def test_wood_frames_load_their_textures(page, picture_path):
    responses = []
    page.on("response", lambda response: responses.append(response) if "/vendor/wood/" in response.url else None)
    upload_and_select(page, picture_path)
    view = page.locator("#frame-view")
    expect(view).to_have_attribute("data-rendered", "old-gold")
    page.locator("#frame").click()
    page.get_by_role("option", name="Mahogany").click()
    expect(view).to_have_attribute("data-rendered", "mahogany")
    assert {response.url.rsplit("/", 1)[1] for response in responses} == {"Wood026_color.jpg", "Wood026_normal.jpg"}
    assert all(response.ok for response in responses)


def changed_pixels(before, after):
    """The share of the pixels of two screenshots which clearly differ, ignoring the noise of the rendering"""
    before, after = (cv2.imdecode(np.frombuffer(shot, np.uint8), cv2.IMREAD_COLOR).astype(int) for shot in (before, after))
    return (np.abs(before - after).max(axis=2) > 20).mean()


def test_wheel_scrolls_the_page_over_the_3d_preview_and_ctrl_wheel_zooms(page, picture_path):
    upload_and_select(page, picture_path)
    view = page.locator("#frame-view")
    expect(view).to_have_attribute("data-rendered", "old-gold")
    view.scroll_into_view_if_needed()
    box = view.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    before, scroll = view.screenshot(), page.evaluate("window.scrollY")
    # The view is at the bottom of the page, which can only scroll up
    page.mouse.wheel(0, -200)
    page.wait_for_function(f"window.scrollY < {scroll}")
    assert changed_pixels(before, view.screenshot()) < 0.01

    view.scroll_into_view_if_needed()
    box = view.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    before, scroll = view.screenshot(), page.evaluate("window.scrollY")
    page.keyboard.down("Control")
    page.mouse.wheel(0, -300)
    page.keyboard.up("Control")
    page.wait_for_timeout(300)
    assert page.evaluate("window.scrollY") == scroll
    assert changed_pixels(before, view.screenshot()) > 0.1
