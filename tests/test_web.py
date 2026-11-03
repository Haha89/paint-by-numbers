# -*- coding: utf-8 -*-

import base64
import http.server
import io
import threading
import urllib.request
import zipfile

import cv2
import numpy as np
import pytest

from paint_by_numbers.web import service
from dash import no_update

from paint_by_numbers.web.app import (FIT_PAPER, create_app, describe_crowding, describe_print, fill_crop_box, frame_size,
                                      next_crop_box, project_settings, saved_project, saved_projects)


@pytest.fixture(autouse=True)
def session_root(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "SESSION_ROOT", tmp_path / "sessions")


@pytest.fixture
def session(picture_path):
    return service.save_source(picture_path.read_bytes(), "my picture.png")


def test_upload_is_decoded():
    contents = "data:image/png;base64," + base64.b64encode(b"data").decode()
    assert service.image_bytes_from_upload(contents) == b"data"


def test_invalid_upload_is_rejected():
    with pytest.raises(service.ImageError):
        service.image_bytes_from_upload("data:image/png;base64,not base64!")


def test_too_large_upload_is_rejected(monkeypatch):
    monkeypatch.setattr(service, "MAX_IMAGE_BYTES", 2)
    with pytest.raises(service.ImageError, match="too large"):
        service.image_bytes_from_upload("data:image/png;base64," + base64.b64encode(b"data").decode())


def test_file_which_is_not_a_picture_is_rejected():
    with pytest.raises(service.ImageError, match="not a supported picture"):
        service.save_source(b"not a picture", "notes.txt")


@pytest.mark.parametrize("name, expected", [("my picture.png", "my_picture"), ("../../evil.png", "evil"),
                                            ("été.jpg", "t"), ("....png", "picture")])
def test_file_names_are_made_safe(name, expected):
    assert service.safe_stem(name) == expected


def test_source_is_stored_shrunk(tmp_path, monkeypatch):
    path = tmp_path / "large.png"
    cv2.imwrite(str(path), np.zeros((50, 100, 3), dtype=np.uint8))
    monkeypatch.setattr(service, "MAX_SOURCE_SIZE", 40)
    session = service.save_source(path.read_bytes(), "large.png")
    assert cv2.imread(str(service.source_path(session))).shape[:2] == (20, 40)


@pytest.mark.parametrize("session_id", ["../..", "abc", None, "g" * 32])
def test_invalid_session_is_rejected(session_id):
    with pytest.raises(service.ImageError, match="Invalid session"):
        service.source_path(session_id)


def test_unknown_session_is_rejected():
    with pytest.raises(service.ImageError, match="not found"):
        service.source_path("0" * 32)


def test_old_sessions_are_purged(session):
    folder = service.source_path(session).parent
    service.purge_old_sessions(folder.stat().st_mtime + service.SESSION_TTL_SECONDS - 1)
    assert folder.exists()
    service.purge_old_sessions(folder.stat().st_mtime + service.SESSION_TTL_SECONDS + 1)
    assert not folder.exists()


@pytest.mark.parametrize("crop, expected", [
    (None, None),
    ([0, 0, 1, 1], None),
    ([0.5, 0.5, 0.51, 0.9], None),
    ([0.2, 0.3, 0.6, 0.8], (0.2, 0.3, 0.6, 0.8)),
    ([0.6, 0.8, 0.2, 0.3], (0.2, 0.3, 0.6, 0.8)),
    ([-1, -1, 0.5, 2], (0.0, 0.0, 0.5, 1.0))])
def test_validate_crop(crop, expected):
    assert service.validate_crop(crop) == expected


def test_working_picture_is_cropped(session):
    path = service.working_picture(session, [0.5, 0, 1, 0.5])
    assert cv2.imread(str(path)).shape[:2] == (200, 300)
    assert path.name == service.source_path(session).name


def test_working_picture_without_crop_is_the_source(session):
    assert service.working_picture(session, None) == service.source_path(session)


def test_previews_compare_the_detail_levels(session):
    previews = service.render_previews(session, 5)
    assert [preview["detail"] for preview in previews] == ["easy", "medium", "detailed"]
    assert all(preview["image"].startswith("data:image/png;base64,") for preview in previews)
    assert previews[0]["regions"] < previews[-1]["regions"]


def test_build_files_then_get_them(session):
    built = service.build_files(session, "easy", 5, pixel_size=600, title="A title")
    assert built["regions"] > 0
    with zipfile.ZipFile(service.built_zip(session)) as archive:
        assert sorted(archive.namelist()) == ["my_picture-canvas.png", "my_picture-colormap.png",
                                              "my_picture-result.png", "my_picture.pdf"]
        assert archive.testzip() is None
        assert archive.read("my_picture.pdf").startswith(b"%PDF")


def test_build_files_uses_the_crop(session):
    service.build_files(session, "easy", 5, crop=[0, 0, 0.25, 1], pixel_size=600)
    with zipfile.ZipFile(service.built_zip(session)) as archive:
        canvas = np.frombuffer(archive.read("my_picture-canvas.png"), dtype=np.uint8)
    # A quarter of the 600x400 picture is 150x400, which is then scaled to 600 pixels high
    assert cv2.imdecode(canvas, cv2.IMREAD_GRAYSCALE).shape == (600, 225)


SOURCE = {"width": 600, "height": 400}


def test_print_estimate_without_a_picture():
    assert "Choose a picture" in describe_print(None, None, "16x20", "letter")


def test_print_estimate_on_a_standard_size():
    # The 600x400 landscape picture is limited by the width of the 20x16 page
    text = describe_print(SOURCE, None, "16x20", "letter")
    assert "20.0 x 13.3 in (51 x 34 cm)" in text
    assert "83% of the 16 x 20 in page" in text


def test_print_estimate_follows_the_crop():
    # Cropped to 500x400: the 5:4 shape of the page
    assert "20.0 x 16.0 in" in (text := describe_print(SOURCE, [0, 0, 5 / 6, 1], "16x20", "letter"))
    assert "filling the page" in text


def test_print_estimate_when_fitting_the_paper():
    text = describe_print(SOURCE, None, FIT_PAPER, "letter")
    # Letter in landscape, with the margins and the header
    assert "9.8 x 6.5 in" in text
    assert "on a Letter sheet" in text
    assert "on a A4 sheet" in describe_print(SOURCE, None, FIT_PAPER, "a4")


@pytest.mark.parametrize("host, expected", [("127.0.0.1", False), ("localhost", False), ("192.168.1.10", False),
                                            ("169.254.169.254", False), ("::1", False), ("8.8.8.8", True)])
def test_only_public_hosts_are_accepted(host, expected):
    assert service.is_public_host(host) is expected


@pytest.mark.parametrize("url, message", [("ftp://example.com/a.png", "must start"), ("file:///etc/passwd", "must start"),
                                          ("not a link", "must start"), ("http://127.0.0.1:8051/a.png", "can't be reached"),
                                          ("http://localhost/a.png", "can't be reached")])
def test_invalid_links_are_rejected(url, message):
    with pytest.raises(service.ImageError, match=message):
        service.image_bytes_from_url(url)


def test_redirect_to_a_private_address_is_rejected():
    request = urllib.request.Request("http://example.com/a.png")
    with pytest.raises(service.ImageError, match="can't be reached"):
        service.PublicRedirectHandler().redirect_request(request, None, 302, "Found", {}, "http://127.0.0.1/secret")


@pytest.fixture
def picture_server(picture_path, monkeypatch):
    """A local server for the picture, seen as public, which also redirects and fails"""
    content = picture_path.read_bytes()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/moved":
                self.send_response(302)
                self.send_header("Location", "/my%20picture.png?size=large")
            elif self.path.startswith("/my%20picture.png"):
                self.send_response(200)
            else:
                self.send_response(404)
            self.end_headers()
            if self.path.startswith("/my%20picture.png"):
                self.wfile.write(content)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(service, "is_public_host", lambda host: True)
    yield f"http://127.0.0.1:{server.server_port}", content
    server.shutdown()


@pytest.mark.parametrize("path", ["/my%20picture.png?size=large", "/moved"])
def test_picture_is_downloaded_from_a_link(picture_server, path):
    base, content = picture_server
    assert service.image_bytes_from_url(f"  {base}{path}  ") == content


def test_missing_link_is_rejected(picture_server):
    with pytest.raises(service.ImageError, match="can't be downloaded"):
        service.image_bytes_from_url(f"{picture_server[0]}/missing.png")


def test_too_large_download_is_rejected(picture_server, monkeypatch):
    monkeypatch.setattr(service, "MAX_IMAGE_BYTES", 10)
    with pytest.raises(service.ImageError, match="too large"):
        service.image_bytes_from_url(f"{picture_server[0]}/my%20picture.png")


@pytest.mark.parametrize("url, expected", [("https://a.com/b/my%20photo.jpg?x=1", "my photo.jpg"),
                                           ("https://a.com/", "picture"), ("https://a.com", "picture")])
def test_name_from_url(url, expected):
    assert service.name_from_url(url) == expected


def test_file_not_built_yet_is_rejected(session):
    with pytest.raises(service.ImageError, match="not been built"):
        service.built_zip(session)


def test_drawn_crop_box_is_clamped_to_the_picture():
    box = next_crop_box({"shapes": [{"x0": -10, "y0": 9.5, "x1": 49.5, "y1": 500}]}, None, 100, 80)
    assert box == (0.0, 10.0, 50.0, 80.0)


def test_moved_crop_box_keeps_the_other_sides():
    current = {"x0": 9.5, "y0": 9.5, "x1": 19.5, "y1": 19.5}
    assert next_crop_box({"shapes[0].x1": 29.5}, current, 100, 80) == (10.0, 10.0, 30.0, 20.0)


def test_erased_crop_box_means_no_crop():
    assert next_crop_box({"shapes": []}, {"x0": 0, "y0": 0, "x1": 1, "y1": 1}, 100, 80) is None


def test_other_graph_events_do_not_touch_the_crop():
    assert next_crop_box({"autosize": True}, None, 100, 80) is False


def test_app_page_and_stylesheet_are_served():
    client = create_app().server.test_client()
    assert client.get("/").status_code == 200
    assert b"Paint by numbers" in client.get("/_dash-layout").data
    assert client.get("/assets/style.css").status_code == 200


def test_built_zip_is_served_for_download(session):
    service.build_files(session, "easy", 5, pixel_size=600)
    response = create_app().server.test_client().get(f"/files/{session}")
    assert response.status_code == 200
    assert zipfile.ZipFile(io.BytesIO(response.data)).testzip() is None
    assert "my_picture.zip" in response.headers["Content-Disposition"]
    assert "attachment" in response.headers["Content-Disposition"]


@pytest.mark.parametrize("url", ["/files/{session}", "/files/not-a-session"])
def test_unavailable_files_are_not_served(session, url):
    assert create_app().server.test_client().get(url.format(session=session)).status_code == 404


def test_path_traversal_only_reaches_the_page(session):
    # The URL is normalized outside of /files, where Dash answers any path with its page
    service.build_files(session, "easy", 5, pixel_size=600)
    response = create_app().server.test_client().get(f"/files/..%2F{session}")
    assert "Content-Disposition" not in response.headers
    assert response.mimetype == "text/html"


CROWDED = {"easy": {"all": {"8x10": 0.1, "a4": 0.9}, "faces": None},
           "medium": {"all": {"8x10": 0.1, "24x36": 0}, "faces": {"8x10": 0.5, "24x36": 0.1}},
           "detailed": {"all": {"8x10": 0.6}, "faces": {"8x10": 0.8}}}


def test_no_crowding_warning_without_a_choice_or_under_the_threshold():
    assert describe_crowding(None, None, "8x10", "a4") is None
    assert describe_crowding(CROWDED, None, "8x10", "a4") is None
    assert describe_crowding(CROWDED, "easy", "8x10", "a4") is None
    assert describe_crowding(CROWDED, "medium", "24x36", "a4") is None


def test_crowding_warning_gives_the_share_for_the_size_or_the_paper():
    assert "60% of the numbers will" in describe_crowding(CROWDED, "detailed", "8x10", "a4").children
    assert "90%" in describe_crowding(CROWDED, "easy", FIT_PAPER, "a4").children


def test_crowding_warning_on_the_faces_when_the_average_is_fine():
    assert "50% of the numbers on the faces" in describe_crowding(CROWDED, "medium", "8x10", "a4").children


def test_previews_give_the_crowded_labels_of_each_size_and_paper(session):
    crowded = service.render_previews(session, 5)[0]["crowded"]
    assert {"8x10", "24x36", "a4", "a3", "letter"} <= set(crowded["all"])
    assert all(0 <= share <= 1 for share in crowded["all"].values())
    assert crowded["all"]["8x10"] >= crowded["all"]["24x36"]
    assert crowded["faces"] is None  # The generated picture has no face


def test_fill_crop_box_centers_the_page_in_the_picture():
    box = fill_crop_box({"width": 600, "height": 400}, None, "16x20")
    assert box == pytest.approx((50, 0, 550, 400))


def test_fill_crop_box_stays_in_the_current_crop():
    box = fill_crop_box({"width": 600, "height": 400}, [0.5, 0, 1, 0.5], "16x20")
    assert box == pytest.approx((325, 0, 575, 200))


def test_frame_fits_the_picture_without_white_bands():
    assert frame_size(None, None, "16x20", "letter") is None
    size = frame_size({"width": 600, "height": 400}, None, "16x20", "letter")
    assert size["picture"] == pytest.approx([20, 40 / 3])
    assert size["page"] == size["picture"]


def test_frame_size_on_the_paper_is_the_picture_itself():
    size = frame_size({"width": 600, "height": 400}, None, FIT_PAPER, "letter")
    assert size["page"] == pytest.approx(size["picture"])


def test_saved_project_keeps_the_picture_and_the_settings(session):
    source = {"session": session, "name": "my picture.png", "width": 600, "height": 400}
    project = saved_project(None, source, [0.1, 0, 0.9, 1], 20, "easy", "8x10", "a4", "", None, "walnut")
    assert len(project["id"]) == 32 and project["name"] == "my picture"
    assert (project["crop"], project["nb_color"], project["detail"]) == ([0.1, 0, 0.9, 1], 20, "easy")
    assert (project["print_size"], project["paper"], project["frame"]) == ("8x10", "a4", "walnut")
    assert project["title"] == project["subtitle"] == ""
    picture = cv2.imdecode(np.frombuffer(service.image_bytes_from_upload(project["picture"]), np.uint8), cv2.IMREAD_COLOR)
    assert picture.shape == (400, 600, 3)
    thumbnail = service.image_bytes_from_upload(project["thumbnail"])
    assert max(cv2.imdecode(np.frombuffer(thumbnail, np.uint8), cv2.IMREAD_COLOR).shape[:2]) == 160
    # Saving again keeps the id, and the title names the project
    again = saved_project(project["id"], source, None, 20, None, FIT_PAPER, "a4", "A title", "", "none")
    assert (again["id"], again["name"]) == (project["id"], "A title")


def test_project_picture_is_reduced(session, monkeypatch):
    monkeypatch.setattr(service, "PROJECT_PICTURE_SIZE", 300)
    picture, _ = service.project_pictures(session)
    decoded = cv2.imdecode(np.frombuffer(service.image_bytes_from_upload(picture), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (200, 300)


def test_project_settings_are_restored():
    settings = project_settings({"crop": [0.1, 0, 0.9, 1], "nb_color": 20, "detail": "easy", "print_size": "8x10",
                                 "paper": "a3", "frame": "walnut", "title": "A title", "subtitle": "A subtitle"})
    assert settings == {"crop": (0.1, 0.0, 0.9, 1.0), "nb_color": 20, "detail": "easy", "print_size": "8x10",
                        "paper": "a3", "frame": "walnut", "title": "A title", "subtitle": "A subtitle"}
    assert project_settings({"print_size": FIT_PAPER})["print_size"] == FIT_PAPER


def test_project_settings_ignore_what_the_page_does_not_offer():
    # The browser storage can be edited by hand, or written by an older version of the page
    settings = project_settings({"crop": [0, "a", 1, 1], "nb_color": 500, "detail": "extreme", "print_size": ["8x10"],
                                 "paper": "a0", "frame": "plastic", "title": 3, "subtitle": "x" * 1000})
    assert settings == {"crop": None, "nb_color": 40, "detail": None, "print_size": no_update, "paper": no_update,
                        "frame": no_update, "title": "", "subtitle": "x" * 200}
    assert project_settings({})["nb_color"] is no_update
    assert project_settings({"nb_color": True})["nb_color"] is no_update


def test_saved_projects_list():
    assert saved_projects(None) is None
    assert saved_projects([]) is None
    assert saved_projects([{"name": "no id"}, "junk"]) is None
    listed = saved_projects([{"id": "a", "name": "First", "thumbnail": "data:image/jpeg;base64,AA"},
                             {"id": "b", "name": "Second", "thumbnail": "javascript:alert(1)"}])
    text = str(listed)
    assert "First" in text and "Second" in text
    assert "javascript:" not in text
