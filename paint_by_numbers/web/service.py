# -*- coding: utf-8 -*-
"""Everything the web interface does, minus the UI: storing the source picture, previewing the detail levels and
building the final files

Sources are kept server-side, one folder per upload under SESSION_ROOT. The browser only holds the opaque session id,
never a file path, so it can't point the app at an arbitrary file on the server.
"""

import base64
import binascii
import http.client
import io
import ipaddress
import re
import shutil
import socket
import tempfile
import threading
import time
import urllib.request
import uuid
import warnings
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
from PIL import Image

from ..canvas import Canvas
from ..export import colormap_picture, crowded_label_share
from ..faces import face_mask
from ..preprocessing import resize

SESSION_ROOT = Path(tempfile.gettempdir()) / "paint-by-numbers-web"
# Sessions older than this are deleted when a new picture comes in
SESSION_TTL_SECONDS = 24 * 3600
MAX_IMAGE_BYTES = 25 * 1024 * 1024
# A small file can hold a huge picture, a blank 20000 x 20000 PNG is 1 MB: its size is checked before decoding it.
# This lets in the 108 megapixel photos of phones, about 330 MB once decoded
MAX_SOURCE_PIXELS = 120_000_000
MAX_SOURCE_SIZE = 6000
URL_TIMEOUT_SECONDS = 10
# The whole download, as the timeout above only applies to each read: a server sending a byte at a time stops here
URL_DOWNLOAD_SECONDS = 60
# Canvases generated at the same time, the others wait: each takes seconds of CPU and hundreds of MB
HEAVY_WORK = threading.BoundedSemaphore(2)
# Size of the pictures compared in the detail cards: a full size canvas takes several seconds
PREVIEW_SIZE = 1200
# Largest dimension of the final pictures shown in the page, the downloads keep their full size
RESULT_DISPLAY_SIZE = 1400
# A crop narrower or shorter than this share of the picture is a stray click
MIN_CROP_FRACTION = 0.04
DETAIL_DESCRIPTIONS = {"easy": "Large regions, quick to paint", "medium": "A balance of detail and effort",
                       "detailed": "Small regions, closest to the picture"}
# Largest dimension of the picture of a project saved in the browser, whose storage holds about 5 MB. The canvas is
# denoised at 1500 pxl, so this keeps all the details of a crop down to 60% of the picture, in a few hundred kB
PROJECT_PICTURE_SIZE = 2400
PROJECT_THUMBNAIL_SIZE = 160
BUILT_SUFFIXES = (".pdf", "-canvas.png", "-result.png", "-colormap.png")

_SESSION_ID = re.compile(r"^[0-9a-f]{32}$")


class ImageError(ValueError):
    """The picture can't be used. The message is shown to the user"""


def image_bytes_from_upload(contents):
    """Decode the data:<mime>;base64,<data> string produced by a Dash upload"""
    _, _, encoded = contents.partition(",")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageError("The file can't be read.") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError(f"The picture is too large (max {MAX_IMAGE_BYTES // 2 ** 20} MB).")
    return data


def is_public_host(host):
    """Whether all the addresses of a host are on the public internet, so that a link can't reach the local network"""
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        return False
    return all(ipaddress.ip_address(info[4][0]).is_global for info in infos)


def check_url(url):
    """Refuse the links which are not http(s) or which point to a private address"""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ImageError("The link must start with http:// or https://.")
    if not is_public_host(parsed.hostname):
        raise ImageError("This link points to an address which can't be reached.")


class PublicRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Check the target of each redirect, as a public link could redirect to a private address"""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class PublicPeer:
    """Check the address a connection really reaches, as the host name can resolve to another address than the one
    check_url saw (DNS rebinding)

    shortcut: a proxy on the local network is refused too, allow it if the app ever runs behind one
    """

    def connect(self):
        super().connect()
        if not is_public_host(self.sock.getpeername()[0]):
            self.close()
            raise ImageError("This link points to an address which can't be reached.")


class PublicHTTPConnection(PublicPeer, http.client.HTTPConnection):
    pass


class PublicHTTPSConnection(PublicPeer, http.client.HTTPSConnection):
    pass


class PublicHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(PublicHTTPConnection, req)


class PublicHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(PublicHTTPSConnection, req, context=self._context)


def image_bytes_from_url(url):
    """Download a picture from an http(s) link, up to MAX_IMAGE_BYTES and URL_DOWNLOAD_SECONDS"""
    url = url.strip()
    check_url(url)
    request = urllib.request.Request(url, headers={"User-Agent": "paint-by-numbers/0.1"})
    opener = urllib.request.build_opener(PublicRedirectHandler, PublicHTTPHandler, PublicHTTPSHandler)
    deadline = time.monotonic() + URL_DOWNLOAD_SECONDS
    data = bytearray()
    try:
        with opener.open(request, timeout=URL_TIMEOUT_SECONDS) as response:
            # read1 returns what has arrived, rather than waiting for the whole chunk
            while len(data) <= MAX_IMAGE_BYTES and (chunk := response.read1(64 * 1024)):
                data += chunk
                if time.monotonic() > deadline:
                    raise ImageError("The picture takes too long to download.")
    except ImageError:
        raise
    except (OSError, ValueError, http.client.HTTPException) as exc:
        raise ImageError("The picture can't be downloaded from this link.") from exc
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageError(f"The picture is too large (max {MAX_IMAGE_BYTES // 2 ** 20} MB).")
    return bytes(data)


def name_from_url(url):
    """The file name at the end of a link"""
    return unquote(urlparse(url.strip()).path).rstrip("/").rsplit("/", 1)[-1] or "picture"


def safe_stem(name):
    """A version of an uploaded file name safe for the file system, ASCII as cv2 can't open other paths on Windows"""
    stem = re.sub(r"[^\w-]+", "_", Path(name).stem, flags=re.ASCII).strip("_")
    return stem[:40] or "picture"


def purge_old_sessions(now):
    """Delete the sessions older than SESSION_TTL_SECONDS"""
    if not SESSION_ROOT.is_dir():
        return
    for session in SESSION_ROOT.iterdir():
        if session.is_dir() and now - session.stat().st_mtime > SESSION_TTL_SECONDS:
            shutil.rmtree(session, ignore_errors=True)


def save_source(data, name):
    """Check that data is a picture, store it as PNG in a new session folder and return the session id"""
    too_large = ImageError(f"The picture is too large (max {MAX_SOURCE_PIXELS // 10 ** 6} million pixels).")
    try:
        # Pillow only reads the header here. It warns about the large pictures, which are checked just after
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as header:
                width, height = header.size
    except Image.DecompressionBombError as exc:  # The huge ones, above twice its own limit
        raise too_large from exc
    except OSError as exc:
        raise ImageError("This file is not a supported picture (PNG, JPEG, WebP...).") from exc
    if width * height > MAX_SOURCE_PIXELS:
        raise too_large
    picture = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if picture is None:
        raise ImageError("This file is not a supported picture (PNG, JPEG, WebP...).")
    if max(picture.shape[:2]) > MAX_SOURCE_SIZE:
        picture = resize(picture, MAX_SOURCE_SIZE)
    purge_old_sessions(time.time())
    session_id = uuid.uuid4().hex
    session = SESSION_ROOT / session_id
    session.mkdir(parents=True)
    if not cv2.imwrite(str(session / f"{safe_stem(name)}.png"), picture):
        raise ImageError("The picture can't be saved.")
    return session_id


def source_path(session_id):
    """Returns the stored source picture of a session"""
    if not isinstance(session_id, str) or not _SESSION_ID.match(session_id):
        raise ImageError("Invalid session.")
    pictures = sorted((SESSION_ROOT / session_id).glob("*.png"))
    if not pictures:
        raise ImageError("Picture not found: it may have expired, please upload it again.")
    return pictures[0]


def data_uri(picture, max_size=None):
    """A PNG data: URI of an RGB or grey picture, for the page"""
    if max_size and max(picture.shape[:2]) > max_size:
        picture = resize(picture, max_size)
    bgr = cv2.cvtColor(picture, cv2.COLOR_RGB2BGR) if picture.ndim == 3 else picture
    ok, png = cv2.imencode(".png", bgr)
    if not ok:
        raise ImageError("The picture can't be displayed.")
    return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode("ascii")


def source_preview(session_id, max_size=800):
    """Returns the source picture as a data: URI, with its displayed width and height"""
    picture = cv2.cvtColor(cv2.imread(str(source_path(session_id))), cv2.COLOR_BGR2RGB)
    if max(picture.shape[:2]) > max_size:
        picture = resize(picture, max_size)
    return data_uri(picture), picture.shape[1], picture.shape[0]


def jpeg_uri(picture, max_size, quality):
    """A JPEG data: URI of a BGR picture, no larger than max_size"""
    if max(picture.shape[:2]) > max_size:
        picture = resize(picture, max_size)
    ok, jpeg = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ImageError("The picture can't be saved.")
    return "data:image/jpeg;base64," + base64.b64encode(jpeg.tobytes()).decode("ascii")


def project_pictures(session_id):
    """The source picture to keep in a saved project, and its thumbnail, as JPEG data: URIs"""
    picture = cv2.imread(str(source_path(session_id)))
    return jpeg_uri(picture, PROJECT_PICTURE_SIZE, 92), jpeg_uri(picture, PROJECT_THUMBNAIL_SIZE, 80)


def validate_crop(crop):
    """Returns the crop (left, top, right, bottom as fractions) clamped to the picture, None if there is no crop or if
    it is too small to be deliberate"""
    if crop is None:
        return None
    left, top, right, bottom = (min(1.0, max(0.0, float(value))) for value in crop)
    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))
    if right - left < MIN_CROP_FRACTION or bottom - top < MIN_CROP_FRACTION:
        return None
    if (left, top, right, bottom) == (0.0, 0.0, 1.0, 1.0):
        return None
    return left, top, right, bottom


def cropped_canvas(session_id, crop, nb_color, **options):
    """The canvas of the source picture, cropped in memory: requests running at the same time don't share any file"""
    canvas = Canvas(source_path(session_id), nb_color, **options)
    crop = validate_crop(crop)
    if crop is not None:
        canvas.crop(crop)
    return canvas


def crowded_labels(canvas):
    """The share of labels touching their borders, for each standard size and each paper the canvas can fit

    Returns it over the whole picture ("all") and over its faces ("faces", None without any face), as the average can
    hide a crowded face: the smallest regions are kept there
    """
    shape = canvas.borders.shape
    faces = face_mask(shape, canvas.faces)

    def shares(mask):
        return {**{size: crowded_label_share(canvas.regions, shape, print_size=size, mask=mask)
                   for size in Canvas.print_sizes},
                **{paper: crowded_label_share(canvas.regions, shape, paper=paper, mask=mask)
                   for paper in Canvas.paper_sizes}}

    return {"all": shares(None), "faces": None if faces is None else shares(faces)}


def render_previews(session_id, nb_color, crop=None):
    """Generate the picture once per detail level, at a low resolution, and return what the detail cards show"""
    previews = []
    for detail in Canvas.detail_levels:
        canvas = cropped_canvas(session_id, crop, nb_color, pixel_size=PREVIEW_SIZE, detail=detail)
        with HEAVY_WORK:
            canvas.generate()
        previews.append({"detail": detail, "description": DETAIL_DESCRIPTIONS[detail],
                         "image": data_uri(canvas.quantified_image), "regions": len(canvas.regions),
                         "colors": len(canvas.colormap), "crowded": crowded_labels(canvas)})
    return previews


def build_files(session_id, detail, nb_color, crop=None, pixel_size=4000, title=None, subtitle="", paper="a4",
                print_size=None):
    """Generate the canvas at full size, save its files and their ZIP in the session and return what the page shows"""
    canvas = cropped_canvas(session_id, crop, nb_color, pixel_size=pixel_size, detail=detail)
    with HEAVY_WORK:
        canvas.generate()
    build = source_path(session_id).parent / "build"
    canvas.save(build, title=title, subtitle=subtitle, paper=paper, print_size=print_size)
    with zipfile.ZipFile(build / f"{canvas.namefile}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for suffix in BUILT_SUFFIXES:
            archive.write(build / f"{canvas.namefile}{suffix}", f"{canvas.namefile}{suffix}")
    return {"canvas": data_uri(canvas.canvas, RESULT_DISPLAY_SIZE),
            "result": data_uri(canvas.quantified_image, RESULT_DISPLAY_SIZE),
            "colormap": data_uri(colormap_picture(canvas.colormap)),
            "regions": len(canvas.regions), "colors": len(canvas.colormap)}


def built_zip(session_id):
    """Returns the path of the ZIP written by build_files"""
    path = source_path(session_id)
    built = path.parent / "build" / f"{path.stem}.zip"
    if not built.is_file():
        raise ImageError("This file has not been built yet.")
    return built
