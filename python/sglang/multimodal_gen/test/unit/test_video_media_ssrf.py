# SPDX-License-Identifier: Apache-2.0
"""SSRF regression tests for client-supplied media URLs in /v1/videos.

Covers sgl-project/sglang#51701: unauthenticated media URLs (video_path,
input_reference/image_path, control_path) must not make the server fetch
private, loopback, link-local, or reserved addresses, or hand FFmpeg
protocol-prefixed strings to ``av.open``. Public http(s) URLs keep working.
"""

import asyncio
import http.server
import threading
import unittest.mock as mock

import pytest
import requests
from fastapi import HTTPException

from sglang.multimodal_gen.runtime.entrypoints.openai.utils import (
    _save_url_image_to_path,
    _validate_remote_media_paths,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3 import (
    _iter_video_frames_pyav,
)
from sglang.multimodal_gen.runtime.utils import vision
from sglang.srt.multimodal.media_io import configure_media_url_security

PRIVATE_URL = "http://127.0.0.1:19881/path1_target?run_id=test"
METADATA_URL = "http://169.254.169.254/latest/meta-data"
PUBLIC_VIDEO_URL = "http://93.184.216.34/clip.mp4"
PUBLIC_IMAGE_URL = "http://93.184.216.34/input.png"

MEDIA_FIELDS = (
    "video_path",
    "video_url",
    "image_path",
    "input_reference",
    "reference_url",
    "control_path",
)


@pytest.fixture(autouse=True)
def _reset_media_url_policy():
    configure_media_url_security([], max_file_size_mb=64)
    yield
    configure_media_url_security([], max_file_size_mb=64)


class TestEntrypointMediaValidation:
    @pytest.mark.parametrize("field", MEDIA_FIELDS)
    @pytest.mark.parametrize("url", [PRIVATE_URL, METADATA_URL])
    def test_private_url_is_rejected(self, field, url):
        with pytest.raises(HTTPException) as excinfo:
            _validate_remote_media_paths({field: url})
        assert excinfo.value.status_code == 400
        assert "disallowed media URL" in excinfo.value.detail

    def test_control_path_list_is_checked_item_by_item(self):
        with pytest.raises(HTTPException) as excinfo:
            _validate_remote_media_paths(
                {"control_path": [PUBLIC_VIDEO_URL, PRIVATE_URL]}
            )
        assert excinfo.value.status_code == 400

    @pytest.mark.parametrize(
        "value",
        [
            "file:///etc/passwd",
            "file:/etc/passwd",
            "concat:clip1.mp4|clip2.mp4",
            "ftp://93.184.216.34/clip.mp4",
        ],
    )
    def test_non_http_scheme_is_rejected(self, value):
        with pytest.raises(HTTPException) as excinfo:
            _validate_remote_media_paths({"video_path": value})
        assert excinfo.value.status_code == 400

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"video_path": PUBLIC_VIDEO_URL},
            {"video_path": "/srv/media/clip.mp4"},
            {"video_path": "relative/clip.mp4"},
            {"image_path": "data:image/png;base64,iVBORw0KGgo="},
            {"control_path": None},
            {},
        ],
    )
    def test_legitimate_values_pass(self, kwargs):
        # Literal public IPs need no DNS, so this needs no network.
        _validate_remote_media_paths(kwargs)


class _FakeResponse:
    """Minimal requests response: context manager + streamed body."""

    def __init__(self, status_code=200, headers=None, chunks=(b"fake-mp4",)):
        self.status_code = status_code
        self.headers = headers or {}
        self.url = "http://93.184.216.34/clip.mp4"
        self._chunks = list(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass

    def iter_content(self, chunk_size=8192):
        yield from self._chunks


class TestLoadVideoDownload:
    def test_private_url_is_rejected_without_fetching(self):
        with mock.patch.object(vision.requests, "get") as fake_get:
            with pytest.raises(ValueError, match="non-public IP"):
                vision.load_video(PRIVATE_URL)
        fake_get.assert_not_called()

    def test_redirect_to_private_ip_is_rejected(self):
        redirect = _FakeResponse(
            status_code=302,
            headers={"Location": PRIVATE_URL},
        )
        with mock.patch.object(
            vision.requests, "get", return_value=redirect
        ) as fake_get:
            with pytest.raises(ValueError, match="non-public IP"):
                vision.load_video(PUBLIC_VIDEO_URL)
        # Only the initial (public) URL was requested; the redirect target
        # was never fetched.
        fake_get.assert_called_once()
        assert fake_get.call_args.args[0] == PUBLIC_VIDEO_URL

    def test_redirect_loop_is_bounded(self):
        redirect = _FakeResponse(
            status_code=302,
            headers={"Location": PUBLIC_VIDEO_URL},
        )
        with mock.patch.object(vision.requests, "get", return_value=redirect):
            with pytest.raises(requests.exceptions.TooManyRedirects):
                vision.load_video(PUBLIC_VIDEO_URL)

    def test_oversized_download_is_rejected(self, monkeypatch):
        monkeypatch.setattr(vision, "_VIDEO_DOWNLOAD_MAX_BYTES", 10)
        response = _FakeResponse(chunks=[b"x" * 11])
        with mock.patch.object(vision.requests, "get", return_value=response):
            with pytest.raises(ValueError, match="download limit"):
                vision.load_video(PUBLIC_VIDEO_URL)

    def test_public_url_still_downloads(self):
        response = _FakeResponse(chunks=[b"fake-mp4"])
        with (
            mock.patch.object(vision.requests, "get", return_value=response),
            mock.patch.object(
                vision, "_read_video_frames", return_value=["frame"]
            ) as read_frames,
        ):
            assert vision.load_video(PUBLIC_VIDEO_URL) == ["frame"]
        read_frames.assert_called_once()


class TestAvOpenGuard:
    def _fake_av(self):
        fake_av = mock.MagicMock()
        container = mock.MagicMock()
        stream = mock.MagicMock()
        container.streams.video = [stream]
        container.decode.return_value = iter([])
        fake_av.open.return_value.__enter__.return_value = container
        return fake_av

    def test_file_scheme_is_rejected_before_av_open(self):
        fake_av = self._fake_av()
        with mock.patch.dict("sys.modules", {"av": fake_av}):
            with pytest.raises(ValueError, match="Unsupported video path scheme"):
                list(_iter_video_frames_pyav("file:///etc/passwd", 1))
        fake_av.open.assert_not_called()

    def test_concat_scheme_is_rejected_before_av_open(self):
        fake_av = self._fake_av()
        with mock.patch.dict("sys.modules", {"av": fake_av}):
            with pytest.raises(ValueError, match="Unsupported video path scheme"):
                list(_iter_video_frames_pyav("concat:clip1.mp4|clip2.mp4", 1))
        fake_av.open.assert_not_called()

    def test_private_http_url_is_rejected_before_av_open(self):
        fake_av = self._fake_av()
        with mock.patch.dict("sys.modules", {"av": fake_av}):
            with pytest.raises(ValueError, match="non-public IP"):
                list(_iter_video_frames_pyav(PRIVATE_URL, 1))
        fake_av.open.assert_not_called()

    def test_local_path_still_reaches_av_open(self):
        fake_av = self._fake_av()
        with mock.patch.dict("sys.modules", {"av": fake_av}):
            assert list(_iter_video_frames_pyav("/srv/media/clip.mp4", 1)) == []
        fake_av.open.assert_called_once_with("/srv/media/clip.mp4")


class _ImageHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/image.png":
            payload = b"\x89PNG\r\n\x1a\nfake-png"
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/redirect-to-private":
            self.send_response(302)
            self.send_header(
                "Location",
                f"http://localhost:{self.server.server_port}/evil.png",
            )
            self.end_headers()
            return
        if self.path == "/evil.png":
            self.server.evil_hits += 1
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.end_headers()
            self.wfile.write(b"must-not-be-fetched")
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *_):
        pass


class TestSaveUrlImageRedirects:
    @pytest.fixture(autouse=True)
    def _no_proxy_env(self, monkeypatch):
        # httpx reads proxy settings from the environment; a deployment
        # proxy (e.g. an egress proxy with bracketed IPv6 no_proxy entries)
        # breaks loopback fetches in this test, so scrub it here.
        for name in (
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
            "NO_PROXY",
            "no_proxy",
        ):
            monkeypatch.delenv(name, raising=False)
        yield

    @classmethod
    def setup_class(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ImageHandler)
        cls.server.evil_hits = 0
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_port

    @classmethod
    def teardown_class(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setup_method(self):
        self.server.evil_hits = 0
        # The test server is allowlisted so the initial URL passes; the
        # redirect target (localhost) is not, and must still be blocked.
        configure_media_url_security(["127.0.0.1"], max_file_size_mb=64)

    def test_redirect_to_non_allowlisted_host_is_blocked(self, tmp_path):
        target = tmp_path / "image"
        with pytest.raises(Exception, match="Failed to download image"):
            asyncio.run(
                _save_url_image_to_path(
                    f"http://127.0.0.1:{self.port}/redirect-to-private",
                    str(target),
                )
            )
        assert self.server.evil_hits == 0
        assert not target.exists()

    def test_same_host_download_still_works(self, tmp_path):
        target = tmp_path / "image"
        saved = asyncio.run(
            _save_url_image_to_path(
                f"http://127.0.0.1:{self.port}/image.png",
                str(target),
            )
        )
        assert saved == str(target) + ".png"
        assert (tmp_path / "image.png").exists()
