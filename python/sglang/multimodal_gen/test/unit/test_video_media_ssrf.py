# SPDX-License-Identifier: Apache-2.0
"""SSRF regression tests for client-supplied media URLs in /v1/videos.

Covers sgl-project/sglang#51701: unauthenticated media URLs (video_path,
input_reference/image_path, control_path) must not make the server fetch
private, loopback, link-local, or reserved addresses, or hand FFmpeg
protocol-prefixed strings to ``av.open``. Public http(s) URLs keep working.
"""

import asyncio
import base64
import http.server
import threading
import unittest.mock as mock
import wave
from contextlib import ExitStack
from types import SimpleNamespace

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
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.cosmos3_simulation.multiview import (
    media_hw,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.material_io import (
    minimax_h3_localize_material_uri,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_animate_2.audio import (
    extract_reference_video_audio,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.wan_s2v import (
    WanS2VBeforeDenoisingStage,
)
from sglang.multimodal_gen.runtime.utils import vision
from sglang.srt.multimodal.media_io import (
    configure_media_url_security,
    get_mm_http_session,
)

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


def _fake_av():
    fake_av = mock.MagicMock()
    container = mock.MagicMock()
    stream = mock.MagicMock()
    container.streams.video = [stream]
    container.decode.return_value = iter([])
    fake_av.open.return_value.__enter__.return_value = container
    return fake_av


class TestEntrypointMediaValidation:
    def test_policy_error_detail_is_not_forwarded(self):
        with (
            mock.patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.utils.check_remote_media_url",
                side_effect=ValueError("synthetic operator configuration"),
            ),
            pytest.raises(HTTPException) as error,
        ):
            _validate_remote_media_paths({"video_path": PUBLIC_VIDEO_URL})
        assert error.value.status_code == 400
        assert error.value.detail == "video_path points to a disallowed media URL"
        assert error.value.__cause__ is None and error.value.__context__ is None

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
            # FFmpeg protocol strings with no URL scheme.
            "subfile,,start,0,end,99,,:http:169.254.169.254/latest",
            "subfile,//:http:169.254.169.254/latest",
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
            {"video_path": "C:\\media\\clip.mp4"},
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
        body = iter(chunks)
        self.raw = SimpleNamespace(
            media_sock=mock.Mock(),
            isclosed=lambda: False,
            read1=lambda amt, decode_content: next(body, b""),
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def close(self):
        pass

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class TestLoadVideoDownload:
    def test_private_url_is_rejected_without_fetching(self):
        with mock.patch.object(get_mm_http_session(), "get") as fake_get:
            with pytest.raises(ValueError, match="non-public IP"):
                vision.load_video(PRIVATE_URL)
        fake_get.assert_not_called()

    def test_redirect_to_private_ip_is_rejected(self):
        redirect = _FakeResponse(
            status_code=302,
            headers={"Location": PRIVATE_URL},
        )
        with mock.patch.object(
            get_mm_http_session(), "get", return_value=redirect
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
        with mock.patch.object(get_mm_http_session(), "get", return_value=redirect):
            with pytest.raises(requests.exceptions.TooManyRedirects):
                vision.load_video(PUBLIC_VIDEO_URL)

    def test_oversized_download_is_rejected(self, monkeypatch):
        monkeypatch.setattr(
            "sglang.srt.multimodal.media_io._media_url_max_file_size_bytes", 10
        )
        response = _FakeResponse(chunks=[b"x" * 11])
        with mock.patch.object(get_mm_http_session(), "get", return_value=response):
            with pytest.raises(ValueError, match="download limit"):
                vision.load_video(PUBLIC_VIDEO_URL)

    def test_public_url_still_downloads(self):
        response = _FakeResponse(chunks=[b"fake-mp4"])
        with (
            mock.patch.object(get_mm_http_session(), "get", return_value=response),
            mock.patch.object(
                vision, "_read_video_frames", return_value=["frame"]
            ) as read_frames,
        ):
            assert vision.load_video(PUBLIC_VIDEO_URL) == ["frame"]
        read_frames.assert_called_once()

    def test_url_is_staged_with_a_fixed_suffix(self):
        url = "https://93.184.216.34/playlist.m3u8"
        with mock.patch.object(
            vision,
            "download_video_to_stream",
            side_effect=lambda _, output: output.write(b"video-bytes"),
        ) as download:
            with vision.local_video_path(url) as local:
                assert local.endswith(".mp4")
                with open(local, "rb") as f:
                    assert f.read() == b"video-bytes"
        download.assert_called_once()
        assert download.call_args.args[0] == url


class TestAvOpenGuard:
    def test_audio_only_media_does_not_echo_its_source(self, tmp_path):
        pytest.importorskip("av")
        path = tmp_path / "synthetic-private-name.wav"
        with wave.open(str(path), "wb") as wav:
            wav.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
            wav.writeframes(bytes(160))
        data_uri = (
            "data:audio/wav;base64," + base64.b64encode(path.read_bytes()).decode()
        )
        for source in (str(path), data_uri):
            with pytest.raises(
                ValueError, match=r"^Input media has no video stream\.$"
            ):
                list(_iter_video_frames_pyav(source, 1))

    def test_file_scheme_is_rejected_before_av_open(self):
        fake_av = _fake_av()
        with mock.patch.dict("sys.modules", {"av": fake_av}):
            with pytest.raises(ValueError, match="Unsupported video path scheme"):
                list(_iter_video_frames_pyav("file:///etc/passwd", 1))
        fake_av.open.assert_not_called()

    def test_concat_scheme_is_rejected_before_av_open(self):
        fake_av = _fake_av()
        with mock.patch.dict("sys.modules", {"av": fake_av}):
            with pytest.raises(ValueError, match="Unsupported video path scheme"):
                list(_iter_video_frames_pyav("concat:clip1.mp4|clip2.mp4", 1))
        fake_av.open.assert_not_called()

    def test_private_http_url_is_rejected_before_av_open(self):
        fake_av = _fake_av()
        with mock.patch.dict("sys.modules", {"av": fake_av}):
            with pytest.raises(ValueError, match="Could not open video source"):
                list(_iter_video_frames_pyav(PRIVATE_URL, 1))
        fake_av.open.assert_not_called()

    def test_approved_transport_failure_does_not_retain_url(self):
        url = PUBLIC_VIDEO_URL + "?signature=synthetic-private-marker"
        fake_av = _fake_av()
        with (
            mock.patch.dict("sys.modules", {"av": fake_av}),
            mock.patch.object(
                get_mm_http_session(), "get", side_effect=requests.Timeout(url)
            ),
            pytest.raises(
                ValueError, match=r"^Could not open video source \(Timeout\)\.$"
            ) as exc,
        ):
            list(_iter_video_frames_pyav(url, 1))
        fake_av.open.assert_not_called()
        assert exc.value.__context__ is None and exc.value.__cause__ is None

    @pytest.mark.parametrize("relative", [False, True])
    def test_unavailable_scratch_fails_before_download(
        self, tmp_path, monkeypatch, relative
    ):
        root = (
            "synthetic-private-root" if relative else str(tmp_path / "missing-volume")
        )
        monkeypatch.setenv("TMPDIR", root)
        fake_av = _fake_av()
        with (
            mock.patch.dict("sys.modules", {"av": fake_av}),
            mock.patch.object(get_mm_http_session(), "get") as get,
            pytest.raises(ValueError, match="Could not open video source") as exc,
        ):
            list(_iter_video_frames_pyav(PUBLIC_VIDEO_URL, 1))
        get.assert_not_called()
        fake_av.open.assert_not_called()
        assert root not in str(exc.value)
        assert exc.value.__context__ is None

    def test_local_path_still_reaches_av_open(self, tmp_path):
        fake_av = _fake_av()
        path = tmp_path / "clip.mp4"
        path.write_bytes(b"fixture")
        with mock.patch.dict("sys.modules", {"av": fake_av}):
            assert list(_iter_video_frames_pyav(str(path), 1)) == []
        source = fake_av.open.call_args.args[0]
        assert source.name == str(path) and source.closed
        assert fake_av.open.call_args.kwargs == {
            "options": {"protocol_whitelist": "data"}
        }

    def test_approved_download_hands_decoder_local_bytes(self):
        fake_av = _fake_av()
        opened = []

        def open_stream(stream, *, options):
            assert stream.seekable() and stream.read() == b"fixture clip"
            assert options == {"protocol_whitelist": "data"}
            opened.append(stream)
            return fake_av.open.return_value

        fake_av.open.side_effect = open_stream
        with (
            mock.patch.dict("sys.modules", {"av": fake_av}),
            mock.patch.object(
                get_mm_http_session(),
                "get",
                return_value=_FakeResponse(chunks=[b"fixture clip"]),
            ) as download,
        ):
            assert list(_iter_video_frames_pyav(PUBLIC_VIDEO_URL, 1)) == []
        download.assert_called_once()
        assert len(opened) == 1 and opened[0].closed


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


class TestModelSpecificFetchPaths:
    """Each loader fetches through the shared policy, so a private URL fails there."""

    def test_cosmos3_multiview_does_not_hand_av_a_url(self):
        fake_av = _fake_av()
        with mock.patch.dict("sys.modules", {"av": fake_av}):
            with pytest.raises(ValueError, match="non-public IP"):
                media_hw(PRIVATE_URL)
        fake_av.open.assert_not_called()

    def test_wan_s2v_remote_input_is_checked(self, tmp_path):
        stage = object.__new__(WanS2VBeforeDenoisingStage)
        with (
            mock.patch.dict("os.environ", {"TMPDIR": str(tmp_path)}),
            ExitStack() as files,
            pytest.raises(ValueError, match="non-public IP"),
        ):
            stage._get_single_path(PRIVATE_URL, "image_path", files)

    def test_wan_animate_2_audio_does_not_hand_av_a_url(self):
        fake_av = _fake_av()
        warnings = []
        with mock.patch.dict("sys.modules", {"av": fake_av}):
            result = extract_reference_video_audio(
                PRIVATE_URL,
                log_info=lambda msg, *args: None,
                log_warning=lambda msg, *args: warnings.append(msg % args),
            )
        assert result == (None, None)
        assert len(warnings) == 1
        fake_av.open.assert_not_called()

    def test_minimax_h3_http_material_is_checked(self, tmp_path):
        batch = SimpleNamespace(extra={})
        with (
            mock.patch("tempfile.tempdir", str(tmp_path)),
            pytest.raises(ValueError, match="non-public IP"),
        ):
            minimax_h3_localize_material_uri(
                batch, PRIVATE_URL, condition_type="image", condition_index=0
            )
