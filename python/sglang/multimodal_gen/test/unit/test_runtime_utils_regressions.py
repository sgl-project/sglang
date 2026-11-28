import base64
import io
import traceback
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import requests
from PIL import Image, UnidentifiedImageError

from sglang.multimodal_gen.runtime.utils import common, vision
from sglang.multimodal_gen.runtime.utils.image_io import save_base64_image_to_path
from sglang.srt.multimodal.media_io import get_mm_http_session

# A public literal avoids DNS; the tests mock the HTTP transfer.
_GIF_URL = "https://93.184.216.34/fixture.gif"


def _gif_bytes() -> bytes:
    buffer = io.BytesIO()
    frames = [Image.new("RGB", (4, 4), color) for color in ("red", "blue")]
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:])
    return buffer.getvalue()


@pytest.mark.parametrize(
    "loader, dependency, category",
    [
        (vision.load_image, "srt_get_image_bytes", requests.HTTPError),
        (vision.load_video, "get", requests.HTTPError),
        (vision.load_video, "get", requests.Timeout),
        (vision.load_video, "get", requests.TooManyRedirects),
    ],
)
def test_media_download_errors_keep_category_not_request(loader, dependency, category):
    marker = "synthetic-media-marker"
    url = f"{_GIF_URL}?ticket={marker}"
    request = requests.Request("GET", url).prepare()
    response = None
    if category is requests.HTTPError:
        response = requests.Response()
        response.status_code = 403
    original = category(url, request=request, response=response)
    original.__cause__ = OSError(marker)
    with patch.object(
        vision if dependency != "get" else get_mm_http_session(),
        dependency,
        side_effect=original,
    ):
        with pytest.raises(category) as caught:
            loader(url)
    error = caught.value
    if response is not None:
        assert "HTTP 403" in str(error)
    assert marker not in "".join(traceback.format_exception(error))
    assert (
        error.request is error.response is error.__cause__ is error.__context__ is None
    )


def test_invalid_media_source_does_not_echo_user_input():
    marker = "synthetic-media-marker"
    with patch.object(vision, "srt_get_image_bytes", side_effect=ValueError(marker)):
        with pytest.raises(ValueError, match="image source") as caught:
            vision.load_image(marker)
    assert marker not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__context__ is None
    with pytest.raises(ValueError, match="existing video file") as caught:
        vision.load_video(marker)
    assert marker not in "".join(traceback.format_exception(caught.value))


@pytest.mark.parametrize(
    "status_code, payload, error",
    [
        (200, _gif_bytes(), None),
        (200, b"not a gif", UnidentifiedImageError),
        (404, b"", requests.HTTPError),
    ],
)
def test_load_video_removes_downloaded_temp_file(
    monkeypatch, tmp_path, status_code, payload, error
):
    """A downloaded URL video must not outlive load_video, on success or failure."""
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    closed = []

    class Response:
        headers = {}
        url = _GIF_URL

        def __init__(self):
            self.status_code = status_code
            body = iter([payload])
            self.raw = SimpleNamespace(
                media_sock=Mock(),
                isclosed=lambda: False,
                read1=lambda amt, decode_content: next(body, b""),
            )

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            closed.append(True)

        def raise_for_status(self):
            if status_code >= 400:
                raise requests.HTTPError(f"{status_code}", response=None)

    monkeypatch.setattr(
        get_mm_http_session(), "get", lambda *args, **kwargs: Response()
    )

    if error is None:
        frames = vision.load_video(_GIF_URL)
        assert [frame.convert("RGB").getpixel((0, 0)) for frame in frames] == [
            (255, 0, 0),
            (0, 0, 255),
        ]
    else:
        with pytest.raises(error):
            vision.load_video(_GIF_URL)

    assert list(tmp_path.iterdir()) == []
    assert closed == [True]


def test_line_wrapped_base64_image_is_saved_in_full(tmp_path):
    """A line-wrapped base64 payload must be decoded in full, not cut at the first line."""
    raw = bytes(range(256)) * 2
    wrapped = base64.encodebytes(raw).decode()
    assert "\n" in wrapped.strip()

    path = save_base64_image_to_path(
        f"data:image/png;base64,{wrapped}", str(tmp_path / "upload")
    )

    with open(path, "rb") as f:
        assert f.read() == raw


def test_media_type_with_newline_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_base64_image_to_path(
            "data:image/p\nng;base64,aGVsbG8=", str(tmp_path / "upload")
        )
    assert not any("\n" in p.name for p in tmp_path.rglob("*"))


def test_bool_env_var_warning_is_deduplicated_per_variable(monkeypatch):
    """Each env var with a bad bool value must be warned about once."""
    monkeypatch.setattr(common, "_warned_bool_env_var_keys", set())
    monkeypatch.setenv("SGLANG_TEST_FLAG_A", "maybe")
    monkeypatch.setenv("SGLANG_TEST_FLAG_B", "maybe")

    with patch.object(common.logger, "warning") as warning:
        common.get_bool_env_var("SGLANG_TEST_FLAG_A")
        common.get_bool_env_var("SGLANG_TEST_FLAG_A")
        common.get_bool_env_var("SGLANG_TEST_FLAG_B")

    assert warning.call_count == 2
