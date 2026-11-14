import base64
import io
from unittest.mock import patch

import pytest
from PIL import Image, UnidentifiedImageError

from sglang.multimodal_gen.runtime.utils import common, vision
from sglang.multimodal_gen.runtime.utils.image_io import save_base64_image_to_path


def _gif_bytes() -> bytes:
    buffer = io.BytesIO()
    frames = [Image.new("RGB", (4, 4), color) for color in ("red", "blue")]
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:])
    return buffer.getvalue()


@pytest.mark.parametrize(
    "payload, error",
    [
        (_gif_bytes(), None),
        (b"not a gif", UnidentifiedImageError),
        (ValueError("Media URL must not point to a local or private address"), ValueError),
    ],
)
def test_load_video_removes_downloaded_temp_file(monkeypatch, tmp_path, payload, error):
    """A downloaded URL video must not outlive load_video, on success or failure."""
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    def download(url, timeout):
        if isinstance(payload, Exception):
            raise payload
        return payload

    monkeypatch.setattr(vision, "download_remote_media", download)

    if error is None:
        frames = vision.load_video("https://example.com/clip.gif")
        assert [frame.convert("RGB").getpixel((0, 0)) for frame in frames] == [
            (255, 0, 0),
            (0, 0, 255),
        ]
    else:
        with pytest.raises(error):
            vision.load_video("https://example.com/clip.gif")

    assert list(tmp_path.iterdir()) == []


def test_load_video_applies_media_url_policy(monkeypatch):
    """Remote video URLs go through the shared private-address check."""
    monkeypatch.delenv("SGLANG_ALLOW_PRIVATE_MEDIA_URLS", raising=False)
    with pytest.raises(ValueError, match="local or private"):
        vision.load_video("http://127.0.0.1/clip.mp4")


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
