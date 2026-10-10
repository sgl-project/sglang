"""Regression tests for small runtime/utils and pipeline helper bugs (CPU only)."""

import base64
import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from sglang.multimodal_gen.runtime.utils import common, vision
from sglang.multimodal_gen.runtime.utils.image_io import save_base64_image_to_path


def _gif_bytes() -> bytes:
    buffer = io.BytesIO()
    frames = [Image.new("RGB", (4, 4), color) for color in ("red", "blue")]
    frames[0].save(buffer, format="GIF", save_all=True, append_images=frames[1:])
    return buffer.getvalue()


@pytest.mark.parametrize("payload", [_gif_bytes(), b"not a gif"])
def test_load_video_removes_downloaded_temp_file(monkeypatch, tmp_path, payload):
    """Every URL video was left in the temp dir for good, whether decoding
    succeeded or not."""
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    response = SimpleNamespace(
        status_code=200, iter_content=lambda chunk_size: iter([payload])
    )
    monkeypatch.setattr(vision.requests, "get", lambda *args, **kwargs: response)

    try:
        vision.load_video("https://example.com/clip.gif")
    except Exception:
        pass

    assert list(tmp_path.iterdir()) == []


def test_line_wrapped_base64_image_is_saved_in_full(tmp_path):
    """A MIME-style base64 payload with newlines was silently cut at the first
    line, and the truncated bytes were written to disk without an error."""
    raw = bytes(range(256)) * 2
    wrapped = base64.encodebytes(raw).decode()
    assert "\n" in wrapped.strip()

    path = save_base64_image_to_path(
        f"data:image/png;base64,{wrapped}", str(tmp_path / "upload")
    )

    with open(path, "rb") as f:
        assert f.read() == raw


def test_bool_env_var_warning_is_deduplicated_per_variable(monkeypatch):
    """The warning was keyed by the bad value, so a second variable holding the
    same bad value was never reported."""
    monkeypatch.setattr(common, "_warned_bool_env_var_keys", set())
    monkeypatch.setenv("SGLANG_TEST_FLAG_A", "maybe")
    monkeypatch.setenv("SGLANG_TEST_FLAG_B", "maybe")

    with patch.object(common.logger, "warning") as warning:
        common.get_bool_env_var("SGLANG_TEST_FLAG_A")
        common.get_bool_env_var("SGLANG_TEST_FLAG_A")
        common.get_bool_env_var("SGLANG_TEST_FLAG_B")

    assert warning.call_count == 2
