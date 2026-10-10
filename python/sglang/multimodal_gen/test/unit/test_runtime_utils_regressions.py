"""Regression tests for small runtime/utils and pipeline helper bugs (CPU only)."""

import io
from types import SimpleNamespace

import pytest
from PIL import Image

from sglang.multimodal_gen.runtime.utils import vision


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
