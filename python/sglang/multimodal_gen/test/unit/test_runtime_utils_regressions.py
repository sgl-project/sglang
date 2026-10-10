"""Regression tests for small runtime/utils and pipeline helper bugs (CPU only)."""

import base64
import io
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.utils import common, vision
from sglang.multimodal_gen.runtime.utils.camera_geometry import get_plucker_embeddings
from sglang.multimodal_gen.runtime.utils.image_io import save_base64_image_to_path
from sglang.multimodal_gen.runtime.utils.mesh3d_utils import (
    ImageProcessorV2,
    MeshRender,
)


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


def test_plucker_rays_for_bf16_poses_match_float32():
    """bf16 cannot represent pixel coordinates above 256, so neighbouring pixels of
    a wide frame collapsed onto the same ray when poses were bf16."""
    n_frames, height, width = 1, 4, 640
    c2ws = torch.eye(4)[None].repeat(n_frames, 1, 1)
    # Values exactly representable in bf16, so only the pixel grid can differ.
    Ks = torch.tensor([[512.0, 512.0, 320.0, 2.0]])

    expected = get_plucker_embeddings(c2ws, Ks, height, width)
    actual = get_plucker_embeddings(c2ws.bfloat16(), Ks.bfloat16(), height, width)

    assert torch.equal(actual.float(), expected)


def test_default_texture_follows_width_height_texture_size():
    """texture_size is (width, height); the blank default texture was built as
    (width, height, 3) instead of the (height, width, 3) of a loaded texture."""
    render = MeshRender.__new__(MeshRender)
    render.tex = None
    render.texture_size = (64, 32)

    assert render.get_texture().shape == (32, 64, 3)


def test_image_processor_rejects_unsupported_input_type():
    """An input that was neither a path nor a PIL image fell through to an
    UnboundLocalError on `mask`."""
    with pytest.raises(TypeError, match="Unsupported image type"):
        ImageProcessorV2(size=8).load_image(torch.zeros(3, 8, 8))
