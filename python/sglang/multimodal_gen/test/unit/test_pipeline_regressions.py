"""Regression tests for pipeline and model-helper bugs (CPU only)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.pipelines.diffusers_pipeline import (
    DiffusersExecutionStage,
)
from sglang.multimodal_gen.runtime.pipelines.sana_video import (
    SanaVideoTextEncodingStage,
)
from sglang.multimodal_gen.runtime.utils.camera_geometry import get_plucker_embeddings
from sglang.multimodal_gen.runtime.utils.mesh3d_utils import (
    ImageProcessorV2,
    MeshRender,
)


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


@pytest.mark.parametrize(
    "data, expected_shape",
    [
        (np.random.rand(2, 5, 16, 16, 3).astype("float32"), (2, 3, 5, 16, 16)),
        ([[Image.new("RGB", (16, 16)) for _ in range(5)]], (1, 3, 5, 16, 16)),
        ([[Image.new("RGB", (16, 16)) for _ in range(3)]], (1, 3, 3, 16, 16)),
        ([Image.new("RGB", (16, 16)) for _ in range(2)], (2, 3, 16, 16)),
    ],
    ids=["np-video", "pil-video", "pil-video-3-frames", "pil-image-batch"],
)
def test_diffusers_output_layout_is_channels_first(data, expected_shape):
    """Video outputs were permuted twice (np) or misread as a batch (PIL lists), and
    several PIL images from an image pipeline were stacked as video frames."""
    stage = DiffusersExecutionStage.__new__(DiffusersExecutionStage)

    output = stage._postprocess_output(stage._convert_to_tensor(data))

    assert tuple(output.shape) == expected_shape


def test_sana_video_negative_prompt_uses_requested_sequence_length():
    """The negative prompt was always encoded at 300 tokens, so a custom
    max_sequence_length gave positive and negative embeddings different lengths."""
    stage = SanaVideoTextEncodingStage.__new__(SanaVideoTextEncodingStage)
    captured = {}
    stage.encode_text = lambda *args, **kwargs: captured.update(kwargs)

    stage._encode_negative_text(
        SimpleNamespace(negative_prompt="Blurry", max_sequence_length=128), None, [0]
    )
    assert captured["max_length"] == 128

    stage._encode_negative_text(
        SimpleNamespace(negative_prompt="Blurry", max_sequence_length=None), None, [0]
    )
    assert captured["max_length"] == 300
