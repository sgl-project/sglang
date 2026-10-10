"""Regression tests for pipeline and model-helper bugs (CPU only)."""

import torch

from sglang.multimodal_gen.runtime.utils.camera_geometry import get_plucker_embeddings
from sglang.multimodal_gen.runtime.utils.mesh3d_utils import (
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
