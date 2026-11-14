"""Regression tests for camera geometry helpers (CPU only)."""

import torch

from sglang.multimodal_gen.runtime.utils.camera_geometry import get_plucker_embeddings


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
