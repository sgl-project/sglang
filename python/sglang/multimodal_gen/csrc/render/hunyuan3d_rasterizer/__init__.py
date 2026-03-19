# SPDX-License-Identifier: Apache-2.0
"""
Custom CUDA rasterizer for Hunyuan3D texture generation.

This module provides JIT-compiled CUDA rasterization for fast mesh rendering.
Adapted from Hunyuan3D-2: https://github.com/Tencent/Hunyuan3D-2
"""

from __future__ import annotations

import os
from typing import List, Tuple

import torch

_abs_path = os.path.dirname(os.path.abspath(__file__))
_custom_rasterizer_kernel = None


def _load_custom_rasterizer():
    """JIT compile and load the custom rasterizer kernel."""
    global _custom_rasterizer_kernel

    if _custom_rasterizer_kernel is not None:
        return _custom_rasterizer_kernel

    from torch.utils.cpp_extension import load

    _custom_rasterizer_kernel = load(
        name="custom_rasterizer_kernel",
        sources=[
            f"{_abs_path}/rasterizer.cpp",
            f"{_abs_path}/rasterizer_gpu.cu",
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )
    return _custom_rasterizer_kernel


def rasterize(
    pos: torch.Tensor,
    tri: torch.Tensor,
    resolution: Tuple[int, int],
    clamp_depth: torch.Tensor = None,
    use_depth_prior: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rasterize mesh to get face indices and barycentric coordinates."""
    kernel = _load_custom_rasterizer()

    if clamp_depth is None:
        clamp_depth = torch.zeros(0, device=pos.device)

    # pos should be [N, 4], remove batch dim if present
    if pos.dim() == 3:
        pos = pos[0]

    findices, barycentric = kernel.rasterize_image(
        pos, tri, clamp_depth, resolution[1], resolution[0], 1e-6, use_depth_prior
    )
    return findices, barycentric


def interpolate(
    col: torch.Tensor,
    findices: torch.Tensor,
    barycentric: torch.Tensor,
    tri: torch.Tensor,
) -> torch.Tensor:
    """Interpolate vertex attributes using barycentric coordinates."""
    # Handle zero indices (background)
    f = findices - 1 + (findices == 0)
    vcol = col[0, tri.long()[f.long()]]
    result = barycentric.view(*barycentric.shape, 1) * vcol
    result = torch.sum(result, axis=-2)
    return result.view(1, *result.shape)


__all__ = ["rasterize", "interpolate"]
