# SPDX-License-Identifier: Apache-2.0
"""
Mesh processor C++ extension for texture inpainting.

This module provides JIT-compiled C++ mesh processing for fast texture inpainting.
Adapted from Hunyuan3D-2: https://github.com/Tencent/Hunyuan3D-2
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

_abs_path = os.path.dirname(os.path.abspath(__file__))
_mesh_processor_kernel = None


def _load_mesh_processor():
    """JIT compile and load the mesh processor kernel."""
    global _mesh_processor_kernel

    if _mesh_processor_kernel is not None:
        return _mesh_processor_kernel

    from torch.utils.cpp_extension import load

    _mesh_processor_kernel = load(
        name="mesh_processor_kernel",
        sources=[
            f"{_abs_path}/mesh_processor.cpp",
        ],
        extra_cflags=["-O3"],
        verbose=False,
    )
    return _mesh_processor_kernel


def meshVerticeInpaint(
    texture: np.ndarray,
    mask: np.ndarray,
    vtx_pos: np.ndarray,
    vtx_uv: np.ndarray,
    pos_idx: np.ndarray,
    uv_idx: np.ndarray,
    method: str = "smooth",
) -> Tuple[np.ndarray, np.ndarray]:
    """Inpaint texture using mesh vertex connectivity."""
    kernel = _load_mesh_processor()

    texture = np.ascontiguousarray(texture, dtype=np.float32)
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    vtx_pos = np.ascontiguousarray(vtx_pos, dtype=np.float32)
    vtx_uv = np.ascontiguousarray(vtx_uv, dtype=np.float32)
    pos_idx = np.ascontiguousarray(pos_idx, dtype=np.int32)
    uv_idx = np.ascontiguousarray(uv_idx, dtype=np.int32)

    return kernel.meshVerticeInpaint(texture, mask, vtx_pos, vtx_uv, pos_idx, uv_idx, method)


__all__ = ["meshVerticeInpaint"]
