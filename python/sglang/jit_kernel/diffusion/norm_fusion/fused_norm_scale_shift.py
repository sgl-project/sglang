from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.diffusion.norm_fusion.norm_fusion import (
    IndexEnum,
    NormEnum,
    get_index_enum,
    get_norm_enum,
)
from sglang.jit_kernel.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _get_kernel_name() -> str:
    return f"fused_norm_scale_shift"


@functools.cache
def _jit_norm_scale_shift_module(
    norm_enum: NormEnum,
    dtype: torch.dtype,
    scale_index_mode: IndexEnum,
    shift_index_mode: IndexEnum,
) -> Module:
    kernel_name = _get_kernel_name()
    args = make_cpp_args(norm_enum, dtype, scale_index_mode, shift_index_mode)
    marker = kernel_name
    export_name = kernel_name

    return load_jit(
        marker,
        cuda_files=["diffusion/fused_norm_scale_shift.cuh"],
        cuda_wrappers=[(export_name, f"{kernel_name}<{args}>")],
    )


def fused_norm_scale_shift(
    x: torch.Tensor,
    gamma: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    LayerNorm(x, gamma, beta) followed by fused scale/shift.
    or RMSNorm(x, gamma) followed by fused scale/shift.
    Expects:
      - x: [B, S, D], contiguous on last dim
      - gamma/beta: None, [D]
      - scale/shift: [D], [1/B, D],  [1/B, 1/S, D] or [B, F, 1, D]
      - norm_type: str, "layer" or "rms"
      - eps: Optional[float], default: 1e-5
    """
    if norm_type == "layer" or norm_type == "rms":
        norm_enum = get_norm_enum(norm_type)
        scale_index_mode = get_index_enum(scale)
        shift_index_mode = get_index_enum(shift)

        y = torch.empty_like(x)
        module = _jit_norm_scale_shift_module(
            norm_enum, x.dtype, scale_index_mode, shift_index_mode
        )
        kernel = getattr(module, _get_kernel_name())
        kernel(y, x, gamma, beta, scale, shift, eps)
        return y
    else:
        raise ValueError(f'norm_type must be one of "layer" and "rms"')
