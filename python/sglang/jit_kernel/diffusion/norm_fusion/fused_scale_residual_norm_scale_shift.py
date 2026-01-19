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
    return f"fused_scale_residual_norm_scale_shift"


@functools.cache
def _jit_scale_residual_norm_scale_shift_module(
    norm_enum: NormEnum,
    dtype: torch.dtype,
    scale_index_enum: IndexEnum,
    shift_index_enum: IndexEnum,
    gate_index_enum: IndexEnum,
    head_dim: int,
) -> Module:
    kernel_name = _get_kernel_name()

    args = make_cpp_args(
        norm_enum, dtype, scale_index_enum, shift_index_enum, gate_index_enum, head_dim
    )
    marker = f"{kernel_name}_d{head_dim}"  # Different marker for different dims
    export_name = kernel_name

    return load_jit(
        marker,
        cuda_files=["diffusion/fused_norm_scale_shift.cuh"],
        cuda_wrappers=[(export_name, f"{kernel_name}<{args}>")],
    )


def fused_scale_residual_norm_scale_shift(
    residual: torch.Tensor,
    x: torch.Tensor,
    gate: Optional[torch.Tensor],
    gamma: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    scale: torch.Tensor,
    shift: torch.Tensor,
    norm_type: str,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Fused: (residual + gate * x) -> LayerNorm(gamma, beta) -> scale/shift.
    or Fused: (residual + gate * x) -> RMSNorm(gamma) -> scale/shift.
    Expects:
      - residual/x: [B, S, D], contiguous on last dim
      - gate: None, [D], [1/B, D],  [1/B, 1/S, D] or [B, F, 1, D]
      - gamma/beta: None, [D]
      - scale/shift: [D], [1/B, D],  [1/B, 1/S, D] or [B, F, 1, D]
      - norm_type: str, "layer" or "rms"
      - eps: Optional[float], default: 1e-5
    """

    if norm_type == "layer" or norm_type == "rms":
        norm_enum = get_norm_enum(norm_type)
        scale_index_enum = get_index_enum(scale)
        shift_index_enum = get_index_enum(shift)
        gate_index_enum = get_index_enum(gate)
        head_dim = x.shape[-1]

        # Validate head_dim is supported by norm.cuh
        if head_dim <= 256:
            assert head_dim in (
                64,
                128,
                256,
            ), f"D={head_dim} not supported, must be 64, 128, or 256 for D <= 256"
        else:
            assert (
                head_dim % 256 == 0 and head_dim <= 8192
            ), f"D={head_dim} not supported, must be multiple of 256 and <= 8192"

        y = torch.empty_like(x)
        residual_output = torch.empty_like(x)
        module = _jit_scale_residual_norm_scale_shift_module(
            norm_enum,
            x.dtype,
            scale_index_enum,
            shift_index_enum,
            gate_index_enum,
            head_dim,
        )
        kernel = getattr(module, _get_kernel_name())
        kernel(y, residual_output, residual, x, gate, gamma, beta, scale, shift, eps)
        return y, residual_output
    else:
        raise ValueError(f'norm_type must be one of "layer" and "rms"')
