from __future__ import annotations

import functools
import os
from typing import TYPE_CHECKING, Optional

import flashinfer
import torch

from sglang.jit_kernel.utils import load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _get_kernel_name(norm_type: str) -> str:
    return f"fused_scale_residual_{norm_type}norm_scale_shift"


@functools.cache
def _jit_scale_residual_norm_scale_shift_module(norm_type: str) -> Module:
    kernel_name = _get_kernel_name(norm_type)
    marker = kernel_name
    export_name = kernel_name
    # TODO: workaround, do not import cutlass from flashinfer
    cutlass_include = os.path.join(
        os.path.dirname(flashinfer.__file__), "data", "cutlass", "include"
    )

    return load_jit(
        marker,
        cuda_files=["diffusion/fused_norm_scale_shift.cuh"],
        cuda_wrappers=[(export_name, kernel_name)],
        extra_include_paths=[cutlass_include],
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
      - residual/x: [M, N], contiguous on last dim
      - gate: None, [1, N], [M, N], [1, 1, N], [B, 1, N], or [B, F, 1, N]
      - gamma/beta: None, [N]
      - scale/shift: [M, N] or [B, F, 1, N]
      - norm_type: str, "layer" or "rms"
      - eps: Optional[float], default: 1e-5
    """
    if norm_type == "layer" or norm_type == "rms":
        y = torch.empty_like(x)
        residual_output = torch.empty_like(x)
        module = _jit_scale_residual_norm_scale_shift_module(norm_type)
        kernel = getattr(module, _get_kernel_name(norm_type))
        kernel(y, residual_output, residual, x, gate, gamma, beta, scale, shift, eps)
        return y, residual_output
    else:
        raise ValueError(f'norm_type must be one of "layer" and "rms"')
