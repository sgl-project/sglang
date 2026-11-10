"""SM70 sigmoid-gated RMSNorm over 128-wide heads, bitwise equal to fla's
Triton layer_norm_gated_fwd_kernel for fp16 (the KDA output norm)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module

HEAD_DIM = 128


@cache_once
def _module() -> Module:
    return load_jit(
        "sm70_norm_gate",
        cuda_files=["elementwise/sm70_norm_gate.cuh"],
        cuda_wrappers=[("norm_gate", "sglang::sm70_norm_gate::norm_gate")],
        extra_cuda_cflags=["--fmad=false"],
    )


@cache_once
def _is_sm70(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) == (7, 0)


def sm70_norm_gate_covered(
    x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor | None, activation: str
) -> bool:
    """x, g [rows, 128] after the caller's reshape."""
    return (
        activation == "sigmoid"
        and weight is not None
        and x.is_cuda
        and x.dim() == 2
        and x.size(0) > 0
        and x.size(1) == HEAD_DIM
        and g.shape == x.shape
        and x.dtype == g.dtype == weight.dtype == torch.float16
        and weight.shape == (HEAD_DIM,)
        and x.is_contiguous()
        and g.is_contiguous()
        and weight.is_contiguous()
        and x.data_ptr() % 16 == 0
        and g.data_ptr() % 16 == 0
        and weight.data_ptr() % 16 == 0
        and _is_sm70(x.device.index)
    )


def sm70_norm_gate(
    x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """In place into x, like layer_norm_gated_fwd with out_dtype=None."""
    _module().norm_gate(x, x, g, weight, float(eps))
    return x
