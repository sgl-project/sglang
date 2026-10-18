from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils.custom_op import register_custom_op

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _gelu_module(dtype: torch.dtype, quick: bool) -> Module:
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(f"Unsupported GELU dtype: {dtype}")
    args = make_cpp_args(dtype, quick, is_arch_support_pdl())
    return load_jit(
        "unary_gelu",
        *args,
        cuda_files=["elementwise/gelu.cuh"],
        cuda_wrappers=[("run", f"gelu<{args}>")],
    )


@register_custom_op(mutates_args=["out"])
def _gelu_inplace(input: torch.Tensor, out: torch.Tensor, quick: bool) -> None:
    module = _gelu_module(input.dtype, quick)
    module.run(input.view(-1), out.view(-1))


def _run_gelu(
    input: torch.Tensor, out: torch.Tensor | None, quick: bool
) -> torch.Tensor:
    if out is not None and out.shape != input.shape:
        raise ValueError("GELU output must have the same shape as input")
    input = input.contiguous()
    if out is None:
        out = torch.empty_like(input)
    _gelu_inplace(input, out, quick)
    return out


def new_gelu(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """NewGELU: ``0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x**3)))``.

    CUDA FP16/BF16/FP32 inputs of any shape; computes in FP32 and rounds once.
    Non-contiguous inputs are copied; a supplied output must be contiguous.
    """
    return _run_gelu(input, out, False)


def quick_gelu(input: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """QuickGELU: ``x*sigmoid(1.702*x)``; same contract as :func:`new_gelu`."""
    return _run_gelu(input, out, True)
