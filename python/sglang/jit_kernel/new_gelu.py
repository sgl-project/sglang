from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_new_gelu_module(dtype: torch.dtype) -> Module:
    """Compile and cache the JIT new_gelu module for a given dtype."""
    args = make_cpp_args(dtype)
    return load_jit(
        "new_gelu",
        *args,
        cuda_files=["elementwise/new_gelu.cuh"],
        cuda_wrappers=[("new_gelu", f"new_gelu<{args}>")],
    )


def new_gelu(src: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Element-wise NewGELU activation.

    NewGELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))

    Supported dtypes: torch.float16, torch.bfloat16, torch.float32.

    Parameters
    ----------
    src : CUDA tensor (FP16 / BF16 / FP32)
    out : optional pre-allocated output tensor (same shape/dtype as src)

    Returns
    -------
    Activated tensor.
    """
    if not src.is_cuda:
        raise RuntimeError("src must be a CUDA tensor")
    if src.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(
            f"Unsupported dtype {src.dtype}. Supported: float16, bfloat16, float32"
        )
    if out is None:
        out = torch.empty_like(src)
    elif not out.is_contiguous():
        raise RuntimeError("out must be contiguous")

    # Kernel operates on flat 1D view; reshape for multi-dim inputs
    src_flat = src.contiguous().view(-1)
    out_flat = out.view(-1)

    module = _jit_new_gelu_module(src.dtype)
    module.new_gelu(out_flat, src_flat)
    return out
