"""Softmax kernel for LLM sampling with large vocabulary sizes.

Two execution paths dispatched at runtime via `num_splits`:
  - **Fused** (num_splits=None): single-block per row
  - **Split** (num_splits>1): multi-block per row with merge

out dtype matches input dtype.  All internal computation is in fp32.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_softmax_module(dtype: torch.dtype) -> Module:
    args = make_cpp_args(is_arch_support_pdl(), dtype)
    return load_jit(
        "softmax",
        *args,
        cuda_files=["elementwise/softmax.cuh"],
        cuda_wrappers=[("softmax", f"SoftmaxKernel<{args}>::run")],
        extra_cuda_cflags=["--use_fast_math"],
    )


def can_use_softmax_sampling(logits: torch.Tensor) -> bool:
    dtype = logits.dtype
    if not (
        logits.is_cuda
        and dtype in (torch.float16, torch.bfloat16, torch.float32)
        and (logits.shape[-1] * dtype.itemsize) % 16 == 0
    ):
        return False
    try:
        _jit_softmax_module(dtype)
        return True
    except RuntimeError:
        return False


def softmax_sampling(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute softmax with fused temperature scaling for sampling.

    Parameters
    ----------
    logits : torch.Tensor
        Input logits of shape ``(batch_size, vocab_size)``.
        Supported dtypes: float16, bfloat16, float32.
    temperatures : torch.Tensor
        Per-row temperature of shape ``(batch_size,)`` in float32.
        Must be > 0 for all elements.
    out : torch.Tensor, optional
        Pre-allocated out tensor of shape ``(batch_size, vocab_size)``
        with the same dtype as logits. If None, one is allocated.

    Returns
    -------
    torch.Tensor
        Probability distribution of shape ``(batch_size, vocab_size)``
        with the same dtype as logits.
    """
    if out is None:
        out = torch.empty_like(logits)
    module = _jit_softmax_module(logits.dtype)
    module.softmax(logits, out, temperatures, 0.0, 0)
    return out
