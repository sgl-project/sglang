from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _validate_grouped_gemma_rmsnorm_args(group_size: int, dtype: torch.dtype) -> None:
    if dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"Unsupported dtype {dtype}. Supported: bfloat16, float16")
    if group_size <= 0 or group_size % 512 != 0:
        raise RuntimeError(
            f"Unsupported group_size {group_size}. Must be a multiple of 512."
        )


@cache_once
def _jit_grouped_gemma_rmsnorm_module(group_size: int, dtype: torch.dtype) -> Module:
    """Compile and cache the JIT grouped Gemma RMSNorm module."""
    # Validation sits under `cache_once`: once per (group_size, dtype) specialisation,
    # not once per call.
    _validate_grouped_gemma_rmsnorm_args(group_size, dtype)
    args = make_cpp_args(group_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "grouped_gemma_rmsnorm",
        *args,
        cuda_files=["elementwise/grouped_gemma_rmsnorm.cuh"],
        cuda_wrappers=[
            ("grouped_gemma_rmsnorm", f"GroupedGemmaRMSNormKernel<{args}>::run")
        ],
    )


def grouped_gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Grouped Gemma-style RMSNorm: out = x * rsqrt(mean(x^2) + eps) * (1 + weight).

    The last dimension is split into groups of `group_size` elements; variance
    is computed per group. With group_size == input.size(-1) this reduces to a
    plain Gemma RMSNorm.

    Supported dtypes: torch.bfloat16, torch.float16.

    Parameters
    ----------
    input      : CUDA tensor [..., hidden_size], hidden_size % group_size == 0
    weight     : CUDA tensor [hidden_size]
    group_size : elements per variance group (multiple of 512)
    eps        : RMSNorm epsilon
    out        : optional pre-allocated output tensor (same shape/dtype as input)

    Returns
    -------
    Normalized tensor, same shape/dtype as input.
    """
    hidden_size = input.size(-1)
    if hidden_size % group_size != 0:
        raise ValueError(
            f"hidden_size ({hidden_size}) must be divisible by group_size ({group_size})"
        )
    if out is not None and out.shape != input.shape:
        raise ValueError(f"out.shape {out.shape} must match input.shape {input.shape}")
    if input.numel() == 0:
        _validate_grouped_gemma_rmsnorm_args(group_size, input.dtype)
        return torch.empty_like(input) if out is None else out

    x = input.reshape(-1, hidden_size)
    if out is None:
        out = torch.empty_like(x)
    else:
        out = out.reshape(-1, hidden_size)

    module = _jit_grouped_gemma_rmsnorm_module(group_size, input.dtype)
    module.grouped_gemma_rmsnorm(x, weight, out, eps)
    return out.reshape(input.shape)
