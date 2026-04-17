"""RMSNorm with HF LlamaRMSNorm semantics (cast to dtype before weight multiply)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_BLOCK_SIZE = 512


def is_supported_rmsnorm_hf_hidden_size(hidden_size: int) -> bool:
    """Return True iff the JIT rmsnorm_hf kernel supports this hidden size."""
    return hidden_size >= _BLOCK_SIZE and hidden_size % _BLOCK_SIZE == 0


@cache_once
def _jit_rmsnorm_hf_module(hidden_size: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(hidden_size, is_arch_support_pdl(), dtype)
    return load_jit(
        "rmsnorm_hf",
        *args,
        cuda_files=["elementwise/rmsnorm_hf.cuh"],
        cuda_wrappers=[("rmsnorm_hf", f"RMSNormHFKernel<{args}>::run")],
    )


def rmsnorm_hf(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """RMSNorm: ``out = weight * cast_dtype(rsqrt(mean(x^2) + eps) * x)``."""
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"rmsnorm_hf: input must be fp16 or bf16, got {input.dtype}")
    hidden_size = input.size(-1)
    if not is_supported_rmsnorm_hf_hidden_size(hidden_size):
        raise RuntimeError(
            f"rmsnorm_hf: unsupported hidden_size={hidden_size} "
            f"(must be a positive multiple of {_BLOCK_SIZE})"
        )
    if out is None:
        out = torch.empty_like(input)
    module = _jit_rmsnorm_hf_module(hidden_size, input.dtype)
    module.rmsnorm_hf(input, weight, out, eps)
    return out
