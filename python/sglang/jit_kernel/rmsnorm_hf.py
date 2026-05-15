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

_CTA_BLOCK_SIZE = 512
_WARP_SIZE = 32


def is_supported_rmsnorm_hf_hidden_size(hidden_size: int) -> bool:
    """Return True iff the JIT rmsnorm_hf kernel supports this hidden size.

    Two launch configs cover the practical range:
      - Warp kernel: ``[32, 512)`` in multiples of 32 (q/k RMSNorm head dims).
      - CTA kernel: ``>= 512`` in multiples of 512 (token RMSNorms).
    """
    if _WARP_SIZE <= hidden_size < _CTA_BLOCK_SIZE and hidden_size % _WARP_SIZE == 0:
        return True
    return hidden_size >= _CTA_BLOCK_SIZE and hidden_size % _CTA_BLOCK_SIZE == 0


@cache_once
def _jit_rmsnorm_hf_module(hidden_size: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(hidden_size, is_arch_support_pdl(), dtype)
    kernel_cls = (
        "HFRMSNormWarpKernel" if hidden_size < _CTA_BLOCK_SIZE else "HFRMSNormKernel"
    )
    return load_jit(
        "rmsnorm_hf",
        *args,
        cuda_files=["elementwise/rmsnorm_hf.cuh"],
        cuda_wrappers=[("rmsnorm_hf", f"{kernel_cls}<{args}>::run")],
    )


def rmsnorm_hf(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """RMSNorm: ``out = weight * cast_dtype(rsqrt(mean(x^2) + eps) * x)``.

    ``input`` must be 2D ``(num_tokens, hidden_size)``; callers with
    higher-rank tensors should reshape first. ``hidden_size`` must satisfy
    :func:`is_supported_rmsnorm_hf_hidden_size`. Empty inputs return an empty
    output without launching the kernel.
    """
    if input.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"rmsnorm_hf: input must be fp16 or bf16, got {input.dtype}")
    if input.dim() != 2:
        raise RuntimeError(f"rmsnorm_hf: input must be 2D, got {input.dim()}D")
    hidden_size = input.size(-1)
    if not is_supported_rmsnorm_hf_hidden_size(hidden_size):
        raise RuntimeError(
            f"rmsnorm_hf: unsupported hidden_size={hidden_size} "
            f"(must be a multiple of {_WARP_SIZE} in [{_WARP_SIZE}, {_CTA_BLOCK_SIZE}) "
            f"or a multiple of {_CTA_BLOCK_SIZE})"
        )
    if out is None:
        out = torch.empty_like(input)
    if input.numel() == 0:
        return out
    module = _jit_rmsnorm_hf_module(hidden_size, input.dtype)
    module.rmsnorm_hf(input, weight, out, eps)
    return out
