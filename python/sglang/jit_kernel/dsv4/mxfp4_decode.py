"""Fused MXFP4 decode attention for DeepSeek V4 — Python entry point.

Reads packed MXFP4 K-cache rows directly in a JIT CUDA kernel, dequantizing
E2M1 noPE + E8M0 block-32 scales on the fly and computing QK^T + softmax +
V-weighted sum in a single pass (no intermediate dequant workspace).

Usage::

    o = mxfp4_decode_attention(
        q, k_cache, page_indices, sm_scale, page_size,
    )
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

from .utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_mxfp4_decode_module() -> Module:
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("mxfp4_decode"),
        *args,
        cuda_files=["deepseek_v4/mxfp4_decode.cuh"],
        cuda_wrappers=[
            ("forward", f"Mxfp4DecodeKernel<{args}>::forward"),
        ],
    )


def mxfp4_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_indices: torch.Tensor,
    sm_scale: float,
    page_size: int = 128,
) -> torch.Tensor:
    """Fused MXFP4 decode attention.

    Args:
        q:            [num_queries, 512] BF16 — flattened Q (batch * num_heads,
                       where each head may correspond to a different page
                       through ``page_indices``).
        k_cache:      [num_rows, 368] uint8 — row-major MXFP4 K-cache.
        page_indices: [num_queries] int32 — which page row each query reads.
        sm_scale:     float — 1/sqrt(head_dim).
        page_size:    int   — tokens per page (128 for DSV4 SWA).

    Returns:
        o: [num_queries, 512] BF16 attention output.
    """
    if q.ndim != 2 or q.shape[1] != 512:
        raise ValueError(f"q must be [N, 512] BF16, got {q.shape}")
    if k_cache.ndim != 2 or k_cache.shape[1] != 368:
        raise ValueError(f"k_cache must be [*, 368] uint8, got {k_cache.shape}")
    if page_indices.ndim != 1 or page_indices.shape[0] != q.shape[0]:
        raise ValueError(
            f"page_indices must be [{q.shape[0]}], got {page_indices.shape}"
        )

    q = q.contiguous()
    k_cache = k_cache.contiguous()
    page_indices = page_indices.contiguous().to(torch.int32)

    o = torch.empty_like(q)

    module = _jit_mxfp4_decode_module()
    module.forward(
        q,
        k_cache,
        page_indices,
        o,
        sm_scale,
        page_size,
    )
    return o


__all__ = ["mxfp4_decode_attention"]
