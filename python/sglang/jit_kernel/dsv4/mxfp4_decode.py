"""Fused MXFP4 decode attention for DeepSeek V4 — Python entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    load_jit,
    make_cpp_args,
)

from .utils import make_name

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_mxfp4_decode_module() -> Module:
    # NOTE: PDL (Programmatic Dependent Launch) disabled for CUDA graph compat.
    args = make_cpp_args(False)
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
    num_valid: int = 0,
    attn_sink: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused MXFP4 decode attention.

    Args:
        q:            [N, 512] BF16 — flattened Q (batch * num_heads).
        k_cache:      [*, 368] uint8 — row-major MXFP4 K-cache.
        page_indices: [N] int32 — page numbers per query.
        sm_scale:     float — 1/sqrt(head_dim).
        page_size:    int — tokens per page (128 for DSV4 SWA).
        num_valid:    int — actual valid tokens (0 = scan all page_size tokens).
        attn_sink:    [N] float32 or None — per-head sink values.

    Returns:
        o: [N, 512] BF16 attention output.
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
    if attn_sink is not None:
        attn_sink = attn_sink.contiguous().to(torch.float32)

    o = torch.empty_like(q)

    module = _jit_mxfp4_decode_module()
    module.forward(
        q,
        k_cache,
        page_indices,
        attn_sink,
        o,
        sm_scale,
        page_size,
        num_valid,
    )
    return o


__all__ = ["mxfp4_decode_attention"]
