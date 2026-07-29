"""
Triton-based sparse attention decode kernels for DeepSeek V4.

This package provides an alternative to the tilelang implementation,
controlled by the environment variable SGLANG_HACK_FLASHMLA_BACKEND=triton.
"""

from typing import Optional, Tuple

import torch

from sglang.srt.layers.attention.nsa.triton_decode.triton_mla_kernels_decode_optimized import (
    triton_sparse_attn_decode,
)


class _KVScopeAdapter:
    """Lightweight adapter providing the kv_scope interface expected by
    ``triton_sparse_attn_decode``.

    The Triton kernels access four fields:
      * ``blocked_k_quantized`` – the raw FP8 KV cache tensor.
      * ``blocked_k``          – only ``blocked_k.shape[1]`` (block size)
                                  is read, so we reuse the same tensor.
      * ``indices_in_kvcache`` – sparse top-k page indices.
      * ``topk_length``        – valid length per batch element.
    """

    __slots__ = [
        "blocked_k",
        "blocked_k_quantized",
        "indices_in_kvcache",
        "topk_length",
    ]

    def __init__(
        self,
        k_cache: torch.Tensor,
        indices: torch.Tensor,
        topk_length: Optional[torch.Tensor],
    ):
        self.blocked_k_quantized = k_cache
        self.blocked_k = k_cache
        self.indices_in_kvcache = indices
        self.topk_length = topk_length


def triton_fp8_attention_fwd(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    head_dim_v: int,
    softmax_scale: float,
    indices: torch.Tensor,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
    **_unused,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse MLA decode via Triton kernels.

    Accepts the same ``**kwargs`` dict that the caller builds for
    ``flash_mla_with_kvcache`` / ``dpsk_v4_fp8_attention_fwd``, but only
    uses the subset of arguments relevant to the Triton implementation.
    Unused keys (``block_table``, ``cache_seqlens``,
    ``tile_scheduler_metadata``, ``num_splits``, ``causal``,
    ``is_fp8_kvcache``) are silently ignored via ``**_unused``.

    Returns:
        ``(output, lse)`` where *output* has shape
        ``[batch, seq_len, num_heads, head_dim_v]`` and *lse* has shape
        ``[batch, seq_len, num_heads]``.
    """
    kv_scope = _KVScopeAdapter(k_cache, indices, topk_length)

    extra_kv_scope = None
    if extra_k_cache is not None:
        extra_kv_scope = _KVScopeAdapter(
            extra_k_cache,
            extra_indices_in_kvcache,
            extra_topk_length,
        )

    output, lse = triton_sparse_attn_decode(
        q=q,
        kv_scope=kv_scope,
        extra_kv_scope=extra_kv_scope,
        sm_scale=softmax_scale,
        d_v=head_dim_v,
        attn_sink=attn_sink,
    )

    # Triton kernel returns lse as (b, h_q, s_q); transpose to
    # (b, s_q, h_q) to match the tilelang / flash_mla convention.
    lse = lse.transpose(1, 2)

    return output, lse
