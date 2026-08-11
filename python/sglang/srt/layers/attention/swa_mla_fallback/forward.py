from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.swa_mla_fallback.ops import (
    apply_swa_score_mask,
    gather_page64_kv_latent,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention


def forward_dense_kvlora_swa_torch_fallback(
    reshape_q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    layer: RadixAttention,
    kv_cache_dim: int,
    head_dim_v: int,
    window_size: int,
) -> torch.Tensor:
    """Page-64 SWA fallback for dense KV-LoRA decode."""
    if layer.tp_k_head_num != 1:
        raise RuntimeError(
            "SWA MLA torch fallback currently supports MLA with one "
            f"KV head, got tp_k_head_num={layer.tp_k_head_num}."
        )

    bs, s_q, num_heads, qk_dim = reshape_q.shape
    if qk_dim != kv_cache_dim:
        raise RuntimeError(
            f"SWA MLA torch fallback got q dim {qk_dim}, "
            f"expected kv_cache_dim {kv_cache_dim}."
        )
    if s_q not in (1, 4):
        raise RuntimeError(
            "SWA MLA torch fallback mask is specialized for s_q=1 "
            f"or s_q=4, got s_q={s_q}."
        )

    # Include the full union of causal windows and align it for BMM.
    kv_latent, kv_valid = gather_page64_kv_latent(
        k_cache,
        block_table,
        cache_seqlens,
        window_size,
        s_q,
        kv_cache_dim,
    )
    gather_len = kv_latent.shape[1]

    # Keep the output in [bs, s_q, num_heads, head_dim_v] order.
    q_for_scores = reshape_q.reshape(bs, s_q * num_heads, qk_dim)
    scores = torch.bmm(q_for_scores, kv_latent.transpose(1, 2)).view(
        bs, s_q, num_heads, gather_len
    )
    scores = scores.float()
    scores.mul_(layer.scaling)

    apply_swa_score_mask(
        scores.transpose(1, 2),
        cache_seqlens,
        kv_valid,
        num_heads,
        window_size,
        s_q,
    )

    probs = torch.softmax(scores, dim=-1).to(reshape_q.dtype)
    return torch.bmm(
        probs.reshape(bs, s_q * num_heads, gather_len),
        kv_latent[..., :head_dim_v],
    ).view(bs, s_q, num_heads, head_dim_v)
