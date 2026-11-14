"""Fused attention over a short sparse suffix and merge with a prefix state."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def can_use_fused_suffix_attention_merge(
    *,
    layer,
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    extra_kwargs: dict,
) -> bool:
    """Whether attention can use the specialized suffix merge."""
    return bool(
        q.dtype in (torch.float16, torch.bfloat16)
        and key_cache.dtype == q.dtype
        and value_cache.dtype == q.dtype
        and layer.head_dim == layer.v_head_dim
        and not layer.is_cross_attention
        and not layer.logit_cap
        and not extra_kwargs
    )


@triton.jit
def _fused_suffix_attention_merge_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    suffix_seqlens_ptr,
    prefix_ptr,
    prefix_lse_ptr,
    scale,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_t,
    k_stride_h,
    k_stride_d,
    v_stride_t,
    v_stride_h,
    v_stride_d,
    page_stride_t,
    page_stride_s,
    prefix_stride_t,
    prefix_stride_h,
    prefix_stride_d,
    prefix_lse_stride_h,
    prefix_lse_stride_t,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SUFFIX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // (NUM_Q_HEADS // NUM_KV_HEADS)

    suffix_offsets = tl.arange(0, BLOCK_SUFFIX)
    suffix_length = tl.load(suffix_seqlens_ptr + token)
    suffix_valid = suffix_offsets < suffix_length
    slots = tl.load(
        page_table_ptr + token * page_stride_t + suffix_offsets * page_stride_s,
        mask=suffix_valid,
        other=0,
    ).to(tl.int64)

    dims = tl.arange(0, BLOCK_D)
    dim_valid = dims < HEAD_DIM
    q = tl.load(
        q_ptr + token * q_stride_t + q_head * q_stride_h + dims * q_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(
        k_cache_ptr
        + slots[:, None] * k_stride_t
        + kv_head * k_stride_h
        + dims[None, :] * k_stride_d,
        mask=suffix_valid[:, None] & dim_valid[None, :],
        other=0.0,
    ).to(tl.float32)
    scores = tl.sum(k * q[None, :], axis=1) * scale
    scores = tl.where(suffix_valid, scores, -float("inf"))
    suffix_max = tl.max(scores, axis=0)

    prefix_lse = tl.load(
        prefix_lse_ptr + q_head * prefix_lse_stride_h + token * prefix_lse_stride_t
    ).to(tl.float32)
    global_max = tl.maximum(prefix_lse, suffix_max)
    prefix_weight = tl.exp(prefix_lse - global_max)
    suffix_weights = tl.exp(scores - global_max)
    denominator = prefix_weight + tl.sum(suffix_weights, axis=0)

    v = tl.load(
        v_cache_ptr
        + slots[:, None] * v_stride_t
        + kv_head * v_stride_h
        + dims[None, :] * v_stride_d,
        mask=suffix_valid[:, None] & dim_valid[None, :],
        other=0.0,
    ).to(tl.float32)
    suffix_numerator = tl.sum(suffix_weights[:, None] * v, axis=0)
    prefix = tl.load(
        prefix_ptr
        + token * prefix_stride_t
        + q_head * prefix_stride_h
        + dims * prefix_stride_d,
        mask=dim_valid,
        other=0.0,
    ).to(tl.float32)
    output = (prefix * prefix_weight + suffix_numerator) / denominator
    tl.store(
        prefix_ptr
        + token * prefix_stride_t
        + q_head * prefix_stride_h
        + dims * prefix_stride_d,
        output,
        mask=dim_valid,
    )


def merge_suffix_attention_in_place(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    suffix_page_table: torch.Tensor,
    suffix_cache_seqlens: torch.Tensor,
    prefix: torch.Tensor,
    prefix_lse: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Compute a short sparse suffix and merge it into ``prefix`` in place.

    ``prefix_lse`` uses FlashAttention's varlen layout ``[num_q_heads,
    num_queries]``. The suffix page table contains physical token slots, one
    row per query, and only its first ``suffix_cache_seqlens[row]`` entries are
    visible.
    """
    if q.ndim != 3:
        raise ValueError("q must have shape [num_queries, num_q_heads, head_dim]")
    num_queries, num_q_heads, head_dim = q.shape
    if prefix.shape != q.shape:
        raise ValueError("prefix output must have the same shape as q")
    if prefix_lse.shape != (num_q_heads, num_queries):
        raise ValueError("prefix_lse must have shape [num_q_heads, num_queries]")
    if k_cache.ndim != 3 or v_cache.ndim != 3:
        raise ValueError("flattened KV caches must have shape [slots, heads, dim]")
    if k_cache.shape != v_cache.shape:
        raise ValueError("K and V caches must have matching shapes")
    num_kv_heads = k_cache.shape[1]
    if k_cache.shape[2] != head_dim:
        raise ValueError("K/V and query head dimensions must match")
    if num_q_heads % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if suffix_page_table.ndim != 2 or suffix_page_table.shape[0] != num_queries:
        raise ValueError("suffix page table must have one row per query")
    if suffix_cache_seqlens.numel() != num_queries:
        raise ValueError("suffix cache lengths must have one value per query")
    if suffix_page_table.shape[1] == 0 or num_queries == 0:
        return prefix

    _fused_suffix_attention_merge_kernel[(num_queries, num_q_heads)](
        q,
        k_cache,
        v_cache,
        suffix_page_table,
        suffix_cache_seqlens,
        prefix,
        prefix_lse,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        suffix_page_table.stride(0),
        suffix_page_table.stride(1),
        prefix.stride(0),
        prefix.stride(1),
        prefix.stride(2),
        prefix_lse.stride(0),
        prefix_lse.stride(1),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SUFFIX=triton.next_power_of_2(suffix_page_table.shape[1]),
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
        num_stages=1,
    )
    return prefix
