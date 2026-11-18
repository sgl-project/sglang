# SPDX-License-Identifier: Apache-2.0
"""Step5 sparse attention over indexer-selected 8-token regions (Triton)."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _sparse_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    packed_ptr,
    counts_ptr,
    lengths_ptr,
    out_ptr,
    k_scale_ptr,
    v_scale_ptr,
    q_stride_row,
    q_stride_head,
    kv_stride,
    packed_stride,
    out_stride_row,
    out_stride_head,
    scale,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    REGION: tl.constexpr,
    FP8_KV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    h = tl.arange(0, BLOCK_H)
    d = tl.arange(0, BLOCK_D)
    n = tl.arange(0, BLOCK_N)
    head_mask = h < HEADS
    dim_mask = d < D
    count = tl.load(counts_ptr + row)
    length = tl.load(lengths_ptr + row)

    q = tl.load(
        q_ptr + row * q_stride_row + h[:, None] * q_stride_head + d[None, :],
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    if FP8_KV:
        k_scale = tl.load(k_scale_ptr)
        v_scale = tl.load(v_scale_ptr)

    m = tl.full((BLOCK_H,), float("-inf"), tl.float32)
    l = tl.zeros((BLOCK_H,), tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)
    for start in range(0, count * REGION, BLOCK_N):
        token = start + n
        region = token // REGION
        offset = token % REGION
        selected = (region < count) & (length > 0)
        entry = tl.load(
            packed_ptr + row * packed_stride + region, mask=selected, other=0
        )
        slot = (entry >> 32) * REGION + offset
        logical = entry - ((entry >> 32) << 32)
        valid = selected & (logical + offset < length)
        kv_mask = valid[:, None] & dim_mask[None, :]
        kv_off = slot[:, None] * kv_stride + d[None, :]
        k = tl.load(k_ptr + kv_off, mask=kv_mask, other=0.0)
        v = tl.load(v_ptr + kv_off, mask=kv_mask, other=0.0)
        if FP8_KV:
            k = (k.to(tl.float32) * k_scale).to(tl.bfloat16)
            v = (v.to(tl.float32) * v_scale).to(tl.bfloat16)

        s = tl.dot(q, tl.trans(k)) * scale
        s = tl.where(valid[None, :], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l = l * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new

    out = acc / tl.where(l > 0, l, 1.0)[:, None]
    tl.store(
        out_ptr + row * out_stride_row + h[:, None] * out_stride_head + d[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=head_mask[:, None] & dim_mask[None, :],
    )


def step5_sparse_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    packed: torch.Tensor,
    counts: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Attention of each query row over its selected 8-token regions.

    q/out: [T, heads, 192]; k/v cache: [slots, 1, 192]. packed: int64 [T, >=513]
    with (physical slot // 8) << 32 | first logical position, one entry per region.
    """
    rows, heads, head_dim = q.shape
    assert head_dim == 192 and k_cache.shape[-1] == v_cache.shape[-1] == head_dim
    assert packed.dtype == torch.int64 and packed.shape[0] == rows
    fp8_kv = k_cache.dtype == torch.float8_e4m3fn
    assert not fp8_kv or (k_scale is not None and v_scale is not None)
    if rows:
        _sparse_attention_kernel[(rows,)](
            q,
            k_cache,
            v_cache,
            packed,
            counts,
            lengths,
            out,
            k_scale,
            v_scale,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            packed.stride(0),
            out.stride(0),
            out.stride(1),
            scale,
            HEADS=heads,
            D=head_dim,
            REGION=8,
            FP8_KV=fp8_kv,
            BLOCK_H=max(16, triton.next_power_of_2(heads)),
            BLOCK_D=256,
            BLOCK_N=64,
            num_warps=8,
        )
    return out
