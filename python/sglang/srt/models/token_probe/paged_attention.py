"""Paged causal attention kernel for the token-probe head.

One flash-style triton kernel serves decode (one query row per request) and
extend (each request's chunk rows). K/V are read straight from the probe
pools through ``req_to_token`` (page size 1) with no host-side plan, so
launches capture into cuda graphs as-is.

Decode launches only ``batch * num_heads`` blocks, which leaves nearly all
SMs idle while each block walks the whole prefix serially. Past a threshold
the sequence is therefore split across blocks (flash-decoding): each block
reduces its own slice, then a combine pass merges the partial softmax
statistics. The split path is numerically equivalent to the single-pass one.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import triton
import triton.language as tl

MAX_SPLITS = 16
_target_blocks = None


@triton.jit
def _attend(
    Q,
    KV,
    ReqToToken,
    Positions,
    offs_m,
    m_mask,
    req,
    lo,
    hi,
    q_stride,
    kv_stride,
    table_stride,
    sm_scale,
    V_OFF: tl.constexpr,
    WINDOW: tl.constexpr,
    HD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    q_head_off,
):
    """Online-softmax scan of keys [lo, hi) -> (running max, denom, acc).

    Every query head reads the one K/V head, so only the query side is
    offset."""
    offs_d = tl.arange(0, HD)
    pos = tl.load(Positions + offs_m, mask=m_mask, other=0).to(tl.int64)
    q = tl.load(
        Q + offs_m[:, None] * q_stride + q_head_off + offs_d[None, :],
        mask=m_mask[:, None],
        other=0.0,
    )
    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, HD], tl.float32)
    for start in range(lo, hi, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < hi
        slots = tl.load(
            ReqToToken + req * table_stride + offs_n, mask=n_mask, other=0
        ).to(tl.int64)
        row = KV + slots[:, None] * kv_stride + offs_d[None, :]
        k = tl.load(row, mask=n_mask[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
        visible = (offs_n[None, :] <= pos[:, None]) & n_mask[None, :] & m_mask[:, None]
        if WINDOW > 0:
            visible &= offs_n[None, :] > pos[:, None] - WINDOW
        s = tl.where(visible, s, float("-inf"))
        # m_safe keeps rows with no visible key yet at weight zero instead of
        # poisoning the accumulator with exp(-inf - -inf).
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp(m_i - m_safe)
        p = tl.exp(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(row + V_OFF, mask=n_mask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_new
    return m_i, l_i, acc


@triton.jit
def _bounds(Positions, offs_m, m_mask, WINDOW: tl.constexpr):
    pos = tl.load(Positions + offs_m, mask=m_mask, other=0).to(tl.int64)
    hi = tl.max(pos) + 1
    lo = tl.zeros([], tl.int64)
    if WINDOW > 0:
        min_pos = tl.min(tl.where(m_mask, pos, 1 << 62))
        lo = tl.maximum(min_pos + 1 - WINDOW, 0)
    return lo, hi


@triton.jit
def _probe_attn_kernel(
    Q,
    KV,
    Out,
    ReqToToken,
    ReqIndices,
    Positions,
    QoIndptr,
    q_stride,
    kv_stride,
    out_stride,
    table_stride,
    sm_scale,
    V_OFF: tl.constexpr,
    WINDOW: tl.constexpr,
    HD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    mb = tl.program_id(2)
    q_start = tl.load(QoIndptr + b).to(tl.int64)
    q_end = tl.load(QoIndptr + b + 1).to(tl.int64)
    base = q_start + mb * BLOCK_M
    if base >= q_end:
        return

    offs_m = base + tl.arange(0, BLOCK_M)
    m_mask = offs_m < q_end
    req = tl.load(ReqIndices + b).to(tl.int64)
    q_head_off = h.to(tl.int64) * HD
    lo, hi = _bounds(Positions, offs_m, m_mask, WINDOW)
    _, l_i, acc = _attend(
        Q,
        KV,
        ReqToToken,
        Positions,
        offs_m,
        m_mask,
        req,
        lo,
        hi,
        q_stride,
        kv_stride,
        table_stride,
        sm_scale,
        V_OFF,
        WINDOW,
        HD,
        BLOCK_M,
        BLOCK_N,
        q_head_off,
    )
    out = acc / l_i[:, None]
    offs_d = tl.arange(0, HD)
    tl.store(
        Out + offs_m[:, None] * out_stride + q_head_off + offs_d[None, :],
        out.to(Out.dtype.element_ty),
        mask=m_mask[:, None],
    )


@triton.jit
def _probe_attn_split_kernel(
    Q,
    KV,
    Mx,
    Lx,
    Acc,
    ReqToToken,
    ReqIndices,
    Positions,
    QoIndptr,
    q_stride,
    kv_stride,
    table_stride,
    sm_scale,
    stat_stride_q,
    stat_stride_h,
    acc_stride_q,
    acc_stride_h,
    acc_stride_s,
    NUM_SPLITS: tl.constexpr,
    V_OFF: tl.constexpr,
    WINDOW: tl.constexpr,
    HD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid0 = tl.program_id(0)
    b = pid0 // NUM_SPLITS
    sp = pid0 % NUM_SPLITS
    h = tl.program_id(1)
    mb = tl.program_id(2)
    q_start = tl.load(QoIndptr + b).to(tl.int64)
    q_end = tl.load(QoIndptr + b + 1).to(tl.int64)
    base = q_start + mb * BLOCK_M
    if base >= q_end:
        return

    offs_m = base + tl.arange(0, BLOCK_M)
    m_mask = offs_m < q_end
    req = tl.load(ReqIndices + b).to(tl.int64)
    q_head_off = h.to(tl.int64) * HD
    lo, hi = _bounds(Positions, offs_m, m_mask, WINDOW)

    # Every split of a given (request, query block) derives the same bounds,
    # so the slices tile [lo, hi) exactly. An empty slice keeps the identity
    # statistics (-inf, 0, 0), which contribute nothing in the combine.
    per = tl.cdiv(hi - lo, NUM_SPLITS)
    s_lo = lo + sp * per
    s_hi = tl.minimum(s_lo + per, hi)

    m_i, l_i, acc = _attend(
        Q,
        KV,
        ReqToToken,
        Positions,
        offs_m,
        m_mask,
        req,
        s_lo,
        s_hi,
        q_stride,
        kv_stride,
        table_stride,
        sm_scale,
        V_OFF,
        WINDOW,
        HD,
        BLOCK_M,
        BLOCK_N,
        q_head_off,
    )
    stat_off = offs_m * stat_stride_q + h * stat_stride_h + sp
    tl.store(Mx + stat_off, m_i, mask=m_mask)
    tl.store(Lx + stat_off, l_i, mask=m_mask)
    offs_d = tl.arange(0, HD)
    tl.store(
        Acc
        + offs_m[:, None] * acc_stride_q
        + h * acc_stride_h
        + sp * acc_stride_s
        + offs_d[None, :],
        acc,
        mask=m_mask[:, None],
    )


@triton.jit
def _probe_attn_combine_kernel(
    Mx,
    Lx,
    Acc,
    Out,
    stat_stride_q,
    stat_stride_h,
    acc_stride_q,
    acc_stride_h,
    acc_stride_s,
    out_stride,
    NUM_SPLITS: tl.constexpr,
    HD: tl.constexpr,
):
    qi = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, NUM_SPLITS)
    offs_d = tl.arange(0, HD)
    stat_off = qi * stat_stride_q + h * stat_stride_h + offs_s
    m = tl.load(Mx + stat_off)
    denom = tl.load(Lx + stat_off)
    # Every row sees at least its own key, so the max over splits is finite.
    m_max = tl.max(m, 0)
    scale = tl.exp(m - m_max)
    acc = tl.load(
        Acc
        + qi * acc_stride_q
        + h * acc_stride_h
        + offs_s[:, None] * acc_stride_s
        + offs_d[None, :]
    )
    out = tl.sum(acc * scale[:, None], 0) / tl.sum(denom * scale, 0)
    tl.store(
        Out + qi * out_stride + h.to(tl.int64) * HD + offs_d,
        out.to(Out.dtype.element_ty),
    )


def _pick_splits(num_blocks: int, device: torch.device, max_kv: int) -> int:
    """Power-of-two split count for ``num_blocks`` query-side blocks over a
    key range of at most ``max_kv`` (tl.arange over the split axis needs a
    power of two).

    Two limits, because occupancy alone gets both ends wrong. Filling the
    device once leaves each block walking a long range serially -- on an A100
    at batch 32 that picked 2 splits where 4 ran 1.6x faster -- so aim at
    several waves instead. But a split shorter than one BLOCK_N tile buys no
    parallelism and still costs its share of the combine, which is what a
    sliding window produces: at window 256 the range is four tiles, and
    splitting it 16 ways ran 2x slower than splitting it 4 ways.
    """
    global _target_blocks
    if _target_blocks is None:
        _target_blocks = (
            4 * torch.cuda.get_device_properties(device).multi_processor_count
        )
    tiles = max(1, max_kv // 64)
    cap = min(MAX_SPLITS, 1 << (tiles.bit_length() - 1))
    splits = 1
    while num_blocks * splits < _target_blocks and splits < cap:
        splits *= 2
    return splits


def probe_paged_attention(
    *,
    q: torch.Tensor,
    kv_pool: torch.Tensor,
    req_to_token: torch.Tensor,
    req_indices: torch.Tensor,
    positions: torch.Tensor,
    qo_indptr: torch.Tensor,
    max_qo_len: int,
    num_heads: int,
    head_dim: int,
    window: Optional[int],
    force_splits: Optional[int] = None,
) -> torch.Tensor:
    """Causal attention of ``q`` rows ([total_q, num_heads * head_dim],
    grouped per request by ``qo_indptr``) over their requests' pooled K/V.
    All query heads share the single K/V head, so ``kv_pool`` rows are just K
    then V, head_dim wide each. Each row's own K/V -- like its whole prefix --
    must already be published at ``req_to_token[req_indices[b]]`` slots.
    """
    total_q = q.shape[0]
    out = torch.empty(total_q, num_heads * head_dim, device=q.device, dtype=q.dtype)
    bs = req_indices.shape[0]
    if total_q == 0 or bs == 0:
        return out
    block_m = 16 if max_qo_len <= 16 else 64
    m_blocks = triton.cdiv(max_qo_len, block_m)
    # Shape-only choice, so a cuda graph captures the same launch it replays:
    # the window is static config and req_to_token's width is the pool's, so
    # neither depends on this batch's actual sequence lengths.
    # force_splits is for warmup, which has to reach every compiled variant.
    splits = force_splits or _pick_splits(
        bs * num_heads * m_blocks,
        q.device,
        window or req_to_token.shape[1],
    )

    if splits == 1:
        _probe_attn_kernel[(bs, num_heads, m_blocks)](
            q,
            kv_pool,
            out,
            req_to_token,
            req_indices,
            positions,
            qo_indptr,
            q.stride(0),
            kv_pool.stride(0),
            out.stride(0),
            req_to_token.stride(0),
            1.0 / math.sqrt(head_dim),
            V_OFF=head_dim,
            WINDOW=window or 0,
            HD=head_dim,
            BLOCK_M=block_m,
            BLOCK_N=64,
            num_warps=4,
            num_stages=2,
        )
        return out

    mx = torch.empty(total_q, num_heads, splits, device=q.device, dtype=torch.float32)
    lx = torch.empty_like(mx)
    acc = torch.empty(
        total_q, num_heads, splits, head_dim, device=q.device, dtype=torch.float32
    )
    _probe_attn_split_kernel[(bs * splits, num_heads, m_blocks)](
        q,
        kv_pool,
        mx,
        lx,
        acc,
        req_to_token,
        req_indices,
        positions,
        qo_indptr,
        q.stride(0),
        kv_pool.stride(0),
        req_to_token.stride(0),
        1.0 / math.sqrt(head_dim),
        mx.stride(0),
        mx.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        NUM_SPLITS=splits,
        V_OFF=head_dim,
        WINDOW=window or 0,
        HD=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=64,
        num_warps=4,
        num_stages=2,
    )
    _probe_attn_combine_kernel[(total_q, num_heads)](
        mx,
        lx,
        acc,
        out,
        mx.stride(0),
        mx.stride(1),
        acc.stride(0),
        acc.stride(1),
        acc.stride(2),
        out.stride(0),
        NUM_SPLITS=splits,
        HD=head_dim,
        num_warps=4,
    )
    return out
