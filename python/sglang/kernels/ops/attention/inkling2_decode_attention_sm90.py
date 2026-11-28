"""Paged decode attention with a compact relative bias for Hopper and SM120: the
split-KV partials of paged_rel_bias_decode (unnormalized o, max, sum), reduced by
combine_gate_sink, for one query per sequence or a causal block of queries per
sequence (MTP verify). Each split takes an equal share of the sequence's own keys."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.arch import current_arch
from sglang.kernels.ops.kvcache.mxfp8_dequant_paged import (
    MXFP8_PAGE,
    mxfp8_dequant_rows,
    mxfp8_scale_words,
)

MAX_SPLITS = 64


def _sm12x_swa_launch_config(
    *,
    batch: int,
    heads_k: int,
    qlen: int,
    rows: int,
    head_dim: int,
    page_size: int,
    span: int,
    sms: int,
    max_splits: int,
) -> tuple[int, int, int] | None:
    if (
        not current_arch().is_sm(12)
        or batch <= 0
        or qlen <= 0
        or heads_k <= 0
        or page_size <= 0
        or sms <= 0
        or rows > 32
        or head_dim > 128
    ):
        return None
    ctas = batch * heads_k
    candidates = (64, 128) if qlen == 1 else (128, 64)
    for block_n in candidates:
        if page_size % block_n != 0 or span < 2 * block_n:
            continue
        splits = triton.cdiv(span + block_n - 1, block_n)
        if splits <= max_splits and ctas * splits <= sms:
            return block_n, splits, 2
    return None


@triton.jit
def _decode_partials(
    Q,
    K,
    V,
    KScale,
    VScale,
    PageTable,
    SeqLens,
    Bias,
    OPartial,
    MPartial,
    LPartial,
    scale,
    batch,
    SQ0: tl.constexpr,
    SQ1: tl.constexpr,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    SK2: tl.constexpr,
    SS0: tl.constexpr,
    SS1: tl.constexpr,
    SP: tl.constexpr,
    SB0: tl.constexpr,
    SB1: tl.constexpr,
    HEADS_K: tl.constexpr,
    QLEN: tl.constexpr,
    GROUP: tl.constexpr,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    EXTENT: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MXFP8: tl.constexpr,
):
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    b = tl.program_id(0) // HEADS_K
    hk = tl.program_id(0) % HEADS_K
    split = tl.program_id(1)
    splits = tl.num_programs(1)
    seqlen = tl.load(SeqLens + b)
    first = 0
    if WINDOW >= 0:
        first = tl.maximum(seqlen - QLEN - WINDOW, 0)
    tiles = tl.cdiv(seqlen - first // BLOCK_N * BLOCK_N, BLOCK_N)
    per_split = tl.cdiv(tiles, splits)
    tile_lo = first // BLOCK_N + split * per_split
    tile_hi = tl.minimum(tile_lo + per_split, first // BLOCK_N + tiles)

    rows = tl.arange(0, ROWS)
    head_mask = rows < QLEN * GROUP
    token = b * QLEN + rows // GROUP
    qh = hk * GROUP + rows % GROUP
    q_pos = seqlen - QLEN + rows // GROUP
    d = tl.arange(0, D)
    q = tl.load(
        Q + token[:, None] * SQ0 + qh[:, None] * SQ1 + d[None, :],
        head_mask[:, None],
        other=0,
    )
    m = tl.full((ROWS,), float("-inf"), tl.float32)
    l = tl.zeros((ROWS,), tl.float32)
    acc = tl.zeros((ROWS, D), tl.float32)
    cols = tl.arange(0, BLOCK_N)
    for tile in range(tile_lo, tile_hi):
        pos = tile * BLOCK_N + cols
        valid = (pos < seqlen) & (pos >= first)
        if PAGE % BLOCK_N == 0:
            page = tl.load(PageTable + b * SP + (tile * BLOCK_N) // PAGE).to(tl.int64)
            slot = (tile * BLOCK_N) % PAGE + cols
            kv_offset = page * SK0 + slot[:, None] * SK1 + hk * SK2 + d[None, :]
        else:
            page = tl.load(PageTable + b * SP + pos // PAGE, valid, other=0)
            slot = pos % PAGE
            kv_offset = (
                page.to(tl.int64)[:, None] * SK0
                + slot[:, None] * SK1
                + hk * SK2
                + d[None, :]
            )
        k = tl.load(K + kv_offset, valid[:, None], other=0.0)
        if MXFP8:
            # PAGE is a multiple of BLOCK_N: page is the tile's one page.
            scale_offset = page * SS0 + hk * SS1
            k = mxfp8_dequant_rows(k, KScale + scale_offset, slot, valid, D)
        s = tl.dot(q, tl.trans(k)) * scale
        dist = q_pos[:, None] - pos[None, :]
        keep = valid[None, :] & (dist >= 0)
        if WINDOW >= 0:
            keep &= dist <= WINDOW
        bias = tl.load(
            Bias + token[:, None] * SB0 + qh[:, None] * SB1 + dist,
            head_mask[:, None] & keep & (dist < EXTENT),
            other=0,
        ).to(tl.float32)
        s = tl.where(keep, s + bias, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        m_ref = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2((m - m_ref) * 1.4426950408889634)
        p = tl.exp2((s - m_ref[:, None]) * 1.4426950408889634)
        l = l * alpha + tl.sum(p, 1)
        v = tl.load(V + kv_offset, valid[:, None], other=0.0)
        if MXFP8:
            v = mxfp8_dequant_rows(v, VScale + scale_offset, slot, valid, D)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m = m_new
    stat = (split * batch * QLEN + token) * (HEADS_K * GROUP) + qh
    tl.store(MPartial + stat, m, head_mask)
    tl.store(LPartial + stat, l, head_mask)
    tl.store(OPartial + stat[:, None] * D + d[None, :], acc, head_mask[:, None])


def decode_partials(
    q,
    k_cache,
    v_cache,
    page_table,
    seqused_k,
    rel_bias,
    softmax_scale,
    *,
    window_size_left=None,
    k_scale=None,
    v_scale=None,
):
    """q (tokens, heads_q, d) with tokens = batch * qlen, the last qlen positions
    of each sequence, attending causally; k/v cache (pages, page_size, heads_k, d);
    rel_bias (tokens, heads_q, rel_extent), indexed by query-key distance. Keys
    beyond the extent get bias 0, or are masked with a window. With k/v_scale, the
    cache is MXFP8: e4m3 data and the pool's interleaved UE8M0 scales
    (pages, heads_k, 32, 4, d // 32), 128-token pages. Returns partials shaped
    (splits, tokens, 1, heads_q[, d])."""
    tokens, heads_q, head_dim = q.shape
    batch = seqused_k.shape[0]
    qlen = tokens // max(batch, 1)
    assert qlen * batch == tokens
    _, page_size, heads_k, _ = k_cache.shape
    group = heads_q // heads_k
    span = (
        window_size_left + qlen
        if window_size_left is not None
        else page_table.shape[1] * page_size
    )
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    max_splits = max(1, min(2 * sms // max(1, batch * heads_k), MAX_SPLITS))
    rows = max(16, triton.next_power_of_2(qlen * group))
    swa_cfg = (
        _sm12x_swa_launch_config(
            batch=batch,
            heads_k=heads_k,
            qlen=qlen,
            rows=rows,
            head_dim=head_dim,
            page_size=page_size,
            span=span,
            sms=sms,
            max_splits=max_splits,
        )
        if window_size_left is not None and window_size_left >= 0
        else None
    )
    if swa_cfg is not None:
        block_n, splits, num_stages = swa_cfg
    else:
        block_n = 128 if qlen == 1 and span >= 128 * max_splits else 32
        splits = min(triton.cdiv(span, block_n), max_splits)
        num_stages = 2 if qlen == 1 else 3
    mxfp8 = k_scale is not None
    if mxfp8:
        assert v_scale is not None and page_size == MXFP8_PAGE
        assert page_size % block_n == 0
        assert k_cache.dtype == v_cache.dtype == torch.float8_e4m3fn
        assert q.dtype == torch.bfloat16
        k_words, v_words = (
            mxfp8_scale_words(k_scale, head_dim),
            mxfp8_scale_words(v_scale, head_dim),
        )
        assert k_words.stride() == v_words.stride()
    else:
        k_words = v_words = k_cache
    o = torch.empty(
        (splits, tokens, 1, heads_q, head_dim), dtype=torch.float32, device=q.device
    )
    m = torch.empty((splits, tokens, 1, heads_q), dtype=torch.float32, device=q.device)
    l = torch.empty_like(m)
    _decode_partials[(batch * heads_k, splits)](
        q,
        k_cache,
        v_cache,
        k_words,
        v_words,
        page_table,
        seqused_k,
        rel_bias,
        o,
        m,
        l,
        softmax_scale,
        batch,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_words.stride(0) if mxfp8 else 0,
        k_words.stride(1) if mxfp8 else 0,
        page_table.stride(0),
        rel_bias.stride(0),
        rel_bias.stride(1),
        heads_k,
        qlen,
        group,
        rows,
        head_dim,
        page_size,
        rel_bias.shape[-1],
        -1 if window_size_left is None else window_size_left,
        block_n,
        mxfp8,
        num_warps=4,
        num_stages=num_stages,
        launch_pdl=True,
    )
    return o, m, l
