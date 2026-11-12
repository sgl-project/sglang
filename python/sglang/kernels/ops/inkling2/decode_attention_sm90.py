"""Paged decode attention with a compact relative bias for Hopper: the split-KV
partials of paged_rel_bias_decode (unnormalized o, max, sum), reduced by
combine_gate_sink, for one query per sequence or a causal block of queries per
sequence (MTP verify). Each split takes an equal share of the sequence's own keys."""

import torch
import triton
import triton.language as tl

MAX_SPLITS = 64


@triton.jit
def _decode_partials(
    Q,
    K,
    V,
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
        page = tl.load(PageTable + b * SP + (tile * BLOCK_N) // PAGE).to(tl.int64)
        slot = (tile * BLOCK_N) % PAGE + cols
        kv_offset = page * SK0 + slot[:, None] * SK1 + hk * SK2 + d[None, :]
        valid = (pos < seqlen) & (pos >= first)
        k = tl.load(K + kv_offset, valid[:, None], other=0)
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
        v = tl.load(V + kv_offset, valid[:, None], other=0)
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
):
    """q (tokens, heads_q, d) with tokens = batch * qlen, the last qlen positions
    of each sequence, attending causally; k/v cache (pages, page_size, heads_k, d);
    rel_bias (tokens, heads_q, rel_extent), indexed by query-key distance. Keys
    beyond the extent get bias 0, or are masked with a window. Returns partials
    shaped (splits, tokens, 1, heads_q[, d])."""
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
    block_n = 128 if qlen == 1 and span >= 128 * max_splits else 32
    splits = min(triton.cdiv(span, block_n), max_splits)
    o = torch.empty(
        (splits, tokens, 1, heads_q, head_dim), dtype=torch.float32, device=q.device
    )
    m = torch.empty((splits, tokens, 1, heads_q), dtype=torch.float32, device=q.device)
    l = torch.empty_like(m)
    _decode_partials[(batch * heads_k, splits)](
        q,
        k_cache,
        v_cache,
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
        page_table.stride(0),
        rel_bias.stride(0),
        rel_bias.stride(1),
        heads_k,
        qlen,
        group,
        max(16, triton.next_power_of_2(qlen * group)),
        head_dim,
        page_size,
        rel_bias.shape[-1],
        -1 if window_size_left is None else window_size_left,
        block_n,
        num_warps=4,
        num_stages=2 if qlen == 1 else 3,
        launch_pdl=True,
    )
    return o, m, l


def fa4_decode_partials(
    q,
    k_cache,
    v_cache,
    page_table,
    seqused_k,
    rel_bias,
    softmax_scale,
    *,
    window_size_left=None,
):
    """decode_partials on FA4's SM90 kernel with the compact relative bias: returns
    (o, lse, None), o normalized per split (splits, tokens, heads, d) and lse a
    (splits, batch, qlen, heads) view, for combine_gate_sink."""
    from sglang.kernels.ops.attention.flash_attn.cute.interface import _flash_attn_fwd

    tokens, heads_q, head_dim = q.shape
    batch = seqused_k.shape[0]
    qlen = tokens // max(batch, 1)
    assert qlen * batch == tokens
    heads_k = k_cache.shape[2]
    wide = window_size_left is None and batch * heads_k >= 64
    tile_n = 128 if wide else 64
    span = (
        window_size_left + qlen
        if window_size_left is not None
        else page_table.shape[1] * k_cache.shape[1]
    )
    splits = max(
        1, min(MAX_SPLITS, 128 // max(1, batch * heads_k), triton.cdiv(span, tile_n))
    )
    o, lse = _flash_attn_fwd(
        q.view(batch, qlen, heads_q, head_dim),
        k_cache,
        v_cache,
        seqused_k=seqused_k,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=True,
        window_size_left=window_size_left,
        window_size_right=0 if window_size_left is not None else None,
        tile_mn=(64, tile_n),
        num_splits=splits,
        pack_gqa=True,
        intra_wg_overlap=wide,
        rel_bias=rel_bias,
        rel_bias_compact=True,
        return_partials=True,
    )
    return o.view(splits, tokens, heads_q, head_dim), lse.permute(0, 1, 3, 2), None
