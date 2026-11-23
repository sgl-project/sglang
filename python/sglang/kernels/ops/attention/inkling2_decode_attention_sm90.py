"""Paged decode attention with a compact relative bias for Hopper: the split-KV
partials of paged_rel_bias_decode (unnormalized o, max, sum), reduced by
combine_gate_sink, for one query per sequence or a causal block of queries per
sequence (MTP verify). Each split takes an equal share of the sequence's own keys;
on a global layer's verify of up to 64 padded rows, an equal share of their cost, where
the keys at least the bias extent behind every query run as larger unmasked tiles."""

import torch
import triton
import triton.language as tl

from sglang.kernels.arch import current_arch

MAX_SPLITS = 64


@triton.jit
def _interior_tile(
    q,
    m,
    l,
    acc,
    start,
    b,
    hk,
    d,
    scale,
    K,
    V,
    PageTable,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    SK2: tl.constexpr,
    SP: tl.constexpr,
    PAGE: tl.constexpr,
    TILE: tl.constexpr,
):
    # Every key is valid and at least EXTENT behind every query row: bias 0, no mask.
    cols = tl.arange(0, TILE)
    if PAGE % TILE == 0:
        page = tl.load(PageTable + b * SP + start // PAGE).to(tl.int64)
        slot = start % PAGE + cols
        kv_offset = page * SK0 + slot[:, None] * SK1 + hk * SK2 + d[None, :]
    else:
        pos = start + cols
        page = tl.load(PageTable + b * SP + pos // PAGE)
        kv_offset = (
            page.to(tl.int64)[:, None] * SK0
            + (pos % PAGE)[:, None] * SK1
            + hk * SK2
            + d[None, :]
        )
    k = tl.load(K + kv_offset)
    s = tl.dot(q, tl.trans(k)) * scale
    # Real keys, no mask: s is finite (padded rows have q = 0), so m_new is never -inf.
    m_new = tl.maximum(m, tl.max(s, 1))
    alpha = tl.exp2((m - m_new) * 1.4426950408889634)
    p = tl.exp2((s - m_new[:, None]) * 1.4426950408889634)
    l = l * alpha + tl.sum(p, 1)
    v = tl.load(V + kv_offset)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    return m_new, l, acc


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
    INTERIOR: tl.constexpr,
    INTERIOR_COST: tl.constexpr,
    MASKED_COST: tl.constexpr,
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
    if INTERIOR == 0:
        tiles = tl.cdiv(seqlen - first // BLOCK_N * BLOCK_N, BLOCK_N)
        per_split = tl.cdiv(tiles, splits)
        tile_lo = first // BLOCK_N + split * per_split
        tile_hi = tl.minimum(tile_lo + per_split, first // BLOCK_N + tiles)
    else:
        # No window: keys [0, inner) are at least EXTENT behind every query row and
        # run as INTERIOR-key tiles, the rest as BLOCK_N-key masked tiles. A split
        # takes an equal share of the cost, a tile costing INTERIOR_COST or
        # MASKED_COST; it runs the tiles whose cost starts inside its share.
        tl.static_assert(WINDOW < 0)
        tl.static_assert(INTERIOR % BLOCK_N == 0)
        inner = tl.maximum(seqlen - QLEN - EXTENT + 1, 0)
        big = inner // INTERIOR
        small_lo = big * (INTERIOR // BLOCK_N)
        cost = big * INTERIOR_COST
        total = cost + (tl.cdiv(seqlen, BLOCK_N) - small_lo) * MASKED_COST
        per_split = tl.cdiv(total, splits)
        lo = tl.minimum(split * per_split, total)
        hi = tl.minimum(lo + per_split, total)
        big_lo = tl.minimum(tl.cdiv(lo, INTERIOR_COST), big)
        big_hi = tl.minimum(tl.cdiv(hi, INTERIOR_COST), big)
        tile_lo = small_lo + tl.cdiv(tl.maximum(lo - cost, 0), MASKED_COST)
        tile_hi = small_lo + tl.cdiv(tl.maximum(hi - cost, 0), MASKED_COST)

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
    if INTERIOR > 0:
        for tile in range(big_lo, big_hi):
            m, l, acc = _interior_tile(
                q, m, l, acc, tile * INTERIOR, b, hk, d, scale, K, V, PageTable,
                SK0, SK1, SK2, SP, PAGE, INTERIOR,
            )  # fmt: skip
    # Keep inline rather than share a helper with _interior_tile:
    # a helper changes the sliding-window verify SASS (1.7% slower at qlen 5 on H200).
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
    sms = current_arch().num_sms
    max_splits = max(1, min(2 * sms // max(1, batch * heads_k), MAX_SPLITS))
    block_n = 128 if qlen == 1 and span >= 128 * max_splits else 32
    splits = min(triton.cdiv(span, block_n), max_splits)
    interior, interior_cost, masked_cost = 0, 1, 1
    num_stages = 2 if qlen == 1 else 3
    # Tuned on blocks of up to 8 tokens x 6 heads, which pad to 64 rows;
    # at 128 rows, interior tiles add spills and ran up to 13% slower at batch 1-4.
    if qlen > 1 and window_size_left is None and qlen * group <= 64:
        interior, interior_cost, masked_cost, num_stages = _global_verify_tiles(
            batch * heads_k
        )
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
        interior,
        interior_cost,
        masked_cost,
        num_warps=4,
        num_stages=num_stages,
        launch_pdl=True,
    )
    return o, m, l


def _global_verify_tiles(units):
    """(interior tile, its cost, masked tile cost, pipeline stages) for global-layer
    verify, by sequences x KV heads. Measured on H200 with 2- to 8-token blocks, 2K
    to 128K keys and the launcher's split counts; re-tune when the kernel changes."""
    if units < 4:
        return 128, 2, 1, 2
    if units < 16:
        return 64, 4, 3, 2
    return 64, 3, 4, 3
