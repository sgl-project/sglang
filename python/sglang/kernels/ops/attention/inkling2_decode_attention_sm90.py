"""Paged decode attention with a compact relative bias for Hopper and SM120: the
split-KV partials of paged_rel_bias_decode (unnormalized o, max, sum), reduced by
combine_gate_sink, for one query per sequence or a causal block of queries per
sequence (MTP verify). Each split takes an equal share of the sequence's own keys. On
a global layer's verify of a block that pads to 32 or 64 rows, each masked tile's bias
is gathered a tile ahead; at 64 rows the keys at least the bias extent behind every
query row also run as larger unmasked tiles, and a split takes an equal share of the
tiles."""

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
def _bias_tile(
    Bias,
    token,
    qh,
    head_mask,
    q_pos,
    start,
    end,
    SB0: tl.constexpr,
    SB1: tl.constexpr,
    EXTENT: tl.constexpr,
    TILE: tl.constexpr,
):
    # The bias of keys [start, start + TILE) below end, as the masked loop adds it.
    pos = start + tl.arange(0, TILE)
    dist = q_pos[:, None] - pos[None, :]
    keep = (pos < end)[None, :] & (dist >= 0) & (dist < EXTENT)
    bias = Bias + token[:, None] * SB0 + qh[:, None] * SB1 + dist
    return tl.load(bias, head_mask[:, None] & keep, other=0)


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
    INTERIOR: tl.constexpr,
    PREFETCH_BIAS: tl.constexpr,
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
        # No window: keys [0, inner) are at least EXTENT behind every query row; whole
        # INTERIOR-key tiles of them run unmasked, the rest as BLOCK_N-key masked tiles.
        # A split takes an equal share of the tiles of both kinds, interior ones first,
        # so a sequence with no interior tile is split as main splits it.
        tl.static_assert(WINDOW < 0)
        tl.static_assert(INTERIOR % BLOCK_N == 0)
        inner = tl.maximum(seqlen - QLEN - EXTENT + 1, 0)
        big = inner // INTERIOR
        small_lo = big * (INTERIOR // BLOCK_N)
        tiles = big + tl.cdiv(seqlen, BLOCK_N) - small_lo
        per_split = tl.cdiv(tiles, splits)
        lo = tl.minimum(split * per_split, tiles)
        hi = tl.minimum(lo + per_split, tiles)
        big_lo = tl.minimum(lo, big)
        big_hi = tl.minimum(hi, big)
        tile_lo = small_lo + tl.maximum(lo - big, 0)
        tile_hi = small_lo + tl.maximum(hi - big, 0)

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
    if PREFETCH_BIAS:
        # A bias gathered one tile ahead overlaps the previous tile and keeps the scores
        # in the MMA layout; gathered in-tile, 64-1024 keys ran 1.09-1.4x slower (H200).
        tl.static_assert(WINDOW < 0)
        end = tl.minimum(seqlen, tile_hi * BLOCK_N)
        bias_next = _bias_tile(
            Bias, token, qh, head_mask, q_pos, tile_lo * BLOCK_N, end,
            SB0, SB1, EXTENT, BLOCK_N,
        )  # fmt: skip
    # Keep inline rather than share a helper with _interior_tile:
    # a helper changes the sliding-window verify SASS (1.7% slower at qlen 5 on H200).
    for tile in range(tile_lo, tile_hi):
        if PREFETCH_BIAS:
            bias_now = bias_next
            bias_next = _bias_tile(
                Bias, token, qh, head_mask, q_pos, (tile + 1) * BLOCK_N, end,
                SB0, SB1, EXTENT, BLOCK_N,
            )  # fmt: skip
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
        if PREFETCH_BIAS:
            bias = bias_now.to(tl.float32)
        else:
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
    if INTERIOR > 0:
        # After the masked loop, so its first loads issue as early as main's. Pipelined:
        # 1.05-1.3x from 32K keys (H200); skipped in splits with no interior tile.
        if big_hi > big_lo:
            for tile in tl.range(big_lo, big_hi, num_stages=2):
                m, l, acc = _interior_tile(
                    q, m, l, acc, tile * INTERIOR, b, hk, d, scale, K, V, PageTable,
                    SK0, SK1, SK2, SP, PAGE, INTERIOR,
                )  # fmt: skip
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
    sms = current_arch().num_sms
    max_splits = max(1, min(2 * sms // max(1, batch * heads_k), MAX_SPLITS))
    block_n = 128 if qlen == 1 and span >= 128 * max_splits else 32
    splits = min(triton.cdiv(span, block_n), max_splits)
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
    interior, prefetch = _verify_path(qlen, group, window_size_left)
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
        max(16, triton.next_power_of_2(qlen * group)),
        head_dim,
        page_size,
        rel_bias.shape[-1],
        -1 if window_size_left is None else window_size_left,
        block_n,
        mxfp8,
        interior,
        prefetch,
        num_warps=4,
        num_stages=2 if qlen == 1 or prefetch else 3,
        # 64-row launches: 203-204 registers (212-217 uncapped), and up to 2.8% faster
        # at 1 sequence x KV head, 4.9% slower at 16 (H200); 32 rows were timed capped.
        maxnreg=232 if prefetch else None,
        launch_pdl=True,
    )
    return o, m, l


def _verify_path(qlen, group, window_size_left):
    """(keys per unmasked interior tile or 0, bias gathered a tile ahead) for a global
    layer's verify; (0, False) keeps main's path. Measured on H200 at 6 heads per KV
    head: interior tiles pay off at 64 rows (6 to 10 tokens) but ran up to 1.4% slower
    than main at 32 rows (3 to 5 tokens) on 1536 keys, and 128-key ones spilled at 128
    rows; at 16 rows (2 tokens) the bias prefetch alone ran up to 4% slower than
    main."""
    if qlen == 1 or window_size_left is not None or not 16 < qlen * group <= 64:
        return 0, False
    return (64 if qlen * group > 32 else 0), True
