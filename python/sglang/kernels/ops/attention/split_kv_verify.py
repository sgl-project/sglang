"""Paged multi-token attention with independent KV splits and a merge pass."""

import torch
import triton
import triton.language as tl


@triton.jit
def _split_attention(
    Q,
    K,
    V,
    Table,
    Lengths,
    Partial,
    Maxima,
    Denom,
    TABLE_STRIDE: tl.constexpr,
    CACHE_PAGE_STRIDE: tl.constexpr,
    CACHE_TOKEN_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    QLEN: tl.constexpr,
    SPLITS: tl.constexpr,
    QK_SCALE: tl.constexpr,
    V_SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BALANCED: tl.constexpr,
    SKIP_EMPTY: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    split, tile, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = tile * BM + tl.arange(0, BM)
    dims = tl.arange(0, 256)
    length = tl.load(Lengths + batch)
    if BALANCED:
        blocks = tl.cdiv(length, BN)
        begin = (split * blocks // SPLITS) * BN
        end = tl.minimum(((split + 1) * blocks // SPLITS) * BN, length)
    else:
        pages_per_split = tl.cdiv(tl.cdiv(length, 64), SPLITS)
        begin = split * pages_per_split * 64
        end = tl.minimum(begin + pages_per_split * 64, length)
    if not SKIP_EMPTY or begin < end:
        query = tl.load(
            Q + (batch * QLEN * HEADS + rows[:, None]) * 256 + dims[None, :],
            rows[:, None] < QLEN * HEADS,
            0.0,
        )
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        denom = tl.zeros((BM,), tl.float32)
        accum = tl.zeros((BM, 256), tl.float32)
        for start in tl.range(begin, end, BN, num_stages=LOOP_STAGES):
            tokens = start + tl.arange(0, BN)
            page = tl.load(
                Table + batch * TABLE_STRIDE + tokens // 64, tokens < length, 0
            )
            cache = page * CACHE_PAGE_STRIDE + (tokens % 64) * CACHE_TOKEN_STRIDE
            keys = tl.load(
                K + cache[None, :] + dims[:, None], tokens[None, :] < end, 0.0
            )
            scores = tl.dot(query, keys) * (QK_SCALE * 1.4426950408889634)
            causal_end = length - QLEN + rows // HEADS
            scores = tl.where(
                (tokens[None, :] < end)
                & (tokens[None, :] <= causal_end[:, None])
                & (rows[:, None] < QLEN * HEADS),
                scores,
                -float("inf"),
            )
            next_max = tl.maximum(maximum, tl.max(scores, 1))
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            alpha = tl.exp2(maximum - safe_max)
            probs = tl.exp2(scores - safe_max[:, None])
            denom = denom * alpha + tl.sum(probs, 1)
            accum = accum * alpha[:, None]
            values = tl.load(
                V + cache[:, None] + dims[None, :], tokens[:, None] < end, 0.0
            )
            accum = tl.dot((probs * 256.0).to(query.dtype), values, accum)
            maximum = next_max
        base = (batch * SPLITS + split) * QLEN * HEADS + rows
        tl.store(
            Partial + base[:, None] * 256 + dims[None, :],
            accum * (V_SCALE / 256.0),
            rows[:, None] < QLEN * HEADS,
        )
        tl.store(Maxima + base, maximum, rows < QLEN * HEADS)
        tl.store(Denom + base, denom, rows < QLEN * HEADS)


@triton.jit
def _split_attention_adaptive(
    Q,
    K,
    V,
    Table,
    Lengths,
    Partial,
    Maxima,
    Denom,
    TABLE_STRIDE: tl.constexpr,
    CACHE_PAGE_STRIDE: tl.constexpr,
    CACHE_TOKEN_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    QLEN: tl.constexpr,
    SPLITS: tl.constexpr,
    QK_SCALE: tl.constexpr,
    V_SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BALANCED: tl.constexpr,
    SKIP_EMPTY: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    length = tl.load(Lengths + tl.program_id(2))
    if length <= 2048:
        _split_attention(
            Q,
            K,
            V,
            Table,
            Lengths,
            Partial,
            Maxima,
            Denom,
            TABLE_STRIDE,
            CACHE_PAGE_STRIDE,
            CACHE_TOKEN_STRIDE,
            HEADS,
            QLEN,
            SPLITS,
            QK_SCALE,
            V_SCALE,
            BM,
            64,
            BALANCED,
            SKIP_EMPTY,
            1,
        )
    else:
        _split_attention(
            Q,
            K,
            V,
            Table,
            Lengths,
            Partial,
            Maxima,
            Denom,
            TABLE_STRIDE,
            CACHE_PAGE_STRIDE,
            CACHE_TOKEN_STRIDE,
            HEADS,
            QLEN,
            SPLITS,
            QK_SCALE,
            V_SCALE,
            BM,
            BN,
            BALANCED,
            SKIP_EMPTY,
            LOOP_STAGES,
        )


@triton.jit
def _split_attention_transposed(
    Q,
    K,
    V,
    Table,
    Lengths,
    Partial,
    Maxima,
    Denom,
    TABLE_STRIDE: tl.constexpr,
    CACHE_PAGE_STRIDE: tl.constexpr,
    CACHE_TOKEN_STRIDE: tl.constexpr,
    HEADS: tl.constexpr,
    QLEN: tl.constexpr,
    SPLITS: tl.constexpr,
    QK_SCALE: tl.constexpr,
    V_SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BALANCED: tl.constexpr,
    SKIP_EMPTY: tl.constexpr,
    LOOP_STAGES: tl.constexpr,
):
    split, tile, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = tile * BM + tl.arange(0, BM)
    dims = tl.arange(0, 256)
    length = tl.load(Lengths + batch)
    if BALANCED:
        blocks = tl.cdiv(length, BN)
        begin = (split * blocks // SPLITS) * BN
        end = tl.minimum(((split + 1) * blocks // SPLITS) * BN, length)
    else:
        pages_per_split = tl.cdiv(tl.cdiv(length, 64), SPLITS)
        begin = split * pages_per_split * 64
        end = tl.minimum(begin + pages_per_split * 64, length)
    if not SKIP_EMPTY or begin < end:
        query = tl.load(
            Q + (batch * QLEN * HEADS + rows[:, None]) * 256 + dims[None, :],
            rows[:, None] < QLEN * HEADS,
            0.0,
        )
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        denom = tl.zeros((BM,), tl.float32)
        accum = tl.zeros((256, BM), tl.float32)
        for start in tl.range(begin, end, BN, num_stages=LOOP_STAGES):
            tokens = start + tl.arange(0, BN)
            page = tl.load(
                Table + batch * TABLE_STRIDE + tokens // 64, tokens < length, 0
            )
            cache = page * CACHE_PAGE_STRIDE + (tokens % 64) * CACHE_TOKEN_STRIDE
            keys = tl.load(
                K + cache[:, None] + dims[None, :], tokens[:, None] < end, 0.0
            )
            scores = tl.dot(keys, tl.trans(query)) * (QK_SCALE * 1.4426950408889634)
            causal_end = length - QLEN + rows // HEADS
            scores = tl.where(
                (tokens[:, None] < end)
                & (tokens[:, None] <= causal_end[None, :])
                & (rows[None, :] < QLEN * HEADS),
                scores,
                -float("inf"),
            )
            next_max = tl.maximum(maximum, tl.max(scores, 0))
            safe_max = tl.where(next_max == -float("inf"), 0.0, next_max)
            alpha = tl.exp2(maximum - safe_max)
            probs = tl.exp2(scores - safe_max[None, :])
            denom = denom * alpha + tl.sum(probs, 0)
            accum = accum * alpha[None, :]
            values = tl.load(
                V + cache[:, None] + dims[None, :], tokens[:, None] < end, 0.0
            )
            accum = tl.dot(tl.trans(values), (probs * 256.0).to(query.dtype), accum)
            maximum = next_max
        base = (batch * SPLITS + split) * QLEN * HEADS + rows
        tl.store(
            Partial + base[:, None] * 256 + dims[None, :],
            tl.trans(accum) * (V_SCALE / 256.0),
            rows[:, None] < QLEN * HEADS,
        )
        tl.store(Maxima + base, maximum, rows < QLEN * HEADS)
        tl.store(Denom + base, denom, rows < QLEN * HEADS)


@triton.jit
def _merge_tile(
    Partial,
    Maxima,
    Denom,
    Lengths,
    ROWS: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
    BALANCED: tl.constexpr,
    SKIP_EMPTY: tl.constexpr,
):
    row, dim_tile, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    splits = tl.arange(0, BS)
    dims = dim_tile * BD + tl.arange(0, BD)
    base = (batch * SPLITS + splits) * ROWS + row
    active = splits < SPLITS
    if SKIP_EMPTY:
        length = tl.load(Lengths + batch)
        if BALANCED:
            blocks = tl.cdiv(length, BN)
            begin = (splits * blocks // SPLITS) * BN
            end = tl.minimum(((splits + 1) * blocks // SPLITS) * BN, length)
        else:
            pages_per_split = tl.cdiv(tl.cdiv(length, 64), SPLITS)
            begin = splits * pages_per_split * 64
            end = tl.minimum(begin + pages_per_split * 64, length)
        active = active & (begin < end)
    maximum = tl.load(Maxima + base, active, -float("inf"))
    denom = tl.load(Denom + base, active, 0.0)
    max_all = tl.max(maximum, 0)
    max_all = tl.where(max_all == -float("inf"), 0.0, max_all)
    weights = tl.exp2(maximum - max_all)
    total = tl.sum(denom * weights, 0)
    partial = tl.load(
        Partial + base[:, None] * 256 + dims[None, :],
        active[:, None] & (dims[None, :] < 256),
        0.0,
    )
    result = tl.sum(partial * weights[:, None], 0) / tl.where(total > 0.0, total, 1.0)
    return result, dims


@triton.jit
def _merge_attention(
    Partial,
    Maxima,
    Denom,
    Lengths,
    Out,
    ROWS: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    BN: tl.constexpr,
    BALANCED: tl.constexpr,
    SKIP_EMPTY: tl.constexpr,
    SHORT_MERGE: tl.constexpr,
):
    if SHORT_MERGE and not BALANCED:
        length = tl.load(Lengths + tl.program_id(2))
        if length <= 2048:
            result, dims = _merge_tile(
                Partial,
                Maxima,
                Denom,
                Lengths,
                ROWS,
                SPLITS,
                32,
                BD,
                BN,
                BALANCED,
                SKIP_EMPTY,
            )
        else:
            result, dims = _merge_tile(
                Partial,
                Maxima,
                Denom,
                Lengths,
                ROWS,
                SPLITS,
                BS,
                BD,
                BN,
                BALANCED,
                SKIP_EMPTY,
            )
    else:
        result, dims = _merge_tile(
            Partial,
            Maxima,
            Denom,
            Lengths,
            ROWS,
            SPLITS,
            BS,
            BD,
            BN,
            BALANCED,
            SKIP_EMPTY,
        )
    row, batch = tl.program_id(0), tl.program_id(2)
    tl.store(Out + (batch * ROWS + row) * 256 + dims, result, dims < 256)


def split_kv_verify(
    q,
    kv,
    table,
    lengths,
    out,
    scratch,
    *,
    splits=64,
    qlen=7,
    block_m=32,
    block_n=64,
    qk_scale=1 / 16,
    v_scale=1.0,
    num_stages=2,
    num_warps=4,
    transposed=False,
    balanced=False,
    skip_empty=False,
    short_merge=False,
    short_tiles=False,
    merge_tile=32,
    merge_warps=4,
):
    """Run attention on one KV head; scratch is caller-owned for graph replay."""
    if short_tiles and (balanced or transposed):
        raise ValueError("Adaptive tiles require unbalanced, non-transposed attention")
    k, v = kv
    heads = q.shape[-2]
    batch = lengths.numel()
    partial, maxima, denom = scratch
    kernel = (
        _split_attention_adaptive
        if short_tiles
        else (_split_attention_transposed if transposed else _split_attention)
    )
    kernel[(splits, triton.cdiv(qlen * heads, block_m), batch)](
        q,
        k,
        v,
        table,
        lengths,
        partial,
        maxima,
        denom,
        table.stride(0),
        k.stride(0),
        k.stride(2),
        heads,
        qlen,
        splits,
        qk_scale,
        v_scale,
        block_m,
        block_n,
        balanced,
        skip_empty,
        num_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _merge_attention[(qlen * heads, triton.cdiv(256, merge_tile), batch)](
        partial,
        maxima,
        denom,
        lengths,
        out,
        qlen * heads,
        splits,
        triton.next_power_of_2(splits),
        merge_tile,
        block_n,
        balanced,
        skip_empty,
        short_merge,
        num_warps=merge_warps,
        enable_fp_fusion=False,
    )
    return out


def allocate_scratch(q, batch, splits, qlen):
    rows = qlen * q.shape[-2]
    return (
        torch.empty((batch, splits, rows, 256), dtype=torch.bfloat16, device=q.device),
        torch.empty((batch, splits, rows), dtype=torch.float32, device=q.device),
        torch.empty((batch, splits, rows), dtype=torch.float32, device=q.device),
    )
