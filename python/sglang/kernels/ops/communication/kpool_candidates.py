"""Exact distributed Top-512 for DCP page-striped, four-token pooled keys."""

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fast_topk import fast_topk


@triton.jit
def _pack_candidates(
    Scores,
    Lengths,
    Indices,
    Packed,
    SCORE_STRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    SHARDS: tl.constexpr,
    RANK: tl.constexpr,
    COLS: tl.constexpr,
    WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
):
    row = tl.program_id(0)
    lane = tl.arange(0, WIDTH)
    if COLS > TOPK:
        index = tl.load(Indices + row * WIDTH + lane)
    else:
        index = lane
    valid = (index >= 0) & (index < tl.load(Lengths + row))
    score = tl.load(Scores + row * SCORE_STRIDE + index, valid, other=0.0)
    bits = score.to(tl.uint32, bitcast=True)
    ordered = tl.where(bits & 0x80000000 != 0, ~bits, bits | 0x80000000)
    logical = (index // PAGE * SHARDS + RANK) * PAGE + index % PAGE
    # FP32 total order, then smaller global logical ID. int32 token lengths
    # leave 31 bits for the ID, keeping every sortable key positive int64.
    packed = (ordered.to(tl.int64) << 31) | (0x7FFFFFFF - logical).to(tl.int64)
    tl.store(Packed + row * WIDTH + lane, tl.where(valid, packed, 0))


@triton.jit
def _token_ids(
    Table,
    Offsets,
    row,
    logical,
    valid,
    TABLE_STRIDE: tl.constexpr,
    OFFSET_STRIDE: tl.constexpr,
    PAGED: tl.constexpr,
    OFFSET: tl.constexpr,
):
    if PAGED:
        return tl.load(Table + row * TABLE_STRIDE + logical, valid, other=-1)
    if OFFSET:
        logical += tl.load(Offsets + row * OFFSET_STRIDE)
    return tl.where(valid, logical, -1)


@triton.jit
def _merge_and_expand(
    Received,
    PoolLengths,
    TokenLengths,
    Table,
    Offsets,
    Output,
    ROWS: tl.constexpr,
    SHARDS: tl.constexpr,
    WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
    KPOOL: tl.constexpr,
    BLOCK: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    OFFSET_STRIDE: tl.constexpr,
    PAGED: tl.constexpr,
    OFFSET: tl.constexpr,
):
    row = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    location = (off // WIDTH * ROWS + row) * WIDTH + off % WIDTH
    values = tl.load(Received + location, mask=off < SHARDS * WIDTH, other=0)
    winners = tl.topk(values, TOPK)
    logical = (0x7FFFFFFF - (winners & 0x7FFFFFFF)).to(tl.int32)
    pool_len = tl.load(PoolLengths + row)
    # Keep the existing selector's sequential short-history output, including
    # its -1 padding. At full length the Top-K output order is unspecified.
    group_lane = tl.arange(0, TOPK)
    logical = tl.where(pool_len <= TOPK, group_lane, logical)
    lanes = tl.arange(0, TOPK * KPOOL)
    group = tl.gather(logical, lanes // KPOOL, 0)
    valid = lanes // KPOOL < pool_len
    base = row * (TOPK * KPOOL + KPOOL - 1)
    token = group * KPOOL + lanes % KPOOL
    history = _token_ids(
        Table, Offsets, row, token, valid, TABLE_STRIDE, OFFSET_STRIDE, PAGED, OFFSET
    )
    tl.store(Output + base + lanes, history)
    tail_lane = tl.arange(0, KPOOL)
    tail = pool_len * KPOOL + tail_lane
    valid_tail = (tail_lane < KPOOL - 1) & (tail < tl.load(TokenLengths + row))
    tail_out = _token_ids(
        Table,
        Offsets,
        row,
        tail,
        valid_tail,
        TABLE_STRIDE,
        OFFSET_STRIDE,
        PAGED,
        OFFSET,
    )
    tl.store(
        Output + base + TOPK * KPOOL + tail_lane, tail_out, mask=tail_lane < KPOOL - 1
    )


def exchange_kpool_candidates(
    scores,
    local_lens,
    pool_lens,
    seq_lens,
    page_table,
    offsets,
    *,
    group,
    shards,
    rank,
    page_size,
):
    """Return this rank's final token IDs, including page mapping and unfinished pools.

    Each owner receives at most 512 (FP32 score, logical group ID) pairs per
    shard and row. The final reduction reads the native all-to-all layout.
    """
    topk, kpool = 512, 4
    rows, cols = scores.shape
    stride = pool_lens.numel()
    assert rows == stride * shards and shards in (2, 4, 8)
    assert page_table is None or offsets is None
    width = min(topk, triton.next_power_of_2(cols))
    indices = fast_topk(scores, local_lens, topk) if cols > topk else local_lens
    packed = scores.new_empty((rows, width), dtype=torch.int64)
    _pack_candidates[(rows,)](
        scores,
        local_lens,
        indices,
        packed,
        scores.stride(0),
        page_size,
        shards,
        rank,
        cols,
        width,
        topk,
    )
    received = torch.empty_like(packed)
    dist.all_to_all_single(received, packed, group=group)
    output = scores.new_empty((stride, topk * kpool + kpool - 1), dtype=torch.int32)
    _merge_and_expand[(stride,)](
        received,
        pool_lens,
        seq_lens,
        page_table,
        offsets,
        output,
        stride,
        shards,
        width,
        topk,
        kpool,
        max(topk, triton.next_power_of_2(shards * width)),
        page_table.stride(0) if page_table is not None else 0,
        offsets.stride(0) if offsets is not None else 0,
        page_table is not None,
        offsets is not None,
        num_warps=8,
    )
    return output
