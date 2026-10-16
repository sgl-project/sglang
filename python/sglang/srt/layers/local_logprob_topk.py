"""Bounded tiled local Top-N with the established row normalizer.

Two selection launches replace multi-launch torch.topk. Keeping the existing
normalizer preserves mixed Top-N/no-Top-N chunk consistency exactly.
Tiles and the row merge use fixed resource bounds; unsupported inputs retain
the established path. No global communication or output lifetime is changed.
"""

import torch
import triton
import triton.language as tl
from sglang.srt.layers.logsumexp import row_logsumexp

TILE = 2048
MAX_MERGE = 2048


@triton.jit
def _keys(value, index):
    bits = value.to(tl.uint32, bitcast=True)
    bits = tl.where(value == 0.0, 0, bits).to(tl.uint32)
    bits = tl.where(value != value, 0x7FC00000, bits).to(tl.uint32)
    ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
    return (ordered.to(tl.uint64) << 32) | (0xFFFFFFFF - index.to(tl.uint32)).to(
        tl.uint64
    )


@triton.jit
def _values(keys):
    ordered = (keys >> 32).to(tl.uint32)
    bits = tl.where((ordered & 0x80000000) != 0, ordered ^ 0x80000000, ~ordered).to(
        tl.uint32
    )
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _tile_topk(
    x,
    candidates,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    COLS: tl.constexpr,
    TILES: tl.constexpr,
    KP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    column = tile * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(
        x + row * ROW_STRIDE + column.to(tl.int64) * COL_STRIDE,
        column < COLS,
        other=-float("inf"),
    ).to(tl.float32)
    index = tl.where(column < COLS, column.to(tl.uint32), 0xFFFFFFFF).to(tl.uint32)
    best = tl.topk(_keys(value, index), KP, descending=True)
    slots = tl.arange(0, KP)
    tl.store(
        candidates + (row * TILES + tile) * KP + slots, best.to(tl.int64, bitcast=True)
    )


@triton.jit
def _merge_local(
    candidates,
    out_values,
    out_ids,
    TILES: tl.constexpr,
    KP: tl.constexpr,
    K: tl.constexpr,
    MERGE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, MERGE)
    sentinel = _keys(
        tl.full((MERGE,), -float("inf"), tl.float32),
        tl.full((MERGE,), 0xFFFFFFFF, tl.uint32),
    )
    key = tl.load(candidates + row * TILES * KP + lane, lane < TILES * KP, other=0).to(
        tl.uint64, bitcast=True
    )
    key = tl.where(lane < TILES * KP, key, sentinel)
    best = tl.topk(key, KP, descending=True)
    slot = tl.arange(0, KP)
    tl.store(out_values + row * K + slot, _values(best), slot < K)
    ids = (0xFFFFFFFF - (best & 0xFFFFFFFF).to(tl.uint32)).to(tl.int64)
    tl.store(out_ids + row * K + slot, ids, slot < K)


def can_use_local_topk(x, k):
    if not (
        x.is_cuda
        and torch.version.hip is None
        and x.ndim == 2
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    ):
        return False
    rows, cols = x.shape
    if not (rows > 0 and 1 <= k <= min(32, cols)):
        return False
    candidates = triton.cdiv(cols, TILE) * triton.next_power_of_2(k)
    return candidates <= MAX_MERGE and rows * candidates <= 2**31 - 1


def local_logsumexp_topk(x, k):
    assert can_use_local_topk(x, k)
    rows, cols = x.shape
    tiles, kp = triton.cdiv(cols, TILE), triton.next_power_of_2(k)
    candidates = torch.empty((rows, tiles, kp), device=x.device, dtype=torch.int64)
    out_max, out_log_sum = row_logsumexp(x)
    values = torch.empty((rows, k), device=x.device, dtype=torch.float32)
    ids = torch.empty((rows, k), device=x.device, dtype=torch.int64)
    _tile_topk[(rows, tiles)](
        x,
        candidates,
        x.stride(0),
        x.stride(1),
        cols,
        tiles,
        kp,
        TILE,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _merge_local[(rows,)](
        candidates,
        values,
        ids,
        tiles,
        kp,
        k,
        triton.next_power_of_2(tiles * kp),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out_max, out_log_sum, values, ids
