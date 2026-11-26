# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache the draft indexer's sorted blocks for paged GBSA.

Block sorting follows qsa_gbsa_sort_counts. Packed outputs store all block
rows contiguously, followed by the counts and capture lengths. Each side
array starts on a 32-byte boundary, even in a partial eight-row tile.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cann import extension as ext

PACK_WIDTH = 536
CACHE_WIDTH = 520


def unpack(packed):
    rows = packed.shape[0]
    flat = packed.view(-1)
    start = rows * 512
    lengths = start + triton.cdiv(rows, 8) * 8
    return (
        flat[:start].view(1, rows, 512),
        flat[start : start + rows].view(1, rows),
        flat[lengths : lengths + rows],
    )


@triton.jit
def _pack(src, lengths, dst, R: tl.constexpr, STRIDE: tl.constexpr, TOKENS: tl.constexpr):
    row = tl.program_id(0) * 8 + tl.arange(0, 8)
    col = tl.arange(0, 512)
    lens = tl.load(lengths + row, row < R, 1)
    pos = col * 4 if TOKENS else col
    vals = tl.load(src + row[:, None] * STRIDE + pos[None, :], row[:, None] < R, -1)
    ids = vals // 4 if TOKENS else vals
    valid = (vals >= 0) & (ids < lens[:, None] // 4)
    keys = tl.where(valid, ids, 16777215).to(tl.float32)
    ids = ext.sort(keys, dim=1, descending=False).to(tl.int32)
    tl.store(dst + row[:, None] * 512 + col[None, :], ids, row[:, None] < R)
    tl.store(dst + R * 512 + row, tl.sum(valid.to(tl.int32), 1), row < R)
    tl.store(dst + R * 512 + tl.cdiv(R, 8) * 8 + row, lens, row < R)


def pack(indices, lengths):
    rows, width = indices.shape
    assert width in (512, 2051) and indices.stride(1) == 1 and lengths.is_contiguous()
    out = indices.new_empty((rows, PACK_WIDTH), dtype=torch.int32)
    _pack[(triton.cdiv(rows, 8),)](
        indices, lengths, out, rows, indices.stride(0), width != 512,
        multibuffer=False, unit_flag=False,
    )
    return out


@triton.jit
def _save(src, anchors, slots, cache, R: tl.constexpr, TRASH: tl.constexpr):
    row = tl.program_id(0)
    anchor = tl.load(anchors + row).to(tl.int32)
    slot = tl.load(slots + row).to(tl.int32)
    valid = (slot > 0) & (slot < TRASH)
    col = tl.arange(0, 512)
    ids = tl.load(src + anchor * 512 + col)
    count = tl.load(src + R * 512 + anchor)
    length = tl.load(src + R * 512 + tl.cdiv(R, 8) * 8 + anchor)
    tl.store(cache + slot * 520 + col, ids, valid)
    meta = tl.arange(0, 8)
    values = tl.where(meta == 0, count, tl.where(meta == 1, length, 0))
    tl.store(cache + slot * 520 + 512 + meta, values, valid)


def save(packed, anchors, slots, cache, trash):
    _save[(slots.numel(),)](
        packed, anchors, slots, cache, packed.shape[0], trash,
        multibuffer=False, unit_flag=False,
    )


@triton.jit
def _lookup(cache, req, positions, dst, R: tl.constexpr):
    tile = tl.program_id(0)
    row = tile * 8 + tl.arange(0, 8)
    col = tl.arange(0, 512)
    slot = tl.load(req + row, row < R, 0).to(tl.int32)
    now = tl.load(positions + row, row < R, 0).to(tl.int32) + 1
    count = tl.load(cache + slot * 520 + 512)
    base = tl.load(cache + slot * 520 + 513)
    # Seed blocks all end before base//4. At most one new block can complete.
    crossed = (now // 4 > base // 4) & (slot > 0)
    full = count == 512
    overflow = crossed & full
    # At capacity the first block is the most distant in logical KV order.
    col_src = tl.minimum(col[None, :] + overflow[:, None].to(tl.int32), 511)
    ids = tl.load(cache + slot[:, None] * 520 + col_src)
    append_col = tl.minimum(count, 511)
    add_block = crossed[:, None] & (col[None, :] == append_col[:, None])
    ids = tl.where(add_block, base[:, None] // 4, ids)
    count += (crossed & ~full).to(tl.int32)
    tl.store(dst + row[:, None] * 512 + col[None, :], ids, row[:, None] < R)
    tl.store(dst + R * 512 + row, count, row < R)
    tl.store(dst + R * 512 + tl.cdiv(R, 8) * 8 + row, now, row < R)


def lookup(cache, slots, positions):
    rows = slots.numel()
    out = cache.new_empty((rows, PACK_WIDTH))
    _lookup[(triton.cdiv(rows, 8),)](
        cache, slots, positions, out, rows, multibuffer=False, unit_flag=False
    )
    return out
