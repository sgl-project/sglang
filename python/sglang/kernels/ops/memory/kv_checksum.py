# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gathered KV-cache checksum.

``gpu_tensor_hash`` hashes one contiguous tensor. PD disaggregation needs the
same idea over a *gather*: the KV rows of one request live at scattered slot
ids, the prefill and decode sides hold them at completely different slots, and
the digest still has to match. So the per-element hash is keyed by the
element's **logical** coordinate -- (buffer tag, position within the
transferred range, word index within the row) -- never by its slot id, and the
per-element hashes are combined with a wrapping add so the result does not
depend on the order the work is scheduled in.

One launch covers a whole batch of requests; ``out[r]`` is request ``r``'s
digest. The grid is (slot tile, sampled buffer, request) and the only host-side
table is one short row per request, so a launch costs a handful of numpy calls
no matter how many rows it digests. Each request names its own subset of the KV
buffers in that row (see ``sglang.srt.disaggregation.kv_checksum``).
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.memory.gpu_tensor_hash import FMIX32_C1, FMIX32_C2, _fmix32

# Odd multipliers that spread a logical coordinate across the 32-bit word
# before it is mixed into the element value. Passed in as constexpr arguments,
# the way gpu_tensor_hash passes its own constants.
TAG_C = 0x9E3779B1
POS_C = 0x85EBCA77
WORD_C = 0xC2B2AE3D

# The per-item 32-bit fold is widened over 64 bits before the cross-item
# wrapping add, so items cannot cancel out in the low half alone.
WIDEN_C = 0x9E3779B97F4A7C15

# Rows hashed per work item. Sized so one item is a healthy chunk of work
# (TILE * row_bytes) while keeping the atomic-add traffic per request low.
DEFAULT_TILE = 32

# Fields of a request's metadata row, then its sampled buffer indices. A
# buffer index doubles as the hash's buffer tag: it is the position in the
# pool's full buffer list, which is stable across both sides of a transfer.
META_SLOT_BASE, META_NUM_ROWS, META_BUFFERS = 0, 1, 2
# Triton only reads a module global that was instantiated as a constexpr.
_META_SLOT_BASE = tl.constexpr(META_SLOT_BASE)
_META_NUM_ROWS = tl.constexpr(META_NUM_ROWS)
_META_BUFFERS = tl.constexpr(META_BUFFERS)


@triton.jit(do_not_specialize=["row_words", "seed"])
def _kv_slot_checksum_kernel(
    buf_ptr_table,  # *int64 [num_bufs]  base address of each KV buffer
    meta_ptr,  # *int32 [num_reqs, meta_stride] per-request row
    slot_ptr,  # *int32/int64 [total_rows] slots, requests concatenated
    out_ptr,  # *int64 [num_reqs] accumulated digests
    meta_stride,
    row_words,  # uint32 words per slot row
    seed,
    C_FM1: tl.constexpr,
    C_FM2: tl.constexpr,
    C_TAG: tl.constexpr,
    C_POS: tl.constexpr,
    C_WORD: tl.constexpr,
    C_WIDEN: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile_id = tl.program_id(axis=0)
    sample = tl.program_id(axis=1)
    req = tl.program_id(axis=2)

    meta_row = meta_ptr + req * meta_stride
    num_rows = tl.load(meta_row + _META_NUM_ROWS)
    pos0 = tile_id * TILE
    # The grid is sized for the longest request; shorter ones stop here.
    if pos0 < num_rows:
        slot_base = tl.load(meta_row + _META_SLOT_BASE).to(tl.int64)
        buf_idx = tl.load(meta_row + _META_BUFFERS + sample)
        base = tl.load(buf_ptr_table + buf_idx).to(tl.pointer_type(tl.uint32))
        tag = buf_idx.to(tl.uint32)

        # Position within the transferred range -- never the slot id, which is
        # what lets the two sides of a transfer agree.
        poss = pos0 + tl.arange(0, TILE)
        row_m = poss < num_rows
        slots = tl.load(slot_ptr + slot_base + poss, mask=row_m, other=0).to(tl.int64)

        s = tl.full((), seed, tl.uint32)
        tag_c = tl.full((), C_TAG, tl.uint32)
        pos_c = tl.full((), C_POS, tl.uint32)
        word_c = tl.full((), C_WORD, tl.uint32)
        # Fixed for a whole row, so it is hoisted out of the word loop.
        row_key = (tag * tag_c) ^ (poss.to(tl.uint32) * pos_c) ^ s

        h = tl.zeros((), dtype=tl.uint32)
        for start in range(0, row_words, BLOCK):
            words = start + tl.arange(0, BLOCK)
            word_m = words < row_words
            m = row_m[:, None] & word_m[None, :]
            addr = slots[:, None] * row_words + words[None, :].to(tl.int64)
            v = tl.load(base + addr, mask=m, other=0).to(tl.uint32)
            k = _fmix32(
                v ^ row_key[:, None] ^ (words[None, :].to(tl.uint32) * word_c),
                C1=C_FM1,
                C2=C_FM2,
            )
            k = tl.where(m, k, tl.zeros_like(k))
            h += tl.sum(tl.sum(k, axis=1), axis=0).to(tl.uint32)

        h = _fmix32(h ^ tag, C1=C_FM1, C2=C_FM2)
        widen = tl.full((), C_WIDEN, tl.uint64)
        tl.atomic_add(out_ptr + req, (h.to(tl.uint64) * widen).to(tl.int64))


def kv_slot_checksum(
    *,
    buf_ptr_table: torch.Tensor,
    row_bytes: int,
    meta: torch.Tensor,
    slots: torch.Tensor,
    num_sampled: int,
    max_rows: int,
    seed: int = 0x243F6A88,
    tile: int = DEFAULT_TILE,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Digest the gathered KV rows of a batch of requests in one launch.

    ``buf_ptr_table`` addresses the pool's slot-indexed KV buffers
    (``make_ptr_table`` of their base pointers): row ``s`` of buffer ``b`` holds
    the KV that slot ``s`` stores for that buffer, ``row_bytes`` wide. ``meta``
    is the ``[num_reqs, 2 + num_sampled]`` int32 table built by
    ``sglang.srt.disaggregation.kv_checksum``, and ``slots`` holds every
    request's rows concatenated in logical order.

    Returns an ``int64`` ``[num_reqs]`` tensor. Nothing is synchronized -- the
    caller decides when to read it.
    """
    assert row_bytes % 4 == 0, f"row_bytes must be 4-byte aligned, got {row_bytes}"
    num_reqs = meta.shape[0]
    if out is None:
        out = torch.zeros(num_reqs, dtype=torch.int64, device=buf_ptr_table.device)
    if max_rows == 0:
        return out

    row_words = row_bytes // 4
    block = min(1 << (row_words - 1).bit_length(), 1024)
    grid = (triton.cdiv(max_rows, tile), num_sampled, num_reqs)
    _kv_slot_checksum_kernel[grid](
        buf_ptr_table,
        meta,
        slots,
        out,
        meta.shape[1],
        row_words,
        seed,
        C_FM1=FMIX32_C1,
        C_FM2=FMIX32_C2,
        C_TAG=TAG_C,
        C_POS=POS_C,
        C_WORD=WORD_C,
        C_WIDEN=WIDEN_C,
        TILE=tile,
        BLOCK=block,
        num_warps=8,
        num_stages=2,
    )
    return out
