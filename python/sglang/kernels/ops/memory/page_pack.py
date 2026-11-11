# Copyright 2023-2026 SGLang Team
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
"""Packing a unified-memory end band's live pages on the device.

The pairs and the remap have fixed shapes sized from host counts, so packing
never reads a page id back on the host. A page is live exactly when its
physical-to-virtual entry is bound (>= 0).
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


def _ranked_positions(
    selected: torch.Tensor, base: int, count: int, *, descending: bool, sink: int
) -> torch.Tensor:
    """``base + i`` for every ``i`` where ``selected`` holds, ascending (or
    descending), in the first entries of a ``count``-long tensor; the rest
    are ``sink``. A selection beyond ``count`` is dropped."""
    rank = torch.cumsum(selected, 0)
    # Inclusive counts: an entry's 0-based rank from the front is its count
    # minus one, from the back the selected entries after it.
    rank = rank[-1:] - rank if descending else rank - 1
    slot = torch.where(selected & (rank < count), rank, count)
    out = torch.full((count + 1,), sink, dtype=torch.int64, device=selected.device)
    positions = torch.arange(
        base, base + selected.numel(), dtype=torch.int64, device=selected.device
    )
    # Slot ``count`` collects everything unselected and is dropped.
    out.scatter_(0, slot, positions)
    return out[:count]


# Every relief packs a span of another size into pairs at another offset of
# one buffer; specializing on those values would compile mid-serving.
@triton.jit(
    do_not_specialize=["first", "span", "split", "count"],
    do_not_specialize_on_alignment=["src_ptr", "dst_ptr"],
)
def pack_pairs_kernel(
    p2v_ptr,  # in: int64 -- physical_to_virtual table
    live_ptr,  # in: [span] int64 -- inclusive count of bound pages from `first`
    src_ptr,  # out: [count] int64, filled with the sink beforehand
    dst_ptr,  # out: [count] int64, filled with the sink beforehand
    first,  # runtime: first page of the span
    span,  # runtime: pages in the span
    split,  # runtime: `packed_end - first`
    count,  # runtime: pairs written at most
    GROW_UP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """`pack_pairs`: each page of the span finds its rank among the sources
    or the free pages from the running count of bound pages, and writes its
    id there; every rank is distinct, so no two lanes write one entry."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < span
    page = first + offs
    bound = tl.load(p2v_ptr + page, mask=mask, other=-1) >= 0
    seen = tl.load(live_ptr + offs, mask=mask, other=0)
    total = tl.load(live_ptr + span - 1)
    below = offs < split
    if GROW_UP:
        # Free pages below the packed end, lowest first; bound pages at or
        # above it, topmost first.
        is_dst = mask & below & ~bound
        is_src = mask & ~below & bound
        dst_rank = offs - seen
        src_rank = total - seen
    else:
        # Bound pages below the packed end, bottommost first; free pages at
        # or above it, topmost first.
        is_src = mask & below & bound
        is_dst = mask & ~below & ~bound
        src_rank = seen - 1
        dst_rank = (span - 1 - offs) - (total - seen)
    tl.store(dst_ptr + dst_rank, page, mask=is_dst & (dst_rank < count))
    tl.store(src_ptr + src_rank, page, mask=is_src & (src_rank < count))


PACK_PAIRS_BLOCK = 1024


def pack_pairs(
    p2v: torch.Tensor,
    *,
    first: int,
    end: int,
    packed_end: int,
    grow_up: bool,
    sink: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The page moves ``(src, dst)`` that pack the live pages of the span
    ``[first, end)`` against the band's base, given that they number
    ``packed_end - first`` (grow-up) or ``end - packed_end`` (grow-down);
    asserted on the device.

    A grow-up end moves its live pages at or above ``packed_end``, topmost
    first, into its free pages below it, lowest first; a grow-down end moves
    those below ``packed_end``, bottommost first, into the free pages at or
    above it, topmost first: the pairs a host compaction walk makes. Both
    tensors are ``min(live, free)`` long, the unused tail ``sink -> sink``.
    O(span) device work, no host sync."""
    span = end - first
    split = packed_end - first
    count = min(split, end - packed_end)
    live = split if grow_up else end - packed_end
    bound = p2v[first:end] >= 0
    seen = torch.cumsum(bound, 0)
    if span:
        torch._assert_async(
            seen[-1] == live, "the bound pages of a span differ from its live count"
        )
    if not p2v.is_cuda:
        # Pure-torch CPU reference for the CUDA kernel.
        if grow_up:
            free, sources = ~bound[:split], bound[split:]
            dst = _ranked_positions(free, first, count, descending=False, sink=sink)
            src = _ranked_positions(
                sources, packed_end, count, descending=True, sink=sink
            )
        else:
            sources, free = bound[:split], ~bound[split:]
            src = _ranked_positions(sources, first, count, descending=False, sink=sink)
            dst = _ranked_positions(free, packed_end, count, descending=True, sink=sink)
        return src, dst
    pairs = torch.full((2, count), sink, dtype=torch.int64, device=p2v.device)
    src, dst = pairs[0], pairs[1]
    if count:
        pack_pairs_kernel[(triton.cdiv(span, PACK_PAIRS_BLOCK),)](
            p2v,
            seen,
            src,
            dst,
            first,
            span,
            split,
            count,
            GROW_UP=grow_up,
            BLOCK=PACK_PAIRS_BLOCK,
        )
    return src, dst


@triton.jit(
    do_not_specialize=["N"],
    do_not_specialize_on_alignment=["src_ptr", "dst_ptr", "moved_ptr"],
)
def remap_packed_pages_kernel(
    src_ptr,  # in: [N] int64 -- physical page moved from
    dst_ptr,  # in: [N] int64 -- physical page moved to
    v2p_ptr,  # in/out: int64 -- virtual_to_physical table
    p2v_ptr,  # in/out: int64 -- physical_to_virtual table
    moved_ptr,  # out: [N] int64 -- the moved virtual page, -1 for a padded pair
    N,
    BLOCK: tl.constexpr,
):
    """Rebind each moved virtual page to its destination and unbind its
    source. Sources and destinations are disjoint and unique apart from
    padded ``sink -> sink`` pairs, which are skipped, so no lane reads an
    entry another lane writes."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    src = tl.load(src_ptr + offs, mask=mask, other=0).to(tl.int64)
    dst = tl.load(dst_ptr + offs, mask=mask, other=0).to(tl.int64)
    moves = mask & (src != dst)
    v = tl.load(p2v_ptr + src, mask=moves, other=-1).to(tl.int64)
    tl.store(v2p_ptr + v, dst, mask=moves)
    tl.store(p2v_ptr + dst, v, mask=moves)
    tl.store(p2v_ptr + src, -1, mask=moves)
    tl.store(moved_ptr + offs, tl.where(moves, v, -1), mask=mask)


REMAP_BLOCK = 256


def remap_packed_pages(
    src: torch.Tensor, dst: torch.Tensor, v2p: torch.Tensor, p2v: torch.Tensor
) -> torch.Tensor:
    """Move the bindings of `pack_pairs`' pairs in both tables, in place;
    return the moved virtual pages (-1 for a padded pair)."""
    n = int(src.numel())
    if n == 0:
        return torch.empty(0, dtype=torch.int64, device=src.device)
    if not src.is_cuda:
        # Pure-torch CPU reference; a padded pair adds 0 to the sink's row,
        # since a boolean mask would sync with the host.
        moves = src != dst
        v = torch.where(moves, p2v[src], -1)
        rows = v.clamp(min=0)
        v2p.index_put_((rows,), torch.where(moves, dst - v2p[rows], 0), accumulate=True)
        p2v.index_put_((dst,), torch.where(moves, v - p2v[dst], 0), accumulate=True)
        p2v.index_put_((src,), torch.where(moves, -1 - p2v[src], 0), accumulate=True)
        return v
    moved = torch.empty(n, dtype=torch.int64, device=src.device)
    remap_packed_pages_kernel[(triton.cdiv(n, REMAP_BLOCK),)](
        src, dst, v2p, p2v, moved, n, BLOCK=REMAP_BLOCK
    )
    return moved
