# SPDX-License-Identifier: Apache-2.0
"""Mark full quads and causal tails with two bits per query.

Each quad uses one bit for lanes before the query's tail length and one
for the remaining lanes. A full quad enables both bits.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["GROUPS", "CAPACITY", "PACK_CAPACITY", "WIDTH"])
def _mark(
    BLOCKS,
    POS,
    SEQ_LENS,
    CU,
    MASK,
    TAILS,
    GROUPS,
    CAPACITY,
    PACK_CAPACITY,
    WIDTH,
    BLOCK: tl.constexpr,
):
    row, batch = tl.program_id(0), tl.program_id(1)
    start, end = tl.load(CU + batch), tl.load(CU + batch + 1)
    query = start + row
    cols = tl.arange(0, BLOCK)
    active = query < end
    block = tl.load(BLOCKS + query * 512 + cols, active & (cols < 512), other=-1)
    seq = tl.load(SEQ_LENS + batch).to(tl.int32)
    position = tl.load(POS + query, active, other=-1).to(tl.int32)
    visible = tl.minimum(tl.maximum(position + 1, 0), seq)
    token = block * 4
    valid = (
        active
        & (block >= 0)
        & (block < CAPACITY)
        & (token + 3 < visible)
        & (token + 3 < WIDTH)
    )
    group = row // 8
    base = (batch.to(tl.int64) * GROUPS + group) * PACK_CAPACITY
    shift = (row % 8) * 2 + (block % 2) * 16
    tl.atomic_or(MASK + base + block // 2, 3 << shift, valid, sem="relaxed")
    tail_quad = visible // 4
    tail_count = visible % 4
    tail_valid = (
        active
        & (tail_count > 0)
        & (tail_quad < CAPACITY)
        & (tail_quad * 4 < seq)
        & (tail_quad * 4 < WIDTH)
    )
    tl.store(TAILS + query, tail_count, active)
    tail_shift = (row % 8) * 2 + (tail_quad % 2) * 16
    tl.atomic_or(
        MASK + base + tail_quad // 2,
        1 << tail_shift,
        tail_valid,
        sem="relaxed",
    )


def build_block_masks(blocks, positions, seq_lens, cu_q, token_slots, max_q):
    batch = cu_q.numel() - 1
    groups = triton.cdiv(max_q, 8)
    width = token_slots.shape[1]
    capacity = triton.cdiv(width, 4)
    packed_capacity = triton.cdiv(capacity, 2)
    masks = torch.zeros(
        (batch, groups, packed_capacity), device=blocks.device, dtype=torch.int32
    )
    tails = torch.empty((blocks.shape[0],), dtype=torch.int8, device=blocks.device)
    _mark[(max_q, batch)](
        blocks,
        positions,
        seq_lens,
        cu_q,
        masks,
        tails,
        groups,
        capacity,
        packed_capacity,
        width,
        512,
        num_warps=4,
    )
    return masks, tails
