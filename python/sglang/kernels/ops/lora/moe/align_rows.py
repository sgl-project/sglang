"""Align routed pairs for Triton and Marlin, dropping negative expert IDs.

Distinct top-k IDs allow a single-token fast path. pair_to_row maps pair p
to row p, or -1 for a pair without an expert.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _align_single_token_kernel(
    topk_ids_ptr,  # int32 [TOPK]
    sorted_ids_ptr,  # int32 [TOPK * BLOCK_SIZE]
    expert_ids_ptr,  # int32 [TOPK]
    num_post_ptr,  # int32 [1]
    pair_to_row_ptr,  # int32 [TOPK] out: pair p is row p, -1 without an expert
    WRITE_PAIR_TO_ROW: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    LANES: tl.constexpr,  # next pow2 >= TOPK
    SLOTS: tl.constexpr,  # next pow2 >= TOPK * BLOCK_SIZE
):
    lane = tl.arange(0, LANES)
    in_range = lane < TOPK
    ids = tl.load(topk_ids_ptr + lane, mask=in_range, other=-1)
    valid = ids >= 0
    # Dropped lanes sort past every valid id; their ties break on lane.
    key = tl.where(valid, ids, 2147483647)
    before = (key[None, :] < key[:, None]) | (
        (key[None, :] == key[:, None]) & (lane[None, :] < lane[:, None])
    )
    rank = tl.sum(before.to(tl.int32), axis=1)
    n_valid = tl.sum(valid.to(tl.int32), axis=0)

    # For one token, the lane is the flat pair index.
    tl.store(expert_ids_ptr + rank, ids, mask=valid)
    tl.store(sorted_ids_ptr + rank * BLOCK_SIZE, lane, mask=valid)
    tl.store(
        expert_ids_ptr + lane,
        tl.full([LANES], -1, tl.int32),
        mask=in_range & (lane >= n_valid),
    )
    # Padding uses numel (= TOPK), matching the general alignment path.
    slot = tl.arange(0, SLOTS)
    pad = (slot < TOPK * BLOCK_SIZE) & (
        (slot % BLOCK_SIZE != 0) | (slot // BLOCK_SIZE >= n_valid)
    )
    tl.store(sorted_ids_ptr + slot, tl.full([SLOTS], TOPK, tl.int32), mask=pad)
    tl.store(num_post_ptr, n_valid * BLOCK_SIZE)
    if WRITE_PAIR_TO_ROW:
        tl.store(pair_to_row_ptr + lane, tl.where(valid, lane, -1), mask=in_range)


def moe_align_single_token(
    topk_ids: torch.Tensor, block_size: int, pair_to_row: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align distinct top-k IDs for one token; accepts int32 [1, topk], topk <= 32."""
    topk = topk_ids.shape[1]
    device = topk_ids.device
    sorted_ids = torch.empty((topk * block_size,), dtype=torch.int32, device=device)
    expert_ids = torch.empty((topk,), dtype=torch.int32, device=device)
    num_post = torch.empty((1,), dtype=torch.int32, device=device)
    _align_single_token_kernel[(1,)](
        topk_ids,
        sorted_ids,
        expert_ids,
        num_post,
        topk_ids if pair_to_row is None else pair_to_row,
        WRITE_PAIR_TO_ROW=pair_to_row is not None,
        TOPK=topk,
        BLOCK_SIZE=block_size,
        LANES=triton.next_power_of_2(topk),
        SLOTS=triton.next_power_of_2(topk * block_size),
        num_warps=1,
    )
    return sorted_ids, expert_ids, num_post


@triton.jit
def _pair_to_row_kernel(topk_ids_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    ids = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
    tl.store(out_ptr + offs, tl.where(ids >= 0, offs, -1).to(tl.int32), mask=mask)


def pair_to_row_map(topk_ids: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Write out[p] = p for routed pairs, else -1; finalize gates on this map."""
    flat = topk_ids.reshape(-1)
    n = flat.numel()
    block = 1024
    _pair_to_row_kernel[(triton.cdiv(n, block),)](
        flat, out, n, BLOCK=block, num_warps=4
    )
    return out
