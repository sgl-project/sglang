"""Align routed pairs for Triton and Marlin, dropping negative expert IDs.

For one token, distinct top-k IDs allow one block per expert in a single launch.
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


def moe_align_single_token(
    topk_ids: torch.Tensor, block_size: int
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
        TOPK=topk,
        BLOCK_SIZE=block_size,
        LANES=triton.next_power_of_2(topk),
        SLOTS=triton.next_power_of_2(topk * block_size),
        num_warps=1,
    )
    return sorted_ids, expert_ids, num_post


def align_rows(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if (
        topk_ids.shape[0] == 1
        and topk_ids.shape[1] <= 32
        and topk_ids.dtype == torch.int32
    ):
        return moe_align_single_token(topk_ids, block_size)
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    return moe_align_block_size(
        topk_ids, block_size, num_experts, ignore_invalid_expert=True
    )
