"""Paged experts, fused host copy for the Triton MoE kernel.

``fused_moe_kernel`` calls the two kernel functions below from blocks that exist only when it is
given ``b_host_ptr``, the copy of every expert in pinned host memory, in B's layout. B then
holds only some experts ("slots"), and ``HOST_IDS_START`` is their number. An expert id below
``HOST_IDS_START`` is a slot, as without the host copy. An id of ``HOST_IDS_START + e`` is
expert e being paged in: the kernel reads it from the host copy and writes it into slot
``expert_slot_ptr[e]`` of B, which no other block reads in this launch.
``missing_expert_count_ptr[0]`` is the number of such experts in the launch.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


def paged_expert_kernel_kwargs(
    B: torch.Tensor, paged_expert_info: Optional[dict]
) -> dict:
    """``fused_moe_kernel``'s arguments for paging experts into the slot table ``B``, none
    without ``paged_expert_info``: ``host_table`` (every expert in pinned host memory, in B's
    layout), ``expert_slot`` (the layer's expert -> slot table) and ``missing_expert_count``
    (the number of experts its current step pages in)."""
    if paged_expert_info is None:
        return {}
    host_table = paged_expert_info["host_table"]
    assert host_table.shape[1:] == B.shape[1:], (
        "--paged-experts-fused-host-copy: the shape of one expert in the host table, "
        f"{tuple(host_table.shape[1:])}, does not match the slot table's, "
        f"{tuple(B.shape[1:])}"
    )
    assert host_table.dtype == B.dtype, (
        "--paged-experts-fused-host-copy: the dtype of the host table, "
        f"{host_table.dtype}, does not match the slot table's, {B.dtype}"
    )
    return {
        "HOST_IDS_START": B.shape[0],
        "b_host_ptr": host_table,
        "expert_slot_ptr": paged_expert_info["expert_slot"],
        "missing_expert_count_ptr": paged_expert_info["missing_expert_count"],
    }


@triton.jit
def spread_missing_expert_blocks(
    pid_m, num_tokens_post_padded, missing_expert_count_ptr, BLOCK_SIZE_M: tl.constexpr
):
    """The block of tokens this thread block takes. The experts read from host memory have the
    highest ids, so their blocks of tokens come last in the sorted order. We spread them evenly
    through the launch, so that the host reads overlap with the work on resident experts."""
    groups = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    hosts = tl.minimum(tl.load(missing_expert_count_ptr), groups)
    if pid_m < groups:
        before = (pid_m * hosts + groups - 1) // groups
        if ((pid_m + 1) * hosts + groups - 1) // groups > before:
            pid_m = groups - hosts + before
        else:
            pid_m = pid_m - before
    return pid_m


@triton.jit
def matmul_with_host_expert(
    accumulator,
    a_ptrs,
    token_mask,
    K,
    stride_ak,
    b_ptr,
    b_host_ptr,
    expert_slot_ptr,
    off_experts,
    offs_k,
    offs_bn,
    stride_be,
    stride_bk,
    stride_bn,
    HOST_IDS_START: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """The kernel's K loop for a block that reads its expert from host memory: it writes each
    tile into the expert's slot as it arrives, and returns 0 for K, so that the kernel's own
    loop has nothing left to do. It is a separate loop so that blocks for resident experts do not
    include any store logic for expert weights."""
    if off_experts >= HOST_IDS_START:
        expert = off_experts - HOST_IDS_START
        slot = tl.load(expert_slot_ptr + expert).to(tl.int64)
        b_tile = offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b_ptrs = b_host_ptr + expert * stride_be + b_tile
        b_copy_ptrs = b_ptr + slot * stride_be + b_tile
        for k_start in range(0, K, BLOCK_SIZE_K):
            if even_Ks:
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                b = tl.load(b_ptrs)
                tl.store(b_copy_ptrs, b)
            else:
                k_mask = offs_k < K - k_start
                a = tl.load(
                    a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0
                )
                b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
                tl.store(b_copy_ptrs, b, mask=k_mask[:, None])
            accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            b_copy_ptrs += BLOCK_SIZE_K * stride_bk
        K = K * 0
    return accumulator, K
