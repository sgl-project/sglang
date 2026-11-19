"""Paged experts, fused host copy for the Triton MoE kernel.

``fused_moe_kernel`` calls the two kernel functions below from blocks that exist only when it is
given ``b_host_ptr``, the copy of every expert in pinned host memory, in B's layout. B then
holds ``HOST_SLOTS`` experts ("slots"). An expert id below ``HOST_SLOTS`` is a slot, as without
the host copy. An id of ``HOST_SLOTS + e`` is expert e being paged in: the kernel reads it from
the host copy and writes it into slot ``expert_slot_ptr[e]`` of B, which no other block reads in
this launch. ``host_count_ptr[0]`` is the number of such experts in the launch.
"""

from __future__ import annotations

from typing import Dict

import torch
import triton
import triton.language as tl

# For a slot table (by its address): the pinned host copy of every expert, the layer's
# expert -> slot table and the number of experts the current step pages in. Set by
# sglang.srt.layers.moe.paged_experts.
HOST_COPIES: Dict[int, tuple] = {}


def host_expert_ids(B: torch.Tensor) -> int:
    """How many expert ids past its own experts the routed entries of ``B`` may use: one per
    expert in its host copy, 0 without one."""
    host = HOST_COPIES.get(B.data_ptr()) if HOST_COPIES else None
    return 0 if host is None else host[0].shape[0]


def host_copy_kwargs(B: torch.Tensor, plain: bool) -> dict:
    """``fused_moe_kernel``'s host-copy arguments for ``B``, none without a host copy.
    ``plain``: the launch has unquantized weights without bias, TMA or swapped operands."""
    host = HOST_COPIES.get(B.data_ptr()) if HOST_COPIES else None
    if host is None:
        return {}
    assert plain, (
        "Paged experts fused host copy: unquantized weights without bias or TMA"
    )
    return {
        "HOST_SLOTS": B.shape[0],
        "b_host_ptr": host[0],
        "expert_slot_ptr": host[1],
        "host_count_ptr": host[2],
    }


@triton.jit
def host_copy_block(
    pid_m, num_tokens_post_padded, host_count_ptr, BLOCK_SIZE_M: tl.constexpr
):
    """The block of tokens this thread block takes. The experts read from host memory have the
    highest ids, so their blocks of tokens come last in the sorted order. We spread them evenly
    through the launch, so that the host reads overlap with the work on resident experts."""
    groups = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    hosts = tl.minimum(tl.load(host_count_ptr), groups)
    if pid_m < groups:
        before = (pid_m * hosts + groups - 1) // groups
        if ((pid_m + 1) * hosts + groups - 1) // groups > before:
            pid_m = groups - hosts + before
        else:
            pid_m = pid_m - before
    return pid_m


@triton.jit
def host_copy_dot(
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
    HOST_SLOTS: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """The kernel's K loop for a block that reads its expert from host memory: it writes each
    tile into the expert's slot as it arrives, and returns 0 for K, so that the kernel's own
    loop has nothing left to do. It is a separate loop so that blocks for resident experts do not
    include any store logic for expert weights."""
    if off_experts >= HOST_SLOTS:
        expert = off_experts - HOST_SLOTS
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
