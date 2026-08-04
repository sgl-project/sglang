from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_xpu

_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_xpu = is_xpu()
_is_musa = is_musa()

if _is_cuda or _is_hip or _is_xpu or _is_musa:
    from sglang.kernels.ops.moe import moe_align_block_size as sgl_moe_align_block_size


@triton.jit
def _deterministic_align_count(
    topk_ids_ptr,
    chunk_counts_ptr,
    num_entries: tl.constexpr,
    num_buckets: tl.constexpr,
    entries_per_chunk: tl.constexpr,
    ignore_invalid_expert: tl.constexpr,
):
    chunk_id = tl.program_id(0)
    chunk_start = chunk_id * entries_per_chunk
    row_offset = (chunk_id + 1) * num_buckets

    for bucket in range(num_buckets):
        tl.store(chunk_counts_ptr + row_offset + bucket, 0)

    for local_offset in range(entries_per_chunk):
        route_index = chunk_start + local_offset
        if route_index < num_entries:
            route = tl.load(topk_ids_ptr + route_index)
            valid = True
            if ignore_invalid_expert:
                valid = route >= 0
            if valid:
                bucket = route + 1
                count_offset = row_offset + bucket
                count = tl.load(chunk_counts_ptr + count_offset)
                tl.store(chunk_counts_ptr + count_offset, count + 1)


@triton.jit
def _deterministic_align_chunk_prefix(
    chunk_counts_ptr,
    num_buckets: tl.constexpr,
):
    bucket = tl.program_id(0)
    prefix = 0
    for chunk in range(1, num_buckets + 1):
        offset = chunk * num_buckets + bucket
        prefix += tl.load(chunk_counts_ptr + offset)
        tl.store(chunk_counts_ptr + offset, prefix)


@triton.jit
def _deterministic_align_expert_prefix(
    chunk_counts_ptr,
    expert_offsets_ptr,
    num_tokens_post_pad_ptr,
    num_buckets: tl.constexpr,
    block_size: tl.constexpr,
):
    offset = 0
    tl.store(expert_offsets_ptr, 0)
    totals_offset = num_buckets * num_buckets
    for bucket in range(num_buckets):
        count = tl.load(chunk_counts_ptr + totals_offset + bucket)
        offset += tl.cdiv(count, block_size) * block_size
        tl.store(expert_offsets_ptr + bucket + 1, offset)
    tl.store(num_tokens_post_pad_ptr, offset)


@triton.jit
def _deterministic_align_scatter(
    topk_ids_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    chunk_counts_ptr,
    expert_offsets_ptr,
    num_entries: tl.constexpr,
    num_buckets: tl.constexpr,
    block_size: tl.constexpr,
    entries_per_chunk: tl.constexpr,
    ignore_invalid_expert: tl.constexpr,
):
    chunk_id = tl.program_id(0)
    bucket = chunk_id

    expert_start = tl.load(expert_offsets_ptr + bucket)
    expert_end = tl.load(expert_offsets_ptr + bucket + 1)
    for block_start in range(expert_start, expert_end, block_size):
        tl.store(expert_ids_ptr + block_start // block_size, bucket - 1)

    chunk_start = chunk_id * entries_per_chunk
    row_offset = chunk_id * num_buckets
    for local_offset in range(entries_per_chunk):
        route_index = chunk_start + local_offset
        if route_index < num_entries:
            route = tl.load(topk_ids_ptr + route_index)
            route_bucket = route + 1
            valid = True
            if ignore_invalid_expert:
                valid = route >= 0
            if valid:
                count_offset = row_offset + route_bucket
                route_rank = tl.load(chunk_counts_ptr + count_offset)
                output_offset = route_rank + tl.load(
                    expert_offsets_ptr + route_bucket
                )
                tl.store(sorted_token_ids_ptr + output_offset, route_index)
                tl.store(chunk_counts_ptr + count_offset, route_rank + 1)


def _moe_align_block_size_deterministic(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    ignore_invalid_expert: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_entries = topk_ids.numel()
    num_buckets = num_experts + 1
    max_num_tokens_padded = num_entries + num_buckets * (block_size - 1)
    if num_entries < num_buckets:
        max_num_tokens_padded = num_entries * block_size

    sorted_ids = torch.full(
        (max_num_tokens_padded,),
        num_entries,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.full(
        (triton.cdiv(max_num_tokens_padded, block_size),),
        -1,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens_post_pad = torch.empty(
        (1,), dtype=torch.int32, device=topk_ids.device
    )
    chunk_counts = torch.zeros(
        (num_buckets + 1, num_buckets),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_offsets = torch.zeros(
        (num_buckets + 1,), dtype=torch.int32, device=topk_ids.device
    )
    entries_per_chunk = triton.cdiv(num_entries, num_buckets)

    _deterministic_align_count[(num_buckets,)](
        topk_ids,
        chunk_counts,
        num_entries,
        num_buckets,
        entries_per_chunk,
        ignore_invalid_expert,
    )
    _deterministic_align_chunk_prefix[(num_buckets,)](
        chunk_counts,
        num_buckets,
    )
    _deterministic_align_expert_prefix[(1,)](
        chunk_counts,
        expert_offsets,
        num_tokens_post_pad,
        num_buckets,
        block_size,
    )
    _deterministic_align_scatter[(num_buckets,)](
        topk_ids,
        sorted_ids,
        expert_ids,
        chunk_counts,
        expert_offsets,
        num_entries,
        num_buckets,
        block_size,
        entries_per_chunk,
        ignore_invalid_expert,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    ignore_invalid_expert: bool = False,
    deterministic: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    - deterministic: Whether to preserve a stable within-expert route order.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    if deterministic:
        return _moe_align_block_size_deterministic(
            topk_ids,
            block_size,
            num_experts,
            ignore_invalid_expert,
        )

    # ===== TO BE REFACTORED ====
    if _SGLANG_EXPERIMENTAL_LORA_OPTI:
        from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs

        if lora_envs.SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN.get() and num_experts <= 8191:
            from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
                _align_block_size_jit,
            )

            return _align_block_size_jit(topk_ids, block_size, num_experts)
    # ===== END TO BE REFACTORED ====

    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)

    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    # In EP, expert_ids for filtered experts are -1. We have num_experts + 1 ids in total.
    cumsum_buffer = torch.empty(
        (num_experts + 2,), dtype=torch.int32, device=topk_ids.device
    )

    # ===== TO BE REFACTORED ====
    use_jit_align = False
    if _SGLANG_EXPERIMENTAL_LORA_OPTI:
        from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs

        use_jit_align = lora_envs.SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN.get()
    if use_jit_align:
        from sglang.kernels.ops.moe.moe_align import (
            moe_align_block_size as jit_moe_align_block_size,
        )

        jit_moe_align_block_size(
            topk_ids,
            num_experts + 1,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            True,
        )
    # ===== END TO BE REFACTORED ====
    else:
        sgl_moe_align_block_size(
            topk_ids,
            num_experts + 1,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            True,
            ignore_invalid_expert,
        )
    return sorted_ids, expert_ids, num_tokens_post_pad
