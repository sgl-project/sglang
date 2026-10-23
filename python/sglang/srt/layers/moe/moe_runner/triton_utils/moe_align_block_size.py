from __future__ import annotations

from typing import Tuple

import torch
import triton

from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_xpu

_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_xpu = is_xpu()
_is_musa = is_musa()

if _is_cuda or _is_hip or _is_xpu or _is_musa:
    from sglang.kernels.ops.moe import moe_align_block_size as sgl_moe_align_block_size

if _is_cuda:
    from sglang.kernels.ops.moe.moe_align_small_numel import (
        SMALL_NUMEL_LIMIT,
        moe_align_small_numel,
    )

# Where the CUDA kernel's own small-batch single-block path stops: its
# per-thread histogram costs 4 * (buckets + 1) ** 2 bytes of shared memory.
_CUDA_SMALL_BATCH_MAX_BUCKETS = 64


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    ignore_invalid_expert: bool = False,
    *,
    return_inactive_prefix_count: bool = False,
    pad_expert_ids: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size. Opting into ``return_inactive_prefix_count``
        returns int32 ``[padded_tokens, inactive_blocks]`` from a supporting local
        producer; the second word counts negative physical blocks, not tokens.

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
    if pad_expert_ids and (not _is_cuda or ignore_invalid_expert):
        raise ValueError(
            "Padding expert IDs requires CUDA and the invalid-expert bucket"
        )

    if return_inactive_prefix_count:
        if not _is_cuda or ignore_invalid_expert or num_experts > 8191:
            raise ValueError(
                "Inactive-prefix counts require CUDA alignment with at most 8191 experts and the +1 bucket convention"
            )
    # ===== TO BE REFACTORED ====
    if _SGLANG_EXPERIMENTAL_LORA_OPTI and not pad_expert_ids:
        from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs

        if lora_envs.SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN.get() and num_experts <= 8191:
            from sglang.kernels.ops.lora.moe.virtual_experts import (
                _align_block_size_jit,
            )

            if return_inactive_prefix_count:
                return _align_block_size_jit(
                    topk_ids,
                    block_size,
                    num_experts,
                    return_inactive_prefix_count=return_inactive_prefix_count,
                    map_invalid_experts_to_sentinel=False,
                )
            return _align_block_size_jit(
                topk_ids,
                block_size,
                num_experts,
                map_invalid_experts_to_sentinel=False,
            )
    # ===== END TO BE REFACTORED ====

    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    # The align kernel pads sorted_ids with int4 stores.
    max_num_tokens_padded = (max_num_tokens_padded + 3) // 4 * 4

    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty(
        (2 if return_inactive_prefix_count else 1,),
        dtype=torch.int32,
        device=topk_ids.device,
    )

    # In EP, expert_ids for filtered experts are -1. We have num_experts + 1 ids in total.
    cumsum_buffer = torch.empty(
        (num_experts + 2,), dtype=torch.int32, device=topk_ids.device
    )

    # Tiny-batch fast path (bs=1 decode): one single-CTA triton launch replaces
    # the generic align + count_and_sort pair, covering the corner the CUDA
    # small-batch kernel cannot reach. Below that bucket limit the CUDA kernel
    # is already a single launch and does O(numel) work where this one does
    # O(numel ** 2) pairwise, so leave that side to it. ignore_invalid_expert is
    # a different contract from the "+1 offset" convention this kernel implements.
    if (
        _is_cuda
        and topk_ids.numel() <= SMALL_NUMEL_LIMIT
        and num_experts + 1 > _CUDA_SMALL_BATCH_MAX_BUCKETS
        and not ignore_invalid_expert
        and not pad_expert_ids
    ):
        moe_align_small_numel(
            topk_ids,
            num_experts + 1,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            write_inactive_prefix_count=return_inactive_prefix_count,
        )
        return sorted_ids, expert_ids, num_tokens_post_pad

    # ===== TO BE REFACTORED ====
    use_jit_align = return_inactive_prefix_count or pad_expert_ids
    if _SGLANG_EXPERIMENTAL_LORA_OPTI:
        from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs

        use_jit_align = (
            use_jit_align or lora_envs.SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN.get()
        )
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
            write_inactive_prefix_count=return_inactive_prefix_count,
            pad_expert_ids=pad_expert_ids,
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
