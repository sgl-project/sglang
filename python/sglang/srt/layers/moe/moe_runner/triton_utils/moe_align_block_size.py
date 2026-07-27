from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
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
def _moe_align_small_numel_kernel(
    topk_ids_ptr,  # [numel] int, flattened (token, slot) expert ids, -1 = filtered
    sorted_token_ids_ptr,  # [max_num_tokens_padded] int32
    expert_ids_ptr,  # [max_num_m_blocks] int32
    num_tokens_post_pad_ptr,  # [1] int32
    num_experts,  # E + 1 (the +1 offset convention's bucket count)
    block_size,
    numel,
    NP: tl.constexpr,  # power-of-2 >= numel
    NB: tl.constexpr,  # power-of-2 >= max blocks used
    USE_GDC: tl.constexpr = False,
):
    """Single-CTA moe_align for tiny batches with MANY experts.

    The CUDA small-batch kernel is gated to num_experts <= 64 (its shared
    memory grows as O(threads x experts)), so bs=1 decode on wide-expert MoEs
    (e.g. 513 experts, 45 pairs) always paid the generic two-kernel path.
    This variant replaces both launches for numel <= NP with one program.

    Everything works on the PAIR axis ([NP, NP] pairwise comparisons plus a
    rank-0 representative per bucket) -- an expert-axis formulation (histogram
    / cumsum over ~1k buckets) is ~3x more single-SM work and measured slower
    than the two-kernel path it replaces.

    Reference semantics reproduced:
    - "+1 offset" convention: expert -1 (EP-filtered) maps to bucket 0 and its
      blocks get expert_ids = -1 (skipped by fused_moe's filter_expert);
    - every bucket is padded to a block_size multiple, offsets in bucket order;
    - pad slots inside [0, num_tokens_post_pad) hold `numel`.
    Intended deviations, both invisible to fused_moe:
    - intra-bucket order is stable in pair index (the reference's atomicAdd
      order is scheduling-dependent; every pair writes its own output row);
    - sorted_token_ids beyond num_tokens_post_pad is left unwritten (the
      reference pre-fills the whole buffer; consumers only read below the
      published total).
    """
    if USE_GDC:
        # Consumer side of the PDL'd router top-k; trigger immediately so the
        # PDL'd fused_moe up-GEMM overlaps its prologue with this kernel.
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()

    offs_p = tl.arange(0, NP)
    mask_p = offs_p < numel
    ids = tl.load(topk_ids_ptr + offs_p, mask=mask_p, other=-2)
    # Padded lanes get an out-of-range bucket and are masked out everywhere.
    bucket = tl.where(mask_p, (ids + 1).to(tl.int32), num_experts)

    # Pairwise stats: stable rank within the bucket and bucket population.
    same = (
        (bucket[None, :] == bucket[:, None]) & mask_p[None, :] & mask_p[:, None]
    )
    earlier = offs_p[None, :] < offs_p[:, None]
    rank = tl.sum((same & earlier).to(tl.int32), axis=1)  # [NP]
    cnt = tl.sum(same.to(tl.int32), axis=1)  # [NP], own-bucket population
    padded_cnt = ((cnt + block_size - 1) // block_size) * block_size
    is_rep = (rank == 0) & mask_p  # one representative pair per bucket

    # Bucket-ordered exclusive offsets: sum the padded counts of every
    # representative with a strictly smaller bucket id.
    smaller_rep = (bucket[None, :] < bucket[:, None]) & is_rep[None, :]
    excl = tl.sum(smaller_rep.to(tl.int32) * padded_cnt[None, :], axis=1)  # [NP]

    total = tl.sum(tl.where(is_rep, padded_cnt, 0), axis=0)
    tl.store(num_tokens_post_pad_ptr, total.to(tl.int32))

    # expert_ids per used block: representative r owns blocks
    # [excl[r], excl[r] + padded_cnt[r]); the written id is bucket - 1
    # (bucket 0 = filtered -> -1).
    offs_b = tl.arange(0, NB)
    block_start = offs_b * block_size
    in_range = (
        (block_start[:, None] >= excl[None, :])
        & (block_start[:, None] < (excl + padded_cnt)[None, :])
        & is_rep[None, :]
    )
    eid = tl.sum(in_range.to(tl.int32) * (bucket[None, :] - 1), axis=1)
    tl.store(expert_ids_ptr + offs_b, eid.to(tl.int32), mask=block_start < total)

    # Fill the used region's pad slots with `numel`, then scatter the real
    # pair indices over them. The barrier is required: fill and scatter run on
    # different warps of this CTA, and a scatter store must not be overtaken
    # by a later-warp fill store to the same address.
    n_fill = (total + NP - 1) // NP
    for it in range(n_fill):
        f_offs = it * NP + offs_p
        tl.store(
            sorted_token_ids_ptr + f_offs,
            tl.full([NP], 0, tl.int32) + numel,
            mask=f_offs < total,
        )
    tl.debug_barrier()
    pos = excl + rank
    tl.store(sorted_token_ids_ptr + pos, offs_p.to(tl.int32), mask=mask_p)


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    ignore_invalid_expert: bool = False,
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

    # Tiny-batch fast path (bs=1 decode): one single-CTA triton launch
    # replaces the generic align + count_and_sort pair. The CUDA small-batch
    # kernel only covers num_experts <= 64 (its smem grows as
    # O(threads x experts)), so e.g. 513-expert decode always paid two
    # launches here. The pair-axis kernel has no expert-count limit, but its
    # [NP, NP] pairwise tensors grow quadratically: NP=64 fits in registers
    # (~4 us, on par with the CUDA pair), while NP=256 spills to local memory
    # and measured ~230 us/launch -- hence the hard numel <= 64 gate.
    # PDL-chained on sm90+ (consumer of the router top-k, early trigger for
    # the fused_moe up-GEMM).
    num_buckets = num_experts + 1
    if _is_cuda and topk_ids.numel() <= 64:
        pdl_kwargs = (
            {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
        )
        _moe_align_small_numel_kernel[(1,)](
            topk_ids,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            num_buckets,
            block_size,
            topk_ids.numel(),
            NP=triton.next_power_of_2(max(topk_ids.numel(), 2)),
            NB=triton.next_power_of_2(max(max_num_m_blocks, 2)),
            num_warps=4,
            **pdl_kwargs,
        )
        return sorted_ids, expert_ids, num_tokens_post_pad

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
