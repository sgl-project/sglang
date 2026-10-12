"""Single-launch moe_align for tiny batches with many experts.

The CUDA small-batch align kernel is gated to ``num_experts <= 64`` (its shared
memory grows as O(threads x experts)), so bs=1 decode on a MoE with a wider
expert dimension always paid the generic two-kernel (align + count_and_sort)
path. This kernel covers that corner in a single launch, at any expert count,
for ``numel <= SMALL_NUMEL_LIMIT``.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl

# Largest numel routed to this kernel. Its [NP, NP] pairwise tensors fit in
# registers at NP=64 (~4 us, on par with the two CUDA launches it replaces) but
# spill to local memory at NP=256 (~230 us measured).
SMALL_NUMEL_LIMIT = 64


@triton.jit
def _moe_align_small_numel_kernel(
    topk_ids_ptr,  # [numel] int, flattened (token, slot) expert ids, -1 = filtered
    sorted_token_ids_ptr,  # [max_num_tokens_padded] int32
    expert_ids_ptr,  # [max_num_m_blocks] int32
    num_tokens_post_pad_ptr,  # [1] int32
    num_experts,  # E + 1 (the "+1 offset" convention's bucket count)
    block_size,
    numel,
    NP: tl.constexpr,  # power-of-2 >= numel
    NB: tl.constexpr,  # power-of-2 >= max blocks used
    USE_GDC: tl.constexpr = False,
):
    """Single-CTA moe_align for tiny batches with MANY experts.

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
        # Consumer side of the router top-k that produced topk_ids.
        tl.extra.cuda.gdc_wait()

    offs_p = tl.arange(0, NP)
    mask_p = offs_p < numel
    ids = tl.load(topk_ids_ptr + offs_p, mask=mask_p, other=-2)
    # Padded lanes get an out-of-range bucket and are masked out everywhere.
    bucket = tl.where(mask_p, (ids + 1).to(tl.int32), num_experts)

    # Pairwise stats: stable rank within the bucket and bucket population.
    same = (bucket[None, :] == bucket[:, None]) & mask_p[None, :] & mask_p[:, None]
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

    if USE_GDC:
        tl.extra.cuda.gdc_launch_dependents()


def moe_align_small_numel(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    """Align and sort expert token ids into block-padded buffers, in one launch.

    Buffer contract matches ``sglang.kernels.ops.moe.moe_align_block_size``
    (minus its ``cumsum_buffer``, which a single CTA does not need):
    ``num_experts`` is the bucket count ``E + 1`` under the "+1 offset"
    convention, and the three output buffers are written in place.

    Callers gate on ``topk_ids.numel() <= SMALL_NUMEL_LIMIT``; the kernel stays
    correct above it, but its pairwise tensors spill and it stops being faster
    than the two-kernel path.
    """
    numel = topk_ids.numel()
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    _moe_align_small_numel_kernel[(1,)](
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        num_experts,
        block_size,
        numel,
        NP=triton.next_power_of_2(max(numel, 2)),
        NB=triton.next_power_of_2(max(expert_ids.numel(), 2)),
        num_warps=4,
        **pdl_kwargs,
    )
