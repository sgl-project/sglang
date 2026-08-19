"""Build the aligned route in three kernels -- fused key + histogram, padded
scan, block labels + pair scatter -- with no memset, no per-call metadata, and
no bucket ceiling.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.lora.moe.routing import virtual_expert_ids_inline
from sglang.srt.lora.moe.workspace import MoeLoraWorkspace

HIST_BLOCK = 512
HIST_WARPS = 8
EXPAND_BLOCK = 128
EXPAND_WARPS = 4
SCAN_CHUNK = 2048
SCAN_WARPS = 4


@triton.jit
def _fused_hist_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    counts_ptr,
    num_pairs,
    routed_expert_id_bound,
    NUM_BUCKETS: tl.constexpr,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    BLOCK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Histogram the fused key over pairs; counts arrive zeroed, no memset pass."""
    pair_ids = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    pair_mask = pair_ids < num_pairs
    virtual_ids = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_lora_mapping_ptr,
        pair_ids,
        pair_mask,
        routed_expert_id_bound,
        LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        SHARED_OUTER=SHARED_OUTER,
    )
    # Invalid pairs land in the sentinel bucket at NUM_BUCKETS - 1; its blocks
    # are labelled -1 downstream so LoRA-A skips them and B zero-fills them.
    buckets = tl.where(virtual_ids < 0, NUM_BUCKETS - 1, virtual_ids)
    tl.atomic_add(counts_ptr + buckets, 1, mask=pair_mask)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _padded_scan_kernel(
    counts_ptr,
    block_cum_ptr,
    cursor_ptr,
    bucket_end_ptr,
    num_pairs_post_padded_ptr,
    num_buckets,
    BLOCK_SIZE_M: tl.constexpr,
    CHUNK: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Exclusive scan of block-padded bucket sizes; re-zeroes counts as it reads."""
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    running = 0
    for base in range(0, num_buckets, CHUNK):
        offs = base + tl.arange(0, CHUNK)
        mask = offs < num_buckets
        counts = tl.load(counts_ptr + offs, mask=mask, other=0)
        tl.store(counts_ptr + offs, 0, mask=mask)
        blocks = (counts + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        block_start = running + tl.cumsum(blocks) - blocks
        tl.store(block_cum_ptr + offs, block_start, mask=mask)
        slot_start = block_start * BLOCK_SIZE_M
        tl.store(cursor_ptr + offs, slot_start, mask=mask)
        # bucket_end must be materialized here; counts are gone by the fill.
        tl.store(bucket_end_ptr + offs, slot_start + counts, mask=mask)
        running += tl.sum(blocks)
    tl.store(block_cum_ptr + num_buckets, running)
    tl.store(num_pairs_post_padded_ptr, running * BLOCK_SIZE_M)
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def _expand_and_scatter_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    cursor_ptr,
    bucket_end_ptr,
    block_cum_ptr,
    sorted_pair_ids_ptr,
    block_virtual_expert_ids_ptr,
    num_pairs,
    routed_expert_id_bound,
    num_blocks,
    num_block_programs,
    NUM_BUCKETS: tl.constexpr,
    NUM_VIRTUAL: tl.constexpr,
    LORA_EXPERTS_PER_ADAPTER: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    TOP_K: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Both scan consumers on one grid, split by program id: labels, then scatter."""
    pid = tl.program_id(0)
    if pid < num_block_programs:
        block_ids = pid * BLOCK + tl.arange(0, BLOCK)
        block_mask = block_ids < num_blocks
        if USE_PDL:
            tl.extra.cuda.gdc_wait()
        low = tl.zeros(block_ids.shape, dtype=tl.int32)
        high = tl.full(block_ids.shape, NUM_BUCKETS, dtype=tl.int32)
        for _ in range(SEARCH_STEPS):
            mid = (low + high) // 2
            bound = tl.load(
                block_cum_ptr + tl.minimum(mid + 1, NUM_BUCKETS),
                mask=block_mask,
                other=0,
            )
            take = block_ids >= bound
            low = tl.where(take & (low < high), mid + 1, low)
            high = tl.where(take | (low >= high), high, mid)
        owner = tl.minimum(low, NUM_BUCKETS - 1)
        total_blocks = tl.load(block_cum_ptr + NUM_BUCKETS)
        in_plan = block_mask & (block_ids < total_blocks)
        labelled = in_plan & (owner < NUM_VIRTUAL)
        tl.store(
            block_virtual_expert_ids_ptr + block_ids,
            tl.where(labelled, owner, -1),
            mask=block_mask,
        )
        # Coalesced 2D fill of every in-plan block's padding tail, sentinel
        # bucket included (a -1 block's slots ARE read by the aligned B kernels).
        real_end = tl.load(bucket_end_ptr + owner, mask=in_plan, other=0)
        slots = block_ids[:, None] * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[None, :]
        tl.store(
            sorted_pair_ids_ptr + slots,
            num_pairs,
            mask=in_plan[:, None] & (slots >= real_end[:, None]),
        )
        return

    # Recomputing the key beats a [T, K] round trip, and needs no scan -- so it
    # runs before the wait below.
    pair_ids = (pid - num_block_programs) * BLOCK + tl.arange(0, BLOCK)
    pair_mask = pair_ids < num_pairs
    virtual_ids = virtual_expert_ids_inline(
        topk_ids_ptr,
        token_lora_mapping_ptr,
        pair_ids,
        pair_mask,
        routed_expert_id_bound,
        LORA_EXPERTS_PER_ADAPTER=LORA_EXPERTS_PER_ADAPTER,
        MAX_LORAS=MAX_LORAS,
        TOP_K=TOP_K,
        SHARED_OUTER=SHARED_OUTER,
    )
    buckets = tl.where(virtual_ids < 0, NUM_BUCKETS - 1, virtual_ids)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    slots = tl.atomic_add(cursor_ptr + buckets, 1, mask=pair_mask)
    tl.store(sorted_pair_ids_ptr + slots, pair_ids, mask=pair_mask)


def fused_align_block_size(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    *,
    num_local_experts: int,
    is_shared_outer: bool,
    max_loras: int,
    block_size: int,
    capacity: int,
    workspace: MoeLoraWorkspace,
    tensor_prefix: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the aligned route: sorted_pair_ids, block_ids, padded_pair_count."""
    device = topk_ids.device
    num_pairs = topk_ids.numel()
    top_k = topk_ids.shape[1]
    lora_experts_per_adapter = 1 if is_shared_outer else num_local_experts
    num_virtual = lora_experts_per_adapter * max_loras
    num_buckets = num_virtual + 1
    if num_buckets >= 2**31 or capacity >= 2**31:
        raise ValueError(
            f"fused align uses int32 plan math: num_buckets={num_buckets} and "
            f"capacity={capacity} must both be < 2**31"
        )
    num_blocks = capacity // block_size

    counts = workspace.tensor(
        f"{tensor_prefix}:counts",
        (num_buckets,),
        dtype=torch.int32,
        device=device,
        zero_on_first_allocation=True,
    )
    block_cumulative = workspace.tensor(
        f"{tensor_prefix}:block_cumulative",
        (num_buckets + 1,),
        dtype=torch.int32,
        device=device,
    )
    cursor = workspace.tensor(
        f"{tensor_prefix}:cursor", (num_buckets,), dtype=torch.int32, device=device
    )
    bucket_end = workspace.tensor(
        f"{tensor_prefix}:bucket_end", (num_buckets,), dtype=torch.int32, device=device
    )
    num_pairs_post_padded = workspace.tensor(
        f"{tensor_prefix}:padded_pairs", (1,), dtype=torch.int32, device=device
    )
    sorted_pair_ids = torch.empty(capacity, dtype=torch.int32, device=device)
    block_virtual_expert_ids = torch.empty(num_blocks, dtype=torch.int32, device=device)

    from sglang.kernels.jit.utils import is_arch_support_pdl

    use_pdl = is_arch_support_pdl()
    pdl_kwargs = {"launch_pdl": True} if use_pdl else {}

    _fused_hist_kernel[(triton.cdiv(max(num_pairs, 1), HIST_BLOCK),)](
        topk_ids,
        token_lora_mapping,
        counts,
        num_pairs,
        num_local_experts,
        NUM_BUCKETS=num_buckets,
        LORA_EXPERTS_PER_ADAPTER=lora_experts_per_adapter,
        MAX_LORAS=max_loras,
        TOP_K=top_k,
        SHARED_OUTER=is_shared_outer,
        BLOCK=HIST_BLOCK,
        USE_PDL=use_pdl,
        num_warps=HIST_WARPS,
    )
    _padded_scan_kernel[(1,)](
        counts,
        block_cumulative,
        cursor,
        bucket_end,
        num_pairs_post_padded,
        num_buckets,
        BLOCK_SIZE_M=block_size,
        CHUNK=SCAN_CHUNK,
        USE_PDL=use_pdl,
        num_warps=SCAN_WARPS,
        **pdl_kwargs,
    )
    num_block_programs = triton.cdiv(max(num_blocks, 1), EXPAND_BLOCK)
    num_pair_programs = triton.cdiv(max(num_pairs, 1), EXPAND_BLOCK)
    _expand_and_scatter_kernel[(num_block_programs + num_pair_programs,)](
        topk_ids,
        token_lora_mapping,
        cursor,
        bucket_end,
        block_cumulative,
        sorted_pair_ids,
        block_virtual_expert_ids,
        num_pairs,
        num_local_experts,
        num_blocks,
        num_block_programs,
        NUM_BUCKETS=num_buckets,
        NUM_VIRTUAL=num_virtual,
        LORA_EXPERTS_PER_ADAPTER=lora_experts_per_adapter,
        MAX_LORAS=max_loras,
        TOP_K=top_k,
        SHARED_OUTER=is_shared_outer,
        BLOCK=EXPAND_BLOCK,
        BLOCK_SIZE_M=block_size,
        # The search picks one of NUM_BUCKETS + 1 answers, so it needs
        # num_buckets.bit_length() steps -- one fewer and a sentinel reads as 0.
        SEARCH_STEPS=num_buckets.bit_length(),
        USE_PDL=use_pdl,
        num_warps=EXPAND_WARPS,
        **pdl_kwargs,
    )
    return sorted_pair_ids, block_virtual_expert_ids, num_pairs_post_padded
