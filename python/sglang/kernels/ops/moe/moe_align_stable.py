"""Block-padded expert routing with a stable order within each expert."""

import torch
import triton
import triton.language as tl


@triton.jit
def _expert_offsets(
    sorted_experts,
    token_offsets,
    padded_offsets,
    total_tokens,
    num_pairs,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    WRITE_INACTIVE_PREFIX: tl.constexpr,
):
    buckets = tl.arange(0, BLOCK_E)
    # Bucket zero contains filtered (-1) experts. The last boundary is the
    # lower bound of NUM_EXPERTS, immediately after the final valid expert.
    lo = tl.full((BLOCK_E,), 0, tl.int32)
    hi = tl.full((BLOCK_E,), num_pairs, tl.int32)
    for _ in range(SEARCH_STEPS):
        mid = (lo + hi) // 2
        value = tl.load(
            sorted_experts + mid,
            mask=(mid < num_pairs) & (buckets <= NUM_EXPERTS + 1),
            other=NUM_EXPERTS,
        )
        right = value < buckets - 1
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    next_lo = tl.gather(lo, tl.minimum(buckets + 1, BLOCK_E - 1), 0)
    counts = tl.where(buckets <= NUM_EXPERTS, next_lo - lo, 0)
    padded_counts = tl.cdiv(counts, BLOCK_M) * BLOCK_M
    starts = tl.cumsum(padded_counts, 0) - padded_counts
    tl.store(token_offsets + buckets, lo, buckets <= NUM_EXPERTS + 1)
    tl.store(padded_offsets + buckets, starts, buckets <= NUM_EXPERTS + 1)
    tl.store(total_tokens, tl.sum(padded_counts))
    if WRITE_INACTIVE_PREFIX:
        inactive_blocks = tl.sum(tl.where(buckets == 0, padded_counts // BLOCK_M, 0))
        tl.store(total_tokens + 1, inactive_blocks)


@triton.jit
def _scatter_tokens(
    sorted_pairs,
    token_offsets,
    padded_offsets,
    sorted_token_ids,
    expert_ids,
    num_pairs,
    BLOCK_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bucket = tl.program_id(0)
    start = tl.load(token_offsets + bucket)
    count = tl.load(token_offsets + bucket + 1) - start
    padded_start = tl.load(padded_offsets + bucket)
    padded_count = tl.load(padded_offsets + bucket + 1) - padded_start
    for base in range(0, padded_count, BLOCK):
        positions = base + tl.arange(0, BLOCK)
        pairs = tl.load(
            sorted_pairs + start + positions,
            positions < count,
            other=num_pairs,
        ).to(tl.int32)
        tl.store(
            sorted_token_ids + padded_start + positions,
            pairs,
            positions < padded_count,
        )
        tl.store(
            expert_ids + (padded_start + positions) // BLOCK_M,
            bucket - 1,
            (positions < padded_count) & (positions % BLOCK_M == 0),
        )


def moe_align_block_size_stable(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    *,
    return_inactive_prefix_count: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group pairs by expert, preserving their flattened token/slot order.

    Expert IDs must be in ``[-1, num_experts)``. Filtered experts retain their
    leading blocks with expert ID -1. Padding within the published token
    count contains ``topk_ids.numel()``; unused capacity is unspecified.
    All shapes depend only on input metadata, so the operation supports CUDA
    graph capture without reading counts back to the host.
    """
    assert topk_ids.is_cuda and topk_ids.dtype in (torch.int32, torch.int64)
    assert topk_ids.ndim == 2 and block_size > 0 and num_experts > 0
    num_pairs = topk_ids.numel()
    capacity = min(
        num_pairs * block_size,
        num_pairs + (num_experts + 1) * (block_size - 1),
    )
    capacity = triton.cdiv(capacity, 4) * 4
    device = topk_ids.device
    sorted_ids = torch.empty(capacity, dtype=torch.int32, device=device)
    expert_ids = torch.empty(
        triton.cdiv(capacity, block_size), dtype=torch.int32, device=device
    )
    total = torch.empty(
        2 if return_inactive_prefix_count else 1, dtype=torch.int32, device=device
    )
    token_offsets = torch.empty(num_experts + 2, dtype=torch.int32, device=device)
    padded_offsets = torch.empty_like(token_offsets)
    sorted_experts, sorted_pairs = torch.sort(topk_ids.reshape(-1), stable=True)
    _expert_offsets[(1,)](
        sorted_experts,
        token_offsets,
        padded_offsets,
        total,
        num_pairs,
        NUM_EXPERTS=num_experts,
        BLOCK_M=block_size,
        BLOCK_E=triton.next_power_of_2(num_experts + 2),
        SEARCH_STEPS=(num_pairs + 1).bit_length(),
        WRITE_INACTIVE_PREFIX=return_inactive_prefix_count,
        num_warps=4,
    )
    _scatter_tokens[(num_experts + 1,)](
        sorted_pairs,
        token_offsets,
        padded_offsets,
        sorted_ids,
        expert_ids,
        num_pairs,
        BLOCK_M=block_size,
        BLOCK=256,
        num_warps=4,
    )
    return sorted_ids, expert_ids, total
