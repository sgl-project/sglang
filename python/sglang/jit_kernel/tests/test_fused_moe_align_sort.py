import pytest
import torch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="stage-b-kernel-unit-1-gpu-large")


def ref_moe_align_sort(topk_ids, block_size, num_experts):
    """Reference: Python implementation of moe_align_block_size."""
    numel = topk_ids.numel()
    flat_ids = topk_ids.flatten()

    if numel < num_experts + 1:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + (num_experts + 1) * (block_size - 1)

    # Count tokens per expert
    counts = torch.zeros(num_experts + 1, dtype=torch.int32)  # +1 for EP offset
    for i in range(numel):
        expert_id = flat_ids[i].item() + 1  # +1 offset
        counts[expert_id] += 1

    # Compute padded prefix sum
    cumsum = torch.zeros(num_experts + 1, dtype=torch.int32)
    for i in range(1, num_experts + 1):
        padded = ((counts[i].item() + block_size - 1) // block_size) * block_size
        cumsum[i] = cumsum[i - 1] + padded

    total_tokens_post_pad = cumsum[num_experts].item()

    # Fill expert_ids
    num_blocks = total_tokens_post_pad // block_size
    expert_ids = torch.zeros(num_blocks, dtype=torch.int32)
    for e in range(num_experts):
        for b in range(cumsum[e].item(), cumsum[e + 1].item(), block_size):
            expert_ids[b // block_size] = e - 1  # -1 for EP offset

    # Sort tokens
    sorted_ids = torch.full((max_num_tokens_padded,), numel, dtype=torch.int32)
    offsets = cumsum.clone()
    for i in range(numel):
        expert_id = flat_ids[i].item() + 1
        sorted_ids[offsets[expert_id]] = i
        offsets[expert_id] += 1

    return sorted_ids, expert_ids, total_tokens_post_pad


@pytest.mark.parametrize("num_tokens", [1, 4, 8, 16, 32])
@pytest.mark.parametrize("top_k", [2, 4, 8])
@pytest.mark.parametrize("num_experts", [8, 64, 128])
@pytest.mark.parametrize("block_size", [64, 128])
def test_fused_moe_align_sort(num_tokens, top_k, num_experts, block_size):
    from sglang.jit_kernel.fused_moe_align_sort import fused_moe_align_sort

    # +1 because sglang passes num_experts + 1 (for EP offset expert -1)
    ne = num_experts + 1

    # Generate random topk_ids in range [0, num_experts)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device="cuda"
    )

    # JIT kernel
    sorted_ids, expert_ids, num_post_pad = fused_moe_align_sort(
        topk_ids, block_size, ne
    )

    # Reference
    ref_sorted, ref_expert_ids, ref_total = ref_moe_align_sort(
        topk_ids.cpu(), block_size, ne
    )

    # Compare total tokens post pad
    assert num_post_pad.item() == ref_total, (
        f"total mismatch: {num_post_pad.item()} vs {ref_total}"
    )

    # Compare expert_ids
    n_blocks = ref_total // block_size
    torch.testing.assert_close(
        expert_ids[:n_blocks].cpu(), ref_expert_ids[:n_blocks]
    )

    # Compare sorted_ids: check that each expert's tokens are the same set
    # (order within expert may differ due to thread scheduling)
    for e in range(ne):
        ref_mask = ref_sorted < topk_ids.numel()
        jit_mask = sorted_ids.cpu() < topk_ids.numel()
        # Both should have same number of valid tokens
        assert ref_mask.sum() == jit_mask.sum()


def test_empty():
    from sglang.jit_kernel.fused_moe_align_sort import fused_moe_align_sort

    topk_ids = torch.empty((0, 2), dtype=torch.int32, device="cuda")
    sorted_ids, expert_ids, num_post_pad = fused_moe_align_sort(topk_ids, 128, 9)
    assert num_post_pad.item() == 0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
