import itertools

import pytest
import torch

from sglang.jit_kernel.moe_align_block_size import moe_align_block_size


def moe_align_block_size_ref(
    topk_ids: torch.Tensor,
    num_experts: int,
    block_size: int,
    pad_sorted_token_ids: bool = True,
):
    numel = topk_ids.numel()
    topk_ids_cpu = topk_ids.flatten().cpu().tolist()

    counts = [0] * num_experts
    for eid in topk_ids_cpu:
        counts[eid + 1] += 1

    padded_counts = [((c + block_size - 1) // block_size) * block_size for c in counts]

    prefix = [0] * (num_experts + 1)
    for i in range(num_experts):
        prefix[i + 1] = prefix[i] + padded_counts[i]

    total_tokens = prefix[num_experts]

    cumsum = list(prefix)

    if numel < num_experts + 1:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + num_experts * (block_size - 1)

    sorted_token_ids = [numel] * max_num_tokens_padded if pad_sorted_token_ids else [0] * max_num_tokens_padded

    cumsum_copy = list(cumsum)
    for i, eid in enumerate(topk_ids_cpu):
        slot = eid + 1
        pos = cumsum_copy[slot]
        cumsum_copy[slot] += 1
        if pos < max_num_tokens_padded:
            sorted_token_ids[pos] = i

    num_blocks = total_tokens // block_size
    expert_ids = [0] * num_blocks
    for b in range(num_blocks):
        block_start = b * block_size
        for j in range(1, num_experts + 1):
            if prefix[j - 1] <= block_start < prefix[j]:
                expert_ids[b] = j - 2
                break

    return sorted_token_ids, expert_ids, total_tokens, cumsum


M_LIST = [1, 4, 16, 64, 128, 256]
TOPK_LIST = [1, 2, 4, 8]
NUM_EXPERTS_LIST = [4, 8, 16, 32, 64]
BLOCK_SIZE_LIST = [16, 32, 64, 128]


def _make_test_params():
    params = []
    for m, topk, num_experts, block_size in itertools.product(
        M_LIST, TOPK_LIST, NUM_EXPERTS_LIST, BLOCK_SIZE_LIST
    ):
        numel = m * topk
        if numel > 2048:
            continue
        params.append((m, topk, num_experts, block_size))
    return params


TEST_PARAMS = _make_test_params()


@pytest.mark.parametrize("m, topk, num_experts, block_size", TEST_PARAMS)
def test_moe_align_block_size(m, topk, num_experts, block_size):
    device = "cuda"
    numel = m * topk

    topk_ids = torch.randint(0, num_experts, (numel,), dtype=torch.int32, device=device)

    ne_param = num_experts + 1

    if numel < ne_param + 1:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + ne_param * (block_size - 1)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)
    cumsum_buffer = torch.empty((ne_param + 1,), dtype=torch.int32, device=device)

    moe_align_block_size(
        topk_ids, ne_param, block_size,
        sorted_ids, expert_ids, num_tokens_post_pad,
        cumsum_buffer, True,
    )
    torch.cuda.synchronize()

    ref_sorted, ref_expert_ids, ref_total, ref_cumsum = moe_align_block_size_ref(
        topk_ids, ne_param, block_size, pad_sorted_token_ids=True,
    )

    total_jit = num_tokens_post_pad.item()
    assert total_jit == ref_total, (
        f"total_tokens_post_pad mismatch: JIT={total_jit}, ref={ref_total}"
    )

    num_blocks = ref_total // block_size
    expert_ids_jit = expert_ids[:num_blocks].cpu().tolist()
    assert expert_ids_jit == ref_expert_ids, (
        f"expert_ids mismatch:\nJIT={expert_ids_jit}\nref={ref_expert_ids}"
    )

    # Token order within each expert range may differ due to atomicAdd,
    # so compare as sets per expert slot.
    sorted_ids_jit = sorted_ids[:total_jit].cpu().tolist()
    ref_sorted_trunc = ref_sorted[:total_jit]

    for slot in range(ne_param):
        start = ref_cumsum[slot]
        end = ref_cumsum[slot + 1] if slot + 1 <= ne_param else total_jit

        jit_set = set()
        ref_set = set()
        for idx in range(start, end):
            v = sorted_ids_jit[idx]
            if v != numel:
                jit_set.add(v)
            v2 = ref_sorted_trunc[idx]
            if v2 != numel:
                ref_set.add(v2)

        assert jit_set == ref_set, (
            f"slot {slot}: token set mismatch\n"
            f"JIT={sorted(jit_set)}\nref={sorted(ref_set)}"
        )


@pytest.mark.parametrize("num_experts", [4, 8, 64])
def test_filtered_experts(num_experts):
    device = "cuda"
    numel = 64
    block_size = 32
    ne_param = num_experts + 1

    topk_ids = torch.randint(-1, num_experts, (numel,), dtype=torch.int32, device=device)

    if numel < ne_param + 1:
        max_num_tokens_padded = numel * block_size
    else:
        max_num_tokens_padded = numel + ne_param * (block_size - 1)

    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=device)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=device)
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=device)
    cumsum_buffer = torch.empty((ne_param + 1,), dtype=torch.int32, device=device)

    moe_align_block_size(
        topk_ids, ne_param, block_size,
        sorted_ids, expert_ids, num_tokens_post_pad,
        cumsum_buffer, True,
    )
    torch.cuda.synchronize()

    ref_sorted, ref_expert_ids, ref_total, ref_cumsum = moe_align_block_size_ref(
        topk_ids, ne_param, block_size, pad_sorted_token_ids=True,
    )

    total_jit = num_tokens_post_pad.item()
    assert total_jit == ref_total


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
