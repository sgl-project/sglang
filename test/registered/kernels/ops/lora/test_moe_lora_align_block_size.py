# Temporarily adapted from https://github.com/vllm-project/vllm/blob/main/tests/lora/test_moe_lora_align_sum.py, will optimize in future refactor
import collections
import random
import sys

import pytest
import torch

# ---------------------------------------------------------
# IMPORT PREBUILT KERNEL
# ---------------------------------------------------------
from sglang.kernels.ops.lora.moe.moe_lora_align import moe_lora_align_block_size
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# test_fused_moe_lora_kernel.py runs this align kernel against a torch reference
# per PR; this wider matrix is opt-in.
register_cuda_ci(est_time=18, stage="extra-a", runner_config="1-gpu-large")
# backend-specific: the HIP JIT build of moe_lora_align_kernel.cu and its
# wave64 per-warp expert counters.
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


def round_up(x, base):
    return ((x + base - 1) // base) * base


def CEILDIV(x, y):
    return (x + y - 1) // y


def sample_data(num_experts, max_loras, num_tokens, topk_num):
    # 1. Generate TopK IDs (Flattened tokens)
    topk_ids = torch.zeros((num_tokens, topk_num), dtype=torch.int32)
    for i in range(num_tokens):
        pool = list(range(num_experts))
        random.shuffle(pool)
        for j in range(topk_num):
            topk_ids[i, j] = pool[j]

    # 2. Generate Random Requests (Segments)
    # We split num_tokens into random chunks to simulate a batch of requests
    remaining_tokens = num_tokens
    seg_lens = []
    while remaining_tokens > 0:
        # Random length between 1 and remaining
        length = random.randint(1, min(32, remaining_tokens))
        if remaining_tokens - length < 0:
            length = remaining_tokens
        seg_lens.append(length)
        remaining_tokens -= length

    # Ensure we cover the full range exactly (cleanup last segment)
    if sum(seg_lens) < num_tokens:
        seg_lens.append(num_tokens - sum(seg_lens))

    # 3. Build seg_indptr [0, len1, len1+len2, ...]
    seg_indptr = torch.cumsum(
        torch.tensor([0] + seg_lens, dtype=torch.int32), dim=0
    ).to(dtype=torch.int32)

    # 4. Assign a LoRA ID to each Request
    num_reqs = len(seg_lens)
    req_to_lora = torch.randint(0, max_loras, (num_reqs,), dtype=torch.int32)

    return (topk_ids.to("cuda"), seg_indptr.to("cuda"), req_to_lora.to("cuda"))


@pytest.mark.parametrize("num_tokens", [200, 4096])
@pytest.mark.parametrize("topk_num", [6])
@pytest.mark.parametrize("num_experts", [64, 512])
@pytest.mark.parametrize("max_loras", [32])
@pytest.mark.parametrize("block_size", [16])
def test_moe_lora_align_block_size(
    num_tokens, topk_num, num_experts, max_loras, block_size
):
    # sample data
    random.seed(1)
    torch.manual_seed(1)

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available, skipping moe_lora_align_block_size test.")
    # UPDATED: Get the new 3-step mapping tensors
    topk_ids, seg_indptr, req_to_lora = sample_data(
        num_experts, max_loras, num_tokens, topk_num
    )

    # compute paddings
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size)

    # init output tensors
    sorted_token_ids = torch.full(
        (max_loras * max_num_tokens_padded,),
        topk_ids.numel(),
        dtype=torch.int32,
        device="cuda",
    )
    expert_ids = torch.full(
        (max_loras * max_num_m_blocks,), num_experts, dtype=torch.int32, device="cuda"
    )
    num_tokens_post_pad = torch.zeros((max_loras,), dtype=torch.int32, device="cuda")
    adapter_enabled = torch.ones((max_loras + 1,), dtype=torch.int32, device="cuda")
    lora_ids = torch.arange(max_loras, dtype=torch.int32, device="cuda")

    # UPDATED: Call kernel with new signature
    moe_lora_align_block_size(
        topk_ids,
        seg_indptr,  # Arg 2: Pointers
        req_to_lora,  # Arg 3: Request Map
        num_experts,
        block_size,
        max_loras,
        max_num_tokens_padded,
        max_num_m_blocks,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_pad,
        adapter_enabled,
        lora_ids,
        None,
    )

    # verify values: per LoRA, every (token, expert) pair of its tokens lands
    # exactly once in a block of its own expert, each block holds only such
    # pairs and padding, and the padded count is ceil(pairs / block_size)
    # blocks per expert. The order inside a block is not part of the contract.
    flat = topk_ids.view(-1).tolist()
    padding = len(flat)
    sorted_rows = sorted_token_ids.view(max_loras, -1).tolist()
    label_rows = expert_ids.view(max_loras, -1).tolist()
    post_pad = num_tokens_post_pad.tolist()

    # Expand req_to_lora back to one owner per token.
    cpu_seg_indptr = seg_indptr.tolist()
    token_owner = [0] * num_tokens
    for r, lora in enumerate(req_to_lora.tolist()):
        token_owner[cpu_seg_indptr[r] : cpu_seg_indptr[r + 1]] = [lora] * (
            cpu_seg_indptr[r + 1] - cpu_seg_indptr[r]
        )
    pairs_by_lora = collections.defaultdict(list)
    for j in range(padding):
        pairs_by_lora[token_owner[j // topk_num]].append(j)

    for lora_idx in range(max_loras):
        total = post_pad[lora_idx]
        assert total % block_size == 0, f"LoRA {lora_idx}: padded count {total}"
        placed, blocks_per_expert = collections.Counter(), collections.Counter()
        for b in range(total // block_size):
            label = label_rows[lora_idx][b]
            block = sorted_rows[lora_idx][b * block_size : (b + 1) * block_size]
            pairs = [j for j in block if j != padding]
            assert 0 <= label < num_experts, f"LoRA {lora_idx}: block label {label}"
            assert pairs and all(
                0 <= j < padding
                and flat[j] == label
                and token_owner[j // topk_num] == lora_idx
                for j in pairs
            ), f"LoRA {lora_idx}, block of expert {label}: {block}"
            placed.update(pairs)
            blocks_per_expert[label] += 1

        expected = pairs_by_lora[lora_idx]
        assert placed == collections.Counter(expected), (
            f"LoRA {lora_idx}: pairs missing or duplicated"
        )
        per_expert = collections.Counter(flat[j] for j in expected)
        assert blocks_per_expert == collections.Counter(
            {e: CEILDIV(n, block_size) for e, n in per_expert.items()}
        ), f"LoRA {lora_idx}: blocks per expert"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
