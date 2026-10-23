# Temporarily adapted from https://github.com/vllm-project/vllm/blob/main/tests/lora/test_fused_moe_lora_kernel.py, will optimize in future refactor
import random
import sys

import pytest
import torch

from sglang.kernels.ops.lora.moe.fused_moe_lora_kernel import fused_moe_lora

# ==============================================================================
# IMPORT PREBUILT KERNEL
# ==============================================================================
from sglang.kernels.ops.lora.moe.moe_lora_align import moe_lora_align_block_size
from sglang.srt.lora.lora_moe_runners import (
    _get_moe_lora_block_config,
    _naive_moe_lora_align_block_size,
)
from sglang.srt.utils import get_device, is_xpu, set_random_seed
from sglang.test.ci.ci_register import register_cuda_ci

# ==============================================================================

register_cuda_ci(est_time=26, stage="base-b-kernel-unit", runner_config="1-gpu-large")

# lora_moe_runners launches both the gate/up and the down delta with
# BLOCK_SIZE_M 64 (also the align block size), shrink BLOCK_SIZE_K and expand
# BLOCK_SIZE_N 64, GROUP_SIZE_M 8, 4 warps, 2 stages and no split-K; shrink N
# and expand K follow the rank (_get_moe_lora_block_config).
BLOCK_SIZE_M = 64


def round_up(x, base):
    return ((x + base - 1) // base) * base


def CEILDIV(x, y):
    return (x + y - 1) // y


def assign_loras_to_tokens(num_tokens: int, num_sequences: int, max_loras: int):
    """
    Split `num_tokens` into `num_sequences` sequences.
    Each sequence randomly selects 1 LoRA index from [0, max_loras),
    and all tokens in that sequence are assigned this LoRA index.

    Args:
        num_tokens (int): Total number of tokens.
        num_sequences (int): Number of sequences to split the tokens into.
        max_loras (int): Total number of available LoRA modules.

    Returns:
        token_lora_mapping (torch.Tensor): 1D tensor of shape [num_tokens]
        seg_indptr (torch.Tensor): 1D tensor of shape [num_sequences + 1]
        req_to_lora (torch.Tensor): 1D tensor of shape [num_sequences]
    """
    assert num_sequences > 0 and max_loras > 0
    assert num_tokens >= num_sequences, "num_tokens must be >= num_sequences"

    # Compute token distribution per sequence (distribute remainder evenly)
    tokens_per_seq = num_tokens // num_sequences
    remainder = num_tokens % num_sequences

    token_lora_mapping = torch.empty(num_tokens, dtype=torch.int32)
    seg_indptr = [0]
    req_to_lora = []

    start = 0
    for seq_idx in range(num_sequences):
        # Determine the token range for this sequence
        end = start + tokens_per_seq + (1 if seq_idx < remainder else 0)

        # Randomly select one LoRA ID for this sequence
        lora_id = random.randint(0, max_loras - 1)

        # Assign the same LoRA ID to all tokens in this sequence
        token_lora_mapping[start:end] = lora_id

        seg_indptr.append(end)
        req_to_lora.append(lora_id)

        start = end

    seg_indptr = torch.tensor(seg_indptr, dtype=torch.int32)
    req_to_lora = torch.tensor(req_to_lora, dtype=torch.int32)

    return token_lora_mapping, seg_indptr, req_to_lora


def assign_experts_to_tokens(num_tokens: int, num_experts: int, top_k_num: int):
    """
    For each token, randomly select `top_k_num` distinct experts out of `num_experts`,
    and assign normalized random weights that sum to 1.

    Args:
        num_tokens (int): Total number of tokens.
        num_experts (int): Total number of available experts.
        top_k_num (int): Number of experts to select per token.

    Returns:
        expert_indices (torch.Tensor): shape [num_tokens, top_k_num],
                                       expert index for each token.
        expert_weights (torch.Tensor): shape [num_tokens, top_k_num],
                                       normalized weights (sum = 1 per row).
    """
    assert top_k_num <= num_experts, "top_k_num must be <= num_experts"

    # Randomly select top_k_num distinct experts for each token
    expert_indices = torch.empty((num_tokens, top_k_num), dtype=torch.int32)
    for i in range(num_tokens):
        # Randomly choose unique expert indices
        selected = torch.randperm(num_experts)[:top_k_num]
        expert_indices[i] = selected

    # Generate random weights and normalize along dim=1
    expert_weights = torch.rand((num_tokens, top_k_num), dtype=torch.float32)
    expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)

    return expert_indices, expert_weights


def sample_data(
    num_tokens: int,
    num_sequences: int,
    max_loras: int,
    num_experts: int,
    top_k_num: int,
):
    topk_ids, topk_weights = assign_experts_to_tokens(
        num_tokens, num_experts, top_k_num
    )
    token_lora_mapping, seg_indptr, req_to_lora = assign_loras_to_tokens(
        num_tokens, num_sequences, max_loras
    )
    return topk_ids, topk_weights, token_lora_mapping, seg_indptr, req_to_lora


def use_fused_moe_lora_kernel(
    topk_ids,
    topk_weights,
    seg_indptr,
    req_to_lora,
    max_lora_rank,
    top_k_num,
    lora_a_stacked,
    lora_b_stacked,
    hidden_states,
    output,
    max_loras,
    num_experts,
    mul_routed_weight,
    fully_sharded=False,
    offset=0,
):
    block_size = BLOCK_SIZE_M
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    max_num_m_blocks = CEILDIV(max_num_tokens_padded, block_size)

    # Important: Ensure output tensors are on the same device as inputs
    device = topk_ids.device

    # init output tensors
    sorted_token_ids = torch.empty(
        (max_loras * max_num_tokens_padded,), dtype=torch.int32, device=device
    )
    expert_ids = torch.empty(
        (max_loras * max_num_m_blocks,), dtype=torch.int32, device=device
    )
    num_tokens_post_padded = torch.empty((max_loras,), dtype=torch.int32, device=device)
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32, device=device)
    lora_ids = torch.arange(max_loras, dtype=torch.int32, device=device)

    # call kernel — the fused CUDA align kernel exists only for CUDA; on XPU
    # use the pure-torch native alignment.
    if is_xpu():
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _naive_moe_lora_align_block_size(
                topk_ids,
                seg_indptr,
                req_to_lora,
                num_experts,
                block_size,
                max_loras,
                max_num_tokens_padded,
                max_num_m_blocks,
                adapter_enabled,
                device,
            )
        )
    else:
        moe_lora_align_block_size(
            topk_ids,
            seg_indptr,
            req_to_lora,
            num_experts,
            block_size,
            max_loras,
            max_num_tokens_padded,
            max_num_m_blocks,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            adapter_enabled,
            lora_ids,
            None,  # maybe_expert_map
        )

    blk = _get_moe_lora_block_config(max_lora_rank)
    # (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M, NUM_WARPS,
    #  NUM_STAGES, SPLIT_K), as lora_moe_runners launches them.
    shrink_tiles = (block_size, blk["shrink_block_size_n"], 64, 8, 4, 2, 1)
    expand_tiles = (block_size, 64, blk["expand_block_size_k"], 8, 4, 2, 1)

    expert_ids = expert_ids.view(max_loras, -1)
    sorted_token_ids = sorted_token_ids.view(max_loras, -1)

    fused_moe_lora(
        output,
        hidden_states,
        lora_a_stacked,
        lora_b_stacked,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        max_lora_rank,
        top_k_num,
        lora_ids,
        adapter_enabled,
        *shrink_tiles,
        *expand_tiles,
        mul_routed_weight,
        fully_sharded=fully_sharded,
        offset=offset,
    )


def use_torch(
    hidden_states,
    token_lora_mapping,
    topk_ids,
    topk_weights,
    lora_a_stacked,
    lora_b_stacked,
    top_k_num,
    mul_routed_weight,
):
    """FP32 per-(token, expert) reference. Slices land side by side in the
    output; a row per (token, expert) pair is the down launch's input."""
    outputs = []
    expanded = hidden_states.shape[0] == topk_ids.shape[0] * top_k_num

    orig_dtype = hidden_states.dtype
    for i in range(topk_ids.shape[0]):
        lora_idx = token_lora_mapping[i]
        tensors = []
        for x in range(top_k_num):
            expert = topk_ids[i, x]
            h_f32 = hidden_states[i * top_k_num + x if expanded else i].to(
                torch.float32
            )
            delta = torch.cat(
                [
                    h_f32
                    @ a[lora_idx, expert].to(torch.float32).T
                    @ b[lora_idx, expert].to(torch.float32).T
                    for a, b in zip(lora_a_stacked, lora_b_stacked)
                ]
            )
            if mul_routed_weight:
                delta = delta * topk_weights[i, x]
            tensors.append(delta.to(orig_dtype))
        outputs.append(torch.stack(tensors, dim=0))
    return torch.stack(outputs, dim=0)


# The two production launches, each at the three rank tiles (16 / 32 / 64 select
# distinct shrink-N and expand-K): the gated gate/up delta reads one row per
# token and writes the gate and up slices without the routed weight; the down
# delta reads one row per (token, expert) pair and applies it. Dtype, top-k and
# slot count alternate so each launch sees both values of each.
CASES = [
    pytest.param("gate_up", 16, torch.bfloat16, 6, 4, id="gate_up-r16-bf16"),
    pytest.param("down", 16, torch.float16, 12, 16, id="down-r16-fp16"),
    pytest.param("gate_up", 32, torch.float16, 12, 16, id="gate_up-r32-fp16"),
    pytest.param("down", 32, torch.bfloat16, 6, 4, id="down-r32-bf16"),
    pytest.param("gate_up", 64, torch.bfloat16, 12, 4, id="gate_up-r64-bf16"),
    pytest.param("down", 64, torch.float16, 6, 16, id="down-r64-fp16"),
]


@pytest.mark.parametrize(
    ("launch", "max_lora_rank", "dtype", "top_k_num", "max_loras"), CASES
)
def test_fused_moe_lora_kernel(launch, max_lora_rank, dtype, top_k_num, max_loras):
    device = get_device(0)
    torch.set_default_device(device)
    set_random_seed(42)
    num_tokens, num_experts, N, K = 100, 64, 1408, 2048
    # the number of randomly generated sentences.
    num_sequences = 10
    # generate data
    topk_ids, topk_weights, token_lora_mapping, seg_indptr, req_to_lora = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num
    )

    # Ensure generated data is on the correct device
    topk_ids = topk_ids.to(device)
    topk_weights = topk_weights.to(device)
    token_lora_mapping = token_lora_mapping.to(device)
    seg_indptr = seg_indptr.to(device)
    req_to_lora = req_to_lora.to(device)

    r = max_lora_rank
    if launch == "gate_up":
        # Gate and up are views into one fused buffer, as the pool lays them out.
        gate_up_a = torch.rand((max_loras, num_experts, 2 * r, K), dtype=dtype)
        gate_up_b = torch.rand((max_loras, num_experts, 2 * N, r), dtype=dtype)
        lora_a_stacked = [gate_up_a[:, :, :r], gate_up_a[:, :, r:]]
        lora_b_stacked = [gate_up_b[:, :, :N], gate_up_b[:, :, N:]]
        hidden_states = torch.rand((num_tokens, K), dtype=dtype)
        mul_routed_weight = False
    else:
        lora_a_stacked = [torch.rand((max_loras, num_experts, r, K), dtype=dtype)]
        lora_b_stacked = [torch.rand((max_loras, num_experts, N, r), dtype=dtype)]
        hidden_states = torch.rand((num_tokens * top_k_num, K), dtype=dtype)
        mul_routed_weight = True

    # fused_moe_lora_kernel output
    output = torch.zeros(
        (num_tokens, top_k_num, len(lora_b_stacked) * N), dtype=dtype, device=device
    )

    use_fused_moe_lora_kernel(
        topk_ids,
        topk_weights,
        seg_indptr,
        req_to_lora,
        max_lora_rank,
        top_k_num,
        lora_a_stacked,
        lora_b_stacked,
        hidden_states,
        output,
        max_loras,
        num_experts,
        mul_routed_weight=mul_routed_weight,
    )
    # pytorch output
    output2 = use_torch(
        hidden_states,
        token_lora_mapping,
        topk_ids,
        topk_weights,
        lora_a_stacked,
        lora_b_stacked,
        top_k_num,
        mul_routed_weight=mul_routed_weight,
    )

    torch.testing.assert_close(output, output2, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
