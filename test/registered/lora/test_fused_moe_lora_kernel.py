# Temporarily adapted from https://github.com/vllm-project/vllm/blob/main/tests/lora/test_fused_moe_lora_kernel.py, will optimize in future refactor
import random
import sys

import pytest
import torch

from sglang.kernels.ops.moe.fused_moe_lora_kernel import fused_moe_lora

# ==============================================================================
# IMPORT PREBUILT KERNEL
# ==============================================================================
from sglang.kernels.ops.moe.moe_lora_align import moe_lora_align_block_size
from sglang.srt.lora.lora_moe_runners import _naive_moe_lora_align_block_size
from sglang.srt.utils import get_device, is_xpu, set_random_seed
from sglang.test.ci.ci_register import register_cuda_ci

# ==============================================================================

register_cuda_ci(est_time=13, stage="base-b", runner_config="1-gpu-large")


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
    block_size,
    mul_routed_weight,
    fully_sharded=False,
    offset=0,
):
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

    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 32,
        "BLOCK_SIZE_K": 64,
        "GROUP_SIZE_M": 1,
        "NUM_WARPS": 4,
        "NUM_STAGES": 3,
        "SPLIT_K": 1,
    }

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
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["GROUP_SIZE_M"],
        config["NUM_WARPS"],
        config["NUM_STAGES"],
        config["SPLIT_K"],
        config["BLOCK_SIZE_M"],
        config["BLOCK_SIZE_N"],
        config["BLOCK_SIZE_K"],
        config["GROUP_SIZE_M"],
        config["NUM_WARPS"],
        config["NUM_STAGES"],
        config["SPLIT_K"],
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
    outputs = []

    orig_dtype = hidden_states.dtype
    for i in range(hidden_states.shape[0]):
        lora_idx = token_lora_mapping[i]
        expert_ids = topk_ids[i]
        expert_weights = topk_weights[i]

        lora_a = lora_a_stacked[0][lora_idx][expert_ids]
        lora_b = lora_b_stacked[0][lora_idx][expert_ids]

        h_f32 = hidden_states[i].to(torch.float32)
        la_f32 = lora_a.to(torch.float32)
        lb_f32 = lora_b.to(torch.float32)

        if mul_routed_weight:
            tensors = [
                ((h_f32 @ la_f32[x].T @ lb_f32[x].T) * expert_weights[x]).to(orig_dtype)
                for x in range(top_k_num)
            ]
        else:
            tensors = [
                (h_f32 @ la_f32[x].T @ lb_f32[x].T).to(orig_dtype)
                for x in range(top_k_num)
            ]
        outputs.append(torch.stack(tensors, dim=0))
    return torch.stack(outputs, dim=0)


DTYPES = [torch.float32, torch.float16, torch.bfloat16]
DEVICES = [get_device(0)]
SEED = [42]


@pytest.mark.parametrize("mul_routed_weight", [False, True])
@pytest.mark.parametrize("num_tokens", [100])
@pytest.mark.parametrize("top_k_num", [6, 12])
@pytest.mark.parametrize("num_experts", [64])
@pytest.mark.parametrize("max_loras", [4, 6, 16])
@pytest.mark.parametrize("N", [1408])
@pytest.mark.parametrize("K", [2048])
@pytest.mark.parametrize("max_lora_rank", [16, 32, 64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_kernel(
    mul_routed_weight,
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    N,
    K,
    max_lora_rank,
    block_size,
    dtype,
    device,
    seed,
):
    torch.set_default_device(device)
    set_random_seed(seed)
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

    # init lora weights
    lora_a_stacked = [
        torch.rand(
            (
                max_loras,
                num_experts,
                max_lora_rank,
                K,
            ),
            dtype=dtype,
            device=device,
        )
    ]
    lora_b_stacked = [
        torch.rand(
            (
                max_loras,
                num_experts,
                N,
                max_lora_rank,
            ),
            dtype=dtype,
            device=device,
        )
    ]
    hidden_states = torch.rand(
        (
            num_tokens,
            K,
        ),
        dtype=dtype,
        device=device,
    )

    # fused_moe_lora_kernel output
    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype, device=device)

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
        block_size,
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


def _run_with_expert_id(
    expert_id_value,
    lora_a_stacked,
    lora_b_stacked,
    hidden_states,
    topk_weights,
    max_loras,
    num_blocks,
    num_tokens,
    top_k_num,
    max_lora_rank,
    N,
    dtype,
    device,
):
    """Run the kernel with adapter 0's blocks routed to `expert_id_value`.

    Only adapter 0 is enabled. Enabling the last adapter as well would make the
    result depend on whatever lies past the end of the weight tensor, which is
    undefined and varies with allocator layout.
    """
    sorted_token_ids = (
        torch.arange(num_tokens, dtype=torch.int32, device=device)
        .repeat(max_loras, 1)
        .contiguous()
    )
    expert_ids = torch.full(
        (max_loras, num_blocks), expert_id_value, dtype=torch.int32, device=device
    )
    num_tokens_post_padded = torch.full(
        (max_loras,), num_tokens, dtype=torch.int32, device=device
    )
    lora_ids = torch.arange(max_loras, dtype=torch.int32, device=device)
    adapter_enabled = torch.zeros(max_loras + 1, dtype=torch.int32, device=device)
    adapter_enabled[0] = 1

    output = torch.zeros((num_tokens, top_k_num, N), dtype=dtype, device=device)

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
        16,  # shrink BLOCK_SIZE_M
        32,  # shrink BLOCK_SIZE_N
        64,  # shrink BLOCK_SIZE_K
        1,  # shrink GROUP_SIZE_M
        4,  # shrink num_warps
        3,  # shrink num_stages
        1,  # shrink SPLIT_K
        16,  # expand BLOCK_SIZE_M
        32,  # expand BLOCK_SIZE_N
        64,  # expand BLOCK_SIZE_K
        1,  # expand GROUP_SIZE_M
        4,  # expand num_warps
        3,  # expand num_stages
        1,  # expand SPLIT_K
        False,  # mul_routed_weight
    )
    return output


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_fused_moe_lora_kernel_expert_id_out_of_range(dtype, device, seed):
    """Routed experts past the end of the LoRA stack must contribute nothing.

    Shared-experts fusion appends the shared expert to the routed set, so
    expert_ids can hold an index equal to num_experts while the LoRA stacks
    only cover 0..num_experts-1. Because stride_bl == num_experts * stride_be,
    the address for (lora_id, num_experts) is exactly the address for
    (lora_id + 1, expert 0), so an unbounded expert_id makes one adapter read
    the next adapter's weights.

    Such experts must be skipped, exactly like the -1 sentinel.
    """
    torch.set_default_device(device)
    set_random_seed(seed)

    max_loras = 2
    num_experts = 4
    max_lora_rank = 16
    N = 32
    K = 64
    top_k_num = 2
    num_tokens = 16
    num_blocks = 1

    lora_a_stacked = [
        torch.rand(
            (max_loras, num_experts, max_lora_rank, K), dtype=dtype, device=device
        )
    ]
    lora_b_stacked = [
        torch.rand(
            (max_loras, num_experts, N, max_lora_rank), dtype=dtype, device=device
        )
    ]
    hidden_states = torch.rand((num_tokens, K), dtype=dtype, device=device)
    topk_weights = torch.rand((num_tokens, top_k_num), dtype=dtype, device=device)

    common = dict(
        lora_a_stacked=lora_a_stacked,
        lora_b_stacked=lora_b_stacked,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        max_loras=max_loras,
        num_blocks=num_blocks,
        num_tokens=num_tokens,
        top_k_num=top_k_num,
        max_lora_rank=max_lora_rank,
        N=N,
        dtype=dtype,
        device=device,
    )

    # Expert 0 is valid, so this run must produce a non-trivial result. It
    # guards the assertion below from passing vacuously on an inert harness.
    in_range = _run_with_expert_id(0, **common)
    assert in_range.abs().max() > 0, "in-range expert produced no output"

    # num_experts is one past the last valid index (0..num_experts-1).
    out_of_range = _run_with_expert_id(num_experts, **common)
    # -1 is the kernel's existing "no expert for this block" sentinel.
    skipped = _run_with_expert_id(-1, **common)

    torch.testing.assert_close(out_of_range, skipped, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
