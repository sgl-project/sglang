# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import random
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.moe_runner.triton import (
    TritonMoeQuantInfo,
    TritonRunnerInput,
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.lora.lora_moe_runners import LoRAInfo
from sglang.srt.utils import set_random_seed


def assign_loras_to_tokens(num_tokens: int, num_sequences: int, max_loras: int):
    assert num_sequences > 0 and max_loras > 0
    assert num_tokens >= num_sequences, "num_tokens must be >= num_sequences"

    tokens_per_seq = num_tokens // num_sequences
    remainder = num_tokens % num_sequences

    token_lora_mapping = torch.empty(num_tokens, dtype=torch.int32)

    start = 0
    for seq_idx in range(num_sequences):
        end = start + tokens_per_seq + (1 if seq_idx < remainder else 0)
        lora_id = random.randint(0, max_loras - 1)
        token_lora_mapping[start:end] = lora_id
        start = end

    return token_lora_mapping


def assign_experts_to_tokens(
    num_tokens: int, num_experts: int, top_k_num: int, dtype=torch.float32
):
    assert top_k_num <= num_experts, "top_k_num must be <= num_experts"

    expert_indices = torch.empty((num_tokens, top_k_num), dtype=torch.int32)
    for i in range(num_tokens):
        selected = torch.randperm(num_experts)[:top_k_num]
        expert_indices[i] = selected

    expert_weights = torch.rand((num_tokens, top_k_num), dtype=dtype)
    expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)

    return expert_indices, expert_weights


def sample_data(
    num_tokens: int,
    num_sequences: int,
    max_loras: int,
    num_experts: int,
    top_k_num: int,
    dtype=torch.float32,
):
    topk_ids, topk_weights = assign_experts_to_tokens(
        num_tokens, num_experts, top_k_num, dtype
    )
    token_lora_mapping = assign_loras_to_tokens(num_tokens, num_sequences, max_loras)
    return topk_ids, topk_weights, token_lora_mapping


def create_lora_info(
    token_lora_mapping,
    topk_ids,
    max_loras,
    num_experts,
    max_lora_rank,
    hidden_dim,
    intermediate_dim,
    gate_up_dim,
    dtype,
    device,
):
    # -------------------------------------------------------------------------
    # 1. Deterministic LoRA A Initialization
    # -------------------------------------------------------------------------
    # We fill A with (1 / input_dim).
    # If input is all 1s, A @ x will result in a vector of all 1s.

    val_gate_up_a = 1.0 / hidden_dim
    gate_up_lora_a_weights = torch.full(
        (max_loras, num_experts, max_lora_rank, hidden_dim),
        val_gate_up_a,
        dtype=dtype,
        device=device,
    )

    val_down_a = 1.0 / intermediate_dim
    down_lora_a_weights = torch.full(
        (max_loras, num_experts, max_lora_rank, intermediate_dim),
        val_down_a,
        dtype=dtype,
        device=device,
    )

    # -------------------------------------------------------------------------
    # 2. Deterministic LoRA B Initialization (Expert-Specific)
    # -------------------------------------------------------------------------
    # We want the output to be safe but noticeable.
    # Let's target a base value of 0.1 per expert index.
    # Formula: fill_value = (target / rank) * (expert_id + 1)

    base_target = 0.01  # Small enough to not explode SiLU, big enough to see

    gate_up_lora_b_weights = torch.zeros(
        (max_loras, num_experts, gate_up_dim, max_lora_rank), dtype=dtype, device=device
    )
    down_lora_b_weights = torch.zeros(
        (max_loras, num_experts, hidden_dim, max_lora_rank), dtype=dtype, device=device
    )

    for i in range(num_experts):
        # Make every expert add a slightly different value so we can check routing
        # Expert 0 adds ~0.01, Expert 10 adds ~0.11
        expert_multiplier = i + 1

        fill_val = (base_target * expert_multiplier) / max_lora_rank

        gate_up_lora_b_weights[:, i, :, :] = fill_val
        down_lora_b_weights[:, i, :, :] = fill_val

    # -------------------------------------------------------------------------
    # 3. Setup Metadata
    # -------------------------------------------------------------------------
    lora_ranks = torch.full(
        (max_loras,), max_lora_rank, dtype=torch.int32, device=device
    )
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32, device=device)

    return LoRAInfo(
        gate_up_lora_a_weights=gate_up_lora_a_weights,
        gate_up_lora_b_weights=gate_up_lora_b_weights,
        down_lora_a_weights=down_lora_a_weights,
        down_lora_b_weights=down_lora_b_weights,
        token_lora_mapping=token_lora_mapping,
        lora_ranks=lora_ranks,
        adapter_enabled=adapter_enabled,
        max_lora_rank=max_lora_rank,
        num_experts=num_experts,
    )


def torch_naive_moe_with_lora(
    hidden_states, w13, w2, b13, b2, topk_weights, topk_ids, lora_info
):
    num_tokens, hidden_dim = hidden_states.shape
    top_k = topk_ids.shape[1]
    num_experts = w13.shape[0]
    intermediate_dim = w2.shape[2]

    hidden_expanded = (
        hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)
    )

    gate_up_out = torch.zeros(
        num_tokens * top_k,
        w13.shape[1],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    for expert_id in range(num_experts):
        mask = (topk_ids == expert_id).flatten()
        if mask.any():
            expert_result = hidden_expanded[mask] @ w13[expert_id].T
            gate_up_out[mask] = expert_result
            if b13 is not None:
                gate_up_out[mask] += b13[expert_id]

    gate_up_out = gate_up_out.view(num_tokens, top_k, -1)

    if lora_info.max_lora_rank > 0:
        for i in range(num_tokens):
            for k in range(top_k):
                expert_id = topk_ids[i, k]
                lora_id = lora_info.token_lora_mapping[i]
                lora_a = lora_info.gate_up_lora_a_weights[lora_id, expert_id]
                lora_b = lora_info.gate_up_lora_b_weights[lora_id, expert_id]
                lora_a_result = lora_a @ hidden_states[i]
                lora_b_result = lora_b @ lora_a_result
                # Using scaling factor of 1.0 since lora_scalings was removed
                lora_delta = 1.0 * lora_b_result
                gate_up_out[i, k] += lora_delta

    gate_up_dim = gate_up_out.shape[-1]
    gate_dim = gate_up_dim // 2
    gate = gate_up_out[..., :gate_dim]
    up = gate_up_out[..., gate_dim:]

    silu_gate = torch.nn.functional.silu(gate)

    intermediate_out = silu_gate * up

    down_out = torch.zeros(
        num_tokens,
        top_k,
        hidden_dim,
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    for expert_id in range(num_experts):
        mask = topk_ids == expert_id
        if mask.any():
            masked_intermediate = intermediate_out[mask]
            expert_down_result = masked_intermediate @ w2[expert_id].T
            down_out[mask] = expert_down_result
            if b2 is not None:
                down_out[mask] += b2[expert_id]

    if lora_info.max_lora_rank > 0:
        for i in range(num_tokens):
            for k in range(top_k):
                expert_id = topk_ids[i, k]
                lora_id = lora_info.token_lora_mapping[i]
                lora_a = lora_info.down_lora_a_weights[lora_id, expert_id]
                lora_b = lora_info.down_lora_b_weights[lora_id, expert_id]
                lora_a_result = lora_a @ intermediate_out[i, k]
                lora_b_result = lora_b @ lora_a_result
                # Using scaling factor of 1.0 since lora_scalings was removed
                lora_delta = 1.0 * lora_b_result
                down_out[i, k] += lora_delta

    weighted_out = down_out * topk_weights.unsqueeze(-1)

    final_out = weighted_out.sum(dim=1)

    return final_out


@pytest.mark.parametrize("num_tokens", [32, 64])
@pytest.mark.parametrize("top_k_num", [1, 2])
@pytest.mark.parametrize("num_experts", [8, 20])
@pytest.mark.parametrize("max_lora_rank", [8, 16])
def test_lora_moe_runner_multi_expert(
    num_tokens, top_k_num, num_experts, max_lora_rank
):
    # Fixed parameters
    max_loras = 2
    hidden_dim = 512
    intermediate_dim = 1024

    dtype = torch.float32
    device = "cuda:0"
    seed = 42

    torch.set_default_device(device)
    set_random_seed(seed)

    # Distribute tokens across all experts (Random Routing)
    topk_ids, topk_weights = assign_experts_to_tokens(
        num_tokens, num_experts, top_k_num, dtype
    )

    # Assign LoRAs randomly
    num_sequences = 4
    token_lora_mapping = assign_loras_to_tokens(num_tokens, num_sequences, max_loras)

    gate_up_dim = intermediate_dim * 2

    # Initialize ALL experts with non-zero random weights
    w13 = torch.randn(num_experts, gate_up_dim, hidden_dim, dtype=dtype) * 0.01
    w2 = torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=dtype) * 0.01
    b13 = torch.randn(num_experts, gate_up_dim, dtype=dtype) * 0.01
    b2 = torch.randn(num_experts, hidden_dim, dtype=dtype) * 0.01

    # Set input to random values
    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype)

    lora_info = create_lora_info(
        token_lora_mapping=token_lora_mapping,
        topk_ids=topk_ids,
        max_loras=max_loras,
        num_experts=num_experts,
        max_lora_rank=max_lora_rank,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        gate_up_dim=gate_up_dim,
        dtype=dtype,
        device=device,
    )

    # Sort tokens by expert ID for the runner
    topk_ids_flat = topk_ids.flatten()
    sorted_indices = torch.argsort(topk_ids_flat)
    sorted_token_ids = sorted_indices // top_k_num
    expert_ids = topk_ids_flat[sorted_indices]

    num_dispatched = num_tokens * top_k_num
    num_tokens_post_padded = torch.tensor(
        [num_dispatched], dtype=torch.int32, device=device
    )

    runner_input = TritonRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )

    quant_info = TritonMoeQuantInfo(
        w13_weight=w13,
        w2_weight=w2,
        b13=b13,
        b2=b2,
    )

    config = MoeRunnerConfig(
        activation="silu",
        is_gated=True,
        inplace=False,
        no_combine=False,
        gemm1_alpha=None,
        gemm1_clamp_limit=None,
        routed_scaling_factor=1.0,
        apply_router_weight_on_input=False,
        num_local_experts=num_experts,
    )

    # Create StandardTopKOutput
    router_logits = torch.randn(num_tokens, num_experts, dtype=dtype, device=device)
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=router_logits,
    )

    # Create StandardDispatchOutput
    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=topk_output,
    )

    class MockServerArgs:
        enable_deterministic_inference = False

    with patch(
        "sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config.get_global_server_args",
        return_value=MockServerArgs(),
    ):
        runner = MoeRunner(MoeRunnerBackend.TRITON, config, lora_enabled=True)
        combine_input = runner.run(dispatch_output, quant_info, lora_info)
        lora_output = combine_input

    torch_output = torch_naive_moe_with_lora(
        hidden_states, w13, w2, b13, b2, topk_weights, topk_ids, lora_info
    )

    print(f"lora_output.hidden_states mean: {lora_output.hidden_states.mean()}")
    print(f"torch_output mean: {torch_output.mean()}")

    # Assert close
    torch.testing.assert_close(
        lora_output.hidden_states, torch_output, atol=1e-2, rtol=1e-2
    )
