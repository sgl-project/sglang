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


def assign_experts_to_tokens(num_tokens: int, num_experts: int, top_k_num: int, dtype=torch.float32):
    assert top_k_num <= num_experts, "top_k_num must be <= num_experts"

    expert_indices = torch.empty((num_tokens, top_k_num), dtype=torch.int32)
    for i in range(num_tokens):
        selected = torch.randperm(num_experts)[:top_k_num]
        expert_indices[i] = selected

    expert_weights = torch.rand((num_tokens, top_k_num), dtype=dtype)
    expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)

    return expert_indices, expert_weights


def sample_data(num_tokens: int, num_sequences: int, max_loras: int, num_experts: int, top_k_num: int, dtype=torch.float32):
    topk_ids, topk_weights = assign_experts_to_tokens(num_tokens, num_experts, top_k_num, dtype)
    token_lora_mapping = assign_loras_to_tokens(num_tokens, num_sequences, max_loras)
    return topk_ids, topk_weights, token_lora_mapping


def create_lora_info(token_lora_mapping, topk_ids, max_loras, num_experts, max_lora_rank, hidden_dim, intermediate_dim, gate_up_dim, dtype, device):
    gate_up_lora_a_weights = torch.randn((max_loras, num_experts, max_lora_rank, hidden_dim), dtype=dtype, device=device)
    gate_up_lora_b_weights = torch.randn((max_loras, num_experts, gate_up_dim, max_lora_rank), dtype=dtype, device=device)
    down_lora_a_weights = torch.randn((max_loras, num_experts, max_lora_rank, intermediate_dim), dtype=dtype, device=device)
    down_lora_b_weights = torch.randn((max_loras, num_experts, hidden_dim, max_lora_rank), dtype=dtype, device=device)

    num_tokens = token_lora_mapping.shape[0]
    dispatched_tokens = []
    dispatched_experts = []
    dispatched_loras = []

    for token_idx in range(num_tokens):
        lora_id = token_lora_mapping[token_idx]
        for k in range(topk_ids.shape[1]):
            expert_id = topk_ids[token_idx, k]
            dispatched_tokens.append(token_idx)
            dispatched_experts.append(expert_id)
            dispatched_loras.append(lora_id)

    token_ids = torch.tensor(dispatched_tokens, dtype=torch.int32, device=device)
    expert_ids = torch.tensor(dispatched_experts, dtype=torch.int32, device=device)
    lora_ids = torch.tensor(dispatched_loras, dtype=torch.int32, device=device)

    lora_ranks = torch.full((max_loras,), max_lora_rank, dtype=torch.int32, device=device)
    lora_scalings = torch.ones(max_loras, dtype=dtype, device=device)
    adapter_enabled = torch.ones(max_loras + 1, dtype=torch.int32, device=device)

    return LoRAInfo(
        gate_up_lora_a_weights=gate_up_lora_a_weights,
        gate_up_lora_b_weights=gate_up_lora_b_weights,
        down_lora_a_weights=down_lora_a_weights,
        down_lora_b_weights=down_lora_b_weights,
        token_ids=token_ids,
        expert_ids=expert_ids,
        lora_ids=lora_ids,
        lora_ranks=lora_ranks,
        lora_scalings=lora_scalings,
        adapter_enabled=adapter_enabled,
        max_lora_rank=max_lora_rank,
        num_experts=num_experts,
    )


def torch_naive_moe_with_lora(hidden_states, w13, w2, b13, b2, topk_weights, topk_ids, lora_info):
    num_tokens, hidden_dim = hidden_states.shape
    top_k = topk_ids.shape[1]
    num_experts = w13.shape[0]
    intermediate_dim = w2.shape[2]

    hidden_expanded = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)
    gate_up_out = torch.zeros(num_tokens * top_k, w13.shape[1], dtype=hidden_states.dtype, device=hidden_states.device)

    for expert_id in range(num_experts):
        mask = (topk_ids == expert_id).flatten()
        if mask.any():
            gate_up_out[mask] = hidden_expanded[mask] @ w13[expert_id].T
            if b13 is not None:
                gate_up_out[mask] += b13[expert_id]

    gate_up_out = gate_up_out.view(num_tokens, top_k, -1)

    if lora_info.max_lora_rank > 0:
        for i in range(num_tokens):
            for k in range(top_k):
                expert_id = topk_ids[i, k]
                lora_id = lora_info.lora_ids[i * top_k + k]
                lora_a = lora_info.gate_up_lora_a_weights[lora_id, expert_id]
                lora_b = lora_info.gate_up_lora_b_weights[lora_id, expert_id]
                lora_delta = lora_info.lora_scalings[lora_id] * (lora_b @ (lora_a @ hidden_states[i]))
                gate_up_out[i, k] += lora_delta

    gate_up_dim = gate_up_out.shape[-1]
    gate_dim = gate_up_dim // 2
    gate = gate_up_out[..., :gate_dim]
    up = gate_up_out[..., gate_dim:]
    intermediate_out = torch.nn.functional.silu(gate) * up

    down_out = torch.zeros(num_tokens, top_k, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device)

    for expert_id in range(num_experts):
        mask = (topk_ids == expert_id)
        if mask.any():
            masked_intermediate = intermediate_out[mask]
            down_out[mask] = masked_intermediate @ w2[expert_id].T
            if b2 is not None:
                down_out[mask] += b2[expert_id]

    if lora_info.max_lora_rank > 0:
        for i in range(num_tokens):
            for k in range(top_k):
                expert_id = topk_ids[i, k]
                lora_id = lora_info.lora_ids[i * top_k + k]
                lora_a = lora_info.down_lora_a_weights[lora_id, expert_id]
                lora_b = lora_info.down_lora_b_weights[lora_id, expert_id]
                lora_delta = lora_info.lora_scalings[lora_id] * (lora_b @ (lora_a @ intermediate_out[i, k]))
                down_out[i, k] += lora_delta

    weighted_out = down_out * topk_weights.unsqueeze(-1)
    final_out = weighted_out.sum(dim=1)

    return final_out


DTYPES = [torch.float16, torch.bfloat16]
DEVICES = ["cuda:0"]
SEED = [42]


@pytest.mark.parametrize("num_tokens", [32])
@pytest.mark.parametrize("top_k_num", [2, 4])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("max_loras", [2, 4])
@pytest.mark.parametrize("hidden_dim", [512])
@pytest.mark.parametrize("intermediate_dim", [1024])
@pytest.mark.parametrize("max_lora_rank", [16, 32])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
def test_lora_moe_runner(
    num_tokens,
    top_k_num,
    num_experts,
    max_loras,
    hidden_dim,
    intermediate_dim,
    max_lora_rank,
    dtype,
    device,
    seed,
):
    torch.set_default_device(device)
    set_random_seed(seed)

    num_sequences = 4
    topk_ids, topk_weights, token_lora_mapping = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num, dtype
    )

    gate_up_dim = intermediate_dim * 2
    w13 = torch.randn(num_experts, gate_up_dim, hidden_dim, dtype=dtype)
    w2 = torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=dtype)
    b13 = torch.randn(num_experts, gate_up_dim, dtype=dtype)
    b2 = torch.randn(num_experts, hidden_dim, dtype=dtype)

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

    num_dispatched = num_tokens * top_k_num
    sorted_token_ids = torch.arange(num_dispatched, dtype=torch.int32, device=device)
    expert_ids = topk_ids.flatten().to(dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.tensor([num_dispatched], dtype=torch.int32, device=device)

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

    # Create StandardTopKOutput for DispatchOutput
    router_logits = torch.randn(num_tokens, num_experts, dtype=dtype, device=device)  # Dummy logits
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

    # Test the full MoeRunner flow with LoRA enabled
    # Mock global server args to avoid dependency on server initialization
    class MockServerArgs:
        enable_deterministic_inference = False

    with patch('sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config.get_global_server_args', return_value=MockServerArgs()):
        runner = MoeRunner(MoeRunnerBackend.TRITON, config, lora_enabled=True)
        combine_input = runner.run(dispatch_output, quant_info, lora_info)
        lora_output = combine_input

    torch_output = torch_naive_moe_with_lora(
        hidden_states, w13, w2, b13, b2, topk_weights, topk_ids, lora_info
    )

    torch.testing.assert_close(lora_output.hidden_states, torch_output, atol=1e-1, rtol=1e-1)
