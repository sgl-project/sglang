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


def generate_request_data(num_tokens: int, num_sequences: int, max_loras: int, device="cuda"):
    """
    Generates segment-based request data instead of token-based data.
    """
    assert num_sequences > 0 and max_loras > 0
    assert num_tokens >= num_sequences, "num_tokens must be >= num_sequences"

    # 1. Generate random segment lengths
    remaining = num_tokens
    seg_lens = []
    for _ in range(num_sequences - 1):
        # Ensure at least 1 token per sequence
        max_len = remaining - (num_sequences - len(seg_lens)) + 1
        length = random.randint(1, min(max_len, num_tokens // num_sequences * 2))
        seg_lens.append(length)
        remaining -= length
    seg_lens.append(remaining) # Last segment gets the rest

    # 2. Build seg_indptr [0, len1, len1+len2, ...]
    seg_indptr = torch.cumsum(torch.tensor([0] + seg_lens, dtype=torch.int32, device=device), dim=0, dtype=torch.int32)

    # 3. Assign one LoRA ID per Request
    req_to_lora = torch.randint(0, max_loras, (num_sequences,), dtype=torch.int32, device=device)

    # 4. Create dense mapping for the Naive verification function
    # (Expand req_to_lora based on seg_lens)
    token_lora_mapping = torch.repeat_interleave(req_to_lora, torch.tensor(seg_lens, device=device))

    return seg_indptr, req_to_lora, token_lora_mapping


def assign_experts_to_tokens(num_tokens: int, num_experts: int, top_k_num: int, dtype=torch.float32):
    assert top_k_num <= num_experts, "top_k_num must be <= num_experts"

    expert_indices = torch.empty((num_tokens, top_k_num), dtype=torch.int32)
    for i in range(num_tokens):
        selected = torch.randperm(num_experts)[:top_k_num]
        expert_indices[i] = selected

    expert_weights = torch.rand((num_tokens, top_k_num), dtype=dtype)
    expert_weights = expert_weights / expert_weights.sum(dim=1, keepdim=True)

    return expert_indices, expert_weights


def sample_data(num_tokens: int, num_sequences: int, max_loras: int, num_experts: int, top_k_num: int, dtype=torch.float32, device="cuda"):
    topk_ids, topk_weights = assign_experts_to_tokens(num_tokens, num_experts, top_k_num, dtype)
    seg_indptr, req_to_lora, token_lora_mapping = generate_request_data(num_tokens, num_sequences, max_loras, device)
    return topk_ids, topk_weights, seg_indptr, req_to_lora, token_lora_mapping


def create_lora_info(seg_indptr, weight_indices, topk_ids, max_loras, num_experts, max_lora_rank, hidden_dim, intermediate_dim, gate_up_dim, dtype, device):
    # -------------------------------------------------------------------------
    # 1. Deterministic LoRA A Initialization
    # -------------------------------------------------------------------------
    val_gate_up_a = 1.0 / hidden_dim
    gate_up_lora_a_weights = torch.full(
        (max_loras, num_experts, max_lora_rank, hidden_dim),
        val_gate_up_a, dtype=dtype, device=device
    )

    val_down_a = 1.0 / intermediate_dim
    down_lora_a_weights = torch.full(
        (max_loras, num_experts, max_lora_rank, intermediate_dim),
        val_down_a, dtype=dtype, device=device
    )

    # -------------------------------------------------------------------------
    # 2. Deterministic LoRA B Initialization
    # -------------------------------------------------------------------------
    base_target = 0.01

    gate_up_lora_b_weights = torch.zeros((max_loras, num_experts, gate_up_dim, max_lora_rank), dtype=dtype, device=device)
    down_lora_b_weights = torch.zeros((max_loras, num_experts, hidden_dim, max_lora_rank), dtype=dtype, device=device)

    for i in range(num_experts):
        expert_multiplier = (i + 1)
        fill_val = (base_target * expert_multiplier) / max_lora_rank

        gate_up_lora_b_weights[:, i, :, :] = fill_val
        down_lora_b_weights[:, i, :, :] = fill_val

    # -------------------------------------------------------------------------
    # 3. Setup Metadata
    # -------------------------------------------------------------------------
    lora_ranks = torch.full((max_loras,), max_lora_rank, dtype=torch.int32, device=device)

    # Enable all adapters referenced in weight_indices
    adapter_enabled = torch.zeros(max_loras + 1, dtype=torch.int32, device=device)
    adapter_enabled.index_fill_(0, weight_indices.long(), 1)

    return LoRAInfo(
        gate_up_lora_a_weights=gate_up_lora_a_weights,
        gate_up_lora_b_weights=gate_up_lora_b_weights,
        down_lora_a_weights=down_lora_a_weights,
        down_lora_b_weights=down_lora_b_weights,
        # UPDATED FIELDS
        seg_indptr=seg_indptr,
        req_to_lora=weight_indices,

        lora_ranks=lora_ranks,
        adapter_enabled=adapter_enabled,
        max_lora_rank=max_lora_rank,
        num_experts=num_experts,
    )


def torch_naive_moe_with_lora(hidden_states, w13, w2, b13, b2, topk_weights, topk_ids, lora_info, token_lora_mapping):
    """
    Naive implementation. Note: We pass 'token_lora_mapping' explicitly because
    lora_info no longer contains it, but the naive token-loop logic needs it.
    """
    num_tokens, hidden_dim = hidden_states.shape
    top_k = topk_ids.shape[1]
    num_experts = w13.shape[0]

    # Expand hidden states for top-k routing
    hidden_expanded = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)

    # 1. Gate/Up Projection (Base)
    gate_up_out = torch.zeros(num_tokens * top_k, w13.shape[1], dtype=hidden_states.dtype, device=hidden_states.device)

    for expert_id in range(num_experts):
        mask = (topk_ids == expert_id).flatten()
        if mask.any():
            expert_result = hidden_expanded[mask] @ w13[expert_id].T
            gate_up_out[mask] = expert_result
            if b13 is not None:
                gate_up_out[mask] += b13[expert_id]

    gate_up_out = gate_up_out.view(num_tokens, top_k, -1)

    # 1.5. LoRA Gate/Up Delta
    if lora_info.max_lora_rank > 0:
        for i in range(num_tokens):
            for k in range(top_k):
                expert_id = topk_ids[i, k]
                lora_id = token_lora_mapping[i] # Use explicit mapping

                # Check if this adapter is enabled/valid
                if lora_id < len(lora_info.lora_ranks):
                     lora_a = lora_info.gate_up_lora_a_weights[lora_id, expert_id]
                     lora_b = lora_info.gate_up_lora_b_weights[lora_id, expert_id]
                     lora_a_result = lora_a @ hidden_states[i]
                     lora_b_result = lora_b @ lora_a_result
                     gate_up_out[i, k] += lora_b_result

    # 2. Activation
    gate_up_dim = gate_up_out.shape[-1]
    gate_dim = gate_up_dim // 2
    gate = gate_up_out[..., :gate_dim]
    up = gate_up_out[..., gate_dim:]

    silu_gate = torch.nn.functional.silu(gate)
    intermediate_out = silu_gate * up

    # 3. Down Projection (Base)
    down_out = torch.zeros(num_tokens, top_k, hidden_dim, dtype=hidden_states.dtype, device=hidden_states.device)

    for expert_id in range(num_experts):
        mask = (topk_ids == expert_id)
        if mask.any():
            masked_intermediate = intermediate_out[mask]
            expert_down_result = masked_intermediate @ w2[expert_id].T
            down_out[mask] = expert_down_result
            if b2 is not None:
                down_out[mask] += b2[expert_id]

    # 3.5. LoRA Down Delta
    if lora_info.max_lora_rank > 0:
        for i in range(num_tokens):
            for k in range(top_k):
                expert_id = topk_ids[i, k]
                lora_id = token_lora_mapping[i] # Use explicit mapping

                if lora_id < len(lora_info.lora_ranks):
                    lora_a = lora_info.down_lora_a_weights[lora_id, expert_id]
                    lora_b = lora_info.down_lora_b_weights[lora_id, expert_id]
                    lora_a_result = lora_a @ intermediate_out[i, k]
                    lora_b_result = lora_b @ lora_a_result
                    down_out[i, k] += lora_b_result

    # 4. Final Reduction
    weighted_out = down_out * topk_weights.unsqueeze(-1)
    final_out = weighted_out.sum(dim=1)

    return final_out


@pytest.mark.parametrize("num_tokens", [32, 64])
@pytest.mark.parametrize("top_k_num", [1, 2])
@pytest.mark.parametrize("num_experts", [8, 20])
@pytest.mark.parametrize("max_lora_rank", [8, 16])
def test_lora_moe_runner_multi_expert(num_tokens, top_k_num, num_experts, max_lora_rank):
    # Fixed parameters
    max_loras = 2
    hidden_dim = 512
    intermediate_dim = 1024

    dtype = torch.float32
    device = "cuda:0"
    seed = 42

    torch.set_default_device(device)
    set_random_seed(seed)

    num_sequences = 4

    # Generate Data using the new Request-Based generator
    topk_ids, topk_weights, seg_indptr, req_to_lora, token_lora_mapping = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num, dtype, device
    )

    gate_up_dim = intermediate_dim * 2

    # Initialize experts
    w13 = torch.randn(num_experts, gate_up_dim, hidden_dim, dtype=dtype) * 0.01
    w2 = torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=dtype) * 0.01
    b13 = torch.randn(num_experts, gate_up_dim, dtype=dtype) * 0.01
    b2 = torch.randn(num_experts, hidden_dim, dtype=dtype) * 0.01

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype)

    # Create LoRA Info using the new fields
    lora_info = create_lora_info(
        seg_indptr=seg_indptr,
        weight_indices=req_to_lora,
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

    # Sort tokens for the runner
    topk_ids_flat = topk_ids.flatten()
    sorted_indices = torch.argsort(topk_ids_flat)
    sorted_token_ids = sorted_indices // top_k_num
    expert_ids = topk_ids_flat[sorted_indices]

    num_dispatched = num_tokens * top_k_num
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

    with patch('sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config.get_global_server_args', return_value=MockServerArgs()):
        runner = MoeRunner(MoeRunnerBackend.TRITON, config, lora_enabled=True)
        # Run SGLang runner (Uses Kernel)
        lora_output = runner.run(dispatch_output, quant_info, lora_info)

    # Run Naive Torch Implementation (Uses dense mapping for verification)
    torch_output = torch_naive_moe_with_lora(
        hidden_states, w13, w2, b13, b2, topk_weights, topk_ids, lora_info, token_lora_mapping
    )

    print(f"lora_output.hidden_states mean: {lora_output.hidden_states.mean()}")
    print(f"torch_output mean: {torch_output.mean()}")

    torch.testing.assert_close(lora_output.hidden_states, torch_output, atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
    pytest.main([__file__])
