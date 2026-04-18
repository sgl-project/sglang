# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
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
)
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.lora.lora_moe_runners import LoRAInfo
from sglang.srt.utils import set_random_seed
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=26, suite="stage-b-test-1-gpu-large")


def generate_request_data(
    num_tokens: int, num_sequences: int, max_loras: int, device="cuda"
):
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
    seg_lens.append(remaining)  # Last segment gets the rest

    # 2. Build seg_indptr [0, len1, len1+len2, ...]
    seg_indptr = torch.cumsum(
        torch.tensor([0] + seg_lens, dtype=torch.int32, device=device),
        dim=0,
        dtype=torch.int32,
    )

    # 3. Assign one LoRA ID per Request
    req_to_lora = torch.randint(
        0, max_loras, (num_sequences,), dtype=torch.int32, device=device
    )

    # 4. Create dense mapping for the Naive verification function
    # (Expand req_to_lora based on seg_lens)
    token_lora_mapping = torch.repeat_interleave(
        req_to_lora, torch.tensor(seg_lens, device=device)
    )

    return seg_indptr, req_to_lora, token_lora_mapping


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
    device="cuda",
):
    topk_ids, topk_weights = assign_experts_to_tokens(
        num_tokens, num_experts, top_k_num, dtype
    )
    seg_indptr, req_to_lora, token_lora_mapping = generate_request_data(
        num_tokens, num_sequences, max_loras, device
    )
    return topk_ids, topk_weights, seg_indptr, req_to_lora, token_lora_mapping


def create_lora_info(
    seg_indptr,
    weight_indices,
    topk_ids,
    max_loras,
    num_experts,
    max_lora_rank,
    hidden_dim,
    intermediate_dim,
    gate_up_dim,
    dtype,
    device,
    lora_use_virtual_experts=False,
):
    # -------------------------------------------------------------------------
    # 1. Deterministic LoRA A Initialization
    # -------------------------------------------------------------------------

    val_gate_up_a = 0.1
    gate_up_lora_a_weights = torch.full(
        (max_loras, num_experts, max_lora_rank * 2, hidden_dim),
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
    # 2. Deterministic LoRA B Initialization
    # -------------------------------------------------------------------------
    base_target = 0.05

    gate_up_lora_b_weights = torch.zeros(
        (max_loras, num_experts, gate_up_dim, max_lora_rank),
        dtype=dtype,
        device=device,
    )
    down_lora_b_weights = torch.zeros(
        (max_loras, num_experts, hidden_dim, max_lora_rank), dtype=dtype, device=device
    )

    for i in range(num_experts):
        expert_multiplier = i + 1
        divisor = max(1, max_lora_rank)
        fill_val = (base_target * expert_multiplier) / divisor

        gate_up_lora_b_weights[:, i, :, :] = fill_val
        down_lora_b_weights[:, i, :, :] = fill_val

    # -------------------------------------------------------------------------
    # 3. Setup Metadata
    # -------------------------------------------------------------------------
    lora_ranks = torch.full(
        (max_loras,), max_lora_rank, dtype=torch.int32, device=device
    )

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
        lora_use_virtual_experts=lora_use_virtual_experts,
    )


def torch_naive_moe_with_lora(
    hidden_states,
    w13,
    w2,
    b13,
    b2,
    topk_weights,
    topk_ids,
    lora_info,
    token_lora_mapping,
):
    """
    Naive implementation. Note: We pass 'token_lora_mapping' explicitly because
    lora_info no longer contains it, but the naive token-loop logic needs it.
    """
    num_tokens, hidden_dim = hidden_states.shape
    top_k = topk_ids.shape[1]
    num_experts = w13.shape[0]

    # Expand hidden states for top-k routing
    hidden_expanded = (
        hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, hidden_dim)
    )

    # 1. Gate/Up Projection (Base)
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

    # 1.5. LoRA Gate/Up Delta
    # gate_up_lora_a is packed as [gate_a; up_a] along rank dim → [2*r, hidden_dim]
    # gate_up_lora_b is packed as [gate_b; up_b] along output dim → [2*inter, r]
    # Correct computation splits them: gate uses first r rows of A with first half of B,
    # up uses last r rows of A with second half of B.
    if lora_info.max_lora_rank > 0:
        r = lora_info.max_lora_rank
        for i in range(num_tokens):
            for k in range(top_k):
                expert_id = topk_ids[i, k]
                lora_id = token_lora_mapping[i]

                if lora_id < len(lora_info.lora_ranks):
                    lora_a = lora_info.gate_up_lora_a_weights[lora_id, expert_id]
                    lora_b = lora_info.gate_up_lora_b_weights[lora_id, expert_id]
                    half = lora_b.shape[0] // 2
                    lora_a_result = lora_a @ hidden_states[i]
                    gate_delta = lora_b[:half, :] @ lora_a_result[:r]
                    up_delta = lora_b[half:, :] @ lora_a_result[r:]
                    gate_up_out[i, k] += torch.cat([gate_delta, up_delta])

    # 2. Activation
    gate_up_dim = gate_up_out.shape[-1]
    gate_dim = gate_up_dim // 2
    gate = gate_up_out[..., :gate_dim]
    up = gate_up_out[..., gate_dim:]

    silu_gate = torch.nn.functional.silu(gate)
    intermediate_out = silu_gate * up

    # 3. Down Projection (Base)
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

    # 3.5. LoRA Down Delta
    if lora_info.max_lora_rank > 0:
        for i in range(num_tokens):
            for k in range(top_k):
                expert_id = topk_ids[i, k]
                lora_id = token_lora_mapping[i]  # Use explicit mapping

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

    num_sequences = 4

    # Generate Data using the new Request-Based generator
    topk_ids, topk_weights, seg_indptr, req_to_lora, token_lora_mapping = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num, dtype, device
    )

    gate_up_dim = intermediate_dim * 2

    # Initialize experts
    w13 = torch.randn(num_experts, gate_up_dim, hidden_dim, dtype=dtype) * 0.1
    w2 = torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=dtype) * 0.1
    b13 = torch.randn(num_experts, gate_up_dim, dtype=dtype) * 0.1
    b2 = torch.randn(num_experts, hidden_dim, dtype=dtype) * 0.1

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype)

    # Create LoRA Info using the new fields
    lora_info_delta = create_lora_info(
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

    lora_info_baseline = create_lora_info(
        seg_indptr=seg_indptr,
        weight_indices=req_to_lora,
        topk_ids=topk_ids,
        max_loras=max_loras,
        num_experts=num_experts,
        max_lora_rank=0,  # Set rank to 0 for baseline
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
    num_tokens_post_padded = torch.tensor(
        [num_dispatched], dtype=torch.int32, device=device
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
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config.get_global_server_args",
        return_value=MockServerArgs(),
    ):
        runner = MoeRunner(MoeRunnerBackend.TRITON, config, lora_enabled=True)

        # 3. Get outputs for both scenarios
        output_with_lora = runner.run(
            dispatch_output, quant_info, lora_info_delta
        ).hidden_states
        output_baseline = runner.run(
            dispatch_output, quant_info, lora_info_baseline
        ).hidden_states

    # Run Naive Torch Implementation (Uses dense mapping for verification)
    torch_output_lora = torch_naive_moe_with_lora(
        hidden_states,
        w13,
        w2,
        b13,
        b2,
        topk_weights,
        topk_ids,
        lora_info_delta,
        token_lora_mapping,
    )

    torch_output_base = torch_naive_moe_with_lora(
        hidden_states,
        w13,
        w2,
        b13,
        b2,
        topk_weights,
        topk_ids,
        lora_info_baseline,
        token_lora_mapping,
    )

    # The actual "Delta" (LoRA effect) for both
    sglang_delta = output_with_lora - output_baseline
    torch_delta = torch_output_lora - torch_output_base

    # Larger expert counts accumulate more numerical drift in Triton kernels on GB300
    tol = 0.15 if num_experts >= 20 else 5e-2
    torch.testing.assert_close(sglang_delta, torch_delta, atol=tol, rtol=tol)


@pytest.mark.parametrize("num_tokens", [32, 64])
@pytest.mark.parametrize("top_k_num", [1, 2])
@pytest.mark.parametrize("num_experts", [8, 20])
@pytest.mark.parametrize("max_lora_rank", [8, 16])
def test_lora_moe_runner_virtual_experts(
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

    num_sequences = 4

    # Generate Data using the new Request-Based generator
    topk_ids, topk_weights, seg_indptr, req_to_lora, token_lora_mapping = sample_data(
        num_tokens, num_sequences, max_loras, num_experts, top_k_num, dtype, device
    )

    gate_up_dim = intermediate_dim * 2

    # Initialize experts
    w13 = torch.randn(num_experts, gate_up_dim, hidden_dim, dtype=dtype) * 0.1
    w2 = torch.randn(num_experts, hidden_dim, intermediate_dim, dtype=dtype) * 0.1
    b13 = torch.randn(num_experts, gate_up_dim, dtype=dtype) * 0.1
    b2 = torch.randn(num_experts, hidden_dim, dtype=dtype) * 0.1

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype)

    # Create LoRA Info with virtual experts enabled
    lora_info_delta = create_lora_info(
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
        lora_use_virtual_experts=True,
    )

    lora_info_baseline = create_lora_info(
        seg_indptr=seg_indptr,
        weight_indices=req_to_lora,
        topk_ids=topk_ids,
        max_loras=max_loras,
        num_experts=num_experts,
        max_lora_rank=0,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        gate_up_dim=gate_up_dim,
        dtype=dtype,
        device=device,
        lora_use_virtual_experts=True,
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

    router_logits = torch.randn(num_tokens, num_experts, dtype=dtype, device=device)
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=router_logits,
    )

    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=topk_output,
    )

    class MockServerArgs:
        enable_deterministic_inference = False

    with patch(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config.get_global_server_args",
        return_value=MockServerArgs(),
    ):
        runner = MoeRunner(MoeRunnerBackend.TRITON, config, lora_enabled=True)

        output_with_lora = runner.run(
            dispatch_output, quant_info, lora_info_delta
        ).hidden_states
        output_baseline = runner.run(
            dispatch_output, quant_info, lora_info_baseline
        ).hidden_states

    # Run Naive Torch Implementation (Uses dense mapping for verification)
    torch_output_lora = torch_naive_moe_with_lora(
        hidden_states,
        w13,
        w2,
        b13,
        b2,
        topk_weights,
        topk_ids,
        lora_info_delta,
        token_lora_mapping,
    )

    torch_output_base = torch_naive_moe_with_lora(
        hidden_states,
        w13,
        w2,
        b13,
        b2,
        topk_weights,
        topk_ids,
        lora_info_baseline,
        token_lora_mapping,
    )

    # The actual "Delta" (LoRA effect) for both
    sglang_delta = output_with_lora - output_baseline
    torch_delta = torch_output_lora - torch_output_base

    # Larger expert counts accumulate more numerical drift in Triton kernels on GB300
    tol = 0.15 if num_experts >= 20 else 5e-2
    torch.testing.assert_close(sglang_delta, torch_delta, atol=tol, rtol=tol)


def _setup_marlin_moe_weights(num_experts, n, k, dtype):
    """Quantize float weights into AWQ Marlin format for testing."""
    from sgl_kernel.scalar_type import scalar_types

    from sglang.test.test_marlin_utils import awq_marlin_quantize

    group_size = 128
    quant_type = scalar_types.uint4

    w = torch.randn((num_experts, n, k), device="cuda", dtype=dtype) / 20

    w_ref_l, qweight_l, scales_l, zeros_l = [], [], [], []
    for i in range(num_experts):
        w_ref, qweight, scales, zeros = awq_marlin_quantize(
            w[i].transpose(1, 0), quant_type, group_size
        )
        w_ref_l.append(w_ref.T)
        qweight_l.append(qweight)
        scales_l.append(scales)
        zeros_l.append(zeros)

    def _stack(tensors):
        dev = tensors[0].device
        return torch.stack(tensors, dim=0).to(dev)

    return (
        _stack(w_ref_l),
        _stack(qweight_l).contiguous(),
        _stack(scales_l),
        _stack(zeros_l),
    )


@pytest.mark.parametrize("num_tokens", [32, 64])
@pytest.mark.parametrize("top_k_num", [1, 2])
@pytest.mark.parametrize("num_experts", [8])
@pytest.mark.parametrize("max_lora_rank", [8, 16])
def test_lora_moe_runner_marlin(num_tokens, top_k_num, num_experts, max_lora_rank):
    from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo

    max_loras = 2
    hidden_dim = 512
    intermediate_dim = 1024
    gate_up_dim = intermediate_dim * 2

    dtype = torch.float16
    device = "cuda:0"
    seed = 42

    torch.set_default_device(device)
    set_random_seed(seed)

    num_sequences = 4

    topk_ids, topk_weights, seg_indptr, req_to_lora, token_lora_mapping = sample_data(
        num_tokens,
        num_sequences,
        max_loras,
        num_experts,
        top_k_num,
        dtype,
        device,
    )

    # Quantize base weights to Marlin format
    _, w13_qweight, w13_scales, w13_qzeros = _setup_marlin_moe_weights(
        num_experts, gate_up_dim, hidden_dim, dtype
    )
    _, w2_qweight, w2_scales, w2_qzeros = _setup_marlin_moe_weights(
        num_experts, hidden_dim, intermediate_dim, dtype
    )

    hidden_states = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)

    lora_info_delta = create_lora_info(
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
        lora_use_virtual_experts=True,
    )

    lora_info_baseline = create_lora_info(
        seg_indptr=seg_indptr,
        weight_indices=req_to_lora,
        topk_ids=topk_ids,
        max_loras=max_loras,
        num_experts=num_experts,
        max_lora_rank=0,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        gate_up_dim=gate_up_dim,
        dtype=dtype,
        device=device,
        lora_use_virtual_experts=True,
    )

    quant_info = MarlinMoeQuantInfo(
        w13_qweight=w13_qweight,
        w2_qweight=w2_qweight,
        w13_scales=w13_scales,
        w2_scales=w2_scales,
        w13_qzeros=w13_qzeros,
        w2_qzeros=w2_qzeros,
        w13_g_idx=None,
        w2_g_idx=None,
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        weight_bits=4,
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

    router_logits = torch.randn(num_tokens, num_experts, dtype=dtype, device=device)
    topk_output = StandardTopKOutput(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        router_logits=router_logits,
    )
    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=topk_output,
    )

    class MockServerArgs:
        enable_deterministic_inference = False

    with patch(
        "sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config.get_global_server_args",
        return_value=MockServerArgs(),
    ):
        runner = MoeRunner(MoeRunnerBackend.MARLIN, config, lora_enabled=True)
        output_with_lora = runner.run(
            dispatch_output, quant_info, lora_info_delta
        ).hidden_states
        output_baseline = runner.run(
            dispatch_output, quant_info, lora_info_baseline
        ).hidden_states

    marlin_delta = output_with_lora - output_baseline

    # Verify the LoRA hooks fired and produced a non-trivial delta
    assert marlin_delta.abs().max().item() > 1e-4, (
        f"LoRA delta is too small ({marlin_delta.abs().max().item():.6f}), "
        "hooks may not be firing"
    )
    assert torch.isfinite(
        output_with_lora
    ).all(), "Marlin+LoRA output contains non-finite values"
    assert torch.isfinite(
        output_baseline
    ).all(), "Marlin baseline output contains non-finite values"


if __name__ == "__main__":
    pytest.main([__file__])
