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

"""
Correctness test: Marlin (int4 base + LoRA) vs Triton (dequantized base + LoRA).

Fake-quantizes random weights to int4/Marlin format and dequantizes them with the
same path, then runs both backends through MoeRunner and compares LoRA deltas.
"""

from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import MarlinMoeQuantInfo
from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.lora.lora_moe_runners import LoRAInfo
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, suite="stage-b-test-1-gpu-large")


# ---------------------------------------------------------------------------
# Fake quantization helpers (symmetric int4, matching Marlin's dequant path)
# ---------------------------------------------------------------------------


def _quantize_per_expert(w_float: torch.Tensor, K: int, group_size: int):
    """Quantize [N, K] float weight to int4. Returns (q_int [N,K], scales_bf16 [N,groups])."""
    N = w_float.shape[0]
    num_groups = K // group_size

    w_grouped = w_float.reshape(N, num_groups, group_size)
    scales_fp32 = w_grouped.abs().amax(dim=-1) / 7.0
    scales_fp32 = scales_fp32.clamp(min=1e-6)
    scales_bf16 = scales_fp32.to(torch.bfloat16)

    scales_for_quant = scales_bf16.float()
    q_int = torch.zeros(N, K, dtype=torch.int32, device=w_float.device)
    for g in range(num_groups):
        s = scales_for_quant[:, g : g + 1]
        sl = slice(g * group_size, (g + 1) * group_size)
        q_int[:, sl] = torch.round(w_float[:, sl] / s).clamp(-8, 7).to(torch.int32) + 8

    return q_int, scales_bf16


def _fake_quantize_to_marlin_int4(weight_bf16: torch.Tensor):
    """Fake-quantize [E, N, K] bf16 weight to Marlin int4 format.

    Returns: (qweight, scales, g_idx, g_idx_sort_indices)
    """
    from sglang.jit_kernel.gptq_marlin_repack import gptq_marlin_repack
    from sglang.srt.layers.quantization.marlin_utils import marlin_permute_scales
    from sglang.srt.layers.quantization.utils import pack_rows

    E, N, K = weight_bf16.shape
    num_bits = 4
    group_size = 128
    device = weight_bf16.device

    all_qweight, all_scales = [], []
    for e in range(E):
        q_int, scales_bf16 = _quantize_per_expert(weight_bf16[e].float(), K, group_size)
        w_quant_t = q_int.t().contiguous()
        packed = pack_rows(w_quant_t, num_bits, K, N)
        perm = torch.arange(K, device=device, dtype=torch.int32)
        all_qweight.append(gptq_marlin_repack(packed.to(device), perm, K, N, num_bits))
        all_scales.append(
            marlin_permute_scales(
                scales_bf16.t().contiguous().to(device), K, N, group_size
            )
        )

    g_idx = (
        (torch.arange(K, device=device, dtype=torch.int32) // group_size)
        .unsqueeze(0)
        .expand(E, -1)
        .contiguous()
    )
    sort_indices = (
        torch.arange(K, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .expand(E, -1)
        .contiguous()
    )

    return torch.stack(all_qweight), torch.stack(all_scales), g_idx, sort_indices


def _dequantize_from_marlin_int4(weight_bf16_orig: torch.Tensor, group_size: int = 128):
    """Dequantize using the same path as _fake_quantize, so Triton reference matches Marlin."""
    E, N, K = weight_bf16_orig.shape
    result = torch.zeros_like(weight_bf16_orig)
    for e in range(E):
        q_int, scales_bf16 = _quantize_per_expert(
            weight_bf16_orig[e].float(), K, group_size
        )
        num_groups = K // group_size
        for g in range(num_groups):
            sl = slice(g * group_size, (g + 1) * group_size)
            s = scales_bf16[:, g : g + 1]
            result[e, :, sl] = (q_int[:, sl] - 8).to(torch.bfloat16) * s
    return result


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", [1, 8, 32])
@pytest.mark.parametrize("top_k", [2, 8])
def test_marlin_vs_triton_lora_correctness(num_tokens, top_k):
    torch.manual_seed(42)

    device = "cuda"
    dtype = torch.bfloat16

    hidden_dim = 7168
    intermediate_dim = 2048
    gate_up_dim = 2 * intermediate_dim
    num_experts = 64
    lora_rank = 32
    num_loras = 1

    hidden = torch.randn(num_tokens, hidden_dim, dtype=dtype, device=device)
    topk_weights = torch.randn(
        num_tokens, top_k, dtype=torch.float32, device=device
    ).softmax(dim=-1)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
    )

    # Base weights (random bf16)
    w13_bf16 = (
        torch.randn(num_experts, gate_up_dim, hidden_dim, dtype=dtype, device=device)
        * 0.01
    )
    w2_bf16 = (
        torch.randn(
            num_experts, hidden_dim, intermediate_dim, dtype=dtype, device=device
        )
        * 0.01
    )

    # LoRA weights (shared across both paths)
    gu_lora_a = (
        torch.randn(num_loras, 1, lora_rank * 2, hidden_dim, dtype=dtype, device=device)
        * 0.01
    )
    gu_lora_b = (
        torch.randn(
            num_loras, num_experts, gate_up_dim, lora_rank, dtype=dtype, device=device
        )
        * 0.01
    )
    dn_lora_a = (
        torch.randn(
            num_loras,
            num_experts,
            lora_rank,
            intermediate_dim,
            dtype=dtype,
            device=device,
        )
        * 0.01
    )
    dn_lora_b = (
        torch.randn(num_loras, 1, hidden_dim, lora_rank, dtype=dtype, device=device)
        * 0.01
    )

    # Token-to-LoRA mapping: all tokens use adapter 0
    seg_indptr = torch.tensor([0, num_tokens], dtype=torch.int32, device=device)
    req_to_lora = torch.tensor([0], dtype=torch.int32, device=device)

    def _make_lora_info(rank):
        return LoRAInfo(
            gate_up_lora_a_weights=gu_lora_a if rank > 0 else gu_lora_a[:, :, :0, :],
            gate_up_lora_b_weights=gu_lora_b if rank > 0 else gu_lora_b[:, :, :, :0],
            down_lora_a_weights=dn_lora_a if rank > 0 else dn_lora_a[:, :, :0, :],
            down_lora_b_weights=dn_lora_b if rank > 0 else dn_lora_b[:, :, :, :0],
            seg_indptr=seg_indptr,
            req_to_lora=req_to_lora,
            lora_ranks=torch.full((num_loras,), rank, dtype=torch.int32, device=device),
            adapter_enabled=torch.ones(num_loras + 1, dtype=torch.int32, device=device),
            max_lora_rank=rank,
            num_experts=num_experts,
            lora_use_virtual_experts=True,
            experts_shared_outer_loras=True,
        )

    lora_info = _make_lora_info(lora_rank)
    lora_baseline = _make_lora_info(0)

    # Quantize for Marlin, dequantize for Triton reference
    w13_qw, w13_sc, w13_gidx, w13_si = _fake_quantize_to_marlin_int4(w13_bf16)
    w2_qw, w2_sc, w2_gidx, w2_si = _fake_quantize_to_marlin_int4(w2_bf16)
    w13_deq = _dequantize_from_marlin_int4(w13_bf16)
    w2_deq = _dequantize_from_marlin_int4(w2_bf16)

    marlin_qi = MarlinMoeQuantInfo(
        w13_qweight=w13_qw,
        w2_qweight=w2_qw,
        w13_scales=w13_sc,
        w2_scales=w2_sc,
        w13_g_idx=w13_gidx,
        w2_g_idx=w2_gidx,
        w13_g_idx_sort_indices=w13_si,
        w2_g_idx_sort_indices=w2_si,
        weight_bits=4,
    )
    triton_qi = TritonMoeQuantInfo(
        w13_weight=w13_deq, w2_weight=w2_deq, b13=None, b2=None
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
        topk_weights=topk_weights, topk_ids=topk_ids, router_logits=router_logits
    )
    dispatch_output = StandardDispatchOutput(
        hidden_states=hidden, hidden_states_scale=None, topk_output=topk_output
    )

    class MockServerArgs:
        enable_deterministic_inference = False

    with patch(
        "sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config.get_global_server_args",
        return_value=MockServerArgs(),
    ):
        marlin_runner = MoeRunner(MoeRunnerBackend.MARLIN, config, lora_enabled=True)
        triton_runner = MoeRunner(MoeRunnerBackend.TRITON, config, lora_enabled=True)

        marlin_out = marlin_runner.run(
            dispatch_output, marlin_qi, lora_info
        ).hidden_states
        marlin_base = marlin_runner.run(
            dispatch_output, marlin_qi, lora_baseline
        ).hidden_states
        triton_out = triton_runner.run(
            dispatch_output, triton_qi, lora_info
        ).hidden_states
        triton_base = triton_runner.run(
            dispatch_output, triton_qi, lora_baseline
        ).hidden_states

    marlin_delta = marlin_out - marlin_base
    triton_delta = triton_out - triton_base

    # Remaining error is from kernel-level accumulation differences
    # (Marlin fp32 reduce vs Triton bf16 dot), not from quantization mismatch.
    torch.testing.assert_close(
        marlin_delta.float(), triton_delta.float(), atol=0.01, rtol=0.05
    )


if __name__ == "__main__":
    pytest.main([__file__])
