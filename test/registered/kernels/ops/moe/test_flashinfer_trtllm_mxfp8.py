# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from sglang.srt.layers.moe.flashinfer_trtllm_moe import (
    trtllm_fp8_block_scale_moe_out_wrapper,
)
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0):
    pytest.skip(
        "FlashInfer TRT-LLM MXFP8 MoE requires compute capability 10 or newer.",
        allow_module_level=True,
    )


def _dequant_mxfp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scale_f32 = (scale.view(torch.uint8).to(torch.int32) << 23).view(torch.float32)
    scale_f32 = scale_f32.reshape(*x.shape[:-1], -1)
    scale_f32 = scale_f32.repeat_interleave(32, dim=-1)
    return x.float() * scale_f32


def _reference_moe(
    hidden_states: torch.Tensor,
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    w31: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    alpha: float,
    beta: float,
    limit: float,
    routed_scaling_factor: float,
) -> torch.Tensor:
    from flashinfer import mxfp8_quantize

    scores = routing_logits.sigmoid()
    topk_ids = (scores + routing_bias).topk(top_k, dim=-1).indices
    expert_weights = scores.gather(-1, topk_ids)
    expert_weights /= expert_weights.sum(dim=-1, keepdim=True) + 1e-20
    selected_w31 = w31[topk_ids]
    up, gate = torch.einsum("th,tkoh->tko", hidden_states, selected_w31).chunk(
        2, dim=-1
    )
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    act = gate * torch.sigmoid(alpha * gate) * (up + beta)
    act_q, act_scale = mxfp8_quantize(
        act.to(torch.bfloat16), is_sf_swizzled_layout=False
    )
    act = _dequant_mxfp8(act_q, act_scale)
    expert_out = torch.einsum("tki,tkhi->tkh", act, w2[topk_ids])
    routed_output = (expert_out * expert_weights[..., None]).sum(dim=1)
    return (routed_output * routed_scaling_factor).to(torch.bfloat16)


def _shuffle_mxfp8_weights(
    w31: torch.Tensor,
    w31_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    from flashinfer import (
        reorder_rows_for_gated_act_gemm,
        shuffle_matrix_a,
        shuffle_matrix_sf_a,
    )

    epilogue_tile_m = 128
    w31_out, w31_scale_out, w2_out, w2_scale_out = [], [], [], []
    for expert_idx in range(w31.shape[0]):
        w31_expert = reorder_rows_for_gated_act_gemm(w31[expert_idx])
        w31_scale_expert = reorder_rows_for_gated_act_gemm(w31_scale[expert_idx])
        w31_out.append(shuffle_matrix_a(w31_expert.view(torch.uint8), epilogue_tile_m))
        w31_scale_out.append(
            shuffle_matrix_sf_a(w31_scale_expert.view(torch.uint8), epilogue_tile_m)
        )
        w2_out.append(
            shuffle_matrix_a(w2[expert_idx].view(torch.uint8), epilogue_tile_m)
        )
        w2_scale_out.append(
            shuffle_matrix_sf_a(w2_scale[expert_idx].view(torch.uint8), epilogue_tile_m)
        )
    return (
        torch.stack(w31_out).view(torch.float8_e4m3fn),
        torch.stack(w31_scale_out),
        torch.stack(w2_out).view(torch.float8_e4m3fn),
        torch.stack(w2_scale_out),
    )


@pytest.mark.parametrize("num_tokens", [1, 8])
@torch.inference_mode()
def test_flashinfer_trtllm_mxfp8_minimax_swiglu(num_tokens: int):
    from flashinfer import mxfp8_quantize
    from flashinfer.fused_moe import Fp8QuantizationType
    from flashinfer.fused_moe.core import ActivationType

    torch.manual_seed(42)
    device = "cuda"
    num_experts, top_k = 4, 2
    hidden_size = intermediate_size = 256
    alpha, beta, limit = 1.702, 1.0, 7.0
    routed_scaling_factor = 2.0

    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16) / 4
    )
    gate = (
        torch.randn(
            num_experts,
            intermediate_size,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / hidden_size**0.5
    )
    up = torch.randn_like(gate) / hidden_size**0.5
    w31 = torch.cat((up, gate), dim=1)
    w2 = (
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        / intermediate_size**0.5
    )
    routing_logits = torch.randn(
        num_tokens, num_experts, device=device, dtype=torch.bfloat16
    )
    routing_bias = torch.randn(num_experts, device=device, dtype=torch.float32) / 8

    w31_q, w31_scale = mxfp8_quantize(w31.flatten(0, 1), False)
    w2_q, w2_scale = mxfp8_quantize(w2.flatten(0, 1), False)
    w31_q = w31_q.view_as(w31)
    w2_q = w2_q.view_as(w2)
    w31_scale = w31_scale.reshape(num_experts, 2 * intermediate_size, -1)
    w2_scale = w2_scale.reshape(num_experts, hidden_size, -1)

    hidden_q, hidden_scale = mxfp8_quantize(hidden_states, False, backend="cute-dsl")
    hidden_ref = _dequant_mxfp8(hidden_q, hidden_scale)
    w31_ref = _dequant_mxfp8(w31_q, w31_scale)
    w2_ref = _dequant_mxfp8(w2_q, w2_scale)
    expected = _reference_moe(
        hidden_ref,
        routing_logits.float(),
        routing_bias,
        w31_ref,
        w2_ref,
        top_k,
        alpha,
        beta,
        limit,
        routed_scaling_factor=routed_scaling_factor,
    )

    w31_kernel, w31_scale_kernel, w2_kernel, w2_scale_kernel = _shuffle_mxfp8_weights(
        w31_q, w31_scale, w2_q, w2_scale
    )

    def expert_param(value: float) -> torch.Tensor:
        return torch.full((num_experts,), value, device=device, dtype=torch.float32)

    output = torch.empty_like(hidden_states)
    trtllm_fp8_block_scale_moe_out_wrapper(
        routing_logits=routing_logits,
        routing_bias=routing_bias,
        hidden_states=hidden_q,
        hidden_states_scale=hidden_scale.reshape(num_tokens, -1),
        gemm1_weights=w31_kernel,
        gemm1_weights_scale=w31_scale_kernel,
        gemm2_weights=w2_kernel,
        gemm2_weights_scale=w2_scale_kernel,
        output=output,
        num_experts=num_experts,
        top_k=top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=intermediate_size,
        local_expert_offset=0,
        local_num_experts=num_experts,
        routed_scaling_factor=routed_scaling_factor,
        routing_method_type=int(RoutingMethodType.MiniMax2),
        use_shuffled_weight=True,
        tune_max_num_tokens=8,
        fp8_quantization_type=int(Fp8QuantizationType.MxFp8),
        activation_type=ActivationType.Swiglu.value,
        gemm1_alpha=expert_param(alpha),
        gemm1_beta=expert_param(beta),
        gemm1_clamp_limit=expert_param(limit),
    )

    relative_error = (
        output.float() - expected.float()
    ).norm() / expected.float().norm()
    assert relative_error.item() < 0.25
