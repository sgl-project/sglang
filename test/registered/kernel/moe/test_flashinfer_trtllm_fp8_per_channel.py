"""Blackwell kernel coverage for FlashInfer FP8 per-channel MoE integration."""

import sys
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from flashinfer.fused_moe.core import ActivationType

import sglang.srt.layers.quantization.fp8  # noqa: F401
from sglang.kernels.ops.quantization.fp8_kernel import scaled_fp8_quant
from sglang.srt.layers.moe.flashinfer_trtllm_moe import (
    trtllm_fp8_per_channel_scale_moe_wrapper,
    trtllm_fp8_per_channel_scale_routed_moe_wrapper,
)
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    align_fp8_per_channel_moe_weights_for_flashinfer_trtllm,
)
from sglang.srt.layers.moe.utils import RoutingMethodType
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _is_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


pytestmark = pytest.mark.skipif(
    not _is_supported(),
    reason="requires an NVIDIA Blackwell GPU",
)


@dataclass
class _Case:
    layer: torch.nn.Module
    hidden_q: torch.Tensor
    hidden_scale: torch.Tensor
    logits: torch.Tensor
    w13_q: torch.Tensor
    w13_scale: torch.Tensor
    w2_q: torch.Tensor
    w2_scale: torch.Tensor
    intermediate_size: int
    is_gated: bool


def _quantize_per_channel(weight: torch.Tensor):
    scale = weight.float().abs().amax(dim=-1).clamp_min(1e-12) / 448.0
    quantized = (weight.float() / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return quantized, scale


def _make_case(*, is_gated: bool, num_experts: int = 8) -> _Case:
    torch.manual_seed(7 + int(is_gated))
    num_tokens, hidden_size, intermediate_size = 8, 1024, 384
    gemm1_rows = (2 if is_gated else 1) * intermediate_size
    w13_q, w13_scale = _quantize_per_channel(
        torch.randn(
            num_experts,
            gemm1_rows,
            hidden_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    w2_q, w2_scale = _quantize_per_channel(
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    layer = torch.nn.Module()
    layer.moe_runner_config = MoeRunnerConfig(
        activation="silu" if is_gated else "relu2", is_gated=is_gated
    )
    layer.w13_weight = torch.nn.Parameter(w13_q.clone(), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(w2_q.clone(), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        w13_scale.unsqueeze(-1).clone(), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        w2_scale.unsqueeze(-1).clone(), requires_grad=False
    )
    align_fp8_per_channel_moe_weights_for_flashinfer_trtllm(layer)

    hidden_q, hidden_scale = scaled_fp8_quant(
        torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16),
        use_per_token_if_dynamic=True,
    )
    return _Case(
        layer=layer,
        hidden_q=hidden_q,
        hidden_scale=hidden_scale,
        logits=torch.randn(num_tokens, num_experts, device="cuda"),
        w13_q=w13_q,
        w13_scale=w13_scale,
        w2_q=w2_q,
        w2_scale=w2_scale,
        intermediate_size=intermediate_size,
        is_gated=is_gated,
    )


def _common_kwargs(case: _Case, *, num_experts: int, local_expert_offset: int = 0):
    activation_type = ActivationType.Swiglu if case.is_gated else ActivationType.Relu2
    return dict(
        routing_bias=None,
        hidden_states=case.hidden_q,
        hidden_states_scale=case.hidden_scale,
        gemm1_weights=case.layer.w13_weight,
        gemm1_per_channel_weight_scale=case.layer.w13_per_channel_weight_scale,
        output1_scale_scalar=case.layer.output1_scales_scalar,
        output1_scale_gate_scalar=case.layer.output1_scales_gate_scalar,
        gemm2_weights=case.layer.w2_weight,
        gemm2_per_channel_weight_scale=case.layer.w2_per_channel_weight_scale,
        output2_scale_scalar=case.layer.output2_scales_scalar,
        num_experts=num_experts,
        top_k=2,
        n_group=None,
        topk_group=None,
        intermediate_size=case.intermediate_size,
        local_expert_offset=local_expert_offset,
        local_num_experts=case.w13_q.shape[0],
        routed_scaling_factor=1.0,
        use_routing_scales_on_input=False,
        routing_method_type=int(RoutingMethodType.Renormalize),
        tune_max_num_tokens=case.hidden_q.shape[0],
        activation_type=int(activation_type),
    )


def _native_reference(case: _Case):
    hidden = case.hidden_q.float() * case.hidden_scale
    topk_logits, topk_ids = torch.topk(case.logits, k=2, dim=-1)
    topk_weights = torch.softmax(topk_logits.float(), dim=-1)
    output = torch.zeros_like(hidden)

    for token in range(hidden.shape[0]):
        for slot in range(2):
            expert = int(topk_ids[token, slot])
            w13 = case.w13_q[expert].float() * case.w13_scale[expert, :, None]
            w2 = case.w2_q[expert].float() * case.w2_scale[expert, :, None]
            gemm1 = hidden[token].float() @ w13.T
            if case.is_gated:
                gate = gemm1[: case.intermediate_size]
                up = gemm1[case.intermediate_size :]
                activation = F.silu(gate) * up
            else:
                activation = F.relu(gemm1).square()
            # SGLang has no calibrated intermediate tensor scale for dynamic
            # compressed-tensors checkpoints, so the integration uses one.
            activation_q = activation.to(torch.float8_e4m3fn).float()
            output[token] += topk_weights[token, slot] * (activation_q @ w2.T)
    return output


@pytest.mark.parametrize("is_gated", [True, False], ids=["swiglu", "relu2"])
def test_logits_and_routed_paths_match_reference(is_gated):
    case = _make_case(is_gated=is_gated)
    kwargs = _common_kwargs(case, num_experts=8)
    logits_output = trtllm_fp8_per_channel_scale_moe_wrapper(
        routing_logits=case.logits, **kwargs
    )

    topk_logits, topk_ids = torch.topk(case.logits, k=2, dim=-1)
    topk_weights = torch.softmax(topk_logits.float(), dim=-1)
    kwargs["routing_method_type"] = int(RoutingMethodType.TopK)
    routed_output = trtllm_fp8_per_channel_scale_routed_moe_wrapper(
        topk_ids=topk_ids.to(torch.int32), topk_weights=topk_weights, **kwargs
    )
    reference = _native_reference(case)
    torch.cuda.synchronize()

    normalized_mae = (
        logits_output.float() - reference
    ).abs().mean() / reference.abs().mean()
    assert normalized_mae < 0.08
    torch.testing.assert_close(routed_output, logits_output, rtol=0.02, atol=0.02)


def test_torch_compile_and_cuda_graph_capture():
    case = _make_case(is_gated=True)
    kwargs = _common_kwargs(case, num_experts=8)

    def invoke(routing_logits, hidden_states, hidden_states_scale):
        return trtllm_fp8_per_channel_scale_moe_wrapper(
            routing_logits=routing_logits,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"hidden_states", "hidden_states_scale"}
            },
        )

    eager_output = invoke(case.logits, case.hidden_q, case.hidden_scale)
    compiled = torch.compile(invoke, backend="eager", fullgraph=True)
    compiled_output = compiled(case.logits, case.hidden_q, case.hidden_scale)
    torch.testing.assert_close(compiled_output, eager_output, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = invoke(case.logits, case.hidden_q, case.hidden_scale)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
