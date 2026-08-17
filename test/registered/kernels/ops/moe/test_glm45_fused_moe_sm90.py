"""Correctness coverage for the H200 GLM-4.5-FP8 fused MoE fast path."""

from __future__ import annotations

import pytest
import torch

from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="8-gpu-h200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(0),
    reason="NVIDIA H200 required",
)

_TOKENS = [1, 16, 64, 113, 119, 129, 1935, 3451, 7497]
_EXPERTS = 161
_HIDDEN = 5120
_GATE_UP = 384
_INTERMEDIATE = 192
_TOP_K = 9


@pytest.fixture(scope="module")
def weights():
    torch.manual_seed(0x45F00D)
    w1 = (
        torch.empty(
            (_EXPERTS, _GATE_UP, _HIDDEN),
            dtype=torch.bfloat16,
            device="cuda",
        )
        .uniform_(-1.0, 1.0)
        .to(torch.float8_e4m3fn)
    )
    w2 = (
        torch.empty(
            (_EXPERTS, _HIDDEN, _INTERMEDIATE),
            dtype=torch.bfloat16,
            device="cuda",
        )
        .uniform_(-1.0, 1.0)
        .to(torch.float8_e4m3fn)
    )
    generator = torch.Generator(device="cuda").manual_seed(0x45CA1E)
    w1_scale = torch.rand(
        (_EXPERTS, _GATE_UP, 1), device="cuda", generator=generator
    ).mul_(0.02)
    w2_scale = torch.rand(
        (_EXPERTS, _HIDDEN, 1), device="cuda", generator=generator
    ).mul_(0.02)
    return w1, w2, w1_scale, w2_scale


def _inputs(tokens: int, weights):
    w1, w2, w1_scale, w2_scale = weights
    generator = torch.Generator(device="cuda").manual_seed(tokens)
    hidden = torch.randn(
        (tokens, _HIDDEN),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    routing = torch.rand((tokens, _EXPERTS - 1), device="cuda", generator=generator)
    routed_ids = routing.topk(_TOP_K - 1, dim=1, sorted=True).indices.to(torch.int32)
    shared_ids = torch.full((tokens, 1), _EXPERTS - 1, dtype=torch.int32, device="cuda")
    topk_ids = torch.cat((routed_ids, shared_ids), dim=1)
    routed_weights = torch.rand(
        (tokens, _TOP_K - 1), device="cuda", generator=generator
    )
    routed_weights.div_(routed_weights.sum(dim=1, keepdim=True))
    shared_weights = torch.full((tokens, 1), 0.4, dtype=torch.float32, device="cuda")
    topk_weights = torch.cat((routed_weights, shared_weights), dim=1)
    return hidden, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale


def _run(inputs, *, enable_fast_path: bool):
    from sglang.srt.layers.moe.moe_runner.triton_utils import fused_moe

    hidden, w1, w2, topk_weights, topk_ids, w1_scale, w2_scale = inputs
    old_value = fused_moe._enable_glm45_fused_moe
    fused_moe._enable_glm45_fused_moe = enable_fast_path
    try:
        return fused_moe.fused_experts_impl(
            hidden,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=True,
            activation="silu",
            is_gated=True,
            apply_router_weight_on_input=False,
            use_fp8_w8a8=True,
            per_channel_quant=True,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            no_combine=False,
            routed_scaling_factor=2.5,
            filter_expert=False,
            gate_up_interleaved=True,
        )
    finally:
        fused_moe._enable_glm45_fused_moe = old_value


@pytest.fixture(scope="module", autouse=True)
def _runtime_context():
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))


@pytest.mark.parametrize("tokens", _TOKENS)
def test_glm45_fast_path_matches_triton(tokens: int, weights):
    baseline_inputs = _inputs(tokens, weights)
    candidate_inputs = (baseline_inputs[0].clone(), *baseline_inputs[1:])

    expected = _run(baseline_inputs, enable_fast_path=False).clone()
    actual = _run(candidate_inputs, enable_fast_path=True).clone()

    torch.testing.assert_close(actual, expected, atol=0.10, rtol=0.03)


def test_glm45_fast_path_rejects_non_capture_shape(weights):
    from sglang.kernels.ops.moe.glm45_fused_moe import covered

    inputs = list(_inputs(16, weights))
    inputs[4] = inputs[4][:, :8].contiguous()
    assert not covered(*inputs)
