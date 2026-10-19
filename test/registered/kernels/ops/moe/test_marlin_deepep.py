"""Marlin's DeepEP adapters: real kernels, masking, weighting and graph replay."""

from dataclasses import replace

import pytest
import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.marlin import (
    fused_experts_deepep_to_marlin,
    fused_experts_none_to_marlin,
)
from sglang.srt.layers.moe.token_dispatcher import (
    DeepEPLLDispatchOutput,
    DeepEPNormalDispatchOutput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.marlin_deepep_utils import (
    assert_reference_close,
    make_experts,
    reference,
    reference_tolerances,
)

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")


@pytest.mark.parametrize("format", ["gptq4", "gptq8", "awq", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("tokens", [0, 1, 17])
def test_normal(format, tokens):
    info, matrices = make_experts(format)
    config = MoeRunnerConfig(routed_scaling_factor=1.7 if format != "mxfp4" else None)
    x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, 4, (tokens, 2), device="cuda")
    ids[:, 1] = -1
    weights = torch.rand(tokens, 2, device="cuda") * 1.3
    if tokens > 1:
        ids[0] = -1  # A fully masked row must be zero, including shared scratch.
    dispatch = DeepEPNormalDispatchOutput(x, None, ids, weights, [])
    # A deliberately wrong mapping detects accidental double mapping.
    ep_info = replace(
        info, expert_map=torch.tensor([3, 2, 1, 0], device="cuda"), global_num_experts=4
    )
    result = fused_experts_deepep_to_marlin(dispatch, ep_info, config).hidden_states
    expected = reference(x, ids, weights, matrices, config)
    assert_reference_close(result, expected, format)
    if tokens > 1:
        assert torch.count_nonzero(result[0]) == 0
    valid_ids = ids.clamp_min(0)
    valid_weights = weights.masked_fill(ids < 0, 0)
    standard = fused_experts_none_to_marlin(
        StandardDispatchOutput(
            x, None, StandardTopKOutput(valid_weights, valid_ids, None)
        ),
        info,
        config,
    )
    torch.testing.assert_close(
        result, standard.hidden_states, **reference_tolerances(format)
    )


@pytest.mark.parametrize("tokens", [1, 17])
def test_standard_expert_parallel_masked_routes(tokens):
    info, matrices = make_experts("awq")
    config = MoeRunnerConfig(num_experts=8, num_local_experts=4)
    x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
    ids = torch.randint(0, 4, (tokens, 2), device="cuda", dtype=torch.int32)
    ids[:, 1] = -1
    weights = torch.rand(tokens, 2, device="cuda")
    output = fused_experts_none_to_marlin(
        StandardDispatchOutput(x, None, StandardTopKOutput(weights, ids, None)),
        info,
        config,
    ).hidden_states
    assert_reference_close(output, reference(x, ids, weights, matrices, config), "awq")


@pytest.mark.parametrize("format", ["gptq4", "gptq8", "awq", "mxfp4", "nvfp4"])
@pytest.mark.parametrize("scaled", [False, True])
def test_low_latency_graph(format, scaled):
    info, matrices = make_experts(format, bias=True)
    config = MoeRunnerConfig(
        routed_scaling_factor=1.7 if scaled and format != "mxfp4" else None
    )
    x = torch.randn(4, 17, 256, device="cuda", dtype=torch.bfloat16)
    counts = torch.tensor([1, 0, 17, 3], device="cuda", dtype=torch.int32)
    ids = torch.tensor([[0, 2]], device="cuda")
    weights = torch.tensor([[0.2, 0.7]], device="cuda")
    dispatch = DeepEPLLDispatchOutput(x, None, ids, weights, counts, 17)

    def run():
        return fused_experts_deepep_to_marlin(dispatch, info, config).hidden_states

    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    for new_counts in (
        [1, 0, 17, 3],
        [0, 7, 2, 17],
        [17, 17, 0, 0],
        [0, 0, 17, 17],
        [0, 0, 0, 0],
    ):
        counts.copy_(torch.tensor(new_counts, device="cuda", dtype=torch.int32))
        x.normal_()
        graph.replay()
        eager = run()
        local_ids = torch.arange(4, device="cuda").repeat_interleave(17).view(-1, 1)
        valid = torch.arange(17, device="cuda")[None, :] < counts[:, None]
        expected = reference(
            x.flatten(0, 1), local_ids, valid.reshape(-1, 1).float(), matrices, config
        ).view_as(x)
        torch.testing.assert_close(
            captured[valid], eager[valid], **reference_tolerances(format)
        )
        assert_reference_close(captured[valid], expected[valid], format)
        # Neither expert computation nor combine may depend on input padding.
        x.masked_fill_(~valid[..., None], float("nan"))
        graph.replay()
        assert_reference_close(captured[valid], expected[valid], format)


@pytest.mark.parametrize(
    "format,gated,activation",
    [
        ("gptq4", False, "relu2"),
        ("gptq8", False, "silu"),
        ("nvfp4", False, "relu2"),
        ("mxfp4", True, "situ"),
    ],
)
def test_activation_and_bias(format, gated, activation):
    info, matrices = make_experts(format, gated=gated, bias=True)
    config = MoeRunnerConfig(is_gated=gated, activation=activation)
    x = torch.randn(7, 256, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor(
        [[0, 2], [1, -1], [2, 3], [-1, -1], [0, 1], [1, 2], [3, 0]], device="cuda"
    )
    weights = torch.rand(7, 2, device="cuda")
    output = fused_experts_deepep_to_marlin(
        DeepEPNormalDispatchOutput(x, None, ids, weights, []), info, config
    )
    torch.testing.assert_close(
        output.hidden_states,
        reference(x, ids, weights, matrices, config),
        **reference_tolerances(format),
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
