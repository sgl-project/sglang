"""Triton MoE runner <-> DeepEP normal permutes (used by Intel XPU EP)."""

import sys

import pytest
import torch

import sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe as fused_moe_module
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig, PermuteMethodPool
from sglang.srt.layers.moe.moe_runner.triton import (
    TritonMoeQuantInfo,
    TritonRunnerCore,
    TritonRunnerInput,
    TritonRunnerOutput,
    post_permute_triton_to_deepep_normal,
    pre_permute_deepep_normal_to_triton,
)
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalDispatchOutput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-c-test-cpu")

NUM_LOCAL_EXPERTS = 2
HIDDEN = 4


def _dispatch_output(hidden_states_scale=None):
    # Local expert ids, with -1 in the slots owned by another EP rank.
    topk_ids = torch.tensor([[0, -1], [-1, 1]], dtype=torch.int64)
    return DeepEPNormalDispatchOutput(
        hidden_states=torch.zeros((2, HIDDEN), dtype=torch.bfloat16),
        hidden_states_scale=hidden_states_scale,
        topk_ids=topk_ids,
        topk_weights=torch.full(topk_ids.shape, 0.5, dtype=torch.float32),
        num_recv_tokens_per_expert=[1, 1],
    )


def _quant_info():
    return TritonMoeQuantInfo(
        w13_weight=torch.empty((NUM_LOCAL_EXPERTS, 2 * HIDDEN, HIDDEN)),
        w2_weight=torch.empty((NUM_LOCAL_EXPERTS, HIDDEN, HIDDEN)),
    )


def _runner_config(**overrides):
    kwargs = dict(
        num_experts=4,
        num_local_experts=NUM_LOCAL_EXPERTS,
        top_k=2,
        routed_scaling_factor=2.5,
    )
    kwargs.update(overrides)
    return MoeRunnerConfig(**kwargs)


def _stub_prepare(monkeypatch):
    """Replace the block-alignment kernels with fixed CPU tensors."""

    def _prepare_fused_moe_run(*args, **kwargs):
        return (
            {"BLOCK_SIZE_M": 16},
            None,
            False,
            False,
            torch.zeros(4, dtype=torch.int32),
            torch.tensor([0, -1], dtype=torch.int32),
            torch.tensor([4], dtype=torch.int32),
        )

    monkeypatch.setattr(
        fused_moe_module, "_prepare_fused_moe_run", _prepare_fused_moe_run
    )


def test_permutes_are_registered():
    assert (
        PermuteMethodPool.get_pre_permute("deepep_normal", "triton")
        is pre_permute_deepep_normal_to_triton
    )
    assert (
        PermuteMethodPool.get_post_permute("triton", "deepep_normal")
        is post_permute_triton_to_deepep_normal
    )


def test_pre_permute_passes_recv_layout_through(monkeypatch):
    _stub_prepare(monkeypatch)
    dispatch_output = _dispatch_output()
    running_state = {}

    runner_input = pre_permute_deepep_normal_to_triton(
        dispatch_output,
        _quant_info(),
        _runner_config(),
        running_state,
    )

    assert isinstance(runner_input, TritonRunnerInput)
    # The recv tokens and their local expert ids are consumed as-is.
    assert runner_input.hidden_states is dispatch_output.hidden_states
    assert runner_input.topk_ids is dispatch_output.topk_ids
    assert runner_input.topk_weights is dispatch_output.topk_weights
    assert running_state["config"] == {"BLOCK_SIZE_M": 16}
    # DeepEP combine still has to reduce across ranks, so the scaling is left
    # to the caller.
    assert runner_input.apply_routed_scaling_factor is False


def test_pre_permute_rejects_quantized_dispatch(monkeypatch):
    _stub_prepare(monkeypatch)

    with pytest.raises(AssertionError, match="bf16 DeepEP dispatch"):
        pre_permute_deepep_normal_to_triton(
            _dispatch_output(hidden_states_scale=torch.ones((2, 1))),
            _quant_info(),
            _runner_config(),
            {},
        )


def test_pre_permute_rejects_router_weight_on_input(monkeypatch):
    _stub_prepare(monkeypatch)

    with pytest.raises(AssertionError, match="apply_router_weight_on_input"):
        pre_permute_deepep_normal_to_triton(
            _dispatch_output(),
            _quant_info(),
            _runner_config(apply_router_weight_on_input=True),
            {},
        )


def test_post_permute_carries_topk_for_combine():
    dispatch_output = _dispatch_output()
    running_state = {
        "topk_ids": dispatch_output.topk_ids,
        "topk_weights": dispatch_output.topk_weights,
    }
    hidden_states = torch.zeros((2, HIDDEN), dtype=torch.bfloat16)

    combine_input = post_permute_triton_to_deepep_normal(
        TritonRunnerOutput(hidden_states=hidden_states),
        _quant_info(),
        _runner_config(),
        running_state,
    )

    assert combine_input.format.value == "deepep_normal"
    assert combine_input.hidden_states is hidden_states
    assert combine_input.topk_ids is dispatch_output.topk_ids
    assert combine_input.topk_weights is dispatch_output.topk_weights


def test_runner_core_honors_apply_routed_scaling_factor(monkeypatch):
    captured = {}

    def _fused_moe_kernel_sequence(*args, **kwargs):
        captured.update(kwargs)
        return args[0]

    monkeypatch.setattr(
        fused_moe_module, "_fused_moe_kernel_sequence", _fused_moe_kernel_sequence
    )

    def _runner_input(**overrides):
        dispatch_output = _dispatch_output()
        return TritonRunnerInput(
            hidden_states=dispatch_output.hidden_states,
            topk_weights=dispatch_output.topk_weights,
            topk_ids=dispatch_output.topk_ids,
            sorted_token_ids=torch.zeros(4, dtype=torch.int32),
            expert_ids=torch.tensor([0, -1], dtype=torch.int32),
            num_tokens_post_padded=torch.tensor([4], dtype=torch.int32),
            **overrides,
        )

    core = TritonRunnerCore(_runner_config())
    running_state = {"config": {"BLOCK_SIZE_M": 16}}

    core.run(
        _runner_input(apply_routed_scaling_factor=False),
        _quant_info(),
        running_state,
    )
    assert captured["routed_scaling_factor"] is None
    # Local expert ids mean the kernel must skip the -1 blocks.
    assert captured["filter_expert"] is True

    captured.clear()
    core.run(_runner_input(), _quant_info(), running_state)
    assert captured["routed_scaling_factor"] == 2.5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
