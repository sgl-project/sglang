import sys

import pytest
import torch

from sglang.kernels.ops.moe.bailing_moe_topk import (
    bailing_moe_biased_grouped_topk,
)
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.srt.layers.moe.topk import (
    biased_grouped_topk_gpu,
    biased_grouped_topk_impl,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _make_inputs(num_tokens: int, seed: int):
    torch.manual_seed(seed)
    hidden_states = torch.empty((num_tokens, 1), dtype=torch.float32, device="cuda")
    gating_output = torch.randn((num_tokens, 512), dtype=torch.float32, device="cuda")
    correction_bias = torch.randn(512, dtype=torch.float32, device="cuda") * 0.1
    return hidden_states, gating_output, correction_bias


def _scatter_by_expert(
    weights: torch.Tensor, ids: torch.Tensor, num_experts: int
) -> torch.Tensor:
    dense = torch.zeros(
        (weights.shape[0], num_experts), dtype=torch.float32, device=weights.device
    )
    dense.scatter_(1, ids.long(), weights)
    return dense


@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
def test_bailing_moe_biased_grouped_topk_matches_triton_router(
    renormalize: bool,
    apply_routed_scaling_factor_on_output: bool,
) -> None:
    _, gating_output, correction_bias = _make_inputs(
        num_tokens=257,
        seed=1000 + int(renormalize) * 10 + int(apply_routed_scaling_factor_on_output),
    )
    routed_scaling_factor = 2.5

    topk_weights, topk_ids = bailing_moe_biased_grouped_topk(
        gating_output,
        correction_bias,
        num_expert_group=8,
        topk_group=4,
        topk=8,
        renormalize=renormalize,
        num_fused_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )
    ref_weights, ref_ids = moe_fused_gate(
        gating_output,
        correction_bias,
        topk=8,
        scoring_func="sigmoid",
        num_fused_shared_experts=0,
        renormalize=renormalize,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        num_expert_group=8,
        topk_group=4,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        _scatter_by_expert(topk_weights, topk_ids, 512),
        _scatter_by_expert(ref_weights, ref_ids, 512),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("apply_routed_scaling_factor_on_output", [False, True])
def test_bailing_moe_biased_grouped_topk_with_fused_shared_expert(
    apply_routed_scaling_factor_on_output: bool,
) -> None:
    hidden_states, gating_output, correction_bias = _make_inputs(
        num_tokens=129,
        seed=2000 + int(apply_routed_scaling_factor_on_output),
    )
    routed_scaling_factor = 2.5

    topk_weights, topk_ids = bailing_moe_biased_grouped_topk(
        gating_output,
        correction_bias,
        num_expert_group=8,
        topk_group=4,
        topk=9,
        renormalize=True,
        num_fused_shared_experts=1,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )
    ref_weights, ref_ids = biased_grouped_topk_impl(
        hidden_states,
        gating_output,
        correction_bias,
        topk=9,
        renormalize=True,
        num_expert_group=8,
        topk_group=4,
        num_fused_shared_experts=1,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        _scatter_by_expert(topk_weights, topk_ids, 513),
        _scatter_by_expert(ref_weights, ref_ids, 513),
        rtol=1e-3,
        atol=1e-3,
    )


def test_bailing_moe_biased_grouped_topk_zero_scores_give_zero_weights() -> None:
    """Logits that underflow the sigmoid to 0 must yield zero weights, not NaN."""
    _, gating_output, correction_bias = _make_inputs(num_tokens=17, seed=4000)
    gating_output.fill_(-1e4)

    topk_weights, _ = bailing_moe_biased_grouped_topk(
        gating_output,
        correction_bias,
        num_expert_group=8,
        topk_group=4,
        topk=9,
        renormalize=True,
        num_fused_shared_experts=1,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=True,
    )
    torch.cuda.synchronize()

    assert torch.equal(topk_weights, torch.zeros_like(topk_weights))


def test_biased_grouped_topk_gpu_dispatches_to_bailing_jit(monkeypatch) -> None:
    hidden_states, gating_output, correction_bias = _make_inputs(
        num_tokens=3,
        seed=3000,
    )
    called = {}

    def fake_bailing_topk(
        patched_gating_output,
        patched_correction_bias,
        num_expert_group,
        topk_group,
        topk,
        renormalize,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    ):
        called["args"] = (
            num_expert_group,
            topk_group,
            topk,
            renormalize,
            num_fused_shared_experts,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        )
        assert patched_gating_output.dtype == torch.float32
        assert patched_correction_bias.dtype == torch.float32
        return (
            torch.empty((patched_gating_output.shape[0], topk), device="cuda"),
            torch.empty(
                (patched_gating_output.shape[0], topk),
                dtype=torch.int32,
                device="cuda",
            ),
        )

    import sglang.kernels.ops.moe.bailing_moe_topk as bailing_moe_topk

    monkeypatch.setattr(
        bailing_moe_topk,
        "bailing_moe_biased_grouped_topk",
        fake_bailing_topk,
    )

    topk_weights, topk_ids = biased_grouped_topk_gpu(
        hidden_states,
        gating_output,
        correction_bias,
        topk=9,
        renormalize=True,
        num_expert_group=8,
        topk_group=4,
        num_fused_shared_experts=1,
        routed_scaling_factor=2.5,
        apply_routed_scaling_factor_on_output=True,
    )

    assert called["args"] == (8, 4, 9, True, 1, 2.5, True)
    assert topk_weights.shape == (3, 9)
    assert topk_ids.shape == (3, 9)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
