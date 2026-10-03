"""The unified Triton router, admitted on ROCm and for single-group routing.

Pins that the router selects what the torch reference selects, and that the
shared expert appears exactly once -- two places can emit it (this router, or
_post_process_topk_ids), and if both do, the id is written twice and evicts a
real routed expert while the model keeps producing plausible logits.

The reference is `biased_grouped_topk_impl`, not `select_experts` with the flag
off: on ROCm that is the aiter path, which casts the correction bias down to the
gating dtype, and GLM-5.2 keeps that bias where bf16 cannot separate neighbours.
It reorders routing on its own.
"""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="jit-kernel-unit-test-amd")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

E, TOPK_ROUTED, SHARED, SCALE = 256, 8, 1, 2.5
HIDDEN = 512


def _jit_routed(monkeypatch, logits, hidden, bias, groups):
    """Routed ids from select_experts with the unified router on, sorted."""
    monkeypatch.setenv("SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK", "1")
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts

    cfg = TopKConfig(
        top_k=TOPK_ROUTED + SHARED,
        renormalize=True,
        use_grouped_topk=True,
        num_expert_group=groups,
        num_fused_shared_experts=SHARED,
        topk_group=1,
        scoring_func="sigmoid",
        correction_bias=bias,
        routed_scaling_factor=SCALE,
        apply_routed_scaling_factor_on_output=False,
    )
    ids = select_experts(hidden, logits, cfg).topk_ids.long()
    return ids, ids[ids < E].view(ids.shape[0], TOPK_ROUTED).sort(-1).values


def _reference_routed(logits, hidden, bias, groups):
    """Routed ids from the torch reference, sorted.

    It overwrites the last slot with the shared id, so the routed experts are
    what survives below E.
    """
    from sglang.srt.layers.moe.topk import biased_grouped_topk_impl

    _, ids = biased_grouped_topk_impl(
        hidden_states=hidden,
        gating_output=logits,
        correction_bias=bias,
        topk=TOPK_ROUTED + SHARED,
        renormalize=True,
        num_expert_group=groups,
        topk_group=1,
        num_fused_shared_experts=SHARED,
        routed_scaling_factor=SCALE,
    )
    ids = ids.long()
    return ids[ids < E].view(ids.shape[0], TOPK_ROUTED).sort(-1).values


def _inputs(tokens, dev="cuda"):
    torch.manual_seed(0)
    hidden = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device=dev)
    # A narrow band at a large offset, as GLM-5.2's bias is: near-equal values
    # are what a router has to keep apart.
    bias = (7.0 + 0.04 * torch.randn(E, device=dev)).float()
    # fp32 logits: in bf16 the two round the sigmoid differently and near-equal
    # rows would flip for that reason alone.
    logits = torch.randn(tokens, E, dtype=torch.float32, device=dev)
    return hidden, bias, logits


def _groups(request_groups: int) -> int:
    """Skip single-group cases where the gate does not admit them.

    biased_grouped_topk_gpu admits one group on ROCm only; CUDA still requires
    num_expert_group > 1, so a groups=1 case there never reaches the router and
    the test would assert against a path it did not exercise.
    """
    from sglang.srt.layers.moe import topk as topk_mod

    if request_groups == 1 and not topk_mod._is_hip:
        pytest.skip("single-group routing is admitted on ROCm only")
    return request_groups


@pytest.mark.parametrize("groups", [1, 8])
@pytest.mark.parametrize("tokens", [6, 48, 256])
def test_jit_router_selects_what_the_reference_selects(monkeypatch, tokens, groups):
    groups = _groups(groups)
    hidden, bias, logits = _inputs(tokens)

    want = _reference_routed(logits, hidden, bias, groups)
    _, got = _jit_routed(monkeypatch, logits, hidden, bias, groups)

    score = logits.sigmoid() + bias
    for r in (want != got).any(-1).nonzero().flatten().tolist():
        only_want = sorted(set(want[r].tolist()) - set(got[r].tolist()))
        only_got = sorted(set(got[r].tolist()) - set(want[r].tolist()))
        # Exact ties may break either way; nothing else may.
        for x, y in zip(only_want, only_got):
            assert score[r, x].item() == score[r, y].item(), (
                f"row {r}: reference took {x} (score {score[r, x].item():.9f}) "
                f"but the router took {y} (score {score[r, y].item():.9f})"
            )


@pytest.mark.parametrize("groups", [1, 8])
def test_shared_expert_appears_exactly_once(monkeypatch, groups):
    groups = _groups(groups)
    hidden, bias, logits = _inputs(48)
    ids, routed = _jit_routed(monkeypatch, logits, hidden, bias, groups)

    assert ids.shape[-1] == TOPK_ROUTED + SHARED
    shared = (ids >= E).sum(-1)
    assert torch.equal(shared, torch.full_like(shared, SHARED)), (
        f"shared expert appears {shared.tolist()} times a row, expected {SHARED}"
    )
    assert (routed[:, 1:] != routed[:, :-1]).all(), "a routed expert repeats"


@pytest.mark.parametrize("use_aiter", [True, False])
def test_router_is_asked_for_the_total_width(monkeypatch, use_aiter):
    """The width handed to the kernel, for both callers.

    select_experts passes `num_routed_topk if _use_aiter else top_k`; the kernel
    wants the total either way. Whichever GPU runs the suite fixes `_use_aiter`
    and can only exercise one half, so pin the arithmetic here.
    """
    from sglang.srt.layers.moe import topk as topk_mod

    seen = {}

    class _Captured(Exception):
        pass

    def _capture(scores, bias, topk, **kwargs):
        seen["topk"] = topk
        raise _Captured

    monkeypatch.setenv("SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK", "1")
    monkeypatch.setattr(topk_mod, "_use_aiter", use_aiter, raising=False)
    monkeypatch.setattr(
        "sglang.kernels.ops.moe.moe_fused_gate.moe_fused_gate", _capture
    )

    hidden, bias, logits = _inputs(4)
    # The aiter caller hands over routed-only; everyone else the total.
    topk_in = TOPK_ROUTED if use_aiter else TOPK_ROUTED + SHARED
    with pytest.raises(_Captured):
        topk_mod.biased_grouped_topk_gpu(
            hidden_states=hidden,
            gating_output=logits,
            correction_bias=bias,
            topk=topk_in,
            renormalize=True,
            # 8 groups, not 1: the arithmetic under test is the same either
            # way, and only this value reaches the router on both platforms.
            num_expert_group=8,
            topk_group=1,
            num_fused_shared_experts=SHARED,
            routed_scaling_factor=SCALE,
        )

    assert seen["topk"] == TOPK_ROUTED + SHARED, (
        f"_use_aiter={use_aiter}: caller passed topk={topk_in}, kernel was asked "
        f"for {seen['topk']} slots, expected {TOPK_ROUTED + SHARED} "
        f"(routed {TOPK_ROUTED} + shared {SHARED})"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
