"""Plan rules of the epilogue down-B family (the base down GEMM adds the
LoRA delta in its epilogue)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from sglang.kernels.ops.lora.moe.activation import ActivationFn
from sglang.srt.lora.moe.plan import (
    ActFamily,
    ActSpec,
    AFamily,
    ASpec,
    BFamily,
    BSpec,
    DownOverlap,
    FinalizeFamily,
    FinalizeSpec,
    GateUpOverlap,
    MoeLoraLaunchConfig,
    MoePlan,
    RouteRequirement,
    Site,
    _validate_plan_launch_config,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _plan(**overrides) -> MoePlan:
    fields = dict(
        gate_up_a=ASpec(Site.GATE_UP, AFamily.GROUPED),
        act=ActSpec(ActFamily.B_ACTIVATION, ActivationFn.SILU),
        down_a=ASpec(Site.DOWN, AFamily.GROUPED),
        down_b=BSpec(Site.DOWN, BFamily.EPILOGUE),
        finalize=FinalizeSpec(FinalizeFamily.MATERIALIZED),
        gate_up_overlap=GateUpOverlap.GATE_UP_A,
    )
    fields.update(overrides)
    return MoePlan(**fields)


def test_epilogue_down_b_is_a_valid_per_expert_prefill_plan():
    plan = _plan()
    # The A kernel places the bridge by provider row; the GEMM needs no route.
    assert plan.route_requirements() == frozenset({RouteRequirement.ALIGNED_PER_EXPERT})
    # The default launch tiles satisfy it: no down_b dot tile is demanded.
    _validate_plan_launch_config(plan, MoeLoraLaunchConfig())


def test_epilogue_down_b_rejects_other_sites_and_shared_outer():
    with pytest.raises(ValueError, match="per-expert down delta"):
        BSpec(Site.GATE_UP, BFamily.EPILOGUE)
    with pytest.raises(ValueError, match="per-expert down delta"):
        BSpec(Site.DOWN, BFamily.EPILOGUE, is_shared_outer=True)


@pytest.mark.parametrize(
    "overrides",
    [
        {"down_overlap": DownOverlap.DOWN_A},
        {"down_overlap": DownOverlap.DOWN_A_B},
        {"down_b_into_base": True},
    ],
)
def test_epilogue_down_b_runs_after_the_bridge_and_owns_the_delta(overrides):
    with pytest.raises(ValueError, match="epilogue down-B"):
        _plan(**overrides)


def test_separate_down_b_plan_is_unchanged():
    plan = replace(
        _plan(),
        down_b=BSpec(Site.DOWN, BFamily.GROUPED),
        down_overlap=DownOverlap.DOWN_A,
    )
    assert RouteRequirement.ALIGNED_PER_EXPERT in plan.route_requirements()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
