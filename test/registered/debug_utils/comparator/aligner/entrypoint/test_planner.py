import sys
from typing import Any, Optional

import pytest

from sglang.srt.debug_utils.comparator.aligner.entrypoint.planner import (
    _compute_per_step_plans,
    compute_aligner_plan,
    compute_per_step_sub_plans,
)
from sglang.srt.debug_utils.comparator.aligner.entrypoint.types import (
    AlignerPerStepPlan,
    AlignerPerStepSubPlan,
    AlignerPlan,
)
from sglang.srt.debug_utils.comparator.aligner.reorderer.types import ReordererPlan
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import UnsharderPlan
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="default", nightly=True)


def _make_meta(
    *,
    step: int = 0,
    dims: Optional[str] = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    cp_rank: int = 0,
    cp_size: int = 1,
) -> dict[str, Any]:
    meta: dict[str, Any] = {"step": step}
    if dims is not None:
        meta["dims"] = dims
    meta["sglang_parallel_info"] = {
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "cp_rank": cp_rank,
        "cp_size": cp_size,
    }
    return meta


class TestComputePerStepSubPlans:
    def test_empty_metas(self) -> None:
        result: list[AlignerPerStepSubPlan] = compute_per_step_sub_plans(metas=[])
        assert result == []

    def test_single_meta(self) -> None:
        result: list[AlignerPerStepSubPlan] = compute_per_step_sub_plans(
            metas=[_make_meta(dims="b h(tp)", tp_size=2)]
        )
        assert result == []

    def test_dims_none(self) -> None:
        result: list[AlignerPerStepSubPlan] = compute_per_step_sub_plans(
            metas=[
                _make_meta(tp_rank=0, tp_size=2),
                _make_meta(tp_rank=1, tp_size=2),
            ]
        )
        assert result == []

    def test_tp_sharded_returns_unsharder_plan(self) -> None:
        result: list[AlignerPerStepSubPlan] = compute_per_step_sub_plans(
            metas=[
                _make_meta(dims="b h(tp)", tp_rank=0, tp_size=2),
                _make_meta(dims="b h(tp)", tp_rank=1, tp_size=2),
            ]
        )
        assert len(result) >= 1
        unsharder_plans: list[UnsharderPlan] = [
            p for p in result if isinstance(p, UnsharderPlan)
        ]
        assert len(unsharder_plans) >= 1

    def test_zigzag_returns_both_plans(self) -> None:
        result: list[AlignerPerStepSubPlan] = compute_per_step_sub_plans(
            metas=[
                _make_meta(dims="b s(cp,zigzag) h", cp_rank=0, cp_size=2),
                _make_meta(dims="b s(cp,zigzag) h", cp_rank=1, cp_size=2),
            ]
        )
        unsharder_plans: list[UnsharderPlan] = [
            p for p in result if isinstance(p, UnsharderPlan)
        ]
        reorderer_plans: list[ReordererPlan] = [
            p for p in result if isinstance(p, ReordererPlan)
        ]
        assert len(unsharder_plans) >= 1
        assert len(reorderer_plans) >= 1


class TestComputePerStepPlans:
    def test_groups_by_step(self) -> None:
        metas: list[dict[str, Any]] = [
            _make_meta(step=0, tp_rank=0, tp_size=2),
            _make_meta(step=0, tp_rank=1, tp_size=2),
            _make_meta(step=1, tp_rank=0, tp_size=1),
        ]
        result: list[AlignerPerStepPlan] = _compute_per_step_plans(metas=metas)

        assert len(result) == 2
        assert result[0].step == 0
        assert result[0].input_object_indices == [0, 1]
        assert result[1].step == 1
        assert result[1].input_object_indices == [2]

    def test_sorted_by_step(self) -> None:
        metas: list[dict[str, Any]] = [
            _make_meta(step=2),
            _make_meta(step=0),
            _make_meta(step=1),
        ]
        result: list[AlignerPerStepPlan] = _compute_per_step_plans(metas=metas)

        steps: list[int] = [p.step for p in result]
        assert steps == [0, 1, 2]

    def test_single_meta_per_step_empty_sub_plans(self) -> None:
        metas: list[dict[str, Any]] = [
            _make_meta(step=0),
            _make_meta(step=1),
        ]
        result: list[AlignerPerStepPlan] = _compute_per_step_plans(metas=metas)

        assert len(result) == 2
        assert all(plan.sub_plans == [] for plan in result)


class TestComputeAlignerPlan:
    def test_wraps_both_sides(self) -> None:
        metas_x: list[dict[str, Any]] = [_make_meta(step=0)]
        metas_y: list[dict[str, Any]] = [_make_meta(step=0)]

        plan: AlignerPlan = compute_aligner_plan(
            metas_pair=Pair(x=metas_x, y=metas_y),
            token_aligner_plan=None,
        )

        assert len(plan.per_step_plans.x) == 1
        assert len(plan.per_step_plans.y) == 1
        assert plan.token_aligner_plan is None

    def test_preserves_token_aligner_plan(self) -> None:
        from sglang.srt.debug_utils.comparator.aligner.token_aligner.types import (
            TokenAlignerPlan,
            TokenLocator,
        )

        ta_plan = TokenAlignerPlan(
            locators=Pair(
                x=TokenLocator(steps=[0], token_index_in_step=[0]),
                y=TokenLocator(steps=[0], token_index_in_step=[0]),
            ),
        )

        plan: AlignerPlan = compute_aligner_plan(
            metas_pair=Pair(x=[_make_meta()], y=[_make_meta()]),
            token_aligner_plan=ta_plan,
        )

        assert plan.token_aligner_plan is ta_plan


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
