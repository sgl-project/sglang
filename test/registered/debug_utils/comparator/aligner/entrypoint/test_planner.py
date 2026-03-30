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
from sglang.srt.debug_utils.comparator.aligner.reorderer.types import (
    ReordererPlan,
    ZigzagToNaturalThdParams,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import (
    CpThdConcatParams,
    UnsharderPlan,
)
from sglang.srt.debug_utils.comparator.dims_spec import TokenLayout
from sglang.srt.debug_utils.comparator.utils import Pair
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="stage-a-test-cpu", nightly=True)


def _make_meta(
    *,
    step: int = 0,
    dims: Optional[str] = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    cp_rank: int = 0,
    cp_size: int = 1,
    extra_parallel_info: Optional[dict[str, int]] = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {"step": step}
    if dims is not None:
        meta["dims"] = dims
    parallel_info: dict[str, int] = {
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "cp_rank": cp_rank,
        "cp_size": cp_size,
    }
    if extra_parallel_info is not None:
        parallel_info.update(extra_parallel_info)
    meta["sglang_parallel_info"] = parallel_info
    return meta


class TestComputePerStepSubPlans:
    def test_empty_metas(self) -> None:
        result: list[AlignerPerStepSubPlan] = compute_per_step_sub_plans(metas=[])
        assert result == []

    def test_single_meta(self) -> None:
        result: list[AlignerPerStepSubPlan] = compute_per_step_sub_plans(
            metas=[_make_meta(dims="b h[tp]", tp_size=2)]
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
                _make_meta(dims="b h[tp]", tp_rank=0, tp_size=2),
                _make_meta(dims="b h[tp]", tp_rank=1, tp_size=2),
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
                _make_meta(dims="b s[cp:zigzag] h", cp_rank=0, cp_size=2),
                _make_meta(dims="b s[cp:zigzag] h", cp_rank=1, cp_size=2),
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
            token_aligner_mode=None,
            token_aligner_plan=None,
        )

        assert len(plan.per_step_plans.x) == 1
        assert len(plan.per_step_plans.y) == 1
        assert plan.token_aligner_plan is None

    def test_preserves_token_aligner_plan(self) -> None:
        from sglang.srt.debug_utils.comparator.aligner.token_aligner.smart.types import (
            TokenAlignerPlan,
            TokenLocator,
        )

        ta_plan = TokenAlignerPlan(
            locators=Pair(
                x=TokenLocator(steps=[0], token_index_in_step=[0]),
                y=TokenLocator(steps=[0], token_index_in_step=[0]),
            ),
            layouts=Pair(x=TokenLayout.T, y=TokenLayout.T),
        )

        plan: AlignerPlan = compute_aligner_plan(
            metas_pair=Pair(x=[_make_meta()], y=[_make_meta()]),
            token_aligner_mode="smart",
            token_aligner_plan=ta_plan,
        )

        assert plan.token_aligner_plan is ta_plan
        assert plan.token_aligner_mode == "smart"


class TestComputePerStepSubPlansThd:
    def test_thd_zigzag_returns_thd_plans(self) -> None:
        """t[cp:zigzag] h[tp] generates THD-typed unsharder + reorderer plans."""
        thd_global_seq_lens: list[int] = [100, 64, 92]
        result: list[AlignerPerStepSubPlan] = compute_per_step_sub_plans(
            metas=[
                _make_meta(
                    dims="t[cp:zigzag] h[tp]",
                    cp_rank=0,
                    cp_size=2,
                    tp_rank=0,
                    tp_size=2,
                ),
                _make_meta(
                    dims="t[cp:zigzag] h[tp]",
                    cp_rank=0,
                    cp_size=2,
                    tp_rank=1,
                    tp_size=2,
                ),
                _make_meta(
                    dims="t[cp:zigzag] h[tp]",
                    cp_rank=1,
                    cp_size=2,
                    tp_rank=0,
                    tp_size=2,
                ),
                _make_meta(
                    dims="t[cp:zigzag] h[tp]",
                    cp_rank=1,
                    cp_size=2,
                    tp_rank=1,
                    tp_size=2,
                ),
            ],
            thd_global_seq_lens=thd_global_seq_lens,
        )

        unsharder_plans: list[UnsharderPlan] = [
            p for p in result if isinstance(p, UnsharderPlan)
        ]
        reorderer_plans: list[ReordererPlan] = [
            p for p in result if isinstance(p, ReordererPlan)
        ]

        # Should have at least one THD concat plan for CP axis
        thd_concat_plans: list[UnsharderPlan] = [
            p for p in unsharder_plans if isinstance(p.params, CpThdConcatParams)
        ]
        assert len(thd_concat_plans) == 1
        assert thd_concat_plans[0].params.seq_lens_per_rank == [50, 32, 46]

        # Should have exactly one THD reorder plan
        assert len(reorderer_plans) == 1
        assert isinstance(reorderer_plans[0].params, ZigzagToNaturalThdParams)
        assert reorderer_plans[0].params.cp_size == 2
        # Reorder seq_lens = global seq_lens (reorder happens after unshard)
        assert reorderer_plans[0].params.seq_lens == [100, 64, 92]


class TestComputePerStepSubPlansDpFiltered:
    """Tests that compute_per_step_sub_plans passes dp_filtered_axis to unsharder,
    so DP axes already handled by the upstream DP filter don't cause validation errors.
    """

    def test_dp2_tp2_does_not_raise(self) -> None:
        """DP2 + TP2, dims='t h[tp]' → should not raise despite DP being active."""
        result: list[AlignerPerStepSubPlan] = compute_per_step_sub_plans(
            metas=[
                _make_meta(
                    dims="t h[tp]",
                    tp_rank=0,
                    tp_size=2,
                    extra_parallel_info={"dp_rank": 0, "dp_size": 2},
                ),
                _make_meta(
                    dims="t h[tp]",
                    tp_rank=1,
                    tp_size=2,
                    extra_parallel_info={"dp_rank": 0, "dp_size": 2},
                ),
            ]
        )
        unsharder_plans: list[UnsharderPlan] = [
            p for p in result if isinstance(p, UnsharderPlan)
        ]
        assert len(unsharder_plans) == 1
        assert unsharder_plans[0].axis.value == "tp"

    def test_dp2_only_no_sharding_does_not_raise(self) -> None:
        """DP2 only, dims='t h' → should not raise, no plans produced."""
        result: list[AlignerPerStepSubPlan] = compute_per_step_sub_plans(
            metas=[
                _make_meta(
                    dims="t h",
                    extra_parallel_info={"dp_rank": 0, "dp_size": 2},
                ),
                _make_meta(
                    dims="t h",
                    extra_parallel_info={"dp_rank": 0, "dp_size": 2},
                ),
            ]
        )
        assert result == []

    def test_dp_alias_passes_correct_filtered_axis(self) -> None:
        """dims with '# dp:=moe_dp', metas have moe_dp → should not raise."""
        result: list[AlignerPerStepSubPlan] = compute_per_step_sub_plans(
            metas=[
                _make_meta(
                    dims="t h[tp] # dp:=moe_dp",
                    tp_rank=0,
                    tp_size=2,
                    extra_parallel_info={"moe_dp_rank": 0, "moe_dp_size": 2},
                ),
                _make_meta(
                    dims="t h[tp] # dp:=moe_dp",
                    tp_rank=1,
                    tp_size=2,
                    extra_parallel_info={"moe_dp_rank": 0, "moe_dp_size": 2},
                ),
            ]
        )
        unsharder_plans: list[UnsharderPlan] = [
            p for p in result if isinstance(p, UnsharderPlan)
        ]
        assert len(unsharder_plans) == 1
        assert unsharder_plans[0].axis.value == "tp"

    def test_dp2_tp2_cp2_does_not_raise(self) -> None:
        """DP2 + TP2 + CP2, dims='s[cp] h[tp]' → should not raise."""
        metas = []
        for cp_rank in range(2):
            for tp_rank in range(2):
                metas.append(
                    _make_meta(
                        dims="s[cp] h[tp]",
                        tp_rank=tp_rank,
                        tp_size=2,
                        cp_rank=cp_rank,
                        cp_size=2,
                        extra_parallel_info={"dp_rank": 0, "dp_size": 2},
                    )
                )
        result: list[AlignerPerStepSubPlan] = compute_per_step_sub_plans(metas=metas)
        unsharder_plans: list[UnsharderPlan] = [
            p for p in result if isinstance(p, UnsharderPlan)
        ]
        axes = {p.axis.value for p in unsharder_plans}
        assert axes == {"cp", "tp"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
