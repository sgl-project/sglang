import sys

import pytest

from sglang.srt.debug_utils.comparator.aligner.unsharder.planner import (
    compute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import (
    AxisInfo,
    ConcatParams,
    PickParams,
)
from sglang.srt.debug_utils.comparator.dims import ParallelAxis, parse_dims
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestComputeUnsharderPlan:
    def test_tp4_plan(self) -> None:
        dim_specs = parse_dims("b s h(tp) d")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=i, axis_size=4)} for i in range(4)
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        assert len(plans) == 1
        assert plans[0].axis == ParallelAxis.TP
        assert plans[0].params.dim_name == "h"
        assert plans[0].groups == [[0, 1, 2, 3]]

    def test_inconsistent_axis_size_raises(self) -> None:
        dim_specs = parse_dims("h(tp)")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=4)},
            {ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2)},
        ]
        with pytest.raises(ValueError, match="Inconsistent axis_size"):
            compute_unsharder_plan(dim_specs, parallel_infos)

    def test_missing_axis_in_all_parallel_infos_skipped(self) -> None:
        """Axis in dims but absent from all parallel_infos -> axis_size=1, auto-skip."""
        dim_specs = parse_dims("h(tp)")
        parallel_infos = [{ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2)}]
        # TP not in any parallel_info → skipped; CP is replicated but only 1 rank
        # with size=2 → incomplete coverage
        with pytest.raises(ValueError, match="axis_rank coverage"):
            compute_unsharder_plan(dim_specs, parallel_infos)

    def test_empty_parallel_infos_raises(self) -> None:
        dim_specs = parse_dims("h(tp)")
        with pytest.raises(ValueError, match="must not be empty"):
            compute_unsharder_plan(dim_specs, [])

    def test_scrambled_world_ranks(self) -> None:
        """world_rank order != axis_rank order."""
        dim_specs = parse_dims("h(tp)")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=2, axis_size=4)},
            {ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=4)},
            {ParallelAxis.TP: AxisInfo(axis_rank=3, axis_size=4)},
            {ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=4)},
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 1
        assert plans[0].groups == [[1, 3, 0, 2]]

    def test_no_sharded_axes_returns_empty(self) -> None:
        dim_specs = parse_dims("b s d")
        parallel_infos = [{}]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert plans == []

    def test_multi_axis_plan(self) -> None:
        """Multi-axis (TP + CP) produces a 2-step plan."""
        dim_specs = parse_dims("s(cp) h(tp)")
        parallel_infos = [
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=1, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=1, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2),
            },
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        assert len(plans) == 2
        assert plans[0].axis == ParallelAxis.CP
        assert plans[1].axis == ParallelAxis.TP

    def test_cp_tp_plan(self) -> None:
        """CP=2 + TP=4 produces correct 2-step plan with correct groups."""
        dim_specs = parse_dims("s(cp) h(tp)")
        parallel_infos = []
        for cp_rank in range(2):
            for tp_rank in range(4):
                parallel_infos.append(
                    {
                        ParallelAxis.CP: AxisInfo(axis_rank=cp_rank, axis_size=2),
                        ParallelAxis.TP: AxisInfo(axis_rank=tp_rank, axis_size=4),
                    }
                )

        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        assert len(plans) == 2

        cp_plan = plans[0]
        assert cp_plan.axis == ParallelAxis.CP
        assert len(cp_plan.groups) == 4
        for group in cp_plan.groups:
            assert len(group) == 2

        tp_plan = plans[1]
        assert tp_plan.axis == ParallelAxis.TP
        assert len(tp_plan.groups) == 1
        assert len(tp_plan.groups[0]) == 4

    def test_cp_tp_scrambled_ranks(self) -> None:
        """Scrambled rank assignment still produces correct plan."""
        dim_specs = parse_dims("s(cp) h(tp)")
        parallel_infos = [
            {
                ParallelAxis.CP: AxisInfo(axis_rank=1, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=1, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        assert len(plans) == 2

        cp_plan = plans[0]
        assert cp_plan.axis == ParallelAxis.CP
        assert len(cp_plan.groups) == 2
        for group in cp_plan.groups:
            assert len(group) == 2

        tp_plan = plans[1]
        assert tp_plan.axis == ParallelAxis.TP
        assert len(tp_plan.groups) == 1
        assert len(tp_plan.groups[0]) == 2

    def test_axis_rank_coverage_incomplete_raises(self) -> None:
        """TP size=4 but only ranks 0,1,3 provided (missing rank 2)."""
        dim_specs = parse_dims("h(tp)")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=4)},
            {ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=4)},
            {ParallelAxis.TP: AxisInfo(axis_rank=3, axis_size=4)},
        ]
        with pytest.raises(ValueError, match="axis_rank coverage.*incomplete"):
            compute_unsharder_plan(dim_specs, parallel_infos)

    def test_reduction_not_implemented_raises(self) -> None:
        dim_specs = parse_dims("h(tp,partial)")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=i, axis_size=2)} for i in range(2)
        ]
        with pytest.raises(NotImplementedError, match="reduction"):
            compute_unsharder_plan(dim_specs, parallel_infos)

    def test_ordering_zigzag_accepted(self) -> None:
        dim_specs = parse_dims("s(cp,zigzag)")
        parallel_infos = [
            {ParallelAxis.CP: AxisInfo(axis_rank=i, axis_size=2)} for i in range(2)
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 1
        assert plans[0].axis == ParallelAxis.CP

    def test_ordering_natural_accepted(self) -> None:
        dim_specs = parse_dims("s(cp,natural)")
        parallel_infos = [
            {ParallelAxis.CP: AxisInfo(axis_rank=i, axis_size=2)} for i in range(2)
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 1
        assert plans[0].axis == ParallelAxis.CP

    def test_three_axis_plan(self) -> None:
        """EP=2 + CP=2 + TP=2 produces a 3-step plan."""
        dim_specs = parse_dims("b e(ep) s(cp) h(tp)")
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = []
        for ep_rank in range(2):
            for cp_rank in range(2):
                for tp_rank in range(2):
                    parallel_infos.append(
                        {
                            ParallelAxis.EP: AxisInfo(axis_rank=ep_rank, axis_size=2),
                            ParallelAxis.CP: AxisInfo(axis_rank=cp_rank, axis_size=2),
                            ParallelAxis.TP: AxisInfo(axis_rank=tp_rank, axis_size=2),
                        }
                    )

        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        assert len(plans) == 3
        assert plans[0].axis == ParallelAxis.EP
        assert plans[1].axis == ParallelAxis.CP
        assert plans[2].axis == ParallelAxis.TP

        # Step 0 (EP): 8 tensors → 4 (groups of 2)
        assert len(plans[0].groups) == 4
        for group in plans[0].groups:
            assert len(group) == 2

        # Step 1 (CP): 4 tensors → 2 (groups of 2)
        assert len(plans[1].groups) == 2
        for group in plans[1].groups:
            assert len(group) == 2

        # Step 2 (TP): 2 tensors → 1 (single group of 2)
        assert len(plans[2].groups) == 1
        assert len(plans[2].groups[0]) == 2

    def test_sp_in_dims_but_not_in_parallel_info(self) -> None:
        """s(sp) in dims but SP absent from parallel_info (SP disabled), should auto-skip."""
        dim_specs = parse_dims("s(sp) b h(tp)")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2)},
            {ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2)},
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 1
        assert plans[0].axis == ParallelAxis.TP

    def test_all_dims_sharded_but_single_gpu(self) -> None:
        """Single GPU (TP=1, CP=1), dims has s(cp) h(tp) but parallel_info is empty."""
        dim_specs = parse_dims("b s(cp) h(tp) d")
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [{}]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert plans == []

    def test_sharded_axis_missing_from_rank_raises(self) -> None:
        """A world_rank missing a sharded axis raises ValueError."""
        dim_specs = parse_dims("s(cp) h(tp)")
        parallel_infos = [
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=1, axis_size=2),
                # missing TP — sharded axis absent from rank
            },
        ]
        with pytest.raises(ValueError, match="missing parallel_info"):
            compute_unsharder_plan(dim_specs, parallel_infos)


class TestReplicatedAxes:
    def test_replicated_tp_with_sharded_cp(self) -> None:
        """CP2 TP2, dims='b s(cp) d' → PickPlan(TP) + ConcatPlan(CP)."""
        dim_specs = parse_dims("b s(cp) d")
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=1, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=1, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2),
            },
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        assert len(plans) == 2
        assert plans[0].axis == ParallelAxis.TP
        assert isinstance(plans[0].params, PickParams)
        assert len(plans[0].groups) == 2
        for group in plans[0].groups:
            assert len(group) == 2

        assert plans[1].axis == ParallelAxis.CP
        assert isinstance(plans[1].params, ConcatParams)
        assert plans[1].params.dim_name == "s"

    def test_fully_replicated(self) -> None:
        """CP2 TP2, dims='b h d' → PickPlan(CP) + PickPlan(TP)."""
        dim_specs = parse_dims("b h d")
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=1, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=1, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2),
            },
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        assert len(plans) == 2
        assert all(isinstance(p.params, PickParams) for p in plans)
        axes = {p.axis for p in plans}
        assert axes == {ParallelAxis.CP, ParallelAxis.TP}

    def test_multiple_replicated_one_sharded(self) -> None:
        """CP2 TP2 EP2, dims='h(tp)' → PickPlan(CP) + PickPlan(EP) + ConcatPlan(TP)."""
        dim_specs = parse_dims("h(tp)")
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = []
        for cp_rank in range(2):
            for ep_rank in range(2):
                for tp_rank in range(2):
                    parallel_infos.append(
                        {
                            ParallelAxis.CP: AxisInfo(axis_rank=cp_rank, axis_size=2),
                            ParallelAxis.EP: AxisInfo(axis_rank=ep_rank, axis_size=2),
                            ParallelAxis.TP: AxisInfo(axis_rank=tp_rank, axis_size=2),
                        }
                    )

        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        assert len(plans) == 3
        pick_plans = [p for p in plans if isinstance(p.params, PickParams)]
        concat_plans = [p for p in plans if isinstance(p.params, ConcatParams)]
        assert len(pick_plans) == 2
        assert len(concat_plans) == 1
        assert concat_plans[0].axis == ParallelAxis.TP

        replicated_axes = {p.axis for p in pick_plans}
        assert replicated_axes == {ParallelAxis.CP, ParallelAxis.EP}

    def test_replicated_scrambled_ranks(self) -> None:
        """Scrambled world_rank order with replicated axis."""
        dim_specs = parse_dims("h(tp)")
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [
            {
                ParallelAxis.CP: AxisInfo(axis_rank=1, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=1, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2),
            },
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        assert len(plans) == 2
        assert plans[0].axis == ParallelAxis.CP
        assert isinstance(plans[0].params, PickParams)
        assert plans[1].axis == ParallelAxis.TP
        assert isinstance(plans[1].params, ConcatParams)

    def test_replicated_axis_inconsistent_size_raises(self) -> None:
        """Replicated axis with inconsistent sizes raises ValueError."""
        dim_specs = parse_dims("h(tp)")
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=4),
                ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2),
            },
        ]
        with pytest.raises(ValueError, match="Inconsistent axis_size"):
            compute_unsharder_plan(dim_specs, parallel_infos)

    def test_replicated_axis_missing_from_rank_raises(self) -> None:
        """A rank missing a replicated axis that other ranks have raises ValueError."""
        dim_specs = parse_dims("h(tp)")
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                # missing CP — replicated axis absent from this rank
                ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2),
            },
        ]
        with pytest.raises(ValueError, match="missing parallel_info"):
            compute_unsharder_plan(dim_specs, parallel_infos)

    def test_recompute_pseudo_replicated(self) -> None:
        """RECOMPUTE_PSEUDO with no dim annotation → replicated → PickParams."""
        dim_specs = parse_dims("h d")
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [
            {ParallelAxis.RECOMPUTE_PSEUDO: AxisInfo(axis_rank=0, axis_size=2)},
            {ParallelAxis.RECOMPUTE_PSEUDO: AxisInfo(axis_rank=1, axis_size=2)},
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        assert len(plans) == 1
        assert plans[0].axis == ParallelAxis.RECOMPUTE_PSEUDO
        assert isinstance(plans[0].params, PickParams)
        assert plans[0].groups == [[0, 1]]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
