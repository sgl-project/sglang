import sys

import pytest

from sglang.srt.debug_utils.comparator.dims import ParallelAxis, parse_dims
from sglang.srt.debug_utils.comparator.unshard import AxisInfo
from sglang.srt.debug_utils.comparator.unshard.plan import compute_unshard_plan
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestComputeUnshardPlan:
    def test_tp4_plan(self) -> None:
        dim_specs = parse_dims("b s h(tp) d")
        parallel_infos = [{"tp": AxisInfo(axis_rank=i, axis_size=4)} for i in range(4)]
        plan = compute_unshard_plan(dim_specs, parallel_infos)

        assert len(plan.steps) == 1
        step = plan.steps[0]
        assert step.axis == ParallelAxis.TP
        assert step.params.dim == 2
        assert step.world_ranks_by_axis_rank == [0, 1, 2, 3]
        assert plan.pick_world_ranks == frozenset({0, 1, 2, 3})

    def test_replicated_axes(self) -> None:
        dim_specs = parse_dims("b s h(tp) d")
        parallel_infos = [
            {
                "tp": AxisInfo(axis_rank=i % 2, axis_size=2),
                "cp": AxisInfo(axis_rank=i // 2, axis_size=2),
            }
            for i in range(4)
        ]
        plan = compute_unshard_plan(dim_specs, parallel_infos)
        assert "cp" in plan.replicated_axes
        assert plan.replicated_axes["cp"].axis_size == 2
        assert plan.pick_world_ranks == frozenset({0, 1})

    def test_inconsistent_axis_size_raises(self) -> None:
        dim_specs = parse_dims("h(tp)")
        parallel_infos = [
            {"tp": AxisInfo(axis_rank=0, axis_size=4)},
            {"tp": AxisInfo(axis_rank=1, axis_size=2)},
        ]
        with pytest.raises(ValueError, match="Inconsistent axis_size"):
            compute_unshard_plan(dim_specs, parallel_infos)

    def test_missing_axis_in_parallel_info_raises(self) -> None:
        dim_specs = parse_dims("h(tp)")
        parallel_infos = [{"cp": AxisInfo(axis_rank=0, axis_size=2)}]
        with pytest.raises(ValueError, match="No parallel_info found"):
            compute_unshard_plan(dim_specs, parallel_infos)

    def test_empty_parallel_infos_raises(self) -> None:
        dim_specs = parse_dims("h(tp)")
        with pytest.raises(ValueError, match="must not be empty"):
            compute_unshard_plan(dim_specs, [])

    def test_scrambled_world_ranks(self) -> None:
        """world_rank order != axis_rank order."""
        dim_specs = parse_dims("h(tp)")
        parallel_infos = [
            {"tp": AxisInfo(axis_rank=2, axis_size=4)},
            {"tp": AxisInfo(axis_rank=0, axis_size=4)},
            {"tp": AxisInfo(axis_rank=3, axis_size=4)},
            {"tp": AxisInfo(axis_rank=1, axis_size=4)},
        ]
        plan = compute_unshard_plan(dim_specs, parallel_infos)
        assert plan.steps[0].world_ranks_by_axis_rank == [1, 3, 0, 2]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
