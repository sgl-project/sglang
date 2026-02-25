import sys

import pytest

from sglang.srt.debug_utils.comparator.dims import ParallelAxis, parse_dims
from sglang.srt.debug_utils.comparator.unshard.planner import compute_unshard_plan
from sglang.srt.debug_utils.comparator.unshard.types import AxisInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestComputeUnshardPlan:
    def test_tp4_plan(self) -> None:
        dim_specs = parse_dims("b s h(tp) d")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=i, axis_size=4)} for i in range(4)
        ]
        plan = compute_unshard_plan(dim_specs, parallel_infos)

        assert plan is not None
        assert plan.axis == ParallelAxis.TP
        assert plan.params.dim == 2
        assert plan.world_ranks_by_axis_rank == [0, 1, 2, 3]

    def test_inconsistent_axis_size_raises(self) -> None:
        dim_specs = parse_dims("h(tp)")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=4)},
            {ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2)},
        ]
        with pytest.raises(ValueError, match="Inconsistent axis_size"):
            compute_unshard_plan(dim_specs, parallel_infos)

    def test_missing_axis_in_parallel_info_raises(self) -> None:
        dim_specs = parse_dims("h(tp)")
        parallel_infos = [{ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2)}]
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
            {ParallelAxis.TP: AxisInfo(axis_rank=2, axis_size=4)},
            {ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=4)},
            {ParallelAxis.TP: AxisInfo(axis_rank=3, axis_size=4)},
            {ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=4)},
        ]
        plan = compute_unshard_plan(dim_specs, parallel_infos)
        assert plan is not None
        assert plan.world_ranks_by_axis_rank == [1, 3, 0, 2]

    def test_no_sharded_axes_returns_none(self) -> None:
        dim_specs = parse_dims("b s d")
        parallel_infos = [{}]
        plan = compute_unshard_plan(dim_specs, parallel_infos)
        assert plan is None

    def test_multi_axis_raises(self) -> None:
        dim_specs = parse_dims("h(tp) s(cp)")
        parallel_infos = [
            {
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2),
                ParallelAxis.CP: AxisInfo(axis_rank=1, axis_size=2),
            },
        ]
        with pytest.raises(NotImplementedError, match="Multi-axis unshard"):
            compute_unshard_plan(dim_specs, parallel_infos)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
