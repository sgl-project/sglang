import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.dims import parse_dims
from sglang.srt.debug_utils.comparator.unshard.execute import execute_unshard_plan
from sglang.srt.debug_utils.comparator.unshard.plan import compute_unshard_plan
from sglang.srt.debug_utils.comparator.unshard.types import AxisInfo, UnshardPlan
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestExecuteUnshardPlan:
    def test_tp4_concat(self) -> None:
        full_tensor = torch.randn(2, 8, 16)
        shards = list(full_tensor.chunk(4, dim=1))

        dim_specs = parse_dims("b h(tp) d")
        parallel_infos = [{"tp": AxisInfo(axis_rank=i, axis_size=4)} for i in range(4)]
        plan = compute_unshard_plan(dim_specs, parallel_infos)
        tensors_by_rank = {i: shards[i] for i in range(4)}

        result = execute_unshard_plan(plan, tensors_by_rank)
        assert torch.allclose(result, full_tensor)

    def test_scrambled_world_ranks_correct_result(self) -> None:
        full_tensor = torch.randn(4, 8)
        shards = list(full_tensor.chunk(4, dim=0))

        parallel_infos = [
            {"tp": AxisInfo(axis_rank=2, axis_size=4)},
            {"tp": AxisInfo(axis_rank=0, axis_size=4)},
            {"tp": AxisInfo(axis_rank=3, axis_size=4)},
            {"tp": AxisInfo(axis_rank=1, axis_size=4)},
        ]
        dim_specs = parse_dims("h(tp) d")
        plan = compute_unshard_plan(dim_specs, parallel_infos)

        tensors_by_rank = {
            0: shards[2],
            1: shards[0],
            2: shards[3],
            3: shards[1],
        }

        result = execute_unshard_plan(plan, tensors_by_rank)
        assert torch.allclose(result, full_tensor)

    def test_no_sharded_dims_single_tensor(self) -> None:
        tensor = torch.randn(4, 4)
        dim_specs = parse_dims("b d")
        parallel_infos = [{}]
        plan = compute_unshard_plan(dim_specs, parallel_infos)
        assert len(plan.steps) == 0

        result = execute_unshard_plan(plan, {0: tensor})
        assert torch.allclose(result, tensor)

    def test_no_steps_multiple_tensors_raises(self) -> None:
        plan = UnshardPlan(
            tensor_name="",
            dims_str="",
            steps=[],
            pick_world_ranks=frozenset({0, 1}),
        )
        with pytest.raises(ValueError, match="No unshard steps"):
            execute_unshard_plan(plan, {0: torch.randn(2), 1: torch.randn(2)})

    def test_tp_with_replicated_cp(self) -> None:
        """TP=2, CP=2. dims="b h(tp) d" means CP is replicated.

        pick_world_ranks filters to cp_rank=0 ranks only.
        """
        full_tensor = torch.randn(2, 8, 4)
        shards = list(full_tensor.chunk(2, dim=1))

        parallel_infos = [
            {
                "tp": AxisInfo(axis_rank=0, axis_size=2),
                "cp": AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                "tp": AxisInfo(axis_rank=1, axis_size=2),
                "cp": AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                "tp": AxisInfo(axis_rank=0, axis_size=2),
                "cp": AxisInfo(axis_rank=1, axis_size=2),
            },
            {
                "tp": AxisInfo(axis_rank=1, axis_size=2),
                "cp": AxisInfo(axis_rank=1, axis_size=2),
            },
        ]
        dim_specs = parse_dims("b h(tp) d")
        plan = compute_unshard_plan(dim_specs, parallel_infos)

        assert plan.pick_world_ranks == frozenset({0, 1})

        tensors_by_rank = {
            0: shards[0],
            1: shards[1],
            2: shards[0],
            3: shards[1],
        }

        result = execute_unshard_plan(plan, tensors_by_rank)
        assert torch.allclose(result, full_tensor)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
