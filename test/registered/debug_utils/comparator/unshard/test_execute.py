import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.dims import parse_dims
from sglang.srt.debug_utils.comparator.unshard.executor import execute_unshard_plan
from sglang.srt.debug_utils.comparator.unshard.planner import compute_unshard_plan
from sglang.srt.debug_utils.comparator.unshard.types import AxisInfo
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestExecuteUnshardPlan:
    def test_tp4_concat(self) -> None:
        full_tensor = torch.randn(2, 8, 16)
        shards = list(full_tensor.chunk(4, dim=1))

        dim_specs = parse_dims("b h(tp) d")
        parallel_infos = [{"tp": AxisInfo(axis_rank=i, axis_size=4)} for i in range(4)]
        plan = compute_unshard_plan(dim_specs, parallel_infos)
        assert plan is not None
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
        assert plan is not None

        tensors_by_rank = {
            0: shards[2],
            1: shards[0],
            2: shards[3],
            3: shards[1],
        }

        result = execute_unshard_plan(plan, tensors_by_rank)
        assert torch.allclose(result, full_tensor)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
