import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.reorder import (
    ReorderPlan,
    _reorder_zigzag_to_natural,
    compute_reorder_plans,
    execute_reorder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unshard.executor import (
    execute_unshard_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unshard.planner import (
    compute_unshard_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unshard.types import AxisInfo
from sglang.srt.debug_utils.comparator.dims import ParallelAxis, parse_dims
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestZigzagToNatural:
    def test_zigzag_to_natural_cp2(self) -> None:
        """cp_size=2: zigzag order [0,3,1,2] -> natural [0,1,2,3]."""
        natural = torch.arange(24).reshape(4, 6)
        chunks = list(natural.chunk(4, dim=0))

        zigzag_order: list[int] = [0, 3, 1, 2]
        zigzagged = torch.cat([chunks[i] for i in zigzag_order], dim=0)

        result = _reorder_zigzag_to_natural(zigzagged, dim=0, cp_size=2)
        assert torch.equal(result, natural)

    def test_zigzag_to_natural_cp3(self) -> None:
        """cp_size=3: zigzag 162534 -> natural 123456 (1-indexed)."""
        natural = torch.arange(60).reshape(6, 10)
        chunks = list(natural.chunk(6, dim=0))

        zigzag_order: list[int] = [0, 5, 1, 4, 2, 3]
        zigzagged = torch.cat([chunks[i] for i in zigzag_order], dim=0)

        result = _reorder_zigzag_to_natural(zigzagged, dim=0, cp_size=3)
        assert torch.equal(result, natural)

    def test_zigzag_to_natural_arbitrary_dim(self) -> None:
        """Reorder along dim=1 instead of dim=0."""
        natural = torch.arange(48).reshape(3, 4, 4)
        chunks = list(natural.chunk(4, dim=1))

        zigzag_order: list[int] = [0, 3, 1, 2]
        zigzagged = torch.cat([chunks[i] for i in zigzag_order], dim=1)

        result = _reorder_zigzag_to_natural(zigzagged, dim=1, cp_size=2)
        assert torch.equal(result, natural)


class TestComputeReorderPlans:
    def test_compute_reorder_plans_zigzag(self) -> None:
        """s(cp,zigzag) produces a ReorderPlan."""
        dim_specs = parse_dims("b s(cp,zigzag) h(tp)")
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
        ]
        plans = compute_reorder_plans(
            dim_specs=dim_specs, parallel_infos=parallel_infos
        )

        assert len(plans) == 1
        assert plans[0].params.op == "zigzag_to_natural"
        assert plans[0].params.dim == 1
        assert plans[0].params.cp_size == 2

    def test_compute_reorder_plans_non_seq_dim_raises(self) -> None:
        """Zigzag on non-sequence dim (e.g. t(cp,zigzag)) raises ValueError."""
        dim_specs = parse_dims("t(cp,zigzag) h(tp)")
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
        ]
        with pytest.raises(ValueError, match="only supported on sequence dims"):
            compute_reorder_plans(dim_specs=dim_specs, parallel_infos=parallel_infos)

    def test_compute_reorder_plans_natural(self) -> None:
        """s(cp) and s(cp,natural) produce no reorder plans."""
        for dims_str in ["b s(cp) h(tp)", "b s(cp,natural) h(tp)"]:
            dim_specs = parse_dims(dims_str)
            parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [
                {
                    ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                    ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
                },
            ]
            plans = compute_reorder_plans(
                dim_specs=dim_specs, parallel_infos=parallel_infos
            )
            assert plans == []


class TestCpZigzagTpE2E:
    def test_cp_zigzag_tp_e2e(self) -> None:
        """CP=2 zigzag + TP=2: full pipeline round-trip."""
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 8, 16)

        # Shard: first split seq dim (dim=1) into CP=2 with zigzag ordering,
        # then split hidden dim (dim=2) into TP=2.
        natural_cp_chunks = list(full_tensor.chunk(4, dim=1))
        zigzag_order: list[int] = [0, 3, 1, 2]
        zigzagged = torch.cat([natural_cp_chunks[i] for i in zigzag_order], dim=1)

        cp_chunks = list(zigzagged.chunk(2, dim=1))
        tensors: list[torch.Tensor] = []
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = []
        for cp_rank in range(2):
            tp_chunks = list(cp_chunks[cp_rank].chunk(2, dim=2))
            for tp_rank in range(2):
                tensors.append(tp_chunks[tp_rank])
                parallel_infos.append(
                    {
                        ParallelAxis.CP: AxisInfo(axis_rank=cp_rank, axis_size=2),
                        ParallelAxis.TP: AxisInfo(axis_rank=tp_rank, axis_size=2),
                    }
                )

        dim_specs = parse_dims("b s(cp,zigzag) h(tp)")

        unshard_plans = compute_unshard_plan(
            dim_specs=dim_specs, parallel_infos=parallel_infos
        )
        reorder_plans = compute_reorder_plans(
            dim_specs=dim_specs, parallel_infos=parallel_infos
        )
        all_plans = [*unshard_plans, *reorder_plans]

        assert len(unshard_plans) == 2
        assert len(reorder_plans) == 1

        current: list[torch.Tensor] = tensors
        for plan in all_plans:
            if isinstance(plan, ReorderPlan):
                current = execute_reorder_plan(plan, current)
            else:
                current, _ = execute_unshard_plan(plan, current)

        assert len(current) == 1
        assert torch.allclose(current[0], full_tensor)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
