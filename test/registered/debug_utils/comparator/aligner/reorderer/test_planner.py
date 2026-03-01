import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.reorderer.executor import (
    execute_reorderer_plan,
)
from sglang.srt.debug_utils.comparator.aligner.reorderer.planner import (
    compute_reorderer_plans,
)
from sglang.srt.debug_utils.comparator.aligner.reorderer.types import ReordererPlan
from sglang.srt.debug_utils.comparator.aligner.unsharder.executor import (
    execute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.planner import (
    compute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import AxisInfo
from sglang.srt.debug_utils.comparator.dims import DimSpec, ParallelAxis, parse_dims
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestComputeReordererPlans:
    def test_compute_reorderer_plans_zigzag(self) -> None:
        """s(cp:zigzag) produces a ReordererPlan."""
        dim_specs = parse_dims("b s(cp:zigzag) h(tp)").dims
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
        ]
        plans = compute_reorderer_plans(
            dim_specs=dim_specs, parallel_infos=parallel_infos
        )

        assert len(plans) == 1
        assert plans[0].params.op == "zigzag_to_natural"
        assert plans[0].params.dim_name == "s"
        assert plans[0].params.cp_size == 2

    def test_compute_reorderer_plans_thd_zigzag(self) -> None:
        """t(cp:zigzag) produces a ZigzagToNaturalThdParams plan."""
        dim_specs = parse_dims("t(cp:zigzag) h(tp)").dims
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
        ]
        thd_global_seq_lens: list[int] = [100, 64, 92]
        plans = compute_reorderer_plans(
            dim_specs=dim_specs,
            parallel_infos=parallel_infos,
            thd_global_seq_lens=thd_global_seq_lens,
        )

        assert len(plans) == 1
        assert plans[0].params.op == "zigzag_to_natural_thd"
        assert plans[0].params.cp_size == 2
        assert plans[0].params.seq_lens == [100, 64, 92]

    def test_non_seq_dim_still_raises(self) -> None:
        """Zigzag on non-sequence/non-token dim (e.g. h(cp:zigzag)) raises ValueError."""
        dim_specs = parse_dims("h(cp:zigzag) d").dims
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [
            {ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2)},
        ]
        with pytest.raises(ValueError, match="only supported on sequence dims"):
            compute_reorderer_plans(dim_specs=dim_specs, parallel_infos=parallel_infos)

    def test_thd_zigzag_without_seq_lens_raises(self) -> None:
        """t(cp:zigzag) without thd_global_seq_lens raises ValueError."""
        dim_specs = parse_dims("t(cp:zigzag) h(tp)").dims
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
        ]
        with pytest.raises(ValueError, match="thd_global_seq_lens is required"):
            compute_reorderer_plans(dim_specs=dim_specs, parallel_infos=parallel_infos)

    def test_thd_natural_no_reorder(self) -> None:
        """t(cp:natural) and t(cp) produce no reorder plans."""
        for dims_str in ["t(cp:natural) h(tp)", "t(cp) h(tp)"]:
            dim_specs = parse_dims(dims_str).dims
            parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [
                {
                    ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                    ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
                },
            ]
            plans = compute_reorderer_plans(
                dim_specs=dim_specs, parallel_infos=parallel_infos
            )
            assert plans == []

    def test_compute_reorderer_plans_natural(self) -> None:
        """s(cp) and s(cp:natural) produce no reorder plans."""
        for dims_str in ["b s(cp) h(tp)", "b s(cp:natural) h(tp)"]:
            dim_specs = parse_dims(dims_str).dims
            parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [
                {
                    ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                    ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
                },
            ]
            plans = compute_reorderer_plans(
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

        dim_specs: list[DimSpec] = parse_dims("b s(cp:zigzag) h(tp)").dims
        dim_names: list[str] = [s.name for s in dim_specs]

        unsharder_plans = compute_unsharder_plan(
            dim_specs=dim_specs, parallel_infos=parallel_infos
        )
        reorderer_plans = compute_reorderer_plans(
            dim_specs=dim_specs, parallel_infos=parallel_infos
        )
        all_plans = [*unsharder_plans, *reorderer_plans]

        assert len(unsharder_plans) == 2
        assert len(reorderer_plans) == 1

        current: list[torch.Tensor] = [t.refine_names(*dim_names) for t in tensors]
        for plan in all_plans:
            if isinstance(plan, ReordererPlan):
                current = execute_reorderer_plan(plan, current)
            else:
                current = execute_unsharder_plan(plan, current).tensors

        assert len(current) == 1
        assert torch.allclose(current[0].rename(None), full_tensor)


class TestCpZigzagSpSameDimE2E:
    """E2E test for t(cp:zigzag,sp) — two axes sharding the same token dim."""

    def test_cp2_sp2_zigzag_e2e(self) -> None:
        """CP=2 zigzag + SP=2 on same token dim: full unshard + reorder round-trip.

        Shard order (outer to inner, matching left-to-right in dims annotation):
          1. CP zigzag splits token dim into 2 CP chunks (zigzag order)
          2. SP splits each CP chunk into 2 SP chunks

        Unshard order (inner to outer, right-to-left):
          1. SP concat (inner): merge SP chunks back
          2. CP concat (outer): merge CP chunks back
          3. Zigzag reorder: restore natural token order
        """
        torch.manual_seed(42)
        total_tokens: int = 16
        hidden: int = 8
        full_tensor: torch.Tensor = torch.randn(total_tokens, hidden)

        # Step 1: CP zigzag split — split into 2*cp_size=4 natural chunks, reorder by zigzag
        cp_size: int = 2
        sp_size: int = 2
        n_natural_chunks: int = cp_size * 2
        natural_chunks: list[torch.Tensor] = list(
            full_tensor.chunk(n_natural_chunks, dim=0)
        )
        zigzag_order: list[int] = [0, 3, 1, 2]
        zigzagged: torch.Tensor = torch.cat(
            [natural_chunks[i] for i in zigzag_order], dim=0
        )
        cp_chunks: list[torch.Tensor] = list(zigzagged.chunk(cp_size, dim=0))

        # Step 2: SP split within each CP chunk
        tensors: list[torch.Tensor] = []
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = []
        for cp_rank in range(cp_size):
            sp_chunks: list[torch.Tensor] = list(
                cp_chunks[cp_rank].chunk(sp_size, dim=0)
            )
            for sp_rank in range(sp_size):
                tensors.append(sp_chunks[sp_rank])
                parallel_infos.append(
                    {
                        ParallelAxis.CP: AxisInfo(axis_rank=cp_rank, axis_size=cp_size),
                        ParallelAxis.SP: AxisInfo(axis_rank=sp_rank, axis_size=sp_size),
                    }
                )

        dim_specs: list[DimSpec] = parse_dims("t(cp:zigzag,sp) h").dims
        dim_names: list[str] = [s.name for s in dim_specs]

        unsharder_plans = compute_unsharder_plan(
            dim_specs=dim_specs, parallel_infos=parallel_infos
        )
        reorderer_plans = compute_reorderer_plans(
            dim_specs=dim_specs,
            parallel_infos=parallel_infos,
            thd_global_seq_lens=[total_tokens],
        )
        all_plans = [*unsharder_plans, *reorderer_plans]

        assert len(unsharder_plans) == 2  # SP concat, CP concat
        assert unsharder_plans[0].axis == ParallelAxis.SP
        assert unsharder_plans[1].axis == ParallelAxis.CP
        assert len(reorderer_plans) == 1  # zigzag reorder

        current: list[torch.Tensor] = [t.refine_names(*dim_names) for t in tensors]
        for plan in all_plans:
            if isinstance(plan, ReordererPlan):
                current = execute_reorderer_plan(plan, current)
            else:
                current = execute_unsharder_plan(plan, current).tensors

        assert len(current) == 1
        assert torch.allclose(current[0].rename(None), full_tensor)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
