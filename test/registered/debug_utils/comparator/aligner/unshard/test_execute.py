import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.unshard.executor import (
    _apply_unshard,
    _verify_replicated_group,
    execute_unshard_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unshard.planner import (
    compute_unshard_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unshard.types import (
    AxisInfo,
    PickParams,
)
from sglang.srt.debug_utils.comparator.dims import ParallelAxis, parse_dims
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


class TestExecuteUnshardPlan:
    def test_tp4_concat(self) -> None:
        full_tensor = torch.randn(2, 8, 16)
        shards = list(full_tensor.chunk(4, dim=1))

        dim_specs = parse_dims("b h(tp) d")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=i, axis_size=4)} for i in range(4)
        ]
        plans = compute_unshard_plan(dim_specs, parallel_infos)
        assert len(plans) == 1

        result, warnings = execute_unshard_plan(plans[0], shards)
        assert len(result) == 1
        assert torch.allclose(result[0], full_tensor)
        assert warnings == []

    def test_scrambled_world_ranks_correct_result(self) -> None:
        full_tensor = torch.randn(4, 8)
        shards = list(full_tensor.chunk(4, dim=0))

        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=2, axis_size=4)},
            {ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=4)},
            {ParallelAxis.TP: AxisInfo(axis_rank=3, axis_size=4)},
            {ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=4)},
        ]
        dim_specs = parse_dims("h(tp) d")
        plans = compute_unshard_plan(dim_specs, parallel_infos)
        assert len(plans) == 1

        tensors_ordered_by_world_rank = [
            shards[2],  # world_rank=0, axis_rank=2
            shards[0],  # world_rank=1, axis_rank=0
            shards[3],  # world_rank=2, axis_rank=3
            shards[1],  # world_rank=3, axis_rank=1
        ]

        result, warnings = execute_unshard_plan(plans[0], tensors_ordered_by_world_rank)
        assert len(result) == 1
        assert torch.allclose(result[0], full_tensor)
        assert warnings == []

    def test_single_step_reduces_tensor_count(self) -> None:
        """8 tensors with 2 groups of 4 produce 2 output tensors."""
        full_a = torch.randn(4, 8)
        full_b = torch.randn(4, 8)
        shards_a = list(full_a.chunk(4, dim=0))
        shards_b = list(full_b.chunk(4, dim=0))

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

        plans = compute_unshard_plan(dim_specs, parallel_infos)
        assert len(plans) == 2

        tensors: list[torch.Tensor] = []
        for cp_rank in range(2):
            source = shards_a if cp_rank == 0 else shards_b
            for tp_rank in range(4):
                tensors.append(source[tp_rank])

        intermediate, _ = execute_unshard_plan(plans[0], tensors)
        assert len(intermediate) == 4

        final, _ = execute_unshard_plan(plans[1], intermediate)
        assert len(final) == 1

    def test_cp_tp_concat(self) -> None:
        """CP=2 + TP=2: multi-step unshard reconstructs original tensor."""
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 8, 16)

        cp_chunks = list(full_tensor.chunk(2, dim=1))
        tensors: list[torch.Tensor] = []
        parallel_infos = []
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

        dim_specs = parse_dims("b s(cp) h(tp)")
        plans = compute_unshard_plan(dim_specs, parallel_infos)
        assert len(plans) == 2

        current = tensors
        for plan in plans:
            current, _ = execute_unshard_plan(plan, current)

        assert len(current) == 1
        assert torch.allclose(current[0], full_tensor)

    def test_cp_tp_scrambled(self) -> None:
        """Scrambled world_ranks for CP=2 + TP=2 still reconstruct correctly."""
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 8, 16)

        cp_chunks = list(full_tensor.chunk(2, dim=1))
        shard_map: dict[tuple[int, int], torch.Tensor] = {}
        for cp_rank in range(2):
            tp_chunks = list(cp_chunks[cp_rank].chunk(2, dim=2))
            for tp_rank in range(2):
                shard_map[(cp_rank, tp_rank)] = tp_chunks[tp_rank]

        scrambled_assignment = [
            (1, 1),  # world_rank=0
            (0, 0),  # world_rank=1
            (1, 0),  # world_rank=2
            (0, 1),  # world_rank=3
        ]

        tensors: list[torch.Tensor] = []
        parallel_infos = []
        for cp_rank, tp_rank in scrambled_assignment:
            tensors.append(shard_map[(cp_rank, tp_rank)])
            parallel_infos.append(
                {
                    ParallelAxis.CP: AxisInfo(axis_rank=cp_rank, axis_size=2),
                    ParallelAxis.TP: AxisInfo(axis_rank=tp_rank, axis_size=2),
                }
            )

        dim_specs = parse_dims("b s(cp) h(tp)")
        plans = compute_unshard_plan(dim_specs, parallel_infos)
        assert len(plans) == 2

        current = tensors
        for plan in plans:
            current, _ = execute_unshard_plan(plan, current)

        assert len(current) == 1
        assert torch.allclose(current[0], full_tensor)

    def test_unsupported_params_type_raises(self) -> None:
        """_apply_unshard raises ValueError for unknown params type."""

        class _FakeParams:
            pass

        with pytest.raises(ValueError, match="Unsupported unshard"):
            _apply_unshard(
                _FakeParams(),
                [torch.randn(2, 2)],
                axis=ParallelAxis.TP,
                group_index=0,
            )

    def test_cp_tp_ep_three_axis_concat(self) -> None:
        """CP=2 + TP=2 + EP=2: three-step unshard reconstructs original tensor."""
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 8, 16, 32)

        ep_chunks = list(full_tensor.chunk(2, dim=1))
        shard_map: dict[tuple[int, int, int], torch.Tensor] = {}
        for ep_rank in range(2):
            cp_chunks = list(ep_chunks[ep_rank].chunk(2, dim=2))
            for cp_rank in range(2):
                tp_chunks = list(cp_chunks[cp_rank].chunk(2, dim=3))
                for tp_rank in range(2):
                    shard_map[(ep_rank, cp_rank, tp_rank)] = tp_chunks[tp_rank]

        tensors: list[torch.Tensor] = []
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = []
        for ep_rank in range(2):
            for cp_rank in range(2):
                for tp_rank in range(2):
                    tensors.append(shard_map[(ep_rank, cp_rank, tp_rank)])
                    parallel_infos.append(
                        {
                            ParallelAxis.EP: AxisInfo(axis_rank=ep_rank, axis_size=2),
                            ParallelAxis.CP: AxisInfo(axis_rank=cp_rank, axis_size=2),
                            ParallelAxis.TP: AxisInfo(axis_rank=tp_rank, axis_size=2),
                        }
                    )

        dim_specs = parse_dims("b e(ep) s(cp) h(tp)")
        plans = compute_unshard_plan(dim_specs, parallel_infos)
        assert len(plans) == 3

        current = tensors
        for plan in plans:
            current, _ = execute_unshard_plan(plan, current)

        assert len(current) == 1
        assert torch.allclose(current[0], full_tensor)

    def test_cp_tp_ep_scrambled_three_axis(self) -> None:
        """Scrambled ranks for CP=2 + TP=2 + EP=2 still reconstruct correctly."""
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 8, 16, 32)

        ep_chunks = list(full_tensor.chunk(2, dim=1))
        shard_map: dict[tuple[int, int, int], torch.Tensor] = {}
        for ep_rank in range(2):
            cp_chunks = list(ep_chunks[ep_rank].chunk(2, dim=2))
            for cp_rank in range(2):
                tp_chunks = list(cp_chunks[cp_rank].chunk(2, dim=3))
                for tp_rank in range(2):
                    shard_map[(ep_rank, cp_rank, tp_rank)] = tp_chunks[tp_rank]

        scrambled_assignment = [
            (1, 0, 1),  # world_rank=0
            (0, 1, 0),  # world_rank=1
            (1, 1, 0),  # world_rank=2
            (0, 0, 0),  # world_rank=3
            (0, 1, 1),  # world_rank=4
            (1, 0, 0),  # world_rank=5
            (0, 0, 1),  # world_rank=6
            (1, 1, 1),  # world_rank=7
        ]

        tensors: list[torch.Tensor] = []
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = []
        for ep_rank, cp_rank, tp_rank in scrambled_assignment:
            tensors.append(shard_map[(ep_rank, cp_rank, tp_rank)])
            parallel_infos.append(
                {
                    ParallelAxis.EP: AxisInfo(axis_rank=ep_rank, axis_size=2),
                    ParallelAxis.CP: AxisInfo(axis_rank=cp_rank, axis_size=2),
                    ParallelAxis.TP: AxisInfo(axis_rank=tp_rank, axis_size=2),
                }
            )

        dim_specs = parse_dims("b e(ep) s(cp) h(tp)")
        plans = compute_unshard_plan(dim_specs, parallel_infos)
        assert len(plans) == 3

        current = tensors
        for plan in plans:
            current, _ = execute_unshard_plan(plan, current)

        assert len(current) == 1
        assert torch.allclose(current[0], full_tensor)


class TestPickOperation:
    def test_pick_single_group(self) -> None:
        """PickParams picks the first tensor from a single group."""
        tensor = torch.randn(4, 8)
        dim_specs = parse_dims("h d")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2)},
            {ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2)},
        ]

        plans = compute_unshard_plan(dim_specs, parallel_infos)
        assert len(plans) == 1
        assert isinstance(plans[0].params, PickParams)

        result, warnings = execute_unshard_plan(plans[0], [tensor, tensor.clone()])
        assert len(result) == 1
        assert torch.allclose(result[0], tensor)
        assert warnings == []

    def test_pick_multiple_groups(self) -> None:
        """PickParams with multiple groups picks one from each."""
        dim_specs = parse_dims("h(tp)")
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = [
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=1, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=0, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2),
            },
            {
                ParallelAxis.CP: AxisInfo(axis_rank=1, axis_size=2),
                ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2),
            },
        ]

        plans = compute_unshard_plan(dim_specs, parallel_infos)
        pick_plans = [p for p in plans if isinstance(p.params, PickParams)]
        assert len(pick_plans) == 1
        assert pick_plans[0].axis == ParallelAxis.CP

        tensor = torch.randn(4)
        tensors = [tensor.clone() for _ in range(4)]

        result, warnings = execute_unshard_plan(pick_plans[0], tensors)
        assert len(result) == 2
        assert warnings == []

    def test_replicated_tp_sharded_cp_e2e(self) -> None:
        """CP2 TP2, dims='b s(cp) d': replicated TP pick + sharded CP concat round-trip."""
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 8, 16)
        cp_chunks = list(full_tensor.chunk(2, dim=1))

        tensors: list[torch.Tensor] = []
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = []
        for cp_rank in range(2):
            for tp_rank in range(2):
                tensors.append(cp_chunks[cp_rank].clone())
                parallel_infos.append(
                    {
                        ParallelAxis.CP: AxisInfo(axis_rank=cp_rank, axis_size=2),
                        ParallelAxis.TP: AxisInfo(axis_rank=tp_rank, axis_size=2),
                    }
                )

        dim_specs = parse_dims("b s(cp) d")
        plans = compute_unshard_plan(dim_specs, parallel_infos)
        assert len(plans) == 2

        current = tensors
        for plan in plans:
            current, _ = execute_unshard_plan(plan, current)

        assert len(current) == 1
        assert torch.allclose(current[0], full_tensor)

    def test_fully_replicated_e2e(self) -> None:
        """CP2 TP2, dims='b h d': fully replicated → 2 pick steps → 1 tensor."""
        torch.manual_seed(42)
        full_tensor = torch.randn(4, 8, 16)

        tensors: list[torch.Tensor] = []
        parallel_infos: list[dict[ParallelAxis, AxisInfo]] = []
        for cp_rank in range(2):
            for tp_rank in range(2):
                tensors.append(full_tensor.clone())
                parallel_infos.append(
                    {
                        ParallelAxis.CP: AxisInfo(axis_rank=cp_rank, axis_size=2),
                        ParallelAxis.TP: AxisInfo(axis_rank=tp_rank, axis_size=2),
                    }
                )

        dim_specs = parse_dims("b h d")
        plans = compute_unshard_plan(dim_specs, parallel_infos)
        assert len(plans) == 2
        assert all(isinstance(p.params, PickParams) for p in plans)

        current = tensors
        for plan in plans:
            current, _ = execute_unshard_plan(plan, current)

        assert len(current) == 1
        assert torch.allclose(current[0], full_tensor)


class TestVerifyReplicatedGroup:
    def test_warns_on_mismatch(self) -> None:
        """_verify_replicated_group produces warning when replicas differ."""
        tensor_a = torch.ones(4)
        tensor_b = torch.ones(4) + 0.1

        warnings = _verify_replicated_group(
            [tensor_a, tensor_b],
            axis=ParallelAxis.TP,
            group_index=0,
        )
        assert len(warnings) == 1
        assert warnings[0].axis == "tp"
        assert warnings[0].group_index == 0
        assert warnings[0].differing_index == 1
        assert warnings[0].baseline_index == 0
        assert warnings[0].max_abs_diff == pytest.approx(0.1, abs=1e-5)

    def test_no_warn_when_identical(self) -> None:
        """_verify_replicated_group produces no warning for identical replicas."""
        tensor = torch.randn(4, 8)

        warnings = _verify_replicated_group(
            [tensor, tensor.clone()],
            axis=ParallelAxis.TP,
            group_index=0,
        )
        assert warnings == []

    def test_multiple_mismatches(self) -> None:
        """_verify_replicated_group reports each differing replica."""
        baseline = torch.zeros(4)
        other_a = torch.ones(4)
        other_b = torch.ones(4) * 2

        warnings = _verify_replicated_group(
            [baseline, other_a, other_b],
            axis=ParallelAxis.CP,
            group_index=1,
        )
        assert len(warnings) == 2
        assert warnings[0].differing_index == 1
        assert warnings[1].differing_index == 2
        assert warnings[1].max_abs_diff == pytest.approx(2.0, abs=1e-5)

    def test_execute_returns_warnings(self) -> None:
        """execute_unshard_plan returns warnings for replicated mismatch."""
        dim_specs = parse_dims("h d")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2)},
            {ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2)},
        ]
        plans = compute_unshard_plan(dim_specs, parallel_infos)

        tensor_a = torch.zeros(4)
        tensor_b = torch.ones(4)

        result, warnings = execute_unshard_plan(plans[0], [tensor_a, tensor_b])
        assert len(result) == 1
        assert len(warnings) == 1
        assert torch.allclose(result[0], tensor_a)

    def test_atol_boundary_within(self) -> None:
        """Difference exactly at atol (1e-6) → torch.allclose passes → no warning."""
        baseline = torch.zeros(4)
        other = torch.full((4,), 1e-6)

        warnings = _verify_replicated_group(
            [baseline, other],
            axis=ParallelAxis.TP,
            group_index=0,
        )
        assert warnings == []

    def test_atol_boundary_exceeded(self) -> None:
        """Difference just above atol (1e-6 + 1e-9) → torch.allclose fails → warning."""
        baseline = torch.zeros(4)
        other = torch.full((4,), 1e-6 + 1e-9)

        warnings = _verify_replicated_group(
            [baseline, other],
            axis=ParallelAxis.TP,
            group_index=0,
        )
        assert len(warnings) == 1
        assert warnings[0].differing_index == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
