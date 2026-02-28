import sys

import pytest
import torch

from sglang.srt.debug_utils.comparator.aligner.unsharder.executor import (
    _apply_unshard,
    _verify_replicated_group,
    execute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.planner import (
    compute_unsharder_plan,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import (
    AxisInfo,
    CpThdConcatParams,
    PickParams,
    UnsharderPlan,
)
from sglang.srt.debug_utils.comparator.dims import (
    DimSpec,
    ParallelAxis,
    parse_dims,
)
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default", nightly=True)


def _name_tensors(
    tensors: list[torch.Tensor], dim_specs: list[DimSpec]
) -> list[torch.Tensor]:
    names: list[str] = [s.name for s in dim_specs]
    return [t.refine_names(*names) for t in tensors]


class TestExecuteUnsharderPlan:
    def test_tp4_concat(self) -> None:
        full_tensor = torch.randn(2, 8, 16)
        shards = list(full_tensor.chunk(4, dim=1))

        dim_specs = parse_dims("b h(tp) d")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=i, axis_size=4)} for i in range(4)
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 1

        named_shards: list[torch.Tensor] = _name_tensors(shards, dim_specs)
        with warning_sink.context() as warnings:
            result = execute_unsharder_plan(plans[0], named_shards)
        assert len(result) == 1
        assert torch.allclose(result[0].rename(None), full_tensor)
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
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 1

        tensors_ordered_by_world_rank = _name_tensors(
            [
                shards[2],  # world_rank=0, axis_rank=2
                shards[0],  # world_rank=1, axis_rank=0
                shards[3],  # world_rank=2, axis_rank=3
                shards[1],  # world_rank=3, axis_rank=1
            ],
            dim_specs,
        )

        with warning_sink.context() as warnings:
            result = execute_unsharder_plan(plans[0], tensors_ordered_by_world_rank)
        assert len(result) == 1
        assert torch.allclose(result[0].rename(None), full_tensor)
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

        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 2

        tensors: list[torch.Tensor] = []
        for cp_rank in range(2):
            source = shards_a if cp_rank == 0 else shards_b
            for tp_rank in range(4):
                tensors.append(source[tp_rank])

        named_tensors: list[torch.Tensor] = _name_tensors(tensors, dim_specs)
        with warning_sink.context():
            intermediate = execute_unsharder_plan(plans[0], named_tensors)
        assert len(intermediate) == 4

        with warning_sink.context():
            final = execute_unsharder_plan(plans[1], intermediate)
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
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 2

        current: list[torch.Tensor] = _name_tensors(tensors, dim_specs)
        with warning_sink.context():
            for plan in plans:
                current = execute_unsharder_plan(plan, current)

        assert len(current) == 1
        assert torch.allclose(current[0].rename(None), full_tensor)

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
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 2

        current: list[torch.Tensor] = _name_tensors(tensors, dim_specs)
        with warning_sink.context():
            for plan in plans:
                current = execute_unsharder_plan(plan, current)

        assert len(current) == 1
        assert torch.allclose(current[0].rename(None), full_tensor)

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
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 3

        current: list[torch.Tensor] = _name_tensors(tensors, dim_specs)
        with warning_sink.context():
            for plan in plans:
                current = execute_unsharder_plan(plan, current)

        assert len(current) == 1
        assert torch.allclose(current[0].rename(None), full_tensor)

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
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 3

        current: list[torch.Tensor] = _name_tensors(tensors, dim_specs)
        with warning_sink.context():
            for plan in plans:
                current = execute_unsharder_plan(plan, current)

        assert len(current) == 1
        assert torch.allclose(current[0].rename(None), full_tensor)


class TestPickOperation:
    def test_pick_single_group(self) -> None:
        """PickParams picks the first tensor from a single group."""
        tensor = torch.randn(4, 8)
        dim_specs = parse_dims("h d")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2)},
            {ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2)},
        ]

        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 1
        assert isinstance(plans[0].params, PickParams)

        with warning_sink.context() as warnings:
            result = execute_unsharder_plan(plans[0], [tensor, tensor.clone()])
        assert len(result) == 1
        assert torch.allclose(result[0].rename(None), tensor)
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

        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        pick_plans = [p for p in plans if isinstance(p.params, PickParams)]
        assert len(pick_plans) == 1
        assert pick_plans[0].axis == ParallelAxis.CP

        tensor = torch.randn(4)
        tensors = [tensor.clone() for _ in range(4)]

        with warning_sink.context() as warnings:
            result = execute_unsharder_plan(pick_plans[0], tensors)
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
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 2

        current: list[torch.Tensor] = _name_tensors(tensors, dim_specs)
        with warning_sink.context():
            for plan in plans:
                current = execute_unsharder_plan(plan, current)

        assert len(current) == 1
        assert torch.allclose(current[0].rename(None), full_tensor)

    def test_fully_replicated_e2e(self) -> None:
        """CP2 TP2, dims='b h d': fully replicated -> 2 pick steps -> 1 tensor."""
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
        plans = compute_unsharder_plan(dim_specs, parallel_infos)
        assert len(plans) == 2
        assert all(isinstance(p.params, PickParams) for p in plans)

        current: list[torch.Tensor] = _name_tensors(tensors, dim_specs)
        with warning_sink.context():
            for plan in plans:
                current = execute_unsharder_plan(plan, current)

        assert len(current) == 1
        assert torch.allclose(current[0].rename(None), full_tensor)


class TestVerifyReplicatedGroup:
    def test_warns_on_mismatch(self) -> None:
        """_verify_replicated_group produces warning when replicas differ."""
        tensor_a = torch.ones(4)
        tensor_b = torch.ones(4) + 0.1

        with warning_sink.context() as warnings:
            _verify_replicated_group(
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

        with warning_sink.context() as warnings:
            _verify_replicated_group(
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

        with warning_sink.context() as warnings:
            _verify_replicated_group(
                [baseline, other_a, other_b],
                axis=ParallelAxis.CP,
                group_index=1,
            )
        assert len(warnings) == 2
        assert warnings[0].differing_index == 1
        assert warnings[1].differing_index == 2
        assert warnings[1].max_abs_diff == pytest.approx(2.0, abs=1e-5)

    def test_execute_returns_warnings(self) -> None:
        """execute_unsharder_plan emits warnings for replicated mismatch."""
        dim_specs = parse_dims("h d")
        parallel_infos = [
            {ParallelAxis.TP: AxisInfo(axis_rank=0, axis_size=2)},
            {ParallelAxis.TP: AxisInfo(axis_rank=1, axis_size=2)},
        ]
        plans = compute_unsharder_plan(dim_specs, parallel_infos)

        tensor_a = torch.zeros(4)
        tensor_b = torch.ones(4)

        with warning_sink.context() as warnings:
            result = execute_unsharder_plan(plans[0], [tensor_a, tensor_b])
        assert len(result) == 1
        assert len(warnings) == 1
        assert torch.allclose(result[0].rename(None), tensor_a)

    def test_atol_boundary_within(self) -> None:
        """Difference exactly at atol (1e-6) -> torch.allclose passes -> no warning."""
        baseline = torch.zeros(4)
        other = torch.full((4,), 1e-6)

        with warning_sink.context() as warnings:
            _verify_replicated_group(
                [baseline, other],
                axis=ParallelAxis.TP,
                group_index=0,
            )
        assert warnings == []

    def test_atol_boundary_exceeded(self) -> None:
        """Difference just above atol (1e-6 + 1e-9) -> torch.allclose fails -> warning."""
        baseline = torch.zeros(4)
        other = torch.full((4,), 1e-6 + 1e-9)

        with warning_sink.context() as warnings:
            _verify_replicated_group(
                [baseline, other],
                axis=ParallelAxis.TP,
                group_index=0,
            )
        assert len(warnings) == 1
        assert warnings[0].differing_index == 1

    def test_recompute_pseudo_mismatch_warns(self) -> None:
        """_verify_replicated_group produces warning for RECOMPUTE_PSEUDO axis mismatch."""
        tensor_a = torch.ones(4)
        tensor_b = torch.ones(4) + 0.1

        with warning_sink.context() as warnings:
            _verify_replicated_group(
                [tensor_a, tensor_b],
                axis=ParallelAxis.RECOMPUTE_PSEUDO,
                group_index=0,
            )
        assert len(warnings) == 1
        assert warnings[0].axis == "recompute_pseudo"
        assert warnings[0].group_index == 0
        assert warnings[0].differing_index == 1
        assert warnings[0].baseline_index == 0
        assert warnings[0].max_abs_diff == pytest.approx(0.1, abs=1e-5)


class TestThdCpConcat:
    def test_single_seq(self) -> None:
        """Single seq THD unshard: 2 ranks → per-seq concat."""
        rank0 = torch.tensor([1, 2, 3]).refine_names("t")
        rank1 = torch.tensor([4, 5, 6]).refine_names("t")

        plan = UnsharderPlan(
            axis=ParallelAxis.CP,
            params=CpThdConcatParams(dim_name="t", seq_lens_per_rank=[3]),
            groups=[[0, 1]],
        )
        with warning_sink.context():
            result = execute_unsharder_plan(plan, [rank0, rank1])

        assert len(result) == 1
        expected = torch.tensor([1, 2, 3, 4, 5, 6])
        assert torch.equal(result[0].rename(None), expected)

    def test_multi_seq(self) -> None:
        """Multi-seq THD unshard: 2 ranks, seq_lens=[50, 32, 46]."""
        # rank0: [seqA_r0(50) | seqB_r0(32) | pad_r0(46)]
        # rank1: [seqA_r1(50) | seqB_r1(32) | pad_r1(46)]
        seq_a_r0 = torch.arange(0, 50)
        seq_b_r0 = torch.arange(100, 132)
        pad_r0 = torch.full((46,), -1)
        rank0 = torch.cat([seq_a_r0, seq_b_r0, pad_r0]).refine_names("t")

        seq_a_r1 = torch.arange(50, 100)
        seq_b_r1 = torch.arange(132, 164)
        pad_r1 = torch.full((46,), -2)
        rank1 = torch.cat([seq_a_r1, seq_b_r1, pad_r1]).refine_names("t")

        plan = UnsharderPlan(
            axis=ParallelAxis.CP,
            params=CpThdConcatParams(dim_name="t", seq_lens_per_rank=[50, 32, 46]),
            groups=[[0, 1]],
        )
        with warning_sink.context():
            result = execute_unsharder_plan(plan, [rank0, rank1])

        assert len(result) == 1
        unsharded: torch.Tensor = result[0].rename(None)

        # seqA: r0(50) + r1(50) = 100 tokens, values 0..99
        assert torch.equal(unsharded[:100], torch.cat([seq_a_r0, seq_a_r1]))
        # seqB: r0(32) + r1(32) = 64 tokens
        assert torch.equal(unsharded[100:164], torch.cat([seq_b_r0, seq_b_r1]))
        # pad: r0(46) + r1(46) = 92 tokens
        assert torch.equal(unsharded[164:256], torch.cat([pad_r0, pad_r1]))

    def test_with_hidden_dim(self) -> None:
        """THD unshard with trailing hidden dim: shape [T, H]."""
        torch.manual_seed(42)
        hidden: int = 4
        # rank0: [seqA_r0(3, 4) | seqB_r0(2, 4)]
        # rank1: [seqA_r1(3, 4) | seqB_r1(2, 4)]
        seq_a_r0 = torch.randn(3, hidden)
        seq_b_r0 = torch.randn(2, hidden)
        rank0 = torch.cat([seq_a_r0, seq_b_r0]).refine_names("t", "h")

        seq_a_r1 = torch.randn(3, hidden)
        seq_b_r1 = torch.randn(2, hidden)
        rank1 = torch.cat([seq_a_r1, seq_b_r1]).refine_names("t", "h")

        plan = UnsharderPlan(
            axis=ParallelAxis.CP,
            params=CpThdConcatParams(dim_name="t", seq_lens_per_rank=[3, 2]),
            groups=[[0, 1]],
        )
        with warning_sink.context():
            result = execute_unsharder_plan(plan, [rank0, rank1])

        assert len(result) == 1
        unsharded: torch.Tensor = result[0].rename(None)

        assert unsharded.shape == (10, hidden)
        assert torch.equal(unsharded[:6], torch.cat([seq_a_r0, seq_a_r1]))
        assert torch.equal(unsharded[6:10], torch.cat([seq_b_r0, seq_b_r1]))

    def test_with_leading_batch_dim(self) -> None:
        """THD unshard with leading batch dim: shape [B, T, H], t is dim=1."""
        torch.manual_seed(42)
        batch: int = 2
        hidden: int = 4
        # rank0: [seqA_r0(3) | seqB_r0(2)] per batch item
        # rank1: [seqA_r1(3) | seqB_r1(2)] per batch item
        seq_a_r0 = torch.randn(batch, 3, hidden)
        seq_b_r0 = torch.randn(batch, 2, hidden)
        rank0 = torch.cat([seq_a_r0, seq_b_r0], dim=1).refine_names("b", "t", "h")

        seq_a_r1 = torch.randn(batch, 3, hidden)
        seq_b_r1 = torch.randn(batch, 2, hidden)
        rank1 = torch.cat([seq_a_r1, seq_b_r1], dim=1).refine_names("b", "t", "h")

        plan = UnsharderPlan(
            axis=ParallelAxis.CP,
            params=CpThdConcatParams(dim_name="t", seq_lens_per_rank=[3, 2]),
            groups=[[0, 1]],
        )
        with warning_sink.context():
            result = execute_unsharder_plan(plan, [rank0, rank1])

        assert len(result) == 1
        unsharded: torch.Tensor = result[0].rename(None)

        assert unsharded.shape == (batch, 10, hidden)
        # seqA: r0(3) + r1(3) = 6 tokens per batch
        assert torch.equal(unsharded[:, :6, :], torch.cat([seq_a_r0, seq_a_r1], dim=1))
        # seqB: r0(2) + r1(2) = 4 tokens per batch
        assert torch.equal(
            unsharded[:, 6:10, :], torch.cat([seq_b_r0, seq_b_r1], dim=1)
        )


class TestThdCpConcat:
    def test_single_seq(self) -> None:
        """Single seq THD unshard: 2 ranks → per-seq concat."""
        rank0 = torch.tensor([1, 2, 3]).refine_names("t")
        rank1 = torch.tensor([4, 5, 6]).refine_names("t")

        plan = UnsharderPlan(
            axis=ParallelAxis.CP,
            params=CpThdConcatParams(dim_name="t", seq_lens_per_rank=[3]),
            groups=[[0, 1]],
        )
        with warning_sink.context():
            result = execute_unsharder_plan(plan, [rank0, rank1])

        assert len(result) == 1
        expected = torch.tensor([1, 2, 3, 4, 5, 6])
        assert torch.equal(result[0].rename(None), expected)

    def test_multi_seq(self) -> None:
        """Multi-seq THD unshard: 2 ranks, seq_lens=[50, 32, 46]."""
        # rank0: [seqA_r0(50) | seqB_r0(32) | pad_r0(46)]
        # rank1: [seqA_r1(50) | seqB_r1(32) | pad_r1(46)]
        seq_a_r0 = torch.arange(0, 50)
        seq_b_r0 = torch.arange(100, 132)
        pad_r0 = torch.full((46,), -1)
        rank0 = torch.cat([seq_a_r0, seq_b_r0, pad_r0]).refine_names("t")

        seq_a_r1 = torch.arange(50, 100)
        seq_b_r1 = torch.arange(132, 164)
        pad_r1 = torch.full((46,), -2)
        rank1 = torch.cat([seq_a_r1, seq_b_r1, pad_r1]).refine_names("t")

        plan = UnsharderPlan(
            axis=ParallelAxis.CP,
            params=CpThdConcatParams(dim_name="t", seq_lens_per_rank=[50, 32, 46]),
            groups=[[0, 1]],
        )
        with warning_sink.context():
            result = execute_unsharder_plan(plan, [rank0, rank1])

        assert len(result) == 1
        unsharded: torch.Tensor = result[0].rename(None)

        # seqA: r0(50) + r1(50) = 100 tokens, values 0..99
        assert torch.equal(unsharded[:100], torch.cat([seq_a_r0, seq_a_r1]))
        # seqB: r0(32) + r1(32) = 64 tokens
        assert torch.equal(unsharded[100:164], torch.cat([seq_b_r0, seq_b_r1]))
        # pad: r0(46) + r1(46) = 92 tokens
        assert torch.equal(unsharded[164:256], torch.cat([pad_r0, pad_r1]))

    def test_with_hidden_dim(self) -> None:
        """THD unshard with trailing hidden dim: shape [T, H]."""
        torch.manual_seed(42)
        hidden: int = 4
        # rank0: [seqA_r0(3, 4) | seqB_r0(2, 4)]
        # rank1: [seqA_r1(3, 4) | seqB_r1(2, 4)]
        seq_a_r0 = torch.randn(3, hidden)
        seq_b_r0 = torch.randn(2, hidden)
        rank0 = torch.cat([seq_a_r0, seq_b_r0]).refine_names("t", "h")

        seq_a_r1 = torch.randn(3, hidden)
        seq_b_r1 = torch.randn(2, hidden)
        rank1 = torch.cat([seq_a_r1, seq_b_r1]).refine_names("t", "h")

        plan = UnsharderPlan(
            axis=ParallelAxis.CP,
            params=CpThdConcatParams(dim_name="t", seq_lens_per_rank=[3, 2]),
            groups=[[0, 1]],
        )
        with warning_sink.context():
            result = execute_unsharder_plan(plan, [rank0, rank1])

        assert len(result) == 1
        unsharded: torch.Tensor = result[0].rename(None)

        assert unsharded.shape == (10, hidden)
        assert torch.equal(unsharded[:6], torch.cat([seq_a_r0, seq_a_r1]))
        assert torch.equal(unsharded[6:10], torch.cat([seq_b_r0, seq_b_r1]))

    def test_with_leading_batch_dim(self) -> None:
        """THD unshard with leading batch dim: shape [B, T, H], t is dim=1."""
        torch.manual_seed(42)
        batch: int = 2
        hidden: int = 4
        # rank0: [seqA_r0(3) | seqB_r0(2)] per batch item
        # rank1: [seqA_r1(3) | seqB_r1(2)] per batch item
        seq_a_r0 = torch.randn(batch, 3, hidden)
        seq_b_r0 = torch.randn(batch, 2, hidden)
        rank0 = torch.cat([seq_a_r0, seq_b_r0], dim=1).refine_names("b", "t", "h")

        seq_a_r1 = torch.randn(batch, 3, hidden)
        seq_b_r1 = torch.randn(batch, 2, hidden)
        rank1 = torch.cat([seq_a_r1, seq_b_r1], dim=1).refine_names("b", "t", "h")

        plan = UnsharderPlan(
            axis=ParallelAxis.CP,
            params=CpThdConcatParams(dim_name="t", seq_lens_per_rank=[3, 2]),
            groups=[[0, 1]],
        )
        with warning_sink.context():
            result = execute_unsharder_plan(plan, [rank0, rank1])

        assert len(result) == 1
        unsharded: torch.Tensor = result[0].rename(None)

        assert unsharded.shape == (batch, 10, hidden)
        # seqA: r0(3) + r1(3) = 6 tokens per batch
        assert torch.equal(unsharded[:, :6, :], torch.cat([seq_a_r0, seq_a_r1], dim=1))
        # seqB: r0(2) + r1(2) = 4 tokens per batch
        assert torch.equal(
            unsharded[:, 6:10, :], torch.cat([seq_b_r0, seq_b_r1], dim=1)
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
