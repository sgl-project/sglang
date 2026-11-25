"""CPU regressions for Ascend async EPLB planning and startup."""

import unittest
from concurrent.futures import wait
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.eplb.eplb_algorithms import EplbAlgorithm
from sglang.srt.hardware_backend.npu.eplb import coordinator, placement
from sglang.srt.hardware_backend.npu.eplb.placement import (
    PlanningSpec,
    plan_migration,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAsyncEplbPlanning(unittest.TestCase):
    @staticmethod
    def _plan_two_replicas(ordinary_peak, total):
        # The 50 fixed hot steps already occupy the entire top 5% of this
        # window. One ordinary step over the old peak would raise the P95.
        current = torch.tensor([[0, 1, 0, 2]])
        spec = PlanningSpec(4, 2, None, 1, EplbAlgorithm.deepseek, "random")

        counts = torch.tensor(
            [[[ordinary_peak, 0, total - ordinary_peak, 0]]] * 950
            + [[[0, total, 0, 0]]] * 50,
            dtype=torch.int32,
        )
        return plan_migration(
            counts,
            current,
            spec,
            2,
            (0,),
            num_logical_experts=3,
            narrowed_layers={0},
        )

    def test_dynamic_replica_checks_full_window_p95(self):
        self.assertIsNone(self._plan_two_replicas(80, 128))
        safer = self._plan_two_replicas(104, 128)
        self.assertIsNotNone(safer)
        self.assertEqual(safer.physical_to_logical.tolist(), [[0, 1, 0, 2]])
        self.assertEqual(safer.logical_to_physical[0, 0].tolist(), [0, 2])

    def test_dynamic_replica_bounds_correlated_source_draws(self):
        # Aligned generators can repeat the first 201 draws across both
        # sources. The independent bound used to admit this route-only plan.
        self.assertIsNone(self._plan_two_replicas(311, 512))

    def test_random_admission_uses_active_sources_per_step(self):
        current = torch.arange(32).reshape(1, 32)
        current[:, 2] = 0
        current[:, 3:] -= 1
        spec = PlanningSpec(32, 2, None, 1, EplbAlgorithm.deepseek, "random")
        counts = torch.zeros(1000, 1, 32, dtype=torch.int32)
        counts[:950, 0, 0] = 104
        counts[:950, 0, 2] = 24
        counts[950:, 0, 1] = 128
        active_sources = torch.tensor([[2]] * 950 + [[16]] * 50)
        args = (counts, current, spec, 16, (0,))
        kwargs = dict(num_logical_experts=31, narrowed_layers={0})
        self.assertIsNone(plan_migration(*args, **kwargs))
        plan = plan_migration(*args, **kwargs, active_sources=active_sources)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.layers, (0,))

    def test_balanced_single_copy_window_skips_solver(self):
        current = torch.tensor([[2, 0, 3, 1]])
        spec = PlanningSpec(4, 2, None, 1, EplbAlgorithm.deepseek)
        # Counts are unequal per expert, but each live rank gets eight.
        counts = torch.tensor([[[5, 3, 1, 7]]], dtype=torch.int32).repeat(16, 1, 1)
        with patch.object(PlanningSpec, "compute", autospec=True) as compute:
            self.assertIsNone(
                plan_migration(counts, current, spec, 2, (0,), num_logical_experts=4)
            )
        compute.assert_not_called()

    def test_sparse_window_keeps_only_eligible_layers_in_the_solver(self):
        current = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
        spec = PlanningSpec(4, 2, None, 1, EplbAlgorithm.deepseek)
        counts = torch.tensor([[[2, 2, 2, 2], [30, 10, 1, 1]]]).repeat(16, 1, 1)
        with (
            patch.object(
                PlanningSpec, "compute", autospec=True, side_effect=PlanningSpec.compute
            ) as solve,
            patch.object(
                placement, "_prefer_fewer_moves", wraps=placement._prefer_fewer_moves
            ) as search,
        ):
            plan = plan_migration(
                counts, current, spec, 2, (0, 1), num_logical_experts=4
            )

        self.assertIsNotNone(plan)
        self.assertEqual(plan.layers, (1,))
        self.assertEqual(plan.physical_to_logical.shape, (1, 4))
        self.assertEqual(solve.call_args.args[1].size(1), 1)
        self.assertEqual(search.call_args.args[-1], (0,))


class TestAsyncEplbSnapshot(unittest.TestCase):
    def test_peer_preparation_or_copy_failure_discards_the_window(self):
        from sglang.srt.hardware_backend.npu.eplb import manager

        point = coordinator.ControlPoint
        for failed_at in (point.SNAPSHOT_PREPARE, point.SNAPSHOT_COMPLETE):
            with self.subTest(failed_at=failed_at):
                counts = torch.ones(2, 1, 4, dtype=torch.int32)
                accumulator = object.__new__(manager._AsyncStatAccumulator)
                accumulator._enable = False
                accumulator._communication_group = object()
                accumulator._previous_layouts = {}
                accumulator._global_physical_count_of_buffered_step = SimpleNamespace(
                    get_all=lambda: counts
                )
                accumulator.reset = counts.zero_
                vote = Mock(side_effect=lambda p, ready, error: (ready, p == failed_at))
                spec = PlanningSpec(4, 2, None, 1, EplbAlgorithm.deepseek)
                with patch.multiple(
                    manager.dist, get_rank=Mock(return_value=1), all_reduce=Mock()
                ):
                    self.assertIsNone(
                        accumulator.take_rebalance_snapshot(
                            spec, lambda _: True, vote, source_local=True
                        )
                    )
                    self.assertEqual(
                        manager.dist.all_reduce.call_count,
                        int(failed_at == point.SNAPSHOT_COMPLETE),
                    )
                self.assertFalse(counts.any().item())
                self.assertEqual(
                    [call.args[0] for call in vote.call_args_list],
                    [point.SNAPSHOT_PREPARE, point.SNAPSHOT_COMPLETE],
                )

    def test_placement_preserves_holes_and_trims_only_changed_layers(self):
        from sglang.srt.hardware_backend.npu.eplb import runtime

        physical = torch.tensor([[0, 1, 0, 1], [0, 1, 0, 1]])
        logical = torch.tensor(
            [[[0, -1, -1, 2], [1, 3, -1, -1]], [[0, -1, 2, -1], [1, 3, -1, -1]]]
        )
        plan = placement.MigrationPlacement((1,), physical[1:], logical[1:])
        with patch.object(runtime.dist, "broadcast") as broadcast:
            rows = runtime._broadcast_placement(plan, None, 0, None, physical.shape, 2)
        self.assertEqual(broadcast.call_args_list[0].args[0].tolist(), [1, 3, 0])
        torch.testing.assert_close(rows.logical_to_physical, logical[1:, :, :3])


class TestAsyncEplbCoordinator(unittest.TestCase):
    def test_slow_cpu_plan_can_drain_after_deadline(self):
        working, release = Event(), Event()
        runtime = Mock()
        runtime.prepare.return_value = (0,)
        control = Mock()
        control.vote.side_effect = lambda epoch, phase, layer, ready, error: (
            ready,
            error,
        )
        eplb = coordinator.AsyncEplbCoordinator(control, runtime)

        def planner():
            working.set()
            release.wait()

        def finish_during_drain(futures, *, timeout):
            self.assertGreater(timeout, 0)
            release.set()
            return wait(futures, timeout=timeout)

        try:
            eplb.start(planner)
            self.assertTrue(working.wait(2))
            eplb._deadline = 0
            with (
                patch.object(coordinator, "wait", side_effect=finish_during_drain),
                self.assertRaises(coordinator.RebalanceAborted),
            ):
                eplb.poll()
            self.assertEqual(eplb.phase, coordinator.Phase.DISABLED)
            runtime.stage.assert_not_called()
            runtime.install.assert_not_called()
            runtime.synchronize.assert_called_once()
        finally:
            release.set()
            eplb._executor.shutdown(wait=True)


class TestAsyncEplbStartup(unittest.TestCase):
    def test_ascend_async_requires_an_expert_parallel_backend(self):
        from sglang.srt.hardware_backend.npu.eplb import validate_async_eplb_args

        cfg = SimpleNamespace(
            device="npu",
            enable_eplb=True,
            ep_size=16,
            tp_size=16,
            moe_a2a_backend="none",
        )
        with self.assertRaisesRegex(ValueError, "expert-parallel MoE backend"):
            validate_async_eplb_args(cfg)

    def test_single_rank_warmup_allocation_failure_closes_control_group(self):
        from sglang.srt.hardware_backend.npu.eplb import runtime

        control_group = object()
        parallel = SimpleNamespace(
            nnodes=1, tp_rank=0, num_dp_ranks=1, dist_timeout=None
        )
        moe = SimpleNamespace(
            eplb_rebalance_num_iterations=20,
            expert_distribution_recorder_buffer_size=10,
            ep_dispatch_algorithm="static",
        )
        metadata = SimpleNamespace(
            physical_to_logical_map_cpu=torch.tensor([[0, 1]]),
            num_logical_experts=2,
        )
        transport = Mock(
            layer_ids=(0,),
            weight_format_digest=b"test",
            use_prefill_admission=False,
            _p2p_batch_size=1,
            narrowed_replica_layers=set(),
            staging=(torch.zeros(1),),
            device="cpu",
        )
        torch_npu = SimpleNamespace(_C=SimpleNamespace())
        torch_npu._C._distributed_c10d = SimpleNamespace(
            ProcessGroupHCCL=SimpleNamespace(Options=SimpleNamespace)
        )
        ctx = SimpleNamespace(
            get_disagg=lambda: SimpleNamespace(disaggregation_mode="decode"),
            get_exec=lambda: SimpleNamespace(moe=moe),
            get_parallel=lambda: parallel,
            get_schedule=Mock(),
            reports_expert_balancedness=lambda: False,
        )

        failure = "warmup allocation failed"
        allocate = Mock()

        def gather(peers, local, *, group):
            self.assertIs(group, control_group)
            if failed_rank == 0:
                self.assertEqual(local, f"RuntimeError: {failure}")
                peers[:] = [local, None]
            else:
                self.assertIsInstance(local, tuple)
                peers[:] = [local, f"RuntimeError: {failure}"]

        with (
            patch.dict(
                "sys.modules",
                {"torch_npu": torch_npu, "sglang.srt.runtime_context": ctx},
            ),
            patch.multiple(
                runtime,
                _validate_migration_topology=Mock(),
                _make_planning_spec=Mock(return_value=SimpleNamespace(local_experts=1)),
                _allocate_migration_warmup_buffers=allocate,
                _hccl_blocking_wait=Mock(return_value=False),
                NpuEplbRuntime=Mock(return_value=transport),
                get_global_expert_location_metadata=Mock(return_value=metadata),
            ),
            patch.multiple(
                runtime.dist,
                get_world_size=Mock(return_value=2),
                new_group=Mock(return_value=control_group),
                destroy_process_group=Mock(),
                all_gather_object=gather,
            ),
        ):
            model = SimpleNamespace(routed_experts_weights_of_layer={})
            for failed_rank in (0, 1):
                with self.subTest(failed_rank=failed_rank):
                    allocate.reset_mock()
                    allocate.side_effect = (
                        RuntimeError(failure) if failed_rank == 0 else None
                    )
                    allocate.return_value = (None, None)
                    runtime.dist.new_group.reset_mock()
                    runtime.dist.destroy_process_group.reset_mock()
                    with self.assertRaisesRegex(
                        RuntimeError, f"rank {failed_rank}:.*{failure}"
                    ):
                        runtime.create_npu_async_eplb(
                            None, model, startup_weight_load=None
                        )
                    allocate.assert_called_once_with(2)
                    self.assertEqual(runtime.dist.new_group.call_count, 1)
                    self.assertEqual(
                        runtime.dist.new_group.call_args.kwargs["backend"], "gloo"
                    )
                    runtime.dist.destroy_process_group.assert_called_once_with(
                        control_group
                    )
                    transport.open_transport.assert_not_called()


if __name__ == "__main__":
    unittest.main()
