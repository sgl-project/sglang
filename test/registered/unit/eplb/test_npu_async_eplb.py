"""CPU regressions for Ascend async EPLB planning and startup."""

import unittest
from concurrent.futures import wait
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.eplb.eplb_algorithms import EplbAlgorithm
from sglang.srt.hardware_backend.npu.eplb import coordinator
from sglang.srt.hardware_backend.npu.eplb.placement import (
    PlanningSpec,
    plan_migration,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAsyncEplbPlanning(unittest.TestCase):
    def test_dynamic_replica_checks_full_window_p95(self):
        # The 50 fixed hot steps already occupy the entire top 5% of this
        # window. One ordinary step over the old peak would raise the P95.
        current = torch.tensor([[0, 1, 0, 2]])
        local_replicas = torch.tensor([[[0], [1], [3]]])
        spec = PlanningSpec(4, 2, None, 1, EplbAlgorithm.deepseek, "random")

        def plan(ordinary_peak):
            counts = torch.tensor(
                [[[ordinary_peak, 0, 128 - ordinary_peak, 0]]] * 950
                + [[[0, 128, 0, 0]]] * 50,
                dtype=torch.int32,
            )
            return plan_migration(
                counts,
                current,
                spec,
                2,
                (0,),
                current_replicas=local_replicas,
                narrowed_layers={0},
            )

        self.assertIsNone(plan(80))
        safer = plan(96)
        self.assertIsNotNone(safer)
        self.assertTrue(torch.equal(safer.physical_to_logical, current))
        self.assertEqual(safer.logical_to_physical[0, 0].tolist(), [0, 2])

    def test_balanced_single_copy_window_skips_solver(self):
        current = torch.tensor([[2, 0, 3, 1]])
        spec = PlanningSpec(4, 2, None, 1, EplbAlgorithm.deepseek)
        # Counts are unequal per expert, but each live rank gets eight.
        counts = torch.tensor([[[3, 7, 5, 1]]], dtype=torch.int32).repeat(16, 1, 1)
        with patch.object(PlanningSpec, "compute", autospec=True) as compute:
            self.assertIsNone(plan_migration(counts, current, spec, 2, (0,)))
        compute.assert_not_called()


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
