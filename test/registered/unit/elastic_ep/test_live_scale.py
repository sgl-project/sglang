"""Live preparation, admission, and graph reuse contracts (no GPU required)."""

import contextlib
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from sglang.srt import runtime_context
from sglang.srt.distributed import utils as distributed_utils
from sglang.srt.elastic_ep import elastic_ep, live_prepare, live_scale
from sglang.srt.environ import envs, exportable_env_vars
from sglang.srt.layers.moe.token_dispatcher import nixl
from sglang.srt.managers.scheduler_components.dp_attn import MLPSyncBatchInfo
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor import model_runner
from sglang.srt.model_executor.model_runner_components import cuda_graph_setup
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)
from sglang.srt.platforms import current_platform
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPreparation(unittest.TestCase):
    def setUp(self):
        self.store = dist.HashStore()
        self.store.set_timeout(timedelta(seconds=2))

    def stage(self, rank, *, manifest="matched", timeout=2):
        stage = live_prepare.LiveScalePreparation(
            self.store, rank, 1, 2, timeout, "operation", manifest
        )
        self.addCleanup(lambda: stage.thread and stage.thread.join(3))
        return stage

    def test_connection_does_not_replace_joiner_local_readiness(self):
        old, joiner = self.stage(0), self.stage(1)
        for stage in (old, joiner):
            stage.start(lambda: None)
            # Neither worker can connect until both publish their manifests.
        self.assertTrue(joiner.connected.wait(1))
        self.assertEqual(old.poll(), 0)
        joiner.mark_ready()
        for stage in (old, joiner):
            self.assertTrue(stage.done.wait(1))
            self.assertEqual(stage.poll(), 1)

    def test_connection_failure_and_manifest_mismatch_propagate(self):
        for mismatch in (False, True):
            with self.subTest(mismatch=mismatch):
                self.store = dist.HashStore()
                old = self.stage(0)
                joiner = self.stage(1, manifest="other" if mismatch else "matched")
                old_connect = Mock()
                joiner_connect = Mock(
                    side_effect=None if mismatch else RuntimeError("connect")
                )
                with self.assertLogs(live_prepare.logger, level="ERROR"):
                    old.start(old_connect)
                    joiner.start(joiner_connect)
                    for stage in (old, joiner):
                        self.assertTrue(stage.done.wait(1))
                        self.assertEqual(stage.poll(), -1)
                if mismatch:
                    old_connect.assert_not_called()
                    joiner_connect.assert_not_called()
                else:
                    self.assertIn("connect", old.error)

    def test_missing_joiner_and_invalid_readiness(self):
        stage = self.stage(1, timeout=0.03)
        with self.assertRaisesRegex(RuntimeError, "before connections"):
            stage.mark_ready()
        with self.assertLogs(live_prepare.logger, level="ERROR"):
            stage.start(lambda: None)
            self.assertTrue(stage.done.wait(1))
        stage.connected.set()
        with self.assertRaises(RuntimeError):
            stage.mark_ready()
        self.assertFalse(stage.store.check(["ready/1"]))

    def test_transport_records_survive_early_finish_and_remain_scoped(self):
        first = live_prepare.PreparationStore(self.store, "first")
        first.set("NIXL_EP/0", b"endpoint")
        first.delete_key("NIXL_EP/0")
        self.assertEqual(first.multi_get(["NIXL_EP/0"]), [b"endpoint"])
        self.assertFalse(
            live_prepare.PreparationStore(self.store, "next").check(["NIXL_EP/0"])
        )
        transport = Mock()
        scoped = live_prepare.PreparationStore(transport, "first")
        timeout = timedelta(seconds=30)
        scoped.wait(["NIXL_EP/0"], timeout)
        transport.wait.assert_called_once_with(["first/NIXL_EP/0"], timeout)
        transport.wait.side_effect = TimeoutError("metadata missing")
        with self.assertRaises(TimeoutError):
            scoped.wait(["NIXL_EP/0"])


class TestLiveScale(unittest.TestCase):
    def setUp(self):
        self.context = contextlib.ExitStack()
        self.addCleanup(self.context.close)
        for variable, value in (
            (envs.SGLANG_ELASTIC_LIVE_SCALE, "split"),
            (envs.SGLANG_ELASTIC_CAPTURE_SERVICE_STEPS, "2"),
            (envs.SGLANG_ENABLE_GRAPH_POOL_BORROW, False),
        ):
            self.context.enter_context(variable.override(value))
        self.moe = SimpleNamespace(
            elastic_ep_backend="mooncake",
            moe_a2a_backend="nixl",
            elastic_ep_reuse_cuda_graph=True,
            elastic_ep_scale_timeout=2,
            ep_dispatch_algorithm="static",
            is_ep_scale_joiner=False,
            ep_join_mode="scale",
        )
        execution = SimpleNamespace(
            moe=self.moe, overlap=SimpleNamespace(enable_two_batch_overlap=False)
        )
        self.parallel = SimpleNamespace(
            attn_tp_size=1,
            attn_cp_size=1,
            dcp_size=1,
            pp_size=1,
            moe_dense_tp_size=1,
            enable_dp_lm_head=True,
            max_ep_size=8,
            num_dp_ranks=4,
            ep_join_rank_offset=4,
            tp_size=4,
            tp_rank=0,
        )
        for module in (runtime_context, model_runner, nixl):
            self.patch(
                module, get_exec=lambda: execution, get_parallel=lambda: self.parallel
            )
        self.patch(
            runtime_context,
            get_schedule=lambda: SimpleNamespace(disable_overlap_schedule=True),
        )
        self.store = dist.HashStore()
        self.store.set_timeout(timedelta(seconds=2))
        self.patch(distributed_utils, get_global_tcp_store=lambda: self.store)
        active = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=torch.int32)
        self.state = elastic_ep.ElasticEPState(
            active, active.clone(), active.clone(), effective_ep_size=4
        )
        self.patch(elastic_ep.ElasticEPStateManager, _instance=self.state)

    def patch(self, target, **values):
        self.context.enter_context(patch.multiple(target, **values))

    def coordinator(self, rank=0):
        runner = Mock(gpu_id=0, graph_time_usage={})
        runner._elastic_global_rank.return_value = rank
        runner._elastic_cuda_graph_enabled.return_value = True
        runner.spec_algorithm.is_speculative.return_value = False
        return live_scale.LiveScaleCoordinator(runner)

    def test_invalid_configuration_fails_early(self):
        exported = exportable_env_vars()
        self.assertEqual(exported["SGLANG_ELASTIC_LIVE_SCALE"], "split")
        self.assertEqual(exported["SGLANG_ELASTIC_CAPTURE_SERVICE_STEPS"], "2")
        for variable, value in (
            (envs.SGLANG_ELASTIC_LIVE_SCALE, "typo"),
            (envs.SGLANG_ELASTIC_CAPTURE_SERVICE_STEPS, "0"),
            (envs.SGLANG_ELASTIC_CAPTURE_SERVICE_STEPS, "invalid"),
        ):
            with self.subTest(value=value), variable.override(value):
                with self.assertRaises(ValueError):
                    self.coordinator()

    def test_repeated_growth_waits_for_fresh_operation_announcement(self):
        self.patch(
            nixl.NixlEPBuffer,
            live_layout_signature=lambda: (8,),
            start_live_prepare=Mock(),
        )
        self.patch(
            cuda_graph_setup,
            resolve_elastic_cuda_graph_config=Mock(
                return_value=Mock(signature=lambda: (1, 2))
            ),
        )
        leader, follower, late_joiner = [self.coordinator(rank) for rank in (0, 1, 4)]
        for coordinator in (leader, follower, late_joiner):
            coordinator.graph_storage_snapshot = lambda: ()
            coordinator.start_preparation(4, 8)
        previous = leader.preparation
        previous.store.set("ready/1", b"1")
        previous.store.set("NIXL_EP/1", b"endpoint")
        leader.old, leader.target = 4, 8
        key = "elastic_ep/live_operation/4/8"
        leader.report(complete=False)
        self.assertTrue(self.store.check([key]))
        leader.report(complete=True)
        follower.complete_preparation()
        self.assertFalse(self.store.check([key]))

        # A prestarted replacement joiner and a faster old rank must both wait
        # for rank zero, rather than reading the previous operation's UUID.
        replacement = self.coordinator(4)

        def start(coordinator, entered):
            entered.set()
            coordinator.start_preparation(4, 8)

        with ThreadPoolExecutor(max_workers=2) as pool:
            events = [threading.Event(), threading.Event()]
            futures = [
                pool.submit(start, c, e)
                for c, e in zip((follower, replacement), events)
            ]
            for event, future in zip(events, futures):
                self.assertTrue(event.wait(1))
                self.assertFalse(future.done())
            leader.start_preparation(4, 8)
            late_joiner.complete_preparation()
            for future in futures:
                future.result(timeout=2)
        operation = leader.preparation.operation_id
        self.assertNotEqual(operation, previous.operation_id)
        self.assertEqual(self.store.get(key).decode(), operation)
        for coordinator in (follower, replacement):
            self.assertEqual(coordinator.preparation.operation_id, operation)
            self.assertFalse(coordinator.preparation.store.check(["ready/1"]))
        self.assertEqual(previous.store.get("NIXL_EP/1"), b"endpoint")
        self.coordinator().complete_preparation()  # Recovery has no preparation.

    def test_target_agreement_and_timeout_votes(self):
        coordinator = self.coordinator()
        self.state.pending_ep_size = 8
        self.state.pending_since = time.monotonic() - 10
        self.assertEqual(coordinator.local_control_target(), -2)
        coordinator.preparation = object()
        self.assertEqual(coordinator.local_control_target(), -1)
        self.state.pending_recover_ranks = [4, 5, 6, 7]
        self.assertEqual(coordinator.local_control_target(), -2)
        for targets, enter in (
            ([0, 0], False),
            ([0, 8], False),
            ([8, 0], False),
            ([8, 8], True),
        ):
            coordinator.runner.maybe_join_ep_ranks.reset_mock()
            coordinator.observe_control_targets(targets)
            coordinator.tick()
            self.assertEqual(coordinator.runner.maybe_join_ep_ranks.called, enter)
        for targets in ([-2, 0], [8, -2]):
            coordinator.observe_control_targets(targets)
            coordinator.tick()
        self.assertEqual(coordinator.runner._fail_scale.call_count, 2)
        for targets in ([8, 16], [-1, 0], [-2, -1]):
            with self.assertRaises(RuntimeError):
                coordinator.observe_control_targets(targets)

    def test_preparation_readiness_uses_collective_vote(self):
        coordinator = self.coordinator()
        coordinator.preparation = Mock(error=None, operation_id="operation")
        coordinator.start_preparation = Mock()
        self.state.active_ranks = torch.ones(4, dtype=torch.int32)
        self.state.pending_since = time.monotonic()
        for vote in (0, 1, -1):
            with (
                self.subTest(vote=vote),
                patch.object(
                    dist,
                    "all_reduce",
                    side_effect=lambda tensor, **_: tensor.fill_(vote),
                ),
            ):
                coordinator.preparation.poll.return_value = 1
                if vote < 0:
                    with self.assertRaisesRegex(RuntimeError, "Peer failed"):
                        coordinator.prepare(4, 8)
                else:
                    self.assertEqual(coordinator.prepare(4, 8), bool(vote))

    def test_joiner_capture_failure_prevents_admission(self):
        self.moe.is_ep_scale_joiner = True
        coordinator = self.coordinator(4)
        coordinator.preparation = Mock(rank=4)
        coordinator.event = Mock()
        coordinator.start_preparation = Mock()
        coordinator.capture_joiner_graphs = Mock(
            side_effect=RuntimeError("capture failed")
        )
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            coordinator.finish_joiner()
        coordinator.preparation.mark_ready.assert_not_called()
        coordinator.runner._initialize_elastic_ep_joiner.assert_not_called()
        coordinator.preparation.store.set.assert_called_once_with(
            "error/4", b"capture failed"
        )

    def test_reuse_checks_storage_and_commits_joiner_after_barrier(self):
        coordinator = self.coordinator(4)
        coordinator.event = Mock()
        coordinator.reuse_snapshot = (1, (1, 2))
        coordinator.graph_storage_snapshot = Mock(return_value=())
        with self.assertRaisesRegex(RuntimeError, "storage changed"):
            coordinator.finish_reused_graphs(8, 4, joiner=True)
        coordinator.runner._elastic_scale_ready_barrier.assert_not_called()
        coordinator.graph_storage_snapshot.return_value = coordinator.reuse_snapshot
        coordinator.complete_preparation = Mock(
            side_effect=lambda: (
                coordinator.runner._elastic_scale_ready_barrier.assert_called_once_with(
                    8, "LIVE_GRAPH_REUSE"
                )
            )
        )
        self.patch(cuda_graph_setup, finalize_cuda_graph_capture=Mock())
        coordinator.finish_reused_graphs(8, 4, joiner=True)
        self.assertTrue(coordinator.joiner_finalized)
        self.assertEqual(self.state.effective_ep_size, 8)
        self.assertEqual(self.state.scale_phase, "warming_up")
        coordinator.complete_preparation.assert_called_once_with()

    def test_split_capture_serves_eager_until_all_shapes_finish(self):
        tokenizer = TokenizerManager.__new__(TokenizerManager)
        tokenizer.attn_replica_size = 1
        tokenizer.elastic_pending_stages = []
        tokenizer.elastic_staged_from = None
        for name in (
            "_dispatch_to_scheduler",
            "update_control_communicator_fan_out",
            "_clear_staged_scale",
            "_open_elastic_shrink_gate",
        ):
            setattr(tokenizer, name, Mock())
        coordinator = self.coordinator()
        coordinator.event = Mock()
        coordinator.complete_preparation = Mock()
        factory = coordinator.runner._decode_cuda_graph_runner_cls.return_value
        candidate = factory.return_value
        candidate.capture_bs = [1, 2]
        self.patch(
            cuda_graph_setup,
            **{
                name: Mock()
                for name in (
                    "resolve_elastic_cuda_graph_config",
                    "sync_elastic_cuda_graph_config",
                    "drop_elastic_cuda_graph_state",
                    "refresh_deep_gemm_layout_memory_budget",
                )
            },
        )
        self.patch(current_platform, synchronize=Mock())
        self.patch(dist, barrier=Mock())
        coordinator.activate(8, 4)
        factory.assert_called_once_with(coordinator.runner, defer_capture=True)
        candidate.warmup.assert_called_once_with()
        candidate.capture.assert_not_called()
        self.assertIsNone(coordinator.runner.decode_cuda_graph_runner)
        self.assertFalse(coordinator.runner._pending_elastic_scale_update.graph_ready)
        tokenizer.forward_elastic_scale_update(
            coordinator.runner._pending_elastic_scale_update
        )
        self.assertEqual(tokenizer.elastic_pending_ep_size, 8)
        self.assertEqual(tokenizer.elastic_scale_phase, "graph_warming")
        for _ in range(3):
            coordinator.tick()
        candidate.capture_step.assert_called_once_with(2)
        coordinator.complete_preparation.assert_not_called()
        coordinator.tick()
        self.assertEqual(
            [call.args for call in candidate.capture_step.call_args_list], [(2,), (1,)]
        )
        self.assertIs(coordinator.runner.decode_cuda_graph_runner, candidate)
        self.assertTrue(coordinator.runner._pending_elastic_scale_update.graph_ready)
        tokenizer.forward_elastic_scale_update(
            coordinator.runner._pending_elastic_scale_update
        )
        self.assertIsNone(tokenizer.elastic_pending_ep_size)
        self.assertEqual(tokenizer.elastic_scale_phase, "serving_expanded")
        coordinator.complete_preparation.assert_called_once_with()

    def test_dp_metadata_keeps_departure_after_live_vote(self):
        info = MLPSyncBatchInfo(4, 4, 1, 1, 1, True, True, False, False, False, 0)
        info.is_shutting_down = True
        info.drain_state = 2
        self.patch(elastic_ep, departure_pending=lambda: True)
        legacy = info._get_local_tensor("cpu").tolist()
        self.assertEqual(len(legacy), 14)
        self.assertEqual(legacy[11:13], [1, 2])
        self.assertEqual(legacy[-1], 1)
        self.assertEqual(len(info._get_fallback_tensor("cpu")), 14)
        info.live_scale_target = 8
        self.assertEqual(info._get_local_tensor("cpu").tolist(), legacy[:-1] + [8, 1])
        self.assertEqual(info._get_fallback_tensor("cpu").tolist()[-2:], [0, 0])

    def test_graph_forward_keeps_bare_fault_recovery(self):
        worker = TpModelWorker.__new__(TpModelWorker)
        worker._model_runner = self.coordinator().runner
        for live, pending, enter in (
            (True, None, True),
            (True, 8, False),
            (False, 8, True),
        ):
            worker.model_runner.live_scale = object() if live else None
            self.state.pending_ep_size = pending
            worker.model_runner.maybe_join_ep_ranks.reset_mock()
            worker._maybe_finalize_elastic_cuda_graph_scale()
            self.assertEqual(worker.model_runner.maybe_join_ep_ranks.called, enter)

    def test_capture_error_restores_plan(self):
        runner = DecodeCudaGraphRunner.__new__(DecodeCudaGraphRunner)
        runner.capture_bs = [1, 2, 4]

        def fail():
            self.assertEqual(runner.capture_bs, [4])
            raise RuntimeError("capture failed")

        runner.capture = fail
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            runner.capture_step(4)
        self.assertEqual(runner.capture_bs, [1, 2, 4])
        with self.assertRaises(ValueError):
            runner.capture_step(7)

    def test_recovery_admission_precedes_live_initialization(self):
        self.moe.ep_join_mode = "recover"
        runner = model_runner.ModelRunner.__new__(model_runner.ModelRunner)
        runner.live_scale = None
        events = []

        def admit():
            self.assertIsNone(runner.live_scale)
            events.append("admit")

        runner._initialize_elastic_ep_joiner = admit
        coordinator = Mock()
        coordinator.initialize.side_effect = lambda: events.append("live")
        self.patch(live_scale, LiveScaleCoordinator=Mock(return_value=coordinator))
        runner._initialize_elastic_ep_serving()
        self.assertEqual(events, ["admit", "live"])
        self.assertIs(runner.live_scale, coordinator)

    def test_live_deadline_uses_agreed_vote(self):
        runner = model_runner.ModelRunner.__new__(model_runner.ModelRunner)
        runner.server_args = SimpleNamespace(elastic_ep_scale_timeout=1)
        runner.live_scale = Mock()
        runner.live_scale.prepare.return_value = False
        runner._fail_scale = Mock()
        self.state.pending_ep_size = 8
        self.state.pending_since = time.monotonic() - 10
        self.state.scale_phase = "joining"
        runner.maybe_join_ep_ranks()
        runner.live_scale.prepare.assert_called_once_with(4, 8)
        runner._fail_scale.assert_not_called()
        runner.live_scale = None
        runner.maybe_join_ep_ranks()
        runner._fail_scale.assert_called_once()

    def test_retained_runner_width_tracks_repeated_resize(self):
        def resize(**kwargs):
            self.parallel.num_dp_ranks = kwargs["physical_ep_size"]
            return self.parallel.num_dp_ranks

        self.patch(model_runner, update_dp_attention_for_elastic_ep=resize)
        runner = model_runner.ModelRunner.__new__(model_runner.ModelRunner)
        runner.live_scale = SimpleNamespace(reuse_graphs=True)
        graph = runner.decode_cuda_graph_runner = SimpleNamespace(num_dp_ranks=4)
        eager = runner.eager_runner = SimpleNamespace(num_dp_ranks=4)
        for width in (5, 6, 4, 8):
            self.assertEqual(runner._apply_dp_size(width, 0), width)
            self.assertIs(runner.decode_cuda_graph_runner, graph)
            self.assertIs(runner.eager_runner, eager)
            self.assertEqual((graph.num_dp_ranks, eager.num_dp_ranks), (width, width))

    def test_inactive_connections_only_admit_at_commit(self):
        state = SimpleNamespace(
            buffer=Mock(),
            connected_ep_size=4,
            scale_to=4,
            dispatch_ep_size=8,
            staged_ep_size=None,
            reuse_graphs=True,
        )
        self.patch(nixl.NixlEPBuffer, _state=lambda: state)
        self.patch(
            torch.cuda,
            set_device=Mock(),
            Stream=Mock(),
            stream=lambda _: contextlib.nullcontext(),
        )
        preparation = Mock(rank=0, old_size=4, target_size=8)
        nixl.NixlEPBuffer.start_live_prepare(preparation, 0)
        preparation.start.call_args.args[0]()
        state.buffer.connect_ranks.assert_called_once_with([4, 5, 6, 7], activate=False)
        state.buffer.update_mask_buffer.assert_not_called()
        self.assertEqual(state.connected_ep_size, 4)
        nixl.NixlEPBuffer.on_scale(4, 8)
        self.assertEqual(state.connected_ep_size, 8)
        self.assertEqual(state.buffer.update_mask_buffer.call_count, 8)
        state.buffer.update_mask_buffer.reset_mock()
        nixl.NixlEPBuffer.on_scale(8, 4)
        state.staged_ep_size = None
        nixl.NixlEPBuffer.on_scale(4, 8)  # Recovery retains lazy connections.
        state.buffer.update_mask_buffer.assert_not_called()
        self.assertEqual(state.dispatch_ep_size, 8)

    def test_joiner_initializes_reuse_before_early_return_and_masks_peers(self):
        self.moe.is_ep_scale_joiner = True
        self.state.ep_join_rank_offset = 4
        state = SimpleNamespace(buffer=None, scale_to=None, reuse_graphs=False)
        buffer = Mock(group_size=8)
        factory = Mock(return_value=buffer)
        factory.get_rdma_size_hint.return_value = 128
        self.patch(
            nixl,
            _load_nixl_ep=lambda: (factory, None),
            get_global_tcp_store=lambda: self.store,
        )
        self.patch(
            nixl.NixlEPBuffer,
            _state=lambda: state,
            _connect_ranks=Mock(),
            _mask_unwired=Mock(),
        )
        self.patch(dist, get_rank=lambda _: 0, get_world_size=lambda _: 2)
        self.patch(torch.cuda, synchronize=Mock())
        mode = Mock()
        mode.enable_normal.return_value = False
        mode.enable_low_latency.return_value = True
        nixl.NixlEPBuffer.get_nixl_buffer(object(), 128, mode, 64, 12, 2)
        self.assertTrue(state.reuse_graphs)
        self.assertEqual(state.dispatch_ep_size, 8)
        self.assertEqual(state.connected_ep_size, 0)
        nixl.NixlEPBuffer._connect_ranks.assert_not_called()
        nixl.NixlEPBuffer._mask_unwired.reset_mock()
        state.staged_ep_size = 6
        nixl.NixlEPBuffer.prepare_local_capture(6, 4)
        nixl.NixlEPBuffer._mask_unwired.assert_called_once_with(
            state, [0, 1, 2, 3, 5, 6, 7], strict=True
        )
        self.assertEqual(state.connected_ep_size, 6)


if __name__ == "__main__":
    unittest.main()
