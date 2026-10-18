"""CPU contract tests; runnable without importing SGLang's GPU dependencies.

python test/registered/unit/elastic_ep/test_live_scale.py -v
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import logging
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOT = Path(__file__).resolve().parents[4]
SRT = ROOT / "python/sglang/srt"
spec = importlib.util.spec_from_file_location(
    "live_prepare", SRT / "elastic_ep/live_prepare.py"
)
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


class MemoryStore:
    def __init__(self):
        self.values = {}
        self.lock = threading.Lock()

    def set(self, key, value):
        with self.lock:
            self.values[key] = value

    def get(self, key):
        with self.lock:
            return self.values[key]

    def check(self, keys):
        with self.lock:
            return all(key in self.values for key in keys)

    def multi_get(self, keys):
        with self.lock:
            return [self.values[key] for key in keys]


def load_class(path, name, namespace, methods=None):
    """Load real class bodies with transport mocks, without GPU imports."""
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )
    if methods is not None:
        cls.body = [
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name in methods
        ]
    cls.bases = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[name]


class TestLivePreparation(unittest.TestCase):
    def test_slow_joiner_keeps_old_ranks_preparing_until_local_init_finishes(self):
        store = MemoryStore()
        stages = [
            live.LiveScalePreparation(
                store, rank, 2, 3, 2, operation_id="attempt-1", manifest="matched"
            )
            for rank in range(3)
        ]
        gate = threading.Event()
        try:
            stages[0].start(lambda: None)
            stages[1].start(lambda: None)
            stages[2].start(lambda: gate.wait(1))
            self.assertTrue(stages[0].connected.wait(1))
            self.assertTrue(stages[1].connected.wait(1))
            for _ in range(100):
                # Serving polls do not wait for the joiner's connection or init.
                self.assertEqual(stages[0].poll(), 0)
                self.assertEqual(stages[1].poll(), 0)
            gate.set()
            self.assertTrue(stages[2].connected.wait(1))
            self.assertEqual(stages[0].poll(), 0)
            stages[2].mark_ready()
            for stage in stages:
                self.assertTrue(stage.done.wait(1))
                self.assertEqual(stage.poll(), 1)
        finally:
            gate.set()
            for stage in stages:
                if stage.thread:
                    stage.thread.join(3)

    def test_failure_propagates_to_other_preparation_workers(self):
        store = MemoryStore()
        old = live.LiveScalePreparation(
            store, 0, 1, 2, 2, operation_id="attempt-1", manifest="matched"
        )
        joiner = live.LiveScalePreparation(
            store, 1, 1, 2, 2, operation_id="attempt-1", manifest="matched"
        )
        with self.assertLogs(live.logger, level="ERROR"):
            old.start(lambda: None)
            joiner.start(Mock(side_effect=RuntimeError("connection failed")))
            self.assertTrue(old.done.wait(1))
            self.assertTrue(joiner.done.wait(1))
        self.assertEqual(old.poll(), -1)
        self.assertIn("connection failed", old.error)

    def test_missing_joiner_times_out(self):
        stage = live.LiveScalePreparation(
            MemoryStore(), 0, 1, 2, 0.03, "attempt-1", "matched"
        )
        with self.assertLogs(live.logger, level="ERROR"):
            stage.start(lambda: None)
            self.assertTrue(stage.done.wait(1))
        self.assertEqual(stage.poll(), -1)

    def test_readiness_requires_connection(self):
        stage = live.LiveScalePreparation(
            MemoryStore(), 1, 1, 2, 1, "attempt-1", "matched"
        )
        with self.assertRaisesRegex(RuntimeError, "before connections"):
            stage.mark_ready()

    def test_late_joiner_cannot_advertise_readiness_after_failure(self):
        stage = live.LiveScalePreparation(
            MemoryStore(), 1, 1, 2, 1, "attempt-1", "matched"
        )
        stage.connected.set()
        stage.error = "preparation expired"
        with self.assertRaisesRegex(RuntimeError, "preparation expired"):
            stage.mark_ready()
        self.assertFalse(stage.store.check(["ready/1"]))

    def test_nixl_wait_preserves_namespace_timeout_and_store_errors(self):
        from datetime import timedelta

        store = Mock()
        scoped = live.PreparationStore(store, "attempt-2")
        timeout = timedelta(seconds=300)
        scoped.wait(["NIXL_EP/0", "NIXL_EP/1"], timeout)
        store.wait.assert_called_once_with(
            ["attempt-2/NIXL_EP/0", "attempt-2/NIXL_EP/1"], timeout
        )
        store.wait.reset_mock()
        scoped.wait(["NIXL_EP/0"])
        store.wait.assert_called_once_with(["attempt-2/NIXL_EP/0"])
        store.wait.side_effect = TimeoutError("peer metadata missing")
        with self.assertRaisesRegex(TimeoutError, "peer metadata missing"):
            scoped.wait(["NIXL_EP/2"], timeout)

    def test_metadata_survives_early_peer_finish_and_isolates_next_scale(self):
        store = MemoryStore()
        first = live.PreparationStore(store, "1/2")
        next_scale = live.PreparationStore(store, "2/3")
        first.set("NIXL_EP/0", b"endpoint")
        first.delete_key("NIXL_EP/0")
        self.assertEqual(first.multi_get(["NIXL_EP/0"]), [b"endpoint"])
        self.assertFalse(next_scale.check(["NIXL_EP/0"]))


class TestNixlPreparation(unittest.TestCase):
    def test_connection_does_not_publish_membership_until_commit(self):
        state = SimpleNamespace(
            buffer=Mock(),
            connected_ep_size=2,
            scale_to=2,
            dispatch_ep_size=2,
            staged_ep_size=None,
            reuse_graphs=False,
        )
        stream = Mock()
        cuda = SimpleNamespace(
            set_device=Mock(),
            Stream=lambda: stream,
            stream=lambda _: contextlib.nullcontext(),
        )
        facade = load_class(
            SRT / "layers/moe/token_dispatcher/nixl.py",
            "NixlEPBuffer",
            {
                "torch": SimpleNamespace(cuda=cuda),
                "live_scale_mode": lambda: "split",
                "get_exec": lambda: SimpleNamespace(
                    moe=SimpleNamespace(elastic_ep_live_prepare=True)
                ),
                "logger": logging.getLogger(__name__),
            },
        )
        facade._state = classmethod(lambda cls: state)
        callbacks = []
        preparation = SimpleNamespace(
            rank=0,
            old_size=2,
            target_size=3,
            store=MemoryStore(),
            start=callbacks.append,
        )
        facade.start_live_prepare(preparation, 0)
        self.assertEqual(state.dispatch_ep_size, 2)
        callbacks[0]()
        state.buffer.connect_ranks.assert_called_once_with([2], activate=False)
        state.buffer.update_mask_buffer.assert_not_called()
        self.assertEqual(state.connected_ep_size, 2)
        self.assertEqual(state.dispatch_ep_size, 2)
        facade.on_scale(2, 3)
        self.assertEqual(state.dispatch_ep_size, 3)
        self.assertEqual(state.connected_ep_size, 3)
        self.assertEqual(state.buffer.update_mask_buffer.call_count, 3)
        # The combined shrink path must not require an inactive growth cohort.
        state.reuse_graphs = True
        state.dispatch_ep_size = 8
        facade.on_scale(3, 2)
        self.assertEqual(state.dispatch_ep_size, 8)
        self.assertEqual(state.scale_to, 2)
        self.assertEqual(state.buffer.update_mask_buffer.call_count, 3)


class TestCaptureProgress(unittest.TestCase):
    def test_serving_turns_between_every_capture_and_largest_first(self):
        spec = importlib.util.spec_from_file_location(
            "coordinator", SRT / "elastic_ep/live_scale.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        plan = module.CaptureProgress([1, 2, 4, 8], 3)
        for i, size in enumerate([8, 4, 2, 1]):
            self.assertIsNone(plan.next_step())
            self.assertIsNone(plan.next_step())
            self.assertEqual(plan.next_step(), size)
            self.assertEqual(plan.complete_step(), i == 3)
        with self.assertRaises(ValueError):
            module.CaptureProgress([1], 0)

    def test_capture_step_restores_plan_after_error(self):
        runner_cls = load_class(
            SRT / "model_executor/runner/decode_cuda_graph_runner.py",
            "DecodeCudaGraphRunner",
            {"model_capture_mode": contextlib.nullcontext},
            methods={"capture_step"},
        )
        runner = runner_cls()
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

    def test_mismatched_manifest_prevents_transport_mutation(self):
        store = MemoryStore()
        a = live.LiveScalePreparation(store, 0, 1, 2, 1, "attempt", "split")
        b = live.LiveScalePreparation(store, 1, 1, 2, 1, "attempt", "prepare")
        connect = Mock()
        with self.assertLogs(live.logger, level="ERROR"):
            a.start(connect)
            b.start(connect)
            self.assertTrue(a.done.wait(1))
            self.assertTrue(b.done.wait(1))
        connect.assert_not_called()
        self.assertEqual(a.poll(), -1)
        self.assertEqual(b.poll(), -1)

    def test_retry_does_not_observe_previous_attempt_readiness(self):
        store = MemoryStore()
        old = live.LiveScalePreparation(store, 0, 1, 2, 1, "attempt-1", "split")
        old.store.set("ready/0", b"1")
        old.store.set("ready/1", b"1")
        new = live.LiveScalePreparation(store, 0, 1, 2, 1, "attempt-2", "split")
        self.assertFalse(new.store.check(["ready/0", "ready/1"]))


class TestServingReadiness(unittest.TestCase):
    def test_joiner_uses_runtime_parallel_context_for_cohort_and_target(self):
        cls = load_class(
            SRT / "elastic_ep/live_scale.py",
            "LiveScaleCoordinator",
            {},
            methods={"initialize", "finish_joiner"},
        )
        coordinator = cls()
        # ModelRunner no longer owns tp_group, tp_rank or tp_size.
        coordinator.runner = SimpleNamespace(
            _elastic_cuda_graph_enabled=lambda: False,
            _elastic_global_rank=lambda: 8,
            _initialize_elastic_ep_joiner=Mock(),
            _finish_joiner_ready=Mock(),
        )
        coordinator.mode = "prepare"
        coordinator.event = Mock()
        coordinator.start_preparation = Mock()
        coordinator.preparation = Mock()
        parallel = SimpleNamespace(
            ep_join_rank_offset=8,
            tp_rank=0,
            tp_size=8,
            tp_group=SimpleNamespace(cpu_group=object()),
        )
        dist = SimpleNamespace(barrier=Mock())
        register = Mock()
        modules = {
            "torch": SimpleNamespace(distributed=dist),
            "torch.distributed": dist,
            "sglang.srt.runtime_context": SimpleNamespace(
                get_exec=lambda: SimpleNamespace(
                    moe=SimpleNamespace(is_ep_scale_joiner=True)
                ),
                get_parallel=lambda: parallel,
            ),
            "sglang.srt.elastic_ep.elastic_ep": SimpleNamespace(
                register_scale_cohort=register,
                ElasticEPStateManager=Mock(),
                clear_expert_map_inbox=Mock(),
                seed_barrier_epochs=Mock(),
            ),
            "sglang.srt.layers.moe.token_dispatcher.nixl": SimpleNamespace(
                NixlEPBuffer=Mock()
            ),
            "sglang.srt.model_executor.model_runner_components.cuda_graph_setup": SimpleNamespace(
                finalize_cuda_graph_capture=Mock()
            ),
            "sglang.srt.platforms": SimpleNamespace(current_platform=Mock()),
        }
        with patch.dict("sys.modules", modules):
            coordinator.initialize()
            coordinator.finish_joiner()
        dist.barrier.assert_called_once_with(group=parallel.tp_group.cpu_group)
        register.assert_called_once_with(8, 16, False)
        coordinator.start_preparation.assert_called_once_with(8, 16)
        coordinator.runner._finish_joiner_ready.assert_called_once_with(16)

    def test_activation_constructs_deferred_runner_and_reports_eager_ready(self):
        spec = importlib.util.spec_from_file_location(
            "coordinator", SRT / "elastic_ep/live_scale.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        coordinator = object.__new__(module.LiveScaleCoordinator)
        coordinator.runner = mr = Mock()
        coordinator.service_steps = 32
        coordinator.event = Mock()
        coordinator.report = Mock()
        runner_factory = mr._decode_cuda_graph_runner_cls.return_value
        candidate = runner_factory.return_value
        candidate.capture_bs = [1, 2, 4]
        state = Mock()
        dist = SimpleNamespace(group=SimpleNamespace(WORLD=0))
        setup = SimpleNamespace(
            drop_elastic_cuda_graph_state=Mock(),
            refresh_deep_gemm_layout_memory_budget=Mock(),
            resolve_elastic_cuda_graph_config=Mock(),
            sync_elastic_cuda_graph_config=Mock(),
        )
        modules = {
            "torch": SimpleNamespace(distributed=dist),
            "torch.distributed": dist,
            "sglang.srt.elastic_ep.elastic_ep": SimpleNamespace(
                ElasticEPStateManager=SimpleNamespace(instance=lambda: state)
            ),
            "sglang.srt.platforms": SimpleNamespace(current_platform=Mock()),
            "sglang.srt.model_executor.model_runner_components.cuda_graph_setup": setup,
            "sglang.srt.model_executor.runner_utils.capture_mode": SimpleNamespace(
                model_capture_mode=contextlib.nullcontext
            ),
        }
        with patch.dict("sys.modules", modules):
            coordinator.activate(16, 8)
        mr._decode_cuda_graph_runner_cls.assert_called_once_with()
        runner_factory.assert_called_once_with(mr, defer_capture=True)
        candidate.warmup.assert_called_once_with()
        candidate.capture.assert_not_called()
        self.assertIsNone(mr.decode_cuda_graph_runner)
        self.assertEqual(state.effective_ep_size, 16)
        self.assertEqual(state.pending_ep_size, 16)
        self.assertEqual(state.scale_phase, "graph_warming")
        coordinator.report.assert_called_once_with(complete=False)

    def test_staggered_control_delivery_waits_for_all_ranks(self):
        cls = load_class(
            SRT / "elastic_ep/live_scale.py",
            "LiveScaleCoordinator",
            {},
            methods={"tick", "observe_control_targets"},
        )
        coordinator = cls()
        coordinator.runner = Mock()
        coordinator.candidate = None
        dist = Mock()
        modules = {
            "torch": SimpleNamespace(distributed=dist),
            "torch.distributed": dist,
            "sglang.srt.elastic_ep.elastic_ep": SimpleNamespace(
                ElasticEPStateManager=Mock()
            ),
            "sglang.srt.platforms": SimpleNamespace(current_platform=Mock()),
        }
        with patch.dict("sys.modules", modules):
            for targets, should_enter in [
                ([0, 0], False),
                ([0, 16], False),
                ([16, 0], False),
                ([16, 16], True),
            ]:
                with self.subTest(targets=targets):
                    coordinator.runner.maybe_join_ep_ranks.reset_mock()
                    coordinator.observe_control_targets(targets)
                    coordinator.tick()
                    dist.all_reduce.assert_not_called()
                    self.assertEqual(
                        coordinator.runner.maybe_join_ep_ranks.called, should_enter
                    )
            with self.assertRaisesRegex(RuntimeError, "Conflicting"):
                coordinator.observe_control_targets([8, 16])
            with self.assertRaisesRegex(RuntimeError, "Timed out"):
                coordinator.observe_control_targets([0, -1])

    def test_dp_metadata_preserves_legacy_width_and_adds_live_vote(self):
        cls = load_class(
            SRT / "managers/scheduler_components/dp_attn.py",
            "MLPSyncBatchInfo",
            {
                "dataclass": lambda c: c,
                "torch": SimpleNamespace(int64=0, tensor=lambda values, **kw: values),
                "ForwardMode": SimpleNamespace(IDLE=SimpleNamespace(value=0)),
            },
            methods={"_get_local_tensor", "_get_fallback_tensor"},
        )
        info = cls()
        for name in [
            "num_tokens",
            "num_tokens_for_logprob",
            "can_run_decode_cuda_graph",
            "is_extend_in_batch",
            "local_can_run_tbo",
            "local_forward_mode",
            "can_run_prefill_cuda_graph",
            "prefill_cuda_graph_max_prefix_len",
            "can_run_draft_cuda_graph",
            "is_health_check",
            "attention_pairs",
        ]:
            setattr(info, name, 1)
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.elastic_ep.elastic_ep": SimpleNamespace(
                    departure_pending=lambda: True
                )
            },
        ):
            info.live_scale_target = None
            legacy = info._get_local_tensor("cpu")
            self.assertEqual(len(legacy), 12)
            self.assertEqual(legacy[-1], 1)
            self.assertEqual(len(info._get_fallback_tensor("cpu")), 12)
            info.live_scale_target = 16
            self.assertEqual(info._get_local_tensor("cpu"), legacy[:-1] + [16, 1])
            self.assertEqual(info._get_fallback_tensor("cpu")[-2:], [0, 0])

    def test_singleton_joiner_buffer_sizing_before_and_after_admission(self):
        tree = ast.parse((SRT / "utils/common.py").read_text())
        fn = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "require_mlp_tp_gather"
        )
        parallel = SimpleNamespace(attn_dp_enabled=True, num_dp_ranks=1)
        moe = SimpleNamespace(is_ep_scale_joiner=True, elastic_ep_backend="mooncake")
        namespace = {
            "get_parallel": lambda: parallel,
            "get_exec": lambda: SimpleNamespace(moe=moe),
        }
        modules = {
            "sglang.srt.layers.moe.utils": SimpleNamespace(
                get_moe_a2a_backend=Mock(), MoeA2ABackend=object
            ),
            "sglang.srt.elastic_ep.elastic_ep": SimpleNamespace(
                elastic_expanded_world_enabled=lambda: True
            ),
        }
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), "singleton", "exec"),
            namespace,
        )
        with patch.dict("sys.modules", modules):
            gather = namespace["require_mlp_tp_gather"]
            self.assertFalse(gather())
            parallel.num_dp_ranks = 5
            self.assertTrue(gather())
            parallel.num_dp_ranks = 1
            moe.is_ep_scale_joiner = False
            with self.assertRaises(AssertionError):
                gather()
            moe.is_ep_scale_joiner = True
            moe.elastic_ep_backend = None
            with self.assertRaises(AssertionError):
                gather()

    def test_expanded_serving_does_not_complete_scale_before_graphs(self):
        cls = load_class(
            SRT / "managers/tokenizer_manager.py",
            "TokenizerManager",
            {"physical_ep_size_to_dp_size": lambda size, width: size // width},
            methods={"forward_elastic_scale_update"},
        )
        manager = cls()
        manager._dispatch_to_scheduler = Mock()
        manager.update_control_communicator_fan_out = Mock()
        manager.attn_replica_size = 2
        manager.elastic_pending_stages = []
        manager.elastic_staged_from = None
        manager._clear_staged_scale = Mock()
        manager._open_elastic_shrink_gate = Mock()
        manager.elastic_pending_ep_size = 16
        msg = SimpleNamespace(
            success=True, effective_ep_size=16, graph_ready=False, direction="grow"
        )
        manager.forward_elastic_scale_update(msg)
        self.assertEqual(manager.elastic_worker_count, 8)
        manager.update_control_communicator_fan_out.assert_called_with(8)
        self.assertEqual(manager.elastic_pending_ep_size, 16)
        self.assertEqual(manager.elastic_scale_phase, "graph_warming")
        msg.graph_ready = True
        manager.forward_elastic_scale_update(msg)
        self.assertIsNone(manager.elastic_pending_ep_size)
        self.assertEqual(manager.elastic_scale_phase, "serving_expanded")


class TestRecoveryCompatibility(unittest.TestCase):
    def test_only_started_live_preparation_votes_fatal_timeout(self):
        import time

        state = SimpleNamespace(
            pending_ep_size=8,
            pending_since=time.monotonic() - 10,
            pending_recover_ranks=[],
        )
        cls = load_class(
            SRT / "elastic_ep/live_scale.py",
            "LiveScaleCoordinator",
            {"time": time},
            methods={"local_control_target"},
        )
        coordinator = cls()
        coordinator.candidate = None
        coordinator.preparation = None
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.elastic_ep.elastic_ep": SimpleNamespace(
                    ElasticEPStateManager=SimpleNamespace(instance=lambda: state)
                ),
                "sglang.srt.runtime_context": SimpleNamespace(
                    get_exec=lambda: SimpleNamespace(
                        moe=SimpleNamespace(elastic_ep_scale_timeout=1)
                    )
                ),
            },
        ):
            self.assertEqual(coordinator.local_control_target(), -2)
            coordinator.preparation = object()
            self.assertEqual(coordinator.local_control_target(), -1)
            state.pending_recover_ranks = [4, 5, 6, 7]
            self.assertEqual(coordinator.local_control_target(), -2)

    def test_recovery_startup_installs_live_control_after_normal_admission(self):
        events = []
        coordinator = SimpleNamespace(initialize=lambda: events.append("live"))
        cls = load_class(
            SRT / "model_executor/model_runner.py",
            "ModelRunner",
            {
                "get_exec": lambda: SimpleNamespace(
                    moe=SimpleNamespace(ep_join_mode="recover")
                )
            },
            methods={"_initialize_elastic_ep_serving"},
        )
        runner = cls()
        runner.live_scale = None

        def admit():
            self.assertIsNone(runner.live_scale)
            events.append("admit")

        runner._initialize_elastic_ep_joiner = admit
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.elastic_ep.live_scale": SimpleNamespace(
                    LiveScaleCoordinator=lambda model_runner: coordinator,
                    live_scale_mode=lambda: "split",
                )
            },
        ):
            runner._initialize_elastic_ep_serving()
        self.assertEqual(events, ["admit", "live"])
        self.assertIs(runner.live_scale, coordinator)

    def test_live_deadlines_use_gathered_vote_instead_of_local_recheck(self):
        import time

        state = SimpleNamespace(
            pending_since=time.monotonic() - 10,
            pending_recover_ranks=[],
            scale_phase="joining",
        )
        manager = SimpleNamespace(
            is_scaling=lambda: True,
            instance=lambda: state,
            get_effective_ep_size=lambda: 4,
            get_pending_ep_size=lambda: 8,
        )
        cls = load_class(
            SRT / "model_executor/model_runner.py",
            "ModelRunner",
            {"ElasticEPStateManager": manager, "time": time},
            methods={"maybe_join_ep_ranks"},
        )
        runner = cls()
        runner.server_args = SimpleNamespace(elastic_ep_scale_timeout=1)
        runner.live_scale = SimpleNamespace(prepare=Mock(return_value=False))
        runner._fail_scale = Mock()
        runner.maybe_join_ep_ranks()
        runner.live_scale.prepare.assert_called_once_with(4, 8)
        runner._fail_scale.assert_not_called()
        runner.live_scale = None
        runner.maybe_join_ep_ranks()
        runner._fail_scale.assert_called_once()

    def test_staggered_delivery_timeout_aborts_all_ranks_without_preparation(self):
        cls = load_class(
            SRT / "elastic_ep/live_scale.py",
            "LiveScaleCoordinator",
            {},
            methods={"observe_control_targets", "tick"},
        )
        coordinator = cls()
        coordinator.candidate = None
        coordinator.runner = SimpleNamespace(
            _fail_scale=Mock(), maybe_join_ep_ranks=Mock()
        )
        dist = SimpleNamespace()
        with patch.dict(
            "sys.modules",
            {
                "torch": SimpleNamespace(distributed=dist),
                "torch.distributed": dist,
                "sglang.srt.elastic_ep.elastic_ep": SimpleNamespace(
                    ElasticEPStateManager=SimpleNamespace(
                        get_effective_ep_size=lambda: 4
                    )
                ),
                "sglang.srt.platforms": SimpleNamespace(current_platform=Mock()),
            },
        ):
            coordinator.observe_control_targets([8, 0])
            coordinator.tick()
            coordinator.runner._fail_scale.assert_not_called()
            for votes in ([-2, 0], [0, -2], [8, -2]):
                coordinator.observe_control_targets(votes)
                coordinator.tick()
            self.assertEqual(coordinator.runner._fail_scale.call_count, 3)
            coordinator.runner.maybe_join_ep_ranks.assert_not_called()
            self.assertEqual(coordinator.agreed_target, 0)
        for votes in ([-1, 0], [-2, -1], [-1, -2]):
            with self.assertRaisesRegex(RuntimeError, "restart required"):
                coordinator.observe_control_targets(votes)

    def test_completed_operation_discards_preparation_before_next_request(self):
        cls = load_class(
            SRT / "elastic_ep/live_scale.py",
            "LiveScaleCoordinator",
            {},
            methods={"report"},
        )
        coordinator = cls()
        coordinator.runner = SimpleNamespace(_elastic_global_rank=lambda: 1)
        coordinator.preparation = object()
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.managers.io_struct": SimpleNamespace(
                    ElasticScaleUpdateReq=Mock()
                ),
            },
        ):
            coordinator.report(complete=False)
            self.assertIsNotNone(coordinator.preparation)
            coordinator.report(complete=True)
            self.assertIsNone(coordinator.preparation)

    def test_graph_forward_still_drives_bare_fault_recovery(self):
        cls = load_class(
            SRT / "managers/tp_worker.py",
            "TpModelWorker",
            {},
            methods={"_maybe_finalize_elastic_cuda_graph_scale"},
        )
        worker = cls()
        worker.model_runner = SimpleNamespace(
            live_scale=object(),
            _elastic_cuda_graph_enabled=lambda: True,
            maybe_join_ep_ranks=Mock(),
        )
        state = SimpleNamespace(get_pending_ep_size=Mock(return_value=None))
        with patch.dict(
            "sys.modules",
            {
                "sglang.srt.elastic_ep.elastic_ep": SimpleNamespace(
                    ElasticEPStateManager=state
                )
            },
        ):
            worker._maybe_finalize_elastic_cuda_graph_scale()
            worker.model_runner.maybe_join_ep_ranks.assert_called_once_with()
            worker.model_runner.maybe_join_ep_ranks.reset_mock()
            state.get_pending_ep_size.return_value = 8
            worker._maybe_finalize_elastic_cuda_graph_scale()
            worker.model_runner.maybe_join_ep_ranks.assert_not_called()
            worker.model_runner.live_scale = None
            worker._maybe_finalize_elastic_cuda_graph_scale()
            worker.model_runner.maybe_join_ep_ranks.assert_called_once_with()

    def test_unstaged_recovery_regrow_keeps_lazy_nixl_connections(self):
        cls = load_class(
            SRT / "layers/moe/token_dispatcher/nixl.py",
            "NixlEPBuffer",
            {"logger": Mock()},
            methods={"on_scale"},
        )
        state = SimpleNamespace(
            staged_ep_size=None,
            reuse_graphs=True,
            connected_ep_size=4,
            dispatch_ep_size=8,
            buffer=Mock(),
        )
        cls._state = lambda: state
        cls.on_scale(4, 8)
        self.assertEqual(state.scale_to, 8)
        self.assertEqual(state.connected_ep_size, 4)
        self.assertEqual(state.dispatch_ep_size, 8)
        state.buffer.update_mask_buffer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
