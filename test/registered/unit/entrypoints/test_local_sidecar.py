"""CPU coverage for local publisher metadata and managed sidecar lifecycle."""

import atexit
import dataclasses
import json
import os
import time
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import zmq

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    KVEventBatch,
    ZmqEventPublisher,
)
from sglang.srt.entrypoints.sidecar import (
    SGLANG_GRPC_ENDPOINT_ENV,
    SGLANG_SIDECAR_CONTEXT_ENV,
    Sidecar,
    _run_sidecar,
    build_sidecar_context,
    start_sidecar,
)
from sglang.srt.entrypoints.sidecar_context import (
    LOCAL_KV_EVENT_SOURCES,
    KvEventSource,
    SidecarContext,
    take_local_kv_event_sources,
)
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


def main(argv):
    """Minimal provider imported by the spawned-process smoke test."""
    context = json.loads(os.environ[SGLANG_SIDECAR_CONTEXT_ENV])
    assert context["version"] == 1
    assert context["mode"] == "telemetry"
    assert SGLANG_GRPC_ENDPOINT_ENV not in os.environ
    source = context["kv_event_sources"][0]
    with zmq.Context() as zmq_context:
        with (
            zmq_context.socket(zmq.SUB) as subscriber,
            zmq_context.socket(zmq.PUSH) as report,
        ):
            subscriber.setsockopt_string(zmq.SUBSCRIBE, source["topic"])
            subscriber.connect(source["endpoint"])
            report.connect(argv[0])
            while True:
                report.send_multipart(subscriber.recv_multipart())


def source(rank):
    return KvEventSource(rank, f"tcp://127.0.0.1:{5557 + rank}", "kv", 64)


class TestSidecarContext(unittest.TestCase):
    def test_scheduler_reports_only_owned_publisher_sources(self):
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.managers.scheduler_components.kv_events_publisher import (
            SchedulerKvEventsPublisher,
        )

        for pp, tp, cp, dp_attention in (
            (0, 0, 0, True),
            (0, 0, 0, False),
            (1, 0, 0, True),
            (0, 1, 0, True),
            (0, 0, 1, True),
        ):
            ps = SimpleNamespace(
                pp_rank=pp,
                attn_tp_rank=tp,
                attn_cp_rank=cp,
                attn_dp_size=8 if dp_attention else 1,
                attn_dp_rank=4 if dp_attention else 0,
                dp_rank=None if dp_attention else 4,
            )
            publisher = MagicMock()
            publisher.describe_local_source.return_value = source(4)
            with (
                self.subTest(ps=ps),
                patch(
                    "sglang.srt.managers.scheduler_components.kv_events_publisher.EventPublisherFactory.create",
                    return_value=publisher,
                ) as create,
            ):
                component = SchedulerKvEventsPublisher(
                    kv_events_config='{"publisher":"zmq"}',
                    ps=ps,
                    attn_tp_rank=tp,
                    attn_cp_rank=cp,
                    attn_dp_rank=ps.attn_dp_rank,
                    dp_rank=ps.dp_rank,
                    tree_cache=None,
                    send_metrics_from_scheduler=None,
                    max_running_requests=1,
                    max_total_num_tokens=64,
                    get_stats=lambda: None,
                )
                scheduler = Scheduler.__new__(Scheduler)
                scheduler.max_total_num_tokens = 64
                scheduler.max_req_input_len = 32
                scheduler.startup_time = {}
                scheduler.page_size = 64
                scheduler.kv_events_publisher = component
                with get_context().override_server_args(
                    sidecar_scope="local-telemetry"
                ):
                    info = scheduler.get_init_info()
                if pp == tp == cp == 0:
                    create.assert_called_once_with('{"publisher":"zmq"}', 4)
                    self.assertEqual(info[LOCAL_KV_EVENT_SOURCES], [source(4)])
                else:
                    create.assert_not_called()
                    self.assertEqual(info[LOCAL_KV_EVENT_SOURCES], [])
                with get_context().override_server_args(sidecar_scope="leader"):
                    self.assertNotIn(LOCAL_KV_EVENT_SOURCES, scheduler.get_init_info())

    def test_engine_extracts_metadata_on_direct_and_dp_controller_paths(self):
        from sglang.srt.entrypoints.engine import Engine

        for dp_size in (1, 8):
            sources = [source(0)] if dp_size == 1 else [source(4), source(5)]
            infos = [{"status": "ready", LOCAL_KV_EVENT_SOURCES: sources}]
            with ExitStack() as stack:
                stack.enter_context(
                    get_context().override_server_args(
                        dp_size=dp_size, tp_size=1, pp_size=1, nnodes=1, node_rank=0
                    )
                )
                stack.enter_context(patch("sglang.srt.entrypoints.engine.mp.Process"))
                stack.enter_context(
                    patch(
                        "sglang.srt.entrypoints.engine.mp.Pipe",
                        return_value=(MagicMock(), MagicMock()),
                    )
                )
                stack.enter_context(
                    patch(
                        "sglang.srt.entrypoints.engine.TorchMemorySaverAdapter.create"
                    )
                )
                stack.enter_context(
                    patch(
                        "sglang.srt.entrypoints.engine.numa_utils.configure_subprocess"
                    )
                )
                stack.enter_context(
                    patch(
                        "sglang.srt.entrypoints.engine.maybe_reindex_device_id",
                        return_value=nullcontext(0),
                    )
                )
                stack.enter_context(
                    patch(
                        "sglang.srt.entrypoints.engine._wait_for_scheduler_ready",
                        return_value=infos,
                    )
                )
                result, _ = Engine._launch_scheduler_processes(
                    MagicMock(), MagicMock(), MagicMock()
                )
                result.wait_for_ready()
            self.assertEqual(result.local_kv_event_sources, sources)
            self.assertEqual(result.scheduler_infos, [{"status": "ready"}])

    def test_extracts_all_local_sources_without_leaking_into_server_info(self):
        # Same helper handles direct scheduler replies and DPC ready replies.
        infos = [
            {"status": "ready", LOCAL_KV_EVENT_SOURCES: [source(5)]},
            {"status": "ready"},
            {"status": "ready", LOCAL_KV_EVENT_SOURCES: [source(4)]},
        ]
        local = take_local_kv_event_sources(infos)
        self.assertEqual(local, [source(4), source(5)])
        self.assertEqual(infos, [{"status": "ready"}] * 3)
        controller_info = {"status": "ready", LOCAL_KV_EVENT_SOURCES: local}
        self.assertEqual(take_local_kv_event_sources([controller_info]), local)
        self.assertEqual(controller_info, {"status": "ready"})

    def test_conflicting_sources_fail_startup(self):
        for conflicting in (
            [
                source(4),
                dataclasses.replace(source(4), endpoint="tcp://localhost:7000"),
            ],
            [source(4), dataclasses.replace(source(4), dp_rank=5)],
        ):
            with self.subTest(conflicting=conflicting), self.assertRaises(ValueError):
                take_local_kv_event_sources([{LOCAL_KV_EVENT_SOURCES: conflicting}])

    def test_leader_and_follower_get_same_group_and_only_local_sources(self):
        contexts = []
        for node_rank, ranks in ((0, [0, 1]), (1, [2, 3])):
            with get_context().override_server_args(
                sidecar_scope="local-telemetry",
                node_rank=node_rank,
                nnodes=2,
                dp_size=4,
                dist_init_addr="leader.example:5000",
            ):
                contexts.append(build_sidecar_context([source(rank) for rank in ranks]))
        self.assertEqual(contexts[0].mode, "full")
        self.assertEqual(contexts[1].mode, "telemetry")
        self.assertEqual(contexts[0].dist_init_addr, "tcp://leader.example:5000")
        self.assertEqual(contexts[0].dist_init_addr, contexts[1].dist_init_addr)
        self.assertEqual(contexts[1].kv_event_sources, [source(2), source(3)])
        self.assertEqual(contexts[1].version, 1)

    def test_legacy_mode_does_not_supply_new_context(self):
        with get_context().override_server_args(sidecar_scope="leader"):
            self.assertIsNone(build_sidecar_context([source(0)]))

    def test_out_of_range_source_fails(self):
        with get_context().override_server_args(
            sidecar_scope="local-telemetry", dp_size=4
        ):
            with self.assertRaisesRegex(ValueError, "outside the DP topology"):
                build_sidecar_context([source(4)])


class TestLocalSidecar(unittest.TestCase):
    def test_spawned_follower_consumes_actual_local_publisher(self):
        probe = zmq.Context.instance().socket(zmq.PUB)
        port = probe.bind_to_random_port("tcp://127.0.0.1")
        probe.close(linger=0)
        publisher = ZmqEventPublisher(
            attn_dp_rank=4, endpoint=f"tcp://*:{port - 4}", topic="kv"
        )
        atexit.unregister(publisher.shutdown)
        self.addCleanup(publisher.shutdown)
        report = zmq.Context.instance().socket(zmq.PULL)
        report.bind("tcp://127.0.0.1:0")
        self.addCleanup(report.close, linger=0)
        context = SidecarContext(
            "telemetry",
            1,
            2,
            8,
            "tcp://leader:5000",
            [publisher.describe_local_source(64)],
        )
        with get_context().override_server_args(
            sidecar="test_local_sidecar",
            sidecar_args=[report.getsockopt_string(zmq.LAST_ENDPOINT)],
        ):
            sidecar = start_sidecar(context)
        self.addCleanup(sidecar.stop)
        # Includes spawned interpreter/provider initialization, without a handshake.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            publisher.publish(KVEventBatch(ts=time.time(), events=[AllBlocksCleared()]))
            if report.poll(100):
                topic, sequence, payload = report.recv_multipart()
                self.assertEqual(topic, b"kv")
                self.assertEqual(len(sequence), 8)
                batch = msgspec.msgpack.decode(payload, type=KVEventBatch)
                self.assertEqual(batch.attn_dp_rank, 4)
                self.assertEqual(batch.events, [AllBlocksCleared()])
                break
        else:
            self.fail("Follower sidecar did not forward its local DP-rank events")
        sidecar.stop()
        self.assertFalse(sidecar.proc.is_alive())

    def test_follower_context_is_installed_before_provider_import(self):
        context = SidecarContext("telemetry", 1, 2, 8, "tcp://leader:5000", [source(4)])
        provider_main = MagicMock()

        def import_provider(name):
            self.assertNotIn(SGLANG_GRPC_ENDPOINT_ENV, os.environ)
            self.assertEqual(
                json.loads(os.environ[SGLANG_SIDECAR_CONTEXT_ENV]),
                dataclasses.asdict(context),
            )
            return SimpleNamespace(main=provider_main)

        with (
            patch.dict(os.environ, {SGLANG_GRPC_ENDPOINT_ENV: "http://stale:1"}),
            patch("sglang.srt.entrypoints.sidecar.kill_itself_when_parent_died"),
            patch(
                "sglang.srt.entrypoints.sidecar.importlib.import_module",
                import_provider,
            ),
        ):
            _run_sidecar("provider", [], None, context)
        provider_main.assert_called_once_with([])

    def test_follower_process_receives_context_without_grpc_endpoint(self):
        context = SidecarContext("telemetry", 1, 2, 8, "tcp://leader:5000", [source(4)])
        with (
            get_context().override_server_args(sidecar="provider"),
            patch("sglang.srt.entrypoints.sidecar.mp.get_context") as mp_context,
            patch("sglang.srt.entrypoints.sidecar.Sidecar") as sidecar_class,
        ):
            start_sidecar(context)
        self.assertEqual(
            mp_context.return_value.Process.call_args.kwargs["args"],
            ("provider", [], None, context),
        )
        self.assertFalse(sidecar_class.call_args.kwargs["allow_clean_exit"])
        mp_context.return_value.Pipe.assert_not_called()
        sidecar_class.return_value.start.assert_called_once_with()

    def test_unexpected_clean_exit_is_fatal(self):
        proc = MagicMock(pid=1234, exitcode=0)
        proc.is_alive.return_value = False
        sidecar = Sidecar(proc, "provider", 1, allow_clean_exit=False)
        with patch("sglang.srt.utils.watchdog.os.kill") as kill:
            self.assertTrue(sidecar._watchdog._check_processes())
        kill.assert_called_once()


class TestFollowerSidecarLifecycle(unittest.TestCase):
    def launch(
        self,
        *,
        sources,
        blocking=True,
        failure=None,
        start_failure=None,
        scope="local-telemetry",
    ):
        from sglang.srt.entrypoints.engine import Engine, SchedulerInitResult

        events = []
        result = SchedulerInitResult(
            scheduler_infos=[{"status": "ready"}],
            local_kv_event_sources=sources,
            wait_for_ready=lambda: events.append("scheduler-ready"),
        )
        sidecar = MagicMock()
        sidecar.stop.side_effect = lambda: events.append("sidecar-stopped")

        def start(context):
            events.append("sidecar-started")
            self.assertEqual(events, ["scheduler-ready", "sidecar-started"])
            if start_failure:
                raise start_failure
            return sidecar

        def block():
            events.append("blocking")
            if failure:
                raise failure

        result.block_until_scheduler_exits = block
        proc = MagicMock(pid=12345)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(
            get_context().override_server_args(
                sidecar="provider",
                sidecar_scope=scope,
                node_rank=1,
                nnodes=2,
                dp_size=8,
                dist_init_addr="leader:5000",
            )
        )
        stack.enter_context(
            patch.dict(
                os.environ,
                {"SGLANG_BLOCK_NONZERO_RANK_CHILDREN": "1" if blocking else "0"},
            )
        )
        for name in (
            "configure_logger",
            "_set_envs_and_config",
            "load_plugins",
            "publish",
        ):
            stack.enter_context(patch(f"sglang.srt.entrypoints.engine.{name}"))
        stack.enter_context(
            patch(
                "sglang.srt.entrypoints.engine.resolving_view",
                return_value=SimpleNamespace(
                    reasoning_parser=None, tool_call_parser=None
                ),
            )
        )
        stack.enter_context(
            patch.object(
                Engine, "_launch_scheduler_processes", return_value=(result, [proc])
            )
        )
        stack.enter_context(
            patch("sglang.srt.entrypoints.sidecar.start_sidecar", side_effect=start)
        )
        health = stack.enter_context(
            patch(
                "sglang.srt.entrypoints.engine.launch_dummy_health_check_server",
                side_effect=lambda *args: events.append("health"),
            )
        )
        kill = stack.enter_context(
            patch("sglang.srt.entrypoints.engine.kill_process_tree")
        )
        launch = lambda: Engine._launch_subprocesses(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), port_args=MagicMock()
        )
        return launch, result, sidecar, events, health, kill

    def test_blocking_follower_starts_sidecar_and_stops_on_exit(self):
        launch, _, sidecar, events, _, _ = self.launch(sources=[source(4)])
        launch()
        self.assertEqual(
            events,
            [
                "scheduler-ready",
                "sidecar-started",
                "health",
                "blocking",
                "sidecar-stopped",
            ],
        )
        sidecar.stop.assert_called_once_with()

    def test_blocking_failure_stops_sidecar(self):
        launch, _, sidecar, _, _, _ = self.launch(
            sources=[source(4)], failure=RuntimeError("scheduler stopped")
        )
        with self.assertRaisesRegex(RuntimeError, "scheduler stopped"):
            launch()
        sidecar.stop.assert_called_once_with()

    def test_source_free_and_legacy_followers_do_not_start_sidecar(self):
        for sources, scope in (([], "local-telemetry"), ([source(4)], "leader")):
            with self.subTest(scope=scope):
                launch, _, sidecar, events, _, _ = self.launch(
                    sources=sources, scope=scope
                )
                launch()
                self.assertNotIn("sidecar-started", events)
                sidecar.stop.assert_not_called()

    def test_nonblocking_follower_transfers_ownership_to_engine(self):
        from sglang.srt.entrypoints.engine import Engine

        launch, result, sidecar, events, health, _ = self.launch(
            sources=[source(4)], blocking=False
        )
        returned = launch()
        self.assertIs(returned[3], result)
        self.assertIs(result.sidecar, sidecar)
        self.assertEqual(events, ["scheduler-ready", "sidecar-started"])
        health.assert_not_called()
        engine = Engine.__new__(Engine)
        engine.tokenizer_manager = None
        engine._scheduler_init_result = result
        engine.shutdown()
        self.assertIsNone(result.sidecar)
        sidecar.stop.assert_called_once_with()

    def test_sidecar_start_failure_reaps_schedulers_before_returning(self):
        launch, _, _, _, health, kill = self.launch(
            sources=[source(4)], start_failure=OSError("cannot start provider")
        )
        with self.assertRaisesRegex(OSError, "cannot start provider"):
            launch()
        health.assert_not_called()
        kill.assert_called_once_with(12345, wait_timeout=60)


if __name__ == "__main__":
    unittest.main()
