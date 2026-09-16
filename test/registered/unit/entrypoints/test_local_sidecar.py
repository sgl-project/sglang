"""CPU coverage for local publisher metadata and managed sidecar lifecycle."""

import atexit
import dataclasses
import json
import os
import signal
import threading
import time
import unittest
from contextlib import ExitStack, contextmanager, nullcontext
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

        for pp, tp, cp, dp_attention, dcp_size in (
            (0, 0, 0, True, 1),
            (0, 0, 0, False, 1),
            (0, 0, 0, True, 4),
            (1, 0, 0, True, 1),
            (0, 1, 0, True, 1),
            (0, 0, 1, True, 1),
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
            publisher.describe_local_source.side_effect = lambda block_size: (
                dataclasses.replace(source(4), block_size=block_size)
            )
            with (
                self.subTest(ps=ps, dcp_size=dcp_size),
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
                    sidecar_scope="local-telemetry", dcp_size=dcp_size
                ):
                    info = scheduler.get_init_info()
                if pp == tp == cp == 0:
                    create.assert_called_once_with('{"publisher":"zmq"}', 4)
                    publisher.describe_local_source.assert_called_once_with(
                        64 * dcp_size
                    )
                    self.assertEqual(
                        info[LOCAL_KV_EVENT_SOURCES],
                        [dataclasses.replace(source(4), block_size=64 * dcp_size)],
                    )
                else:
                    create.assert_not_called()
                    self.assertEqual(info[LOCAL_KV_EVENT_SOURCES], [])
                with get_context().override_server_args(sidecar_scope="leader"):
                    self.assertNotIn(LOCAL_KV_EVENT_SOURCES, scheduler.get_init_info())

    def test_engine_extracts_sources_from_scheduler_or_controller_reply(self):
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

    def test_dp_controller_collects_and_forwards_local_scheduler_sources(self):
        from sglang.srt.managers import data_parallel_controller as dpc

        # Node 1 owns DP ranks 2 and 3; their CP companions own no publisher.
        infos = [
            dict(
                status="ready",
                max_total_num_tokens=64,
                max_req_input_len=32,
                **{LOCAL_KV_EVENT_SOURCES: sources},
            )
            for sources in ([source(2)], [], [source(3)], [])
        ]
        pipes = [(MagicMock(), MagicMock()) for _ in infos]
        for (reader, _), info in zip(pipes, infos):
            reader.recv.return_value = info
        controller = dpc.DataParallelController.__new__(dpc.DataParallelController)
        controller.env_lock = threading.Lock()
        controller.scheduler_procs = []
        controller.local_kv_event_sources = []
        controller.run_scheduler_process_func = MagicMock()

        def initialize(server_args, port_args, run_scheduler):
            controller.launch_tensor_parallel_group(server_args, port_args, 0, None)
            return controller

        ready = MagicMock()
        with (
            get_context().override_server_args(
                node_rank=1,
                nnodes=2,
                tp_size=8,
                pp_size=1,
                dp_size=4,
                enable_dp_attention=True,
                attn_cp_size=2,
            ),
            patch.object(dpc, "DataParallelController", side_effect=initialize),
            patch.object(dpc.mp, "Pipe", side_effect=pipes),
            patch.object(dpc.mp, "Process"),
            patch.object(dpc.PortArgs, "init_new", return_value=MagicMock()),
            patch.object(dpc.TorchMemorySaverAdapter, "create"),
            patch.object(dpc.numa_utils, "configure_subprocess"),
            patch.object(dpc, "maybe_reindex_device_id", return_value=nullcontext(0)),
            patch.object(dpc, "publish"),
            patch.object(dpc, "configure_logger"),
            patch.object(dpc, "kill_itself_when_parent_died"),
            patch.object(dpc.setproctitle, "setproctitle"),
            patch.object(dpc.psutil, "Process") as parent,
        ):
            dpc.run_data_parallel_controller_process(
                MagicMock(), MagicMock(), ready, controller.run_scheduler_process_func
            )
        parent.return_value.parent.return_value.send_signal.assert_not_called()
        ready.send.assert_called_once()
        self.assertEqual(
            ready.send.call_args.args[0][LOCAL_KV_EVENT_SOURCES],
            [source(2), source(3)],
        )
        self.assertTrue(all(LOCAL_KV_EVENT_SOURCES not in info for info in infos))

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

    def test_conflicting_sources_are_rejected(self):
        for conflicting in (
            [
                source(4),
                dataclasses.replace(source(4), endpoint="tcp://localhost:7000"),
            ],
            [source(4), dataclasses.replace(source(4), dp_rank=5)],
        ):
            with self.subTest(conflicting=conflicting), self.assertRaises(ValueError):
                take_local_kv_event_sources([{LOCAL_KV_EVENT_SOURCES: conflicting}])

    def test_context_selects_mode_and_preserves_supplied_sources(self):
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
            # A provider crash must fail this test, not SIGQUIT the whole runner.
            real_kill = os.kill
            crashed = threading.Event()

            def intercept_kill(pid, sig):
                if pid == os.getpid() and sig == signal.SIGQUIT:
                    crashed.set()
                else:
                    real_kill(pid, sig)

            kill_patch = patch("sglang.srt.utils.watchdog.os.kill", intercept_kill)
            kill_patch.start()
            self.addCleanup(kill_patch.stop)
            sidecar = start_sidecar(context)
        self.addCleanup(sidecar.stop)
        # Includes spawned interpreter/provider initialization, without a handshake.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            self.assertFalse(crashed.is_set(), "Follower provider exited unexpectedly")
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
        self.assertFalse(crashed.is_set(), "Follower provider exited unexpectedly")
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

    def test_unexpected_clean_exit_is_fatal(self):
        proc = MagicMock(pid=1234, exitcode=0)
        proc.is_alive.return_value = False
        sidecar = Sidecar(proc, "provider", 1, allow_clean_exit=False)
        with patch("sglang.srt.utils.watchdog.os.kill") as kill:
            self.assertTrue(sidecar._watchdog._check_processes())
        kill.assert_called_once_with(os.getpid(), signal.SIGQUIT)


class TestFollowerSidecarLifecycle(unittest.TestCase):
    @contextmanager
    def launch(self, *, sources, blocking=True, scope="local-telemetry"):
        from sglang.srt.entrypoints.engine import Engine, SchedulerInitResult

        events = []
        result = SchedulerInitResult(
            scheduler_infos=[{"status": "ready"}],
            local_kv_event_sources=sources,
            wait_for_ready=MagicMock(
                side_effect=lambda: events.append("scheduler-ready")
            ),
            block_until_scheduler_exits=MagicMock(
                side_effect=lambda: events.append("blocking")
            ),
        )
        sidecar = MagicMock()
        sidecar.stop.side_effect = lambda: events.append("sidecar-stopped")

        def start(context):
            events.append("sidecar-started")
            return sidecar

        with (
            get_context().override_server_args(
                sidecar="provider",
                sidecar_scope=scope,
                node_rank=1,
                nnodes=2,
                dp_size=8,
                dist_init_addr="leader:5000",
            ),
            patch.dict(
                os.environ,
                {"SGLANG_BLOCK_NONZERO_RANK_CHILDREN": "1" if blocking else "0"},
            ),
            patch("sglang.srt.entrypoints.engine.configure_logger"),
            patch("sglang.srt.entrypoints.engine._set_envs_and_config"),
            patch("sglang.srt.entrypoints.engine.load_plugins"),
            patch("sglang.srt.entrypoints.engine.publish"),
            patch(
                "sglang.srt.entrypoints.engine.resolving_view",
                return_value=SimpleNamespace(
                    reasoning_parser=None, tool_call_parser=None
                ),
            ),
            patch.object(
                Engine,
                "_launch_scheduler_processes",
                return_value=(result, [MagicMock(pid=12345)]),
            ),
            patch(
                "sglang.srt.entrypoints.sidecar.start_sidecar", side_effect=start
            ) as start_mock,
            patch(
                "sglang.srt.entrypoints.engine.launch_dummy_health_check_server",
                side_effect=lambda *args: events.append("health"),
            ) as health,
            patch("sglang.srt.entrypoints.engine.kill_process_tree") as kill,
        ):
            yield SimpleNamespace(
                run=lambda: Engine._launch_subprocesses(
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    MagicMock(),
                    port_args=MagicMock(),
                ),
                result=result,
                sidecar=sidecar,
                start=start_mock,
                events=events,
                health=health,
                kill=kill,
            )

    def test_blocking_follower_starts_sidecar_and_stops_on_exit(self):
        with self.launch(sources=[source(4)]) as follower:
            follower.run()
        follower.start.assert_called_once_with(
            SidecarContext("telemetry", 1, 2, 8, "tcp://leader:5000", [source(4)])
        )
        self.assertEqual(
            follower.events,
            [
                "scheduler-ready",
                "sidecar-started",
                "health",
                "blocking",
                "sidecar-stopped",
            ],
        )

    def test_blocking_failure_stops_sidecar(self):
        with self.launch(sources=[source(4)]) as follower:
            follower.result.block_until_scheduler_exits.side_effect = RuntimeError(
                "scheduler stopped"
            )
            with self.assertRaisesRegex(RuntimeError, "scheduler stopped"):
                follower.run()
        follower.sidecar.stop.assert_called_once_with()

    def test_source_free_and_legacy_followers_do_not_start_sidecar(self):
        for sources, scope in (([], "local-telemetry"), ([source(4)], "leader")):
            with (
                self.subTest(scope=scope),
                self.launch(sources=sources, scope=scope) as follower,
            ):
                follower.run()
                follower.start.assert_not_called()
                follower.sidecar.stop.assert_not_called()

    def test_nonblocking_follower_transfers_ownership_to_engine(self):
        from sglang.srt.entrypoints.engine import Engine

        with self.launch(sources=[source(4)], blocking=False) as follower:
            returned = follower.run()
            self.assertIs(returned[3], follower.result)
            self.assertIs(follower.result.sidecar, follower.sidecar)
            self.assertEqual(follower.events, ["scheduler-ready", "sidecar-started"])
            follower.health.assert_not_called()
            engine = Engine.__new__(Engine)
            engine.tokenizer_manager = None
            engine._scheduler_init_result = follower.result
            engine.shutdown()
        self.assertIsNone(follower.result.sidecar)
        follower.sidecar.stop.assert_called_once_with()

    def test_sidecar_start_failure_reaps_schedulers_before_returning(self):
        with self.launch(sources=[source(4)]) as follower:
            follower.start.side_effect = OSError("cannot start provider")
            with self.assertRaisesRegex(OSError, "cannot start provider"):
                follower.run()
        follower.health.assert_not_called()
        follower.kill.assert_called_once_with(12345, wait_timeout=60)


if __name__ == "__main__":
    unittest.main()
