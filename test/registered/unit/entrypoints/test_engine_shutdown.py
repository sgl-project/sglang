"""Engine.shutdown lets the schedulers release in userspace before reaping them."""

import asyncio
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import zmq
import zmq.asyncio

from sglang.srt.entrypoints import engine as engine_module
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.managers import tokenizer_manager as tokenizer_manager_module
from sglang.srt.managers.io_struct import sock_recv
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Recorder:
    def __init__(self, events: list, name: str):
        self.events = events
        self.name = name

    def stop(self):
        self.events.append(f"stop {self.name}")

    def shutdown(self):
        self.events.append(f"shutdown {self.name}")


class TestEngineShutdown(CustomTestCase):
    def setUp(self):
        self.events = []
        self.pp_size = 1
        self.wait_error = None

        # The tokenizer's real zmq.asyncio PUSH socket and the scheduler's PULL end.
        address = f"ipc://{tempfile.mkdtemp()}/scheduler_input"
        async_context = zmq.asyncio.Context()
        sync_context = zmq.Context()
        self.to_scheduler = async_context.socket(zmq.PUSH)
        self.to_scheduler.bind(address)
        self.scheduler = sync_context.socket(zmq.PULL)
        self.scheduler.connect(address)
        # A blocking send returns once the scheduler's end is connected.
        zmq.Socket.shadow(self.to_scheduler.underlying).send(b"connected")
        self.assertEqual(self.scheduler.recv(), b"connected")
        self.addCleanup(async_context.destroy, linger=0)
        self.addCleanup(sync_context.destroy, linger=0)

        self.manager = object.__new__(TokenizerManager)
        self.manager._subprocess_watchdog = _Recorder(self.events, "watchdog")
        self.manager._shutdown_req_sent = False
        self.manager.tokenizer_ipc_name = None
        self.manager.send_to_scheduler = self.to_scheduler
        self.manager.mm_processor = None
        self.manager.cuda_vmm_feature_transport = _Recorder(self.events, "transport")
        self.engine = object.__new__(Engine)
        self.engine.tokenizer_manager = self.manager
        self.engine._weight_cache_daemon_procs = ["daemon"]
        self.engine._terminate_weight_cache_daemons = lambda procs: self.events.append(
            "terminate daemons"
        )

        for target, name, fake in (
            (
                tokenizer_manager_module,
                "wait_for_schedulers_to_exit",
                self._wait_for_schedulers_to_exit,
            ),
            (
                engine_module,
                "kill_process_tree",
                lambda *args, **kwargs: self.events.append("kill"),
            ),
            (
                engine_module,
                "get_parallel",
                lambda: SimpleNamespace(pp_size=self.pp_size),
            ),
        ):
            patcher = patch.object(target, name, fake)
            patcher.start()
            self.addCleanup(patcher.stop)

    def _wait_for_schedulers_to_exit(self):
        # The schedulers exit on the request they read.
        self.assertTrue(self.scheduler.poll(5000))
        self.events.append(type(sock_recv(self.scheduler)).__name__)
        if self.wait_error is not None:
            raise self.wait_error

    def test_schedulers_get_shutdown_req_before_the_kill(self):
        """Engine.shutdown used to SIGKILL the schedulers without a ShutdownReq,
        so they never released GPU and pinned host memory in userspace."""
        self.engine.shutdown()

        self.assertEqual(
            self.events,
            [
                "stop watchdog",
                "ShutdownReq",
                "terminate daemons",
                "kill",
                "shutdown transport",
            ],
        )

    def test_shutdown_after_asyncio_run_still_sends_the_request(self):
        """asyncio.run() leaves the thread without a current event loop, which
        a send on the zmq.asyncio socket needs."""
        asyncio.run(asyncio.sleep(0))

        self.engine.shutdown()

        self.assertIn("ShutdownReq", self.events)

    def test_repeated_shutdown_sends_nothing(self):
        """atexit calls shutdown() again after an explicit one."""
        self.engine.shutdown()
        self.events.clear()

        self.engine.shutdown()

        self.assertEqual(self.events, ["stop watchdog", "kill", "shutdown transport"])
        self.assertEqual(self.scheduler.poll(100), 0)

    def test_request_does_not_depend_on_local_scheduler_processes(self):
        """Ray scheduler actors are not children of this process."""
        with patch.object(
            tokenizer_manager_module, "collect_scheduler_processes", lambda: []
        ):
            self.manager.shutdown_schedulers()

        self.assertEqual(self.events, ["ShutdownReq"])

    def test_pipeline_parallel_skips_the_request(self):
        """The pipeline-parallel event loops do not honor ShutdownReq yet, so
        sending one would only delay the kill by the grace period."""
        self.pp_size = 2

        self.engine.shutdown()

        self.assertEqual(
            self.events,
            ["stop watchdog", "terminate daemons", "kill", "shutdown transport"],
        )
        self.assertEqual(self.scheduler.poll(100), 0)

    def test_request_waits_for_a_scheduler_still_connecting(self):
        """The schedulers connect before the tokenizer binds its socket, and ZMQ
        retries a connect every 100 ms, so an Engine shut down right after
        startup can find no connected scheduler yet. The request used to fail at
        once there, and the schedulers were killed without releasing."""
        address = f"ipc://{tempfile.mkdtemp()}/bound_late"
        self.scheduler = self.scheduler.context.socket(zmq.PULL)
        self.scheduler.connect(address)
        # The first connect attempt fails: nothing is bound yet.
        time.sleep(0.05)
        self.manager.send_to_scheduler = self.to_scheduler.context.socket(zmq.PUSH)
        self.manager.send_to_scheduler.bind(address)

        self.manager.shutdown_schedulers()

        self.assertEqual(self.events, ["ShutdownReq"])

    def test_unsendable_request_still_reaps_everything(self):
        """No scheduler connects within the grace period."""
        no_peer = self.to_scheduler.context.socket(zmq.PUSH)
        no_peer.bind(f"ipc://{tempfile.mkdtemp()}/no_scheduler")
        self.manager.send_to_scheduler = no_peer

        with patch.object(
            tokenizer_manager_module, "_SCHEDULER_EXIT_TIMEOUT_SECS", 0.1
        ):
            self.engine.shutdown()

        self.assertEqual(
            self.events,
            ["stop watchdog", "terminate daemons", "kill", "shutdown transport"],
        )

    def test_failed_wait_still_reaps_everything(self):
        self.wait_error = RuntimeError("wait failed")

        with self.assertRaisesRegex(RuntimeError, "wait failed"):
            self.engine.shutdown()

        self.assertEqual(
            self.events,
            [
                "stop watchdog",
                "ShutdownReq",
                "terminate daemons",
                "kill",
                "shutdown transport",
            ],
        )

    def test_partial_startup_stops_the_router_watchdog(self):
        """With several tokenizer workers, a startup that fails after the router
        is up leaves Engine.tokenizer_manager pointing at the router; its
        watchdog must stop before the kill, or it reports the killed children."""
        self.engine.tokenizer_manager = SimpleNamespace(
            _subprocess_watchdog=_Recorder(self.events, "router watchdog")
        )

        self.engine.shutdown()

        self.assertEqual(
            self.events, ["stop router watchdog", "terminate daemons", "kill"]
        )


if __name__ == "__main__":
    unittest.main()
