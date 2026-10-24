import multiprocessing as mp
import os
import pickle
import signal
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.multimodal_gen.runtime import launch_server as ls
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.entrypoints.control_requests import ShutdownReq


def _startup_worker(spec):
    outcome = pickle.loads(spec.server_args._pickle).outcome
    if spec.rank == 11:
        if outcome == "exit":
            os._exit(23)
        if outcome == "error":
            spec.pipe_writer.send({"status": "error", "error": "bad checkpoint"})
        elif outcome == "ready_exit":
            spec.pipe_writer.send({"status": "ready"})
            os._exit(23)
        elif outcome != "eof":
            spec.pipe_writer.send({"status": "ready"})
    elif outcome == "ready":
        time.sleep(0.05)
        spec.pipe_writer.send({"status": "ready"})
    else:
        time.sleep(60)
    spec.pipe_writer.close()
    time.sleep(60)


class _FakeProcess:
    name = "fake-worker"

    def __init__(self, *, exit_on_join: bool = False):
        self.alive = True
        self.exit_on_join = exit_on_join
        self.join_timeouts = []
        self.terminated = False
        self.killed = False

    def join(self, timeout=None):
        self.join_timeouts.append(timeout)
        if self.exit_on_join:
            self.alive = False

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.alive = False


class TestLaunchServerShutdown(unittest.TestCase):
    def test_pool_failure_releases_previously_started_roles(self):
        args = SimpleNamespace(
            host="127.0.0.1",
            enable_metrics=False,
            scheduler_port=15000,
            disagg_dispatch_policy="round_robin",
            disagg_timeout=30,
            get_role_parallelism=lambda role: dict(
                tp_size=None, sp_degree=None, ulysses_degree=None, ring_degree=None
            ),
        )
        for failure in ("later_role", "router"):
            workers = [_FakeProcess() for _ in range(3)]
            router = Mock()
            router.wait_ready.side_effect = RuntimeError("bind failed")
            starts = [[p] for p in workers]
            if failure == "later_role":
                starts[1] = RuntimeError("worker failed")
            with (
                self.subTest(failure=failure),
                patch.object(ls, "configure_logger"),
                patch.object(ls, "_find_available_port", side_effect=lambda port: port),
                patch.object(ls.dataclasses, "fields", return_value=[]),
                patch.object(
                    ls.ServerArgs, "from_kwargs", return_value=SimpleNamespace()
                ),
                patch.object(ls, "_start_scheduler_processes", side_effect=starts),
                patch.object(ls, "DiffusionServer", return_value=router),
            ):
                with self.assertRaises(RuntimeError):
                    ls.launch_pool_disagg_server(args, [[0]], [[1]], [[0]])
                for worker in workers[:1] if failure == "later_role" else workers:
                    self.assertTrue(worker.terminated and worker.killed)
                if failure == "router":
                    router.stop.assert_called_once()

    @unittest.skipUnless(sys.platform == "linux", "uses CPU-only fork and SIGALRM")
    def test_startup_waits_for_all_ranks_and_cleans_up_failures(self):
        context = mp.get_context("fork")

        def timed_out(*_):
            raise TimeoutError("startup did not notice a failed peer")

        for outcome in (
            "ready",
            "exit",
            "error",
            "eof",
            "ready_exit",
            "spawn_error",
            "interrupt",
        ):
            processes, pipes = [], []

            def make_pipe(**kwargs):
                pair = context.Pipe(**kwargs)
                pipes.extend(pair)
                return pair

            def make_process(**kwargs):
                if len(processes) == 1 and outcome == "interrupt":
                    raise KeyboardInterrupt
                process = context.Process(**kwargs)
                if len(processes) == 1 and outcome == "spawn_error":
                    process.start = Mock(side_effect=OSError("cannot spawn"))
                processes.append(process)
                return process

            previous = signal.signal(signal.SIGALRM, timed_out)
            try:
                with (
                    self.subTest(outcome=outcome),
                    patch.object(ls, "bootstrap_scheduler_process", _startup_worker),
                    patch.object(
                        ls.mp,
                        "get_context",
                        return_value=SimpleNamespace(
                            Pipe=make_pipe, Process=make_process
                        ),
                    ),
                ):
                    signal.alarm(5)
                    payload = ls.ServerArgsPayload.capture(
                        SimpleNamespace(outcome=outcome)
                    )
                    if outcome == "ready":
                        self.assertEqual(
                            ls._start_scheduler_processes(
                                payload, [(3, 10), (4, 11)], "test-r"
                            ),
                            processes,
                        )
                        self.assertTrue(all(p.is_alive() for p in processes))
                    else:
                        error = {
                            "spawn_error": OSError,
                            "interrupt": KeyboardInterrupt,
                        }.get(outcome, RuntimeError)
                        with self.assertRaises(error):
                            ls._start_scheduler_processes(
                                payload, [(3, 10), (4, 11)], "test-r"
                            )
                        self.assertFalse(any(p.is_alive() for p in processes))
                    self.assertTrue(all(pipe.closed for pipe in pipes))
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, previous)
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    if process.pid is not None:
                        process.join(5)
                    process.close()

    def test_http_graceful_shutdown_timeout_is_forwarded(self):
        for timeout in (None, 0, 30):
            with self.subTest(timeout=timeout):
                server_args = SimpleNamespace(
                    log_level="info",
                    host="127.0.0.1",
                    port=30000,
                    http_graceful_shutdown_timeout=timeout,
                )
                with (
                    patch.object(ls, "init_diffusion_tracing"),
                    patch.object(ls, "set_global_server_args"),
                    patch.object(ls, "create_app"),
                    patch.object(ls.uvicorn, "run") as run,
                ):
                    ls.launch_http_server_only(server_args)

                self.assertEqual(
                    run.call_args.kwargs["timeout_graceful_shutdown"], timeout
                )

    def test_monolithic_shutdown_requests_scheduler_then_forces_worker(self):
        process = _FakeProcess()
        server_args = SimpleNamespace(disagg_role=RoleType.MONOLITHIC)
        client = Mock()

        with patch.object(ls, "SchedulerClient", return_value=client):
            ls.shutdown_scheduler_processes(server_args, [process])

        client.initialize.assert_called_once_with(server_args)
        request = client.forward.call_args.args[0]
        self.assertIsInstance(request, ShutdownReq)
        self.assertEqual(client.forward.call_args.kwargs, {"timeout_ms": 5000})
        client.close.assert_called_once_with()

        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertAlmostEqual(process.join_timeouts[0], 10, delta=0.1)
        self.assertAlmostEqual(process.join_timeouts[1], 1, delta=0.1)
        self.assertAlmostEqual(process.join_timeouts[2], 1, delta=0.1)

    def test_scheduler_shutdown_error_still_forces_worker(self):
        process = _FakeProcess()
        server_args = SimpleNamespace(disagg_role=RoleType.MONOLITHIC)
        client = Mock()
        client.forward.side_effect = TimeoutError("blocked")

        with patch.object(ls, "SchedulerClient", return_value=client):
            ls.shutdown_scheduler_processes(server_args, [process])

        client.close.assert_called_once_with()
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)

    def test_disagg_role_does_not_send_monolithic_shutdown_request(self):
        process = _FakeProcess(exit_on_join=True)
        server_args = SimpleNamespace(disagg_role=RoleType.ENCODER)

        with patch.object(ls, "SchedulerClient") as scheduler_client:
            ls.shutdown_scheduler_processes(server_args, [process])

        scheduler_client.assert_not_called()
        self.assertFalse(process.terminated)
        self.assertFalse(process.killed)


if __name__ == "__main__":
    unittest.main()
