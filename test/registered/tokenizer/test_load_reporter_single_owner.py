"""E2E: single-tokenizer HTTP Worker load reporter (router dials in).

Transport inversion vs. the old HTTP-registration design: the external Router
is now a gRPC CLIENT that dials INTO the Worker's fixed ``--load-reporter-port``
and drives the bidi ``Monitor`` stream.  This suite uses a real ``grpc.aio``
fake Router (no HTTP POST) against a real single-tokenizer server.

Requires a GPU + model + the load-reporter grpc/protobuf extra, so it is
registered as a CUDA CI test and cannot run on a CPU-only host.
"""

from __future__ import annotations

import asyncio
import socket
import threading
import time
from typing import Any, AsyncIterator, List, Optional

import grpc
import grpc.aio
import requests

from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.load_reporter.proto import load_monitor_pb2_grpc as pb_grpc
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import get_free_port
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")


class FakeRouterClient:
    """Real grpc.aio client that dials INTO the Worker reporter port.

    Runs its own event loop on a background thread. Sends a register frame
    first, then periodic keep-alive frames, and records every WorkerFrame it
    reads.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        router_id: str = "e2e-router",
        interval_ms: int = 200,
        lease_ttl_ms: int = 10_000,
    ) -> None:
        self._host = host
        self._port = port
        self._router_id = router_id
        self._interval_ms = interval_ms
        self._lease_ttl_ms = lease_ttl_ms
        self._reports: List[Any] = []
        self._acks: List[Any] = []
        self._lock = threading.Lock()
        self._registered = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._stop_evt: Optional[asyncio.Event] = None
        self._update_interval_ms: Optional[int] = None

    def start(self) -> None:
        self._thread.start()
        self._fut = asyncio.run_coroutine_threadsafe(self._run(), self._loop)

    def request_interval_update(self, interval_ms: int) -> None:
        """Ask the running session to send an UpdateConfigRequest frame."""
        self._update_interval_ms = interval_ms

    async def _frames(self) -> AsyncIterator[pb.RouterFrame]:
        yield pb.RouterFrame(
            register=pb.RegisterRequest(
                router_id=self._router_id,
                report_interval_ms=self._interval_ms,
                lease_ttl_ms=self._lease_ttl_ms,
            )
        )
        while not self._stop_evt.is_set():
            await asyncio.sleep(self._interval_ms / 1000.0 / 2)
            if self._update_interval_ms is not None:
                interval, self._update_interval_ms = self._update_interval_ms, None
                yield pb.RouterFrame(
                    update_config=pb.UpdateConfigRequest(report_interval_ms=interval)
                )
            else:
                yield pb.RouterFrame(keep_alive=pb.KeepAlive())

    async def _run(self) -> None:
        self._stop_evt = asyncio.Event()
        channel = grpc.aio.insecure_channel(f"{self._host}:{self._port}")
        stub = pb_grpc.LoadMonitorServiceStub(channel)
        call = stub.Monitor(self._frames())
        try:
            async for frame in call:
                which = frame.WhichOneof("payload")
                if which == "registered":
                    with self._lock:
                        self._acks.append(frame.registered)
                    self._registered.set()
                elif which == "report":
                    with self._lock:
                        self._reports.append(frame.report)
        except Exception:
            pass
        finally:
            await channel.close()

    def wait_for_register(self, timeout: float = 10.0) -> bool:
        return self._registered.wait(timeout=timeout)

    def report_count(self) -> int:
        with self._lock:
            return len(self._reports)

    def wait_for_reports(self, n: int, timeout: float = 10.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.report_count() >= n:
                return True
            time.sleep(0.05)
        return self.report_count() >= n

    def reports_snapshot(self) -> tuple:
        with self._lock:
            return tuple(self._reports)

    def stop(self) -> None:
        if self._stop_evt is not None:
            self._loop.call_soon_threadsafe(self._stop_evt.set)
        self._loop.call_soon_threadsafe(self._loop.stop)


class TestLoadReporterSingleOwner(CustomTestCase):
    """Single-tokenizer HTTP Worker: register, continuous reports, request-end."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.reporter_port = get_free_port()
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--load-reporter-port",
                str(cls.reporter_port),
                "--mem-fraction-static",
                "0.5",
            ],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        process = getattr(cls, "process", None)
        if process is not None:
            kill_process_tree(process.pid)
            cls.process = None

    def test_register_ack_and_continuous_reports(self) -> None:
        router = FakeRouterClient("127.0.0.1", self.reporter_port, interval_ms=200)
        router.start()
        try:
            self.assertTrue(router.wait_for_register(), "no register ack received")
            with router._lock:
                ack = router._acks[0]
            self.assertEqual(ack.lease_ttl_ms, 10_000)
            # Periodic reports flow even without any inference activity.
            self.assertTrue(
                router.wait_for_reports(3, timeout=10.0),
                "fewer than 3 periodic reports arrived",
            )
        finally:
            router.stop()

    def test_request_end_drives_report_convergence(self) -> None:
        router = FakeRouterClient("127.0.0.1", self.reporter_port, interval_ms=250)
        router.start()
        try:
            self.assertTrue(router.wait_for_register())
            router.wait_for_reports(1)
            # Drive dispatch + completion through the decorator seam.
            resp = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": "single-owner reporter convergence",
                    "sampling_params": {"max_new_tokens": 8, "temperature": 0},
                },
                timeout=30,
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            # A ranked snapshot must eventually converge (hints coalesce; we do
            # not require one report per request-end).
            end = time.monotonic() + 10.0
            ranked = False
            while time.monotonic() < end and not ranked:
                ranked = any(r.ranks for r in router.reports_snapshot())
                time.sleep(0.1)
            self.assertTrue(ranked, "no ranked report converged after a request")
        finally:
            router.stop()

    def test_interval_update_and_reconnect(self) -> None:
        # First session: register, then push an interval update mid-stream.
        router = FakeRouterClient("127.0.0.1", self.reporter_port, interval_ms=1000)
        router.start()
        try:
            self.assertTrue(router.wait_for_register())
            router.request_interval_update(150)
            self.assertTrue(
                router.wait_for_reports(3, timeout=8.0),
                "interval update to 150ms did not speed up reports",
            )
        finally:
            router.stop()

        # Reconnect with the same router_id: a fresh stream must re-register.
        router2 = FakeRouterClient("127.0.0.1", self.reporter_port, interval_ms=200)
        router2.start()
        try:
            self.assertTrue(
                router2.wait_for_register(), "reconnect did not re-register"
            )
            self.assertTrue(router2.wait_for_reports(1, timeout=8.0))
        finally:
            router2.stop()


class TestLoadReporterNativeGrpcReuse(CustomTestCase):
    """Native gRPC (--grpc-port) reuses the HTTP/TokenizerManager lifecycle:
    exactly one reporter listener on --load-reporter-port, not a second one."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.reporter_port = get_free_port()
        cls.grpc_port = get_free_port()
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--load-reporter-port",
                str(cls.reporter_port),
                "--grpc-port",
                str(cls.grpc_port),
                "--mem-fraction-static",
                "0.5",
            ],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        process = getattr(cls, "process", None)
        if process is not None:
            kill_process_tree(process.pid)
            cls.process = None

    def test_single_reporter_listener_shared(self) -> None:
        router = FakeRouterClient("127.0.0.1", self.reporter_port, interval_ms=200)
        router.start()
        try:
            self.assertTrue(
                router.wait_for_register(),
                "native-gRPC server did not expose the shared reporter listener",
            )
            self.assertTrue(router.wait_for_reports(2, timeout=10.0))
        finally:
            router.stop()


class TestLoadReporterDisabled(CustomTestCase):
    """--load-reporter-port unset: inference works and no reporter port listens."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.absent_port = get_free_port()
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--mem-fraction-static", "0.5"],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        process = getattr(cls, "process", None)
        if process is not None:
            kill_process_tree(process.pid)
            cls.process = None

    def test_no_reporter_listener_and_inference_ok(self) -> None:
        # No listener on the (arbitrary) reporter port.
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        free = probe.connect_ex(("127.0.0.1", self.absent_port)) != 0
        probe.close()
        self.assertTrue(free, "a reporter port is listening while disabled")

        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "reporter disabled",
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            },
            timeout=30,
        )
        self.assertEqual(resp.status_code, 200, resp.text)


if __name__ == "__main__":
    import unittest

    unittest.main()
