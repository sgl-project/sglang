"""CUDA end-to-end tests for load-reporter server integrations."""

from __future__ import annotations

import asyncio
import socket
import subprocess
import threading
import time
import unittest
import uuid
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

register_cuda_ci(est_time=580, stage="base-b", runner_config="1-gpu-small")

# ============================================================================
# HTTP and native-gRPC ownership
# ============================================================================


class FakeRouterClient:
    """Real grpc.aio client dialing INTO the Worker reporter port; runs its own loop on a background thread."""

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

    def wait_for_ranked_report(self, timeout: float = 12.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if any(report.ranks for report in self.reports_snapshot()):
                return True
            time.sleep(0.1)
        return False

    def stop(self) -> None:
        if self._stop_evt is not None:
            self._loop.call_soon_threadsafe(self._stop_evt.set)
        self._loop.call_soon_threadsafe(self._loop.stop)


class TestLoadReporterSingleOwner(CustomTestCase):
    """Single-tokenizer HTTP Worker: register and periodic reporting."""

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

    def test_inference_and_periodic_reporting_coexist(self) -> None:
        router = FakeRouterClient("127.0.0.1", self.reporter_port, interval_ms=250)
        router.start()
        try:
            self.assertTrue(router.wait_for_register())
            self.assertTrue(router.wait_for_reports(1), "no initial report received")
            resp = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": "single-owner periodic reporter",
                    "sampling_params": {"max_new_tokens": 8, "temperature": 0},
                },
                timeout=30,
            )
            self.assertEqual(resp.status_code, 200, resp.text)
            reports_after_inference = router.report_count()
            self.assertTrue(
                router.wait_for_reports(reports_after_inference + 1, timeout=10.0),
                "periodic reporting stopped after inference",
            )
            self.assertTrue(
                any(report.ranks for report in router.reports_snapshot()),
                "periodic reports never observed a scheduler snapshot",
            )
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
    """Native gRPC reuses the HTTP lifecycle: exactly one reporter listener on --load-reporter-port."""

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
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            reporter_is_disabled = (
                probe.connect_ex(("127.0.0.1", self.absent_port)) != 0
            )
        finally:
            probe.close()
        self.assertTrue(
            reporter_is_disabled, "a reporter port is listening while disabled"
        )

        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "reporter disabled",
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            },
            timeout=30,
        )
        self.assertEqual(resp.status_code, 200, resp.text)


# ============================================================================
# Multi-tokenizer ownership
# ============================================================================


class TestLoadReporterMultiOwner(CustomTestCase):
    """Two tokenizer workers, one router-owned reporter listener."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.reporter_port = get_free_port()
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tokenizer-worker-num",
                "2",
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

    def test_sole_owner_periodic_reports_continue_through_inference(self) -> None:
        # If any HTTP worker had also bound the port, launch would have failed; a working stream proves sole ownership.
        router = FakeRouterClient("127.0.0.1", self.reporter_port, interval_ms=250)
        router.start()
        try:
            self.assertTrue(
                router.wait_for_register(),
                "router-owned reporter listener never accepted the stream",
            )
            self.assertTrue(
                router.wait_for_reports(2),
                "router-owned fire loop produced fewer than 2 reports",
            )
            # Spread requests across the 2 HTTP workers; activity is independent of the router-owned fire loop.
            for i in range(6):
                resp = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "text": f"multi-owner reporter {i}",
                        "sampling_params": {"max_new_tokens": 8, "temperature": 0},
                    },
                    timeout=30,
                )
                self.assertEqual(resp.status_code, 200, resp.text)
            reports_after_inference = router.report_count()
            self.assertTrue(
                router.wait_for_reports(reports_after_inference + 1),
                "periodic reporting stopped after multi-worker inference",
            )
            self.assertTrue(
                router.wait_for_ranked_report(),
                "shared fire loop produced no ranked report",
            )
        finally:
            router.stop()


# ============================================================================
# Standalone SMG gRPC
# ============================================================================


def port_open(host: str, port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        return probe.connect_ex((host, port)) == 0
    finally:
        probe.close()


class TestLoadReporterStandaloneGrpc(CustomTestCase):
    """Standalone SMG RPC exposes the reporter on its own port, no FastAPI."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.host = "127.0.0.1"
        cls.smg_port = get_free_port()
        cls.sidecar_port = get_free_port()
        cls.reporter_port = get_free_port()
        cls.process = subprocess.Popen(
            [
                "python3",
                "-m",
                "sglang.launch_server",
                "--model-path",
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                "--host",
                cls.host,
                "--port",
                str(cls.smg_port),
                "--smg-grpc-mode",
                "--smg-http-sidecar-port",
                str(cls.sidecar_port),
                "--load-reporter-port",
                str(cls.reporter_port),
                "--mem-fraction-static",
                "0.5",
            ]
        )
        # Wait for the reporter listener, started in the ready callback before the gRPC server accepts requests.
        deadline = time.monotonic() + DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        cls.reporter_up = False
        while time.monotonic() < deadline:
            if cls.process.poll() is not None:
                break
            if port_open(cls.host, cls.reporter_port):
                cls.reporter_up = True
                break
            time.sleep(1.0)

    @classmethod
    def tearDownClass(cls) -> None:
        process = getattr(cls, "process", None)
        if process is not None:
            kill_process_tree(process.pid)
            cls.process = None

    def test_inference_and_reporting_coexist(self) -> None:
        """Inference and periodic reporting run independently — requests never trigger a snapshot pull."""
        from smg_grpc_proto import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc

        self.assertTrue(
            self.reporter_up, "standalone reporter port never started listening"
        )
        # Reporter port is distinct from the SMG inference port and sidecar.
        self.assertNotEqual(self.reporter_port, self.smg_port)
        self.assertNotEqual(self.reporter_port, self.sidecar_port)

        report_interval_ms = 500
        router = FakeRouterClient(
            self.host,
            self.reporter_port,
            interval_ms=report_interval_ms,
            lease_ttl_ms=30_000,
        )
        router.start()
        try:
            self.assertTrue(
                router.wait_for_register(),
                "standalone reporter did not accept the register stream",
            )
            self.assertTrue(
                router.wait_for_reports(1, timeout=8.0),
                "no initial report after register",
            )

            # Run a real inference request — must succeed without affecting the reporter stream.
            channel = grpc.insecure_channel(f"{self.host}:{self.smg_port}")
            try:
                stub = sglang_scheduler_pb2_grpc.SglangSchedulerStub(channel)
                request = sglang_scheduler_pb2.GenerateRequest(
                    request_id=f"load-reporter-e2e-{uuid.uuid4().hex}",
                    tokenized=sglang_scheduler_pb2.TokenizedInput(
                        input_ids=[123, 456, 789, 234],
                        original_text="load reporter e2e",
                    ),
                    sampling_params=sglang_scheduler_pb2.SamplingParams(
                        temperature=0.0,
                        max_new_tokens=1,
                    ),
                    stream=False,
                )
                responses = list(stub.Generate(request, timeout=60))
            finally:
                channel.close()
            self.assertTrue(responses and responses[-1].HasField("complete"))

            # Periodic reports must continue arriving after inference.
            reports_after_inference = router.report_count()
            self.assertTrue(
                router.wait_for_reports(
                    reports_after_inference + 1,
                    timeout=report_interval_ms / 1000 + 3,
                ),
                "no deadline report after real standalone inference",
            )
            report = router.reports_snapshot()[-1]
            self.assertTrue(report.ranks, "post-inference report has no rank snapshot")
        finally:
            router.stop()


if __name__ == "__main__":
    unittest.main()
