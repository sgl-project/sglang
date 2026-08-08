"""E2E: standalone SMG RPC (``--smg-grpc-mode``) load reporter.

Standalone SMG RPC bypasses FastAPI: SGLang imports ``smg-grpc-servicer`` and,
in the ``on_request_manager_ready`` callback, starts the SAME reporter runtime
+ ``grpc.aio`` service on ``--load-reporter-port`` and applies the SAME
``enable_load_monitor("request_lifecycle")`` decorator to the current
``GrpcRequestManager.generate_request`` bound method. A real ``grpc.aio`` fake
Router dials into the reporter port (distinct from the SMG inference port), and
the external SMG inference stub drives a real generation request. The next
deadline report must contain a snapshot collected by the request-end wake,
rather than one collected by ordinary interval sampling at that deadline.

Requires a GPU + model + ``smg-grpc-servicer`` + the load-reporter extra (CUDA
CI); it cannot run on a CPU-only host without those packages.
"""

from __future__ import annotations

import asyncio
import socket
import subprocess
import threading
import time
import uuid
from typing import Any, AsyncIterator, List, Optional

import grpc
import grpc.aio
from smg_grpc_proto import sglang_scheduler_pb2, sglang_scheduler_pb2_grpc

from sglang.srt.load_reporter.proto import load_monitor_pb2 as pb
from sglang.srt.load_reporter.proto import load_monitor_pb2_grpc as pb_grpc
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import get_free_port
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    CustomTestCase,
)

register_cuda_ci(est_time=200, stage="base-b", runner_config="1-gpu-small")


def port_open(host: str, port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        return probe.connect_ex((host, port)) == 0
    finally:
        probe.close()


class FakeRouterClient:
    """Real grpc.aio client dialing INTO the Worker reporter port."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        router_id: str = "e2e-standalone-router",
        interval_ms: int = 60_000,  # long: prove initial report is not timer-driven
        lease_ttl_ms: int = 120_000,
    ) -> None:
        self._host = host
        self._port = port
        self._router_id = router_id
        self._interval_ms = interval_ms
        self._lease_ttl_ms = lease_ttl_ms
        self._reports: List[Any] = []
        self._lock = threading.Lock()
        self._registered = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._stop_evt: Optional[asyncio.Event] = None

    def start(self) -> None:
        self._thread.start()
        self._fut = asyncio.run_coroutine_threadsafe(self._run(), self._loop)

    async def _frames(self) -> AsyncIterator[pb.RouterFrame]:
        yield pb.RouterFrame(
            register=pb.RegisterRequest(
                router_id=self._router_id,
                report_interval_ms=self._interval_ms,
                lease_ttl_ms=self._lease_ttl_ms,
            )
        )
        while not self._stop_evt.is_set():
            await asyncio.sleep(1.0)
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

    def reports_snapshot(self) -> tuple:
        with self._lock:
            return tuple(self._reports)

    def wait_for_reports(self, n: int, timeout: float = 10.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.report_count() >= n:
                return True
            time.sleep(0.05)
        return self.report_count() >= n

    def stop(self) -> None:
        if self._stop_evt is not None:
            self._loop.call_soon_threadsafe(self._stop_evt.set)
        self._loop.call_soon_threadsafe(self._loop.stop)


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
        # Wait for the reporter listener (started in the ready callback, before
        # the gRPC server accepts requests) to come up.
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

    def test_real_inference_refreshes_next_deadline_report(self) -> None:
        self.assertTrue(
            self.reporter_up, "standalone reporter port never started listening"
        )
        # Reporter port is distinct from the SMG inference port and sidecar.
        self.assertNotEqual(self.reporter_port, self.smg_port)
        self.assertNotEqual(self.reporter_port, self.sidecar_port)

        report_interval_ms = 10_000
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
                "no initial report after register under a long interval",
            )

            inference_started_ms = int(time.time() * 1000)
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
            inference_finished_ms = int(time.time() * 1000)

            self.assertTrue(
                router.wait_for_reports(2, timeout=report_interval_ms / 1000 + 3),
                "no deadline report after real standalone inference",
            )
            report = router.reports_snapshot()[1]
            self.assertTrue(report.ranks, "post-inference report has no rank snapshot")
            snapshot_time_ms = max(rank.snapshot_time_unix_ms for rank in report.ranks)
            self.assertGreaterEqual(snapshot_time_ms, inference_started_ms)
            self.assertLessEqual(snapshot_time_ms, inference_finished_ms + 2_000)
        finally:
            router.stop()


if __name__ == "__main__":
    import unittest

    unittest.main()
