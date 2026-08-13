"""E2E: multi-tokenizer ownership — only the sole MultiTokenizerRouter binds --load-reporter-port."""

from __future__ import annotations

import asyncio
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

register_cuda_ci(est_time=200, stage="base-b", runner_config="1-gpu-small")


class FakeRouterClient:
    """Real grpc.aio client dialing INTO the Worker reporter port."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        router_id: str = "e2e-multi-router",
        interval_ms: int = 200,
        lease_ttl_ms: int = 10_000,
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
            await asyncio.sleep(self._interval_ms / 1000.0 / 2)
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

    def reports_snapshot(self) -> tuple:
        with self._lock:
            return tuple(self._reports)

    def report_count(self) -> int:
        with self._lock:
            return len(self._reports)

    def wait_for_reports(self, n: int, timeout: float = 12.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.report_count() >= n:
                return True
            time.sleep(0.05)
        return self.report_count() >= n

    def wait_for_ranked_report(self, timeout: float = 12.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if any(r.ranks for r in self.reports_snapshot()):
                return True
            time.sleep(0.1)
        return False

    def stop(self) -> None:
        if self._stop_evt is not None:
            self._loop.call_soon_threadsafe(self._stop_evt.set)
        self._loop.call_soon_threadsafe(self._loop.stop)


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


if __name__ == "__main__":
    import unittest

    unittest.main()
