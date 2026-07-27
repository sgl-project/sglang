"""GPU E2E proof for the multi-tokenizer load-reporter ownership boundary."""

from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, Optional

import grpc
import requests
from google.protobuf import empty_pb2

from sglang.srt.load_reporter.proto import load_monitor_pb2_grpc as pb_grpc
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-small")


class FakeLoadReporterRouter(pb_grpc.LoadMonitorServiceServicer):
    """Minimal gRPC Router that counts streams and retains received reports."""

    def __init__(self) -> None:
        """Initialize an unbound fake Router.

        Returns:
            None.
        """
        self.port = 0
        self._stream_count = 0
        self._reports: list[Any] = []
        self._server: Optional[grpc.Server] = None
        self._condition = threading.Condition()

    @property
    def stream_count(self) -> int:
        """Return the number of client streams opened so far.

        Returns:
            The total stream count.
        """
        with self._condition:
            return self._stream_count

    def reports_snapshot(self) -> tuple[Any, ...]:
        """Return a thread-safe immutable snapshot of received reports.

        Returns:
            All reports received so far.
        """
        with self._condition:
            return tuple(self._reports)

    def start(self) -> None:
        """Bind an ephemeral loopback port and start the gRPC server.

        Returns:
            None.

        Raises:
            RuntimeError: If gRPC fails to allocate a loopback port.
        """
        self._server = grpc.server(ThreadPoolExecutor(max_workers=4))
        pb_grpc.add_LoadMonitorServiceServicer_to_server(self, self._server)
        self.port = self._server.add_insecure_port("127.0.0.1:0")
        if self.port == 0:
            self._server = None
            raise RuntimeError("failed to bind fake load reporter Router")
        self._server.start()

    def stop(self) -> None:
        """Stop the fake Router and wait for its worker threads.

        Returns:
            None.
        """
        server = self._server
        self._server = None
        if server is not None:
            server.stop(grace=1.0).wait()

    def wait_for_ranked_report(self, timeout: float = 10.0) -> bool:
        """Wait until a report containing at least one rank arrives.

        Args:
            timeout: Maximum wait in seconds.

        Returns:
            Whether a ranked report arrived before the timeout.
        """
        with self._condition:
            return self._condition.wait_for(
                lambda: any(report.ranks for report in self._reports),
                timeout=timeout,
            )

    def Report(
        self,
        request_iterator: Iterator[Any],
        context: grpc.ServicerContext,
    ) -> empty_pb2.Empty:
        """Consume one client stream and record every report.

        Args:
            request_iterator: Reports sent over the client-streaming RPC.
            context: gRPC server context for the stream.

        Returns:
            An empty acknowledgement after the client closes the stream.
        """
        del context
        with self._condition:
            self._stream_count += 1
            self._condition.notify_all()

        try:
            for report in request_iterator:
                with self._condition:
                    self._reports.append(report)
                    self._condition.notify_all()
        except grpc.RpcError:
            pass
        return empty_pb2.Empty()


class TestLoadReporterSingleOwner(CustomTestCase):
    """Prove that two HTTP/tokenizer workers share one reporter runtime."""

    @classmethod
    def setUpClass(cls) -> None:
        """Start the fake Router and a two-tokenizer SGLang server."""
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.admin_api_key = "load-reporter-e2e-admin-key"
        cls.fake_router = FakeLoadReporterRouter()
        cls.fake_router.start()
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tokenizer-worker-num",
                "2",
                "--admin-api-key",
                cls.admin_api_key,
                "--mem-fraction-static",
                "0.5",
            ],
        )

    @classmethod
    def tearDownClass(cls) -> None:
        """Stop the SGLang process tree and fake Router defensively."""
        process = getattr(cls, "process", None)
        if process is not None:
            kill_process_tree(process.pid)
            cls.process = None
        fake_router = getattr(cls, "fake_router", None)
        if fake_router is not None:
            fake_router.stop()
            cls.fake_router = None

    def test_two_tokenizer_workers_open_one_report_stream(self) -> None:
        """Register once, generate once, and observe exactly one gRPC stream."""
        response = requests.post(
            f"{self.base_url}/v1/start_reporting",
            headers={"Authorization": f"Bearer {self.admin_api_key}"},
            json={
                "ip": "127.0.0.1",
                "port": self.fake_router.port,
                "report_interval_ms": 250,
                "lease_ttl_ms": 10000,
            },
            timeout=10,
        )
        self.assertEqual(response.status_code, 200, response.text)

        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Load reporter single-owner verification",
                "sampling_params": {"max_new_tokens": 8, "temperature": 0},
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 200, response.text)

        self.assertTrue(
            self.fake_router.wait_for_ranked_report(),
            "no ranked load report arrived before the E2E timeout",
        )
        self.assertEqual(self.fake_router.stream_count, 1)

        reports = self.fake_router.reports_snapshot()
        ranked_report = next(report for report in reports if report.ranks)
        self.assertTrue(ranked_report.worker.worker_addr)


if __name__ == "__main__":
    unittest.main()
