"""Test tracing in PD disaggregation mode."""

import os

# Configure OTLP exporter for faster test execution
# Must be set before importing sglang trace module
os.environ.setdefault("SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS", "50")
os.environ.setdefault("SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE", "4")

import logging
import shlex
import time
import unittest
from urllib.parse import urlparse

import requests

# Import the lightweight collector from the main tracing test module
from test_tracing import LightweightOtlpCollector

from sglang.srt.observability.req_time_stats import RequestStage
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import get_rdma_devices_args
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_pd_server,
    popen_with_error_check,
)
from sglang.utils import wait_for_http_ready

logger = logging.getLogger(__name__)

# CI registration - PD disaggregation requires 2 GPUs
register_cuda_ci(est_time=51, suite="stage-b-test-2-gpu-large")


class TestTraceDisaggregation(CustomTestCase):
    """Test tracing in PD disaggregation mode."""

    @classmethod
    def setUpClass(cls):
        # Initialize collector first
        cls.collector = LightweightOtlpCollector()
        cls.collector.start()
        time.sleep(0.2)

        # Setup PD disaggregation server addresses
        parsed_url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed_url.hostname
        base_port = str(parsed_url.port)
        cls.lb_port = base_port
        cls.prefill_port = f"{int(base_port) + 100}"
        cls.decode_port = f"{int(base_port) + 200}"
        cls.bootstrap_port = f"{int(base_port) + 500}"
        cls.prefill_url = f"http://{cls.base_host}:{cls.prefill_port}"
        cls.decode_url = f"http://{cls.base_host}:{cls.decode_port}"
        cls.lb_url = f"http://{cls.base_host}:{cls.lb_port}"
        cls.process_lb = None
        cls.process_decode = None
        cls.process_prefill = None
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST

        # Config transfer backend
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
        cls.rdma_devices = ["--disaggregation-ib-device", get_rdma_devices_args()]

        # Start prefill server with trace enabled
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--enable-trace",
            "--otlp-traces-endpoint",
            "localhost:4317",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

        # Start decode server with trace enabled
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "1",
            "--base-gpu-id",
            "1",
            "--enable-trace",
            "--otlp-traces-endpoint",
            "localhost:4317",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

        # Wait for servers to be ready
        wait_for_http_ready(
            url=cls.prefill_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process_prefill,
        )
        wait_for_http_ready(
            url=cls.decode_url + "/health",
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            process=cls.process_decode,
        )

        # Start load balancer
        lb_command = [
            "python3",
            "-m",
            "sglang_router.launch_router",
            "--pd-disaggregation",
            "--mini-lb",
            "--prefill",
            cls.prefill_url,
            "--decode",
            cls.decode_url,
            "--host",
            cls.base_host,
            "--port",
            cls.lb_port,
        ]
        print("Starting load balancer:", shlex.join(lb_command))
        cls.process_lb = popen_with_error_check(lb_command)
        wait_for_http_ready(url=cls.lb_url + "/health", process=cls.process_lb)

        # Wait for warmup spans and clear
        time.sleep(1)
        cls.collector.clear()

    @classmethod
    def tearDownClass(cls):
        for process in [cls.process_lb, cls.process_decode, cls.process_prefill]:
            if process:
                try:
                    kill_process_tree(process.pid)
                except Exception as e:
                    print(f"Error killing process {process.pid}: {e}")
        if cls.collector:
            cls.collector.stop()
        time.sleep(5)

    def setUp(self):
        """Wait for spans to be drained before each test."""
        max_wait_seconds = 10
        check_interval = 0.2
        elapsed = 0
        consecutive_zero_count = 0
        required_consecutive_zeros = 3

        while elapsed < max_wait_seconds:
            span_count = self.collector.count_spans()
            if span_count == 0:
                consecutive_zero_count += 1
                if consecutive_zero_count >= required_consecutive_zeros:
                    break
            else:
                consecutive_zero_count = 0
                self.collector.clear()
            time.sleep(check_interval)
            elapsed += check_interval
        else:
            raise RuntimeError(
                f"Timeout waiting for spans to drain after {max_wait_seconds}s. "
                f"Remaining spans: {self.collector.count_spans()}"
            )

    def test_disaggregation_transfer_spans(self):
        """Test that disaggregation produces PREFILL_TRANSFER_KV_CACHE and DECODE_TRANSFERRED spans."""
        # Set trace level
        response = requests.get(f"{self.prefill_url}/set_trace_level?level=1")
        self.assertEqual(response.status_code, 200)
        response = requests.get(f"{self.decode_url}/set_trace_level?level=1")
        self.assertEqual(response.status_code, 200)
        self.collector.clear()

        # Send a request through load balancer
        response = requests.post(
            f"{self.lb_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 10,
                },
                "stream": False,
            },
        )
        self.assertEqual(response.status_code, 200)

        # Wait for async export
        time.sleep(1)

        # Verify spans were collected
        self.assertGreater(
            self.collector.count_spans(),
            0,
            "No spans collected from disaggregation request",
        )

        # Verify disaggregation-specific spans exist
        span_names = self.collector.get_span_names()

        # Check for transfer-related spans
        self.assertTrue(
            self.collector.has_any_span(
                [
                    RequestStage.PREFILL_TRANSFER_KV_CACHE.stage_name,
                    RequestStage.DECODE_TRANSFERRED.stage_name,
                ]
            ),
            f"Expected disaggregation transfer spans, got {sorted(span_names)}",
        )


if __name__ == "__main__":
    unittest.main()
