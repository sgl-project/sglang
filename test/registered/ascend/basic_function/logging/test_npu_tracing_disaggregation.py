import os
import time
import unittest

import requests

from sglang.srt.observability.req_time_stats import RequestStage
from sglang.test.ascend.disaggregation_utils import TestDisaggregationBase
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.otel_collector import LightweightOtlpCollector
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

# CI registration - PD disaggregation requires 2 NPUs
register_npu_ci(est_time=120, suite="full-2-npu-a3", nightly=True)


class TestNPUTracingDisaggregation(TestDisaggregationBase):
    """Testcase：Verify --enable-trace exports spans correctly in PD disaggregation mode on NPU.
        Validate that disaggregation-specific transfer spans are exported as expected,
        with --trace-modules=request explicitly configured to clarify the traced module scope.
        Inference requests are successfully processed during testing.

    [Test Category] Functionality
    [Test Target] --enable-trace; --trace-modules; --otlp-traces-endpoint
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.model = QWEN3_0_6B_WEIGHTS_PATH

        os.environ["ASCEND_MF_STORE_URL"] = "tcp://127.0.0.1:24666"
        os.environ["SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS"] = "50"
        os.environ["SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE"] = "4"

        # Initialize collector first
        cls.collector = LightweightOtlpCollector()
        cls.collector.start()
        time.sleep(0.2)

        # Start prefill and decode servers, then launch LB
        cls.start_prefill()
        cls.start_decode()
        cls.wait_server_ready(cls.prefill_url + "/health")
        cls.wait_server_ready(cls.decode_url + "/health")
        cls.launch_lb()

        # Wait for warmup spans to be exported and clear them
        time.sleep(1)
        cls.collector.clear()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--disaggregation-mode",
            "prefill",
            "--tp-size",
            "1",
            "--disaggregation-transfer-backend",
            "ascend",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--dist-init-addr",
            "127.0.0.1:10100",
            "--enable-trace",
            "--otlp-traces-endpoint",
            "localhost:4317",
            "--trace-modules",
            "request",
        ]

        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--disaggregation-mode",
            "decode",
            "--tp-size",
            "1",
            "--base-gpu-id",
            "1",
            "--attention-backend",
            "ascend",
            "--disaggregation-transfer-backend",
            "ascend",
            "--disable-cuda-graph",
            "--enable-trace",
            "--otlp-traces-endpoint",
            "localhost:4317",
            "--dist-init-addr",
            "127.0.0.1:10000",
            "--trace-modules",
            "request",
        ]

        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
            env={
                "SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS": "50",
                "SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE": "4",
            },
        )

    @classmethod
    def tearDownClass(cls):
        if cls.collector:
            cls.collector.stop()
            cls.collector = None
        super().tearDownClass()

    def setUp(self):
        """Wait for spans to be drained before each test."""
        max_wait_seconds = 10
        check_interval = 0.2
        elapsed = 0
        consecutive_zero_count = 0
        required_consecutive_zeros = 3

        # Poll the collector until no new spans arrive for several
        # consecutive checks, ensuring leftover spans from previous
        # tests are fully drained before the current test starts.
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
        """Test that PD disaggregation exports request module spans (e.g., prefill_forward,
        decode_forward) and disaggregation-specific transfer spans."""
        # Set trace level on both prefill and decode servers
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
        self.assertTrue(
            self.collector.has_any_span(
                [
                    RequestStage.PREFILL_FORWARD.stage_name,
                    RequestStage.DECODE_FORWARD.stage_name,
                ]
            ),
            f"Expected request module spans in PD disaggregation, got {sorted(span_names)}",
        )

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
