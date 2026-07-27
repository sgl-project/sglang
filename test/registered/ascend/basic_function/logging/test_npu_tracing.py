import time
import unittest

import requests

from sglang import Engine
from sglang.srt.observability.req_time_stats import RequestStage
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.otel_collector import LightweightOtlpCollector
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=120, suite="full-1-npu-a3", nightly=True)

# Pre-computed expected span names for each trace level
EXPECTED_SPANS_LEVEL_1 = [
    RequestStage.PREFILL_FORWARD.stage_name,
    RequestStage.DECODE_FORWARD.stage_name,
]

EXPECTED_SPANS_LEVEL_2 = EXPECTED_SPANS_LEVEL_1 + [
    RequestStage.REQUEST_PROCESS.stage_name,
]

EXPECTED_SPANS_LEVEL_3 = EXPECTED_SPANS_LEVEL_2 + [
    RequestStage.DECODE_LOOP.stage_name,
]


class TestNPUTracing(TestNPULoggingBase):
    """Testcase：Verify --enable-trace exports spans correctly on single NPU. Validate that different trace levels
    control span granularity as expected, with --trace-modules=request explicitly configured to clarify the traced
    module scope. Inference requests are successfully processed during testing.

    [Test Category] Functionality
    [Test Target] --enable-trace; --trace-modules; --otlp-traces-endpoint
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        """Start collector and server once for all tests."""
        cls.collector = LightweightOtlpCollector()
        cls.collector.start()
        time.sleep(0.2)

        cls.other_args.extend(
            [
                "--enable-trace",
                "--otlp-traces-endpoint",
                "127.0.0.1:4317",
                "--trace-modules",
                "request",
            ]
        )
        # Speed up OTLP export for faster test execution
        cls.env = {
            "SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS": "50",
            "SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE": "4",
        }
        cls.launch_server()

        response = requests.get(f"{cls.base_url}/health_generate")
        assert response.status_code == 200

        # Wait for warmup spans to be exported
        cls.collector.clear()

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

    def _send_request_and_wait(self, text, max_new_tokens=32, trace_level=None):
        """Send a generate request and wait for spans to be collected."""
        if trace_level is not None:
            response = requests.get(
                f"{self.base_url}/set_trace_level?level={trace_level}"
            )
            self.assertEqual(response.status_code, 200)
            self.collector.clear()

        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "stream": True,
            },
            stream=True,
        )
        # Must consume the streaming response to ensure the server
        # fully processes the request and emits all trace spans.
        for _ in response.iter_lines(decode_unicode=False):
            pass

        time.sleep(1)

    def _test_trace_level(
        self, prompt, trace_level, expected_spans=None, max_new_tokens=32
    ):
        """Helper to test a specific trace level.

        Args:
            prompt: The text prompt to send.
            trace_level: The trace level to set (0-3).
            expected_spans: Optional list of expected span names to verify.
            max_new_tokens: Maximum number of tokens to generate.
        """
        self._send_request_and_wait(
            prompt, trace_level=trace_level, max_new_tokens=max_new_tokens
        )

        if trace_level == 0:
            self.assertEqual(
                self.collector.count_spans(),
                0,
                f"Spans collected but expected none: {sorted(self.collector.get_span_names())}",
            )
        else:
            self.assertGreater(
                self.collector.count_spans(),
                0,
                "No spans collected but expected some",
            )

            if expected_spans:
                span_names = self.collector.get_span_names()
                matched = [name for name in expected_spans if name in span_names]
                self.assertGreater(
                    len(matched),
                    0,
                    f"No expected spans found. Expected any of {expected_spans}, "
                    f"got {sorted(span_names)}",
                )

    def test_trace_level_0(self):
        """Trace level 0 should not export any spans."""
        self._test_trace_level("Hello world", trace_level=0, max_new_tokens=5)

    def test_trace_level_1(self):
        """Trace level 1 should export basic prefill and decode spans."""
        self._test_trace_level(
            "The capital of France is",
            trace_level=1,
            expected_spans=EXPECTED_SPANS_LEVEL_1,
        )

    def test_trace_level_2(self):
        """Trace level 2 should export more detailed spans including request_process."""
        self._test_trace_level(
            "What is AI?",
            trace_level=2,
            expected_spans=EXPECTED_SPANS_LEVEL_2,
        )

    def test_trace_level_3(self):
        """Trace level 3 should export the most detailed spans including decode_loop."""
        self._test_trace_level(
            "Explain quantum computing",
            trace_level=3,
            expected_spans=EXPECTED_SPANS_LEVEL_3,
        )

    def test_batch_request(self):
        """Batch requests should produce one prefill span per prompt."""
        response = requests.get(f"{self.base_url}/set_trace_level?level=1")
        self.assertEqual(response.status_code, 200)
        self.collector.clear()

        batch_size = 4
        prompts = ["The capital of France is"] * batch_size
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompts,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 10,
                },
                "stream": False,
            },
        )
        self.assertEqual(response.status_code, 200)

        time.sleep(2)

        self.assertGreater(
            self.collector.count_spans(),
            0,
            "No spans collected from batch request",
        )

        all_spans = self.collector.get_spans()
        request_spans = [
            s for s in all_spans if s.name == RequestStage.PREFILL_FORWARD.stage_name
        ]
        self.assertEqual(
            len(request_spans),
            batch_size,
            f"Expected {batch_size} prefill_forward spans, got {len(request_spans)}",
        )

    def test_parallel_sample(self):
        """Parallel sampling should produce at least one prefill span."""
        response = requests.get(f"{self.base_url}/set_trace_level?level=1")
        self.assertEqual(response.status_code, 200)
        self.collector.clear()

        parallel_num = 4
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0.5,
                    "max_new_tokens": 10,
                    "n": parallel_num,
                },
                "stream": False,
            },
        )
        self.assertEqual(response.status_code, 200)

        time.sleep(2)

        self.assertGreater(
            self.collector.count_spans(),
            0,
            "No spans collected from parallel sample request",
        )

        all_spans = self.collector.get_spans()
        request_spans = [
            s for s in all_spans if s.name == RequestStage.PREFILL_FORWARD.stage_name
        ]
        self.assertGreaterEqual(
            len(request_spans),
            1,
            f"Expected at least 1 prefill_forward span, got {len(request_spans)}",
        )


class TestTraceEngine(CustomTestCase):
    """Testcase：Verify tracing functionality with Engine API on NPU. Ensure request-scoped spans are exported correctly
    via OTLP when enable_trace is set, while the inference and embedding requests are successfully processed.

    [Test Category] Parameter
    [Test Target] enable_trace; trace_modules; otlp_traces_endpoint
    """

    def setUp(self):
        self.collector = None

    def tearDown(self):
        if self.collector:
            self.collector.stop()
            self.collector = None

    def _start_collector(self):
        """Start the lightweight OTLP collector."""
        self.collector = LightweightOtlpCollector()
        self.collector.start()
        time.sleep(0.2)

    def test_trace_engine_enable(self):
        """Test that Engine.generate() exports request module spans with tracing enabled."""
        self._start_collector()

        prompt = "Today is a sunny day and I like"
        model_path = QWEN3_0_6B_WEIGHTS_PATH
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = Engine(
            model_path=model_path,
            random_seed=42,
            enable_trace=True,
            otlp_traces_endpoint="localhost:4317",
            trace_modules="request",
        )

        try:
            engine.generate(prompt, sampling_params)
            time.sleep(0.5)

            self.assertGreater(
                self.collector.count_spans(),
                0,
                "No spans collected from Engine.generate",
            )
            self.assertTrue(
                self.collector.has_any_span([RequestStage.PREFILL_FORWARD.stage_name]),
                f"Expected prefill_forward span, got {self.collector.get_span_names()}",
            )
        finally:
            engine.shutdown()

    def test_trace_engine_encode(self):
        """Test that Engine.encode() exports request module spans with tracing enabled."""
        self._start_collector()

        prompt = "Today is a sunny day and I like"
        model_path = QWEN3_0_6B_WEIGHTS_PATH

        engine = Engine(
            model_path=model_path,
            random_seed=42,
            enable_trace=True,
            otlp_traces_endpoint="localhost:4317",
            trace_modules="request",
            is_embedding=True,
        )

        try:
            engine.encode(prompt)
            time.sleep(0.5)

            self.assertGreater(
                self.collector.count_spans(),
                0,
                "No spans collected from Engine.encode",
            )
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
