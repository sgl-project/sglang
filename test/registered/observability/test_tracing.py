"""Integration tests for tracing with a lightweight in-process OTLP collector.

This module implements a minimal OTLP collector that receives traces via gRPC
and stores them in memory for test assertions, eliminating the need for
Docker-based opentelemetry-collector and file I/O.
"""

import os

# Configure OTLP exporter for faster test execution
# Must be set before importing sglang trace module
os.environ.setdefault("SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS", "50")
os.environ.setdefault("SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE", "4")

import logging
import multiprocessing as mp
import time
import unittest
from dataclasses import dataclass
from typing import List, Optional, Union

import requests
import zmq

from sglang import Engine
from sglang.srt.observability.req_time_stats import RequestStage
from sglang.srt.observability.trace import (
    TraceReqContext,
    TraceSliceContext,
    get_cur_time_ns,
    process_tracing_init,
    set_global_trace_level,
    trace_set_thread_info,
)
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.network import get_zmq_socket
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

logger = logging.getLogger(__name__)

# CI registration
register_cuda_ci(est_time=113, stage="stage-b", runner_config="1-gpu-small")


# ============================================================================
# Lightweight OTLP Collector (shared across tracing tests)
# ============================================================================

from sglang.test.otel_collector import LightweightOtlpCollector, Span  # noqa: F401

# ============================================================================
# Test Helper Functions
# ============================================================================


def _get_span_names_by_level(level: int) -> List[str]:
    """Get expected span names for a given trace level.

    Based on RequestStage definitions in req_time_stats.py:
    - Each RequestStage has a level attribute indicating minimum trace level required
    - Spans with level <= current trace level will be exported
    """
    span_names = []
    # RequestStage is a class with class attributes that are RequestStageConfig instances
    for attr_name in dir(RequestStage):
        if attr_name.startswith("_"):
            continue
        attr = getattr(RequestStage, attr_name)
        # Check if it's a RequestStageConfig (has stage_name and level attributes)
        if hasattr(attr, "stage_name") and hasattr(attr, "level"):
            if attr.level <= level and attr.stage_name:
                span_names.append(attr.stage_name)
    return span_names


# Pre-computed span names by level for efficiency
SPAN_NAMES_LEVEL_1 = _get_span_names_by_level(1)
SPAN_NAMES_LEVEL_2 = _get_span_names_by_level(2)
SPAN_NAMES_LEVEL_3 = _get_span_names_by_level(3)

# Common span names expected in typical inference requests
# Level 1: Basic request lifecycle
EXPECTED_SPANS_LEVEL_1 = [
    RequestStage.PREFILL_FORWARD.stage_name,
    RequestStage.DECODE_FORWARD.stage_name,
]

# Level 2: More detailed including dispatch
EXPECTED_SPANS_LEVEL_2 = EXPECTED_SPANS_LEVEL_1 + [
    RequestStage.REQUEST_PROCESS.stage_name,
]

# Level 3: Most detailed including internal operations
EXPECTED_SPANS_LEVEL_3 = EXPECTED_SPANS_LEVEL_2 + [
    RequestStage.DECODE_LOOP.stage_name,
]


@dataclass
class Req:
    rid: int
    req_context: Optional[Union[TraceReqContext]] = None


def _subprocess_worker():
    """Worker function for subprocess trace context propagation test.
    Must be at module level for pickle compatibility with spawn.
    """
    process_tracing_init("127.0.0.1:4317", "test")
    trace_set_thread_info("Sub Process")

    context = zmq.Context(2)
    recv_from_main = get_zmq_socket(context, zmq.PULL, "ipc:///tmp/zmq_test.ipc", True)

    try:
        req = recv_from_main.recv_pyobj()
        req.req_context.rebuild_thread_context()
        req.req_context.trace_slice_start("work", level=1)
        time.sleep(0.2)
        req.req_context.trace_slice_end("work", level=1, thread_finish_flag=True)
    finally:
        recv_from_main.close()
        context.term()


# ============================================================================
# Test Cases
# ============================================================================


class TestTracePackage(CustomTestCase):
    """Unit tests for tracing package API without server/engine."""

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

    def test_slice_simple(self):
        """Unit test: simple slice trace API."""
        self._start_collector()

        try:
            process_tracing_init("127.0.0.1:4317", "test")
            trace_set_thread_info("Test")
            set_global_trace_level(3)
            req_context = TraceReqContext(0)
            req_context.trace_req_start()
            req_context.trace_slice_start("test slice", level=1)
            time.sleep(0.1)
            req_context.trace_slice_end("test slice", level=1)
            req_context.trace_req_finish()

            time.sleep(0.3)

            self.assertTrue(
                self.collector.has_span("test slice"),
                f"Expected span 'test slice', got {self.collector.get_span_names()}",
            )
        finally:
            pass

    def test_slice_complex(self):
        """Unit test: complex slice trace with events."""
        self._start_collector()

        try:
            process_tracing_init("127.0.0.1:4317", "test")
            trace_set_thread_info("Test")
            set_global_trace_level(3)
            req_context = TraceReqContext(0)
            req_context.trace_req_start()

            t1 = get_cur_time_ns()
            time.sleep(0.1)
            req_context.trace_event("event test", 1)
            t2 = get_cur_time_ns()
            time.sleep(0.1)
            t3 = get_cur_time_ns()

            slice1 = TraceSliceContext("slice A", t1, t2)
            slice2 = TraceSliceContext("slice B", t2, t3)
            req_context.trace_slice(slice1)
            req_context.trace_slice(slice2, thread_finish_flag=True)
            req_context.trace_req_finish()

            time.sleep(0.3)

            self.assertTrue(
                self.collector.has_all_spans(["slice A", "slice B"]),
                f"Expected spans 'slice A' and 'slice B', got {self.collector.get_span_names()}",
            )
        finally:
            pass

    def test_context_propagate(self):
        """Unit test: trace context propagation across processes via ZMQ."""
        self._start_collector()

        ctx = mp.get_context("spawn")

        context = zmq.Context(2)
        send_to_subproc = get_zmq_socket(
            context, zmq.PUSH, "ipc:///tmp/zmq_test.ipc", False
        )

        try:
            process_tracing_init("127.0.0.1:4317", "test")
            trace_set_thread_info("Main Process")

            subproc = ctx.Process(target=_subprocess_worker)
            subproc.start()

            time.sleep(0.3)

            req = Req(rid=0)
            req.req_context = TraceReqContext(0)
            req.req_context.trace_req_start()
            req.req_context.trace_slice_start("dispatch", level=1)
            time.sleep(0.2)
            send_to_subproc.send_pyobj(req)
            req.req_context.trace_slice_end("dispatch", level=1)

            subproc.join()
            req.req_context.trace_req_finish()

            time.sleep(0.5)

            self.assertTrue(
                self.collector.has_all_spans(["dispatch", "work"]),
                f"Expected spans 'dispatch' and 'work', got {self.collector.get_span_names()}",
            )
        finally:
            send_to_subproc.close()
            context.term()


class TestTraceServer(CustomTestCase):
    """Integration tests for tracing with server - starts server once for all tests."""

    @classmethod
    def setUpClass(cls):
        """Start collector and server once for all tests."""
        cls.collector = LightweightOtlpCollector()
        cls.collector.start()
        time.sleep(0.2)

        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-trace",
                "--otlp-traces-endpoint",
                "127.0.0.1:4317",
            ],
        )

        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
        assert response.status_code == 200

        # Wait for warmup spans to be exported
        cls.collector.clear()

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)
        if cls.collector:
            cls.collector.stop()

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

    def _send_request_and_wait(
        self, text, max_new_tokens=32, stream=True, trace_level=None
    ):
        """Helper to send a request and wait for spans."""
        if trace_level is not None:
            response = requests.get(
                f"{DEFAULT_URL_FOR_TEST}/set_trace_level?level={trace_level}"
            )
            self.assertEqual(response.status_code, 200)
            self.collector.clear()

        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "stream": stream,
            },
            stream=stream,
        )
        if stream:
            for _ in response.iter_lines(decode_unicode=False):
                pass
        else:
            self.assertEqual(response.status_code, 200)

        time.sleep(1)

    def test_trace_level_0(self):
        """Test trace level 0 does not export any spans."""
        self._send_request_and_wait("Hello world", max_new_tokens=5, trace_level=0)
        self.assertEqual(
            self.collector.count_spans(),
            0,
            f"Spans collected but expected none: {sorted(self.collector.get_span_names())}",
        )

    def test_trace_level_1(self):
        """Test trace level 1 exports basic request lifecycle spans."""
        self._send_request_and_wait("The capital of France is", trace_level=1)

        self.assertGreater(
            self.collector.count_spans(),
            0,
            "No spans collected but expected some",
        )

        span_names = self.collector.get_span_names()
        matched = [name for name in EXPECTED_SPANS_LEVEL_1 if name in span_names]
        self.assertGreater(
            len(matched),
            0,
            f"No expected spans found. Expected any of {EXPECTED_SPANS_LEVEL_1}, "
            f"got {sorted(span_names)}",
        )

    def test_trace_level_2(self):
        """Test trace level 2 exports more detailed spans."""
        self._send_request_and_wait("What is AI?", trace_level=2)

        span_names = self.collector.get_span_names()
        matched = [name for name in EXPECTED_SPANS_LEVEL_2 if name in span_names]
        self.assertGreater(
            len(matched),
            0,
            f"No expected spans found. Expected any of {EXPECTED_SPANS_LEVEL_2}, "
            f"got {sorted(span_names)}",
        )

    def test_trace_level_3(self):
        """Test trace level 3 exports most detailed spans."""
        self._send_request_and_wait("Explain quantum computing", trace_level=3)

        span_names = self.collector.get_span_names()
        matched = [name for name in EXPECTED_SPANS_LEVEL_3 if name in span_names]
        self.assertGreater(
            len(matched),
            0,
            f"No expected spans found. Expected any of {EXPECTED_SPANS_LEVEL_3}, "
            f"got {sorted(span_names)}",
        )

    def test_batch_request(self):
        """Test tracing with batch requests (multiple prompts in one request)."""
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/set_trace_level?level=1")
        self.assertEqual(response.status_code, 200)
        self.collector.clear()

        batch_size = 4
        prompts = ["The capital of France is"] * batch_size
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
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

        time.sleep(0.5)

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
        """Test tracing with parallel sampling (n > 1 in sampling_params)."""
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/set_trace_level?level=1")
        self.assertEqual(response.status_code, 200)
        self.collector.clear()

        # parallel_sample_num is controlled by 'n' in sampling_params
        parallel_num = 4
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0.5,  # Need non-zero temp for parallel sampling
                    "max_new_tokens": 10,
                    "n": parallel_num,
                },
                "stream": False,
            },
        )
        self.assertEqual(response.status_code, 200)

        time.sleep(0.5)

        self.assertGreater(
            self.collector.count_spans(),
            0,
            "No spans collected from parallel sample request",
        )

        # With parallel sampling, we expect prefill spans for each parallel sample
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
    """Integration tests for tracing with Engine API - each test creates its own engine."""

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
        """Test tracing with Engine API."""
        self._start_collector()

        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = Engine(
            model_path=model_path,
            random_seed=42,
            enable_trace=True,
            otlp_traces_endpoint="localhost:4317",
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
        """Test tracing with Engine encode API."""
        self._start_collector()

        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

        engine = Engine(
            model_path=model_path,
            random_seed=42,
            enable_trace=True,
            otlp_traces_endpoint="localhost:4317",
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
