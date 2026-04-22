"""Integration test for OpenTelemetry tracing in the diffusion pipeline.

Spins up a lightweight in-process OTLP collector, launches a diffusion server
with ``--enable-trace``, sends an image-generation request with a
``traceparent`` header, and asserts that the expected spans
(``scheduler_dispatch``, ``gpu_forward``) are exported.
"""

import os

# Configure OTLP exporter for faster test execution.
# Must be set before importing any sglang trace module.
os.environ.setdefault("SGLANG_OTLP_EXPORTER_SCHEDULE_DELAY_MILLIS", "50")
os.environ.setdefault("SGLANG_OTLP_EXPORTER_MAX_EXPORT_BATCH_SIZE", "4")

import logging
import time

import pytest
import requests

from sglang.multimodal_gen.test.server.test_server_utils import ServerManager
from sglang.multimodal_gen.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST
from sglang.test.otel_collector import LightweightOtlpCollector

logger = logging.getLogger(__name__)

# Expected diffusion trace span names (from DiffStage in trace_wrapper.py)
EXPECTED_DIFF_SPANS = ["scheduler_dispatch", "gpu_forward"]

COLLECTOR_PORT = 4317
SERVER_PORT = 39812


@pytest.fixture(scope="module")
def tracing_env():
    """Start the OTLP collector and diffusion server once for all tests."""
    collector = LightweightOtlpCollector(port=COLLECTOR_PORT)
    collector.start()
    time.sleep(0.3)

    mgr = ServerManager(
        model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        port=SERVER_PORT,
        extra_args=f"--enable-trace --otlp-traces-endpoint 127.0.0.1:{COLLECTOR_PORT}",
    )
    ctx = mgr.start()

    # Clear any warmup spans
    time.sleep(2)
    collector.clear()

    yield collector, ctx

    ctx.cleanup()
    collector.stop()


def _generate_image(headers=None):
    """Send a single image-generation request."""
    resp = requests.post(
        f"http://127.0.0.1:{SERVER_PORT}/v1/images/generations",
        json={
            "model": DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            "prompt": "A white cat",
            "size": "256x256",
            "n": 1,
        },
        headers=headers or {},
        timeout=300,
    )
    assert resp.status_code == 200, f"Generation failed: {resp.text}"
    return resp


def _wait_for_spans(collector, required_names=None, min_count=1, timeout=30):
    """Wait until collector has the required span names (or at least ``min_count`` spans)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if required_names:
            if all(collector.has_span(n) for n in required_names):
                return
        elif collector.count_spans() >= min_count:
            return
        time.sleep(0.5)


def test_spans_exported(tracing_env):
    """After a generation request the expected diffusion spans appear."""
    collector, _ = tracing_env
    collector.clear()

    # W3C Trace Context traceparent header
    _generate_image(
        headers={
            "traceparent": "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
        }
    )
    _wait_for_spans(collector, required_names=EXPECTED_DIFF_SPANS)

    span_names = collector.get_span_names()
    for expected in EXPECTED_DIFF_SPANS:
        assert (
            expected in span_names
        ), f"Missing span '{expected}'. Collected: {sorted(span_names)}"


def test_spans_without_traceparent(tracing_env):
    """Requests without a traceparent header still produce spans as a new
    root trace (not linked to any upstream)."""
    collector, _ = tracing_env
    collector.clear()

    _generate_image()
    _wait_for_spans(collector, required_names=EXPECTED_DIFF_SPANS)

    span_names = collector.get_span_names()
    for expected in EXPECTED_DIFF_SPANS:
        assert (
            expected in span_names
        ), f"Missing span '{expected}'. Collected: {sorted(span_names)}"


def test_batch_requests(tracing_env):
    """Multiple requests each produce their own set of spans."""
    collector, _ = tracing_env
    collector.clear()

    batch_size = 3
    for i in range(batch_size):
        # Each request gets a unique trace-id
        trace_id = f"0af7651916cd43dd8448eb211c8031{i:02x}"
        _generate_image(headers={"traceparent": f"00-{trace_id}-b7ad6b7169203331-01"})

    # Wait until all scheduler_dispatch spans have arrived (they come from a
    # separate process so may lag behind gpu_forward).
    deadline = time.time() + 60
    while time.time() < deadline:
        if all(
            len(collector.get_spans_by_name(n)) >= batch_size
            for n in EXPECTED_DIFF_SPANS
        ):
            break
        time.sleep(0.5)

    for span_name in EXPECTED_DIFF_SPANS:
        matching = collector.get_spans_by_name(span_name)
        assert len(matching) >= batch_size, (
            f"Expected at least {batch_size} '{span_name}' spans, "
            f"got {len(matching)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
