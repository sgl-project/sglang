from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

import asyncio
import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import prometheus_client
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY, CollectorRegistry

import sglang.srt.entrypoints.http_server as http_server
from sglang.srt.constants import ROUTING_KEY_HEADER
from sglang.srt.observability import http_request_metrics
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase

UNKNOWN_PATH = "/__no_such_route__"

REQUEST_LABELS = {"endpoint": "/generate", "method": "POST"}


class _FaultInjector:
    """Substitutes the TokenizerManager backend boundary only. Requests are
    legal public /generate payloads; failures model an external backend
    error, not an inference correctness claim."""

    def __init__(self, mode):
        self.mode = mode
        self.admitted = []

    async def generate_request(self, obj, request):
        self.admitted.append((obj.text, obj.stream))
        if self.mode in ("ok", "partial_error", "partial_cancel"):
            yield {"text": "hello", "meta_info": {"finish_reason": None}}
        if self.mode in ("error", "partial_error"):
            raise RuntimeError("injected backend failure")
        if self.mode in ("handled", "handled_disconnect"):
            raise ValueError("injected handled failure")
        if self.mode in ("cancel", "partial_cancel"):
            raise asyncio.CancelledError("injected task cancellation")

    def create_abort_task(self, obj):
        return None


@contextmanager
def _private_metrics():
    """Install a real _HttpRequestMetrics bound to a private registry, and
    restore the process-global singleton afterwards."""
    registry = CollectorRegistry()
    real_counter = prometheus_client.Counter
    real_gauge = prometheus_client.Gauge

    def counter(**kwargs):
        return real_counter(registry=registry, **kwargs)

    def gauge(**kwargs):
        return real_gauge(registry=registry, **kwargs)

    with (
        patch.object(prometheus_client, "Counter", counter),
        patch.object(prometheus_client, "Gauge", gauge),
    ):
        metrics = http_request_metrics._HttpRequestMetrics()
    original = http_request_metrics._http_request_metrics
    http_request_metrics._http_request_metrics = metrics
    try:
        yield registry
    finally:
        http_request_metrics._http_request_metrics = original


def _sample(registry, name, labels=None):
    return registry.get_sample_value(name, labels) or 0.0


def _response_snapshot(registry):
    result = {
        "requests_total": _sample(
            registry, "sglang:http_requests_total", REQUEST_LABELS
        ),
        "requests_active": _sample(
            registry, "sglang:http_requests_active", REQUEST_LABELS
        ),
        "routing_active": _sample(registry, "sglang:routing_keys_active"),
    }
    for status in (200, 400, 500):
        result[str(status)] = _sample(
            registry,
            "sglang:http_responses_total",
            {**REQUEST_LABELS, "status_code": str(status)},
        )
    return result


async def _invoke(mode, stream, disconnect=False):
    """Drive the real module-level app (middleware stack included) over raw
    ASGI, so observed wire messages are the independent status oracle."""
    backend = _FaultInjector(mode)
    payload = json.dumps(
        {"text": "Hello", "stream": stream, "sampling_params": {"max_new_tokens": 1}}
    ).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/generate",
        "raw_path": b"/generate",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (ROUTING_KEY_HEADER.encode("ascii"), b"unit-regression"),
        ],
        "client": ("127.0.0.1", 10000),
        "server": ("127.0.0.1", 30000),
    }
    messages = []
    state = {"delivered": False, "disconnect_sent": False}

    async def receive():
        if not state["delivered"]:
            state["delivered"] = True
            return {"type": "http.request", "body": payload, "more_body": False}
        if disconnect and not state["disconnect_sent"]:
            state["disconnect_sent"] = True
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)

    error = None
    with patch.object(
        http_server, "_global_state", SimpleNamespace(tokenizer_manager=backend)
    ):
        try:
            await http_server.app(scope, receive, send)
        except BaseException as exc:
            error = exc
    return backend, messages, error


class TestHttpResponseAccounting(CustomTestCase):
    def setUp(self):
        super().setUp()
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.addCleanup(reset_context)

    def _assert_case(
        self, registry, mode, stream, status, error_type, disconnect=False
    ):
        before = _response_snapshot(registry)
        backend, messages, error = asyncio.run(_invoke(mode, stream, disconnect))
        after = _response_snapshot(registry)
        delta = {k: after[k] - before[k] for k in before}
        wire_status = [
            m["status"] for m in messages if m["type"] == "http.response.start"
        ]
        self.assertEqual(backend.admitted, [("Hello", stream)])
        if error_type is None:
            self.assertIsNone(error)
        else:
            self.assertIsInstance(error, error_type)
        self.assertEqual(wire_status, [] if status is None else [status])
        self.assertEqual(delta["requests_total"], 1)
        self.assertEqual(delta["requests_active"], 0)
        self.assertEqual(delta["routing_active"], 0)
        for code in (200, 400, 500):
            self.assertEqual(
                delta[str(code)],
                int(code == status),
                f"{mode}/{stream}: wire status {status}, metric {code}",
            )
        return messages

    def test_unhandled_backend_exception_counts_submitted_status(self):
        with _private_metrics() as registry:
            # Ordinary exception before headers: the outer default error
            # handler answers 500; the response must be counted as 500.
            self._assert_case(registry, "error", False, 500, RuntimeError)
            # A response that already started keeps its submitted status even
            # when the body or generator subsequently fails.
            self._assert_case(registry, "error", True, 200, RuntimeError)
            self._assert_case(registry, "partial_error", True, 200, RuntimeError)

    def test_success_and_handled_errors_count_once(self):
        with _private_metrics() as registry:
            self._assert_case(registry, "ok", False, 200, None)
            self._assert_case(registry, "ok", True, 200, None)
            self._assert_case(registry, "handled", False, 400, None)
            self._assert_case(registry, "handled", True, 200, None)

    def test_cancellation_after_start_keeps_submitted_status(self):
        with _private_metrics() as registry:
            self._assert_case(
                registry, "partial_cancel", True, 200, asyncio.CancelledError
            )

    def test_cancellation_before_response_is_not_a_fictional_500(self):
        # Pre-header cancellation stays a known, deliberately unchanged
        # server boundary: Uvicorn may synthesize a 500 at the transport
        # level, which this middleware cannot observe. Only propagation and
        # cleanup are asserted here.
        with _private_metrics() as registry:
            self._assert_case(registry, "cancel", False, None, asyncio.CancelledError)

    def test_receive_disconnect_still_detected(self):
        # A handled stream error on a disconnected client takes the
        # is_disconnected() early-return path: the stream closes with
        # neither [DONE] nor an error SSE payload. This only works if the metrics
        # middleware passes the original receive through.
        with _private_metrics() as registry:
            messages = self._assert_case(
                registry, "handled_disconnect", True, 200, None, disconnect=True
            )
        body = b"".join(m.get("body", b"") for m in messages)
        # The is_disconnected() early return skips both the error payload
        # and the [DONE] terminator; a plain handled error would emit both.
        self.assertNotIn(b"invalid_request_error", body)
        self.assertNotIn(b"[DONE]", body)

    def test_metrics_disabled_passes_through(self):
        original = http_request_metrics._http_request_metrics
        http_request_metrics._http_request_metrics = None
        try:
            backend, messages, error = asyncio.run(_invoke("ok", False))
        finally:
            http_request_metrics._http_request_metrics = original
        self.assertIsNone(error)
        self.assertEqual(backend.admitted, [("Hello", False)])
        self.assertEqual(
            [m["status"] for m in messages if m["type"] == "http.response.start"],
            [200],
        )

    def test_non_http_scope_passes_through(self):
        with _private_metrics() as registry:
            seen = []

            async def inner(scope, receive, send):
                seen.append(scope["type"])

            middleware = http_request_metrics._HttpRequestMetricsMiddleware(inner)
            asyncio.run(middleware({"type": "websocket"}, None, None))
            self.assertEqual(seen, ["websocket"])
            self.assertEqual(_response_snapshot(registry)["requests_total"], 0)


class TestHttpRequestMetrics(CustomTestCase):
    def test_module_level_app_records_requests(self):
        # Tokenizer workers re-import http_server and serve its module-level app,
        # so the middleware must be on it, while the metrics must wait for
        # init_http_request_metrics().
        self.assertIsNone(http_request_metrics._http_request_metrics)

        http_request_metrics.init_http_request_metrics()
        publish(ServerArgs(model_path="dummy"), role="tokenizer")
        self.addCleanup(reset_context)
        TestClient(http_server.app).get(UNKNOWN_PATH)

        self.assertEqual(
            1.0,
            REGISTRY.get_sample_value(
                "sglang:http_requests_total",
                {"endpoint": UNKNOWN_PATH, "method": "GET"},
            ),
        )


if __name__ == "__main__":
    unittest.main()
