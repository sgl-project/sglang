import asyncio
import gzip
import logging
import threading
from types import SimpleNamespace
from unittest.mock import patch

from prometheus_client import CollectorRegistry, Counter, make_asgi_app

import sglang.srt.utils.common as common_module
from sglang.srt.utils.common import _make_guarded_metrics_app, add_prometheus_middleware
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


async def _call_asgi(app, *, headers=(), query_string=b"", receive_override=None):
    messages = []

    if receive_override is None:

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

    else:
        receive = receive_override

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "http",
            "method": "GET",
            "path": "/metrics",
            "headers": list(headers),
            "query_string": query_string,
        },
        receive,
        send,
    )
    return messages


def _status(messages):
    for message in messages:
        if message["type"] == "http.response.start":
            return message["status"]
    raise AssertionError("missing response start")


def _body(messages):
    return b"".join(
        message.get("body", b"")
        for message in messages
        if message["type"] == "http.response.body"
    )


def _headers(messages):
    for message in messages:
        if message["type"] == "http.response.start":
            return message["headers"]
    raise AssertionError("missing response start")


def _header_dict(messages):
    return {name.lower(): value for name, value in _headers(messages)}


def _slow_app(app, started, finish):
    async def slow(scope, receive, send):
        started.set()
        await asyncio.to_thread(finish.wait, 1)
        await app(scope, receive, send)

    return slow


def test_add_prometheus_middleware_keeps_native_content_negotiation():
    async def run():
        app = SimpleNamespace(routes=[])
        with patch("prometheus_client.multiprocess.MultiProcessCollector"):
            add_prometheus_middleware(app)
        metrics_app = app.routes[-1].app

        openmetrics = await _call_asgi(
            metrics_app,
            headers=[(b"accept", b"application/openmetrics-text; version=1.0.0")],
        )
        assert _status(openmetrics) == 200
        assert _header_dict(openmetrics)[b"content-type"].startswith(
            b"application/openmetrics-text"
        )
        assert _body(openmetrics).endswith(b"# EOF\n")

        compressed = await _call_asgi(
            metrics_app, headers=[(b"accept-encoding", b"gzip")]
        )
        assert _status(compressed) == 200
        assert _header_dict(compressed)[b"content-encoding"] == b"gzip"
        gzip.decompress(_body(compressed))

    asyncio.run(run())


def test_metrics_guard_preserves_native_prometheus_protocol():
    async def run():
        registry = CollectorRegistry()
        requests = Counter("requests_total", "Requests", ["route"], registry=registry)
        requests.labels(route="/generate").inc(3)
        Counter("other_total", "Other", registry=registry).inc()
        native_app = make_asgi_app(registry=registry)
        guarded_app = _make_guarded_metrics_app(native_app)
        cases = [
            (),
            ((b"accept", b"application/openmetrics-text; version=1.0.0"),),
            ((b"accept-encoding", b"gzip"),),
            ((b"accept", b"text/plain; version=1.0.0; escaping=underscores"),),
        ]

        for headers in cases:
            expected = await _call_asgi(native_app, headers=headers)
            actual = await _call_asgi(guarded_app, headers=headers)
            assert _status(actual) == _status(expected) == 200
            assert _headers(actual) == _headers(expected)
            expected_body = _body(expected)
            actual_body = _body(actual)
            if _header_dict(expected).get(b"content-encoding") == b"gzip":
                expected_body = gzip.decompress(expected_body)
                actual_body = gzip.decompress(actual_body)
            assert actual_body == expected_body

        expected = await _call_asgi(native_app, query_string=b"name[]=requests_total")
        actual = await _call_asgi(guarded_app, query_string=b"name[]=requests_total")
        assert _status(actual) == _status(expected) == 200
        assert _headers(actual) == _headers(expected)
        assert _body(actual) == _body(expected)

    asyncio.run(run())


def test_metrics_guard_rejects_concurrent_scrape():
    async def run():
        started = threading.Event()
        finish = threading.Event()
        registry = CollectorRegistry()
        app = _make_guarded_metrics_app(
            _slow_app(make_asgi_app(registry=registry), started, finish),
            scrape_timeout_s=1.0,
        )
        first = asyncio.create_task(_call_asgi(app))
        assert await asyncio.to_thread(started.wait, 1)

        second_messages = await _call_asgi(app)
        finish.set()
        first_messages = await first

        assert _status(second_messages) == 503
        assert _status(first_messages) == 200

    asyncio.run(run())


def test_metrics_guard_singleflight_covers_request_read_window():
    async def run():
        arrivals = 0
        invocations = 0
        release = asyncio.Event()
        started = threading.Event()
        finish = threading.Event()

        async def receive():
            nonlocal arrivals
            arrivals += 1
            if arrivals == 2:
                release.set()
            await release.wait()
            return {"type": "http.request", "body": b"", "more_body": False}

        async def metrics_app(scope, receive, send):
            nonlocal invocations
            invocations += 1
            started.set()
            await asyncio.to_thread(finish.wait, 1)
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok\n"})

        app = _make_guarded_metrics_app(metrics_app, scrape_timeout_s=1.0)
        requests = [
            asyncio.create_task(_call_asgi(app, receive_override=receive))
            for _ in range(2)
        ]
        assert await asyncio.to_thread(started.wait, 1)
        finish.set()
        results = await asyncio.gather(*requests)

        assert sorted(_status(result) for result in results) == [200, 503]
        assert invocations == 1

    asyncio.run(run())


def test_metrics_guard_times_out_slow_scrape():
    async def run():
        started = threading.Event()
        finish = threading.Event()
        registry = CollectorRegistry()
        app = _make_guarded_metrics_app(
            _slow_app(make_asgi_app(registry=registry), started, finish),
            scrape_timeout_s=0.01,
        )
        messages = await _call_asgi(app)
        assert started.is_set()
        finish.set()

        assert _status(messages) == 504
        assert _body(messages) == b"metrics scrape timed out\n"

    asyncio.run(run())


def test_metrics_guard_logs_background_failure_after_timeout(caplog):
    async def run():
        started = threading.Event()
        finish = threading.Event()
        failed = threading.Event()

        async def metrics_app(scope, receive, send):
            started.set()
            await asyncio.to_thread(finish.wait, 1)
            failed.set()
            raise RuntimeError("registry read failed")

        app = _make_guarded_metrics_app(metrics_app, scrape_timeout_s=0.01)
        caplog.set_level(logging.ERROR, logger=common_module.logger.name)
        messages = await _call_asgi(app)
        assert _status(messages) == 504
        assert started.is_set()
        finish.set()
        assert await asyncio.to_thread(failed.wait, 1)

        deadline = asyncio.get_running_loop().time() + 1
        while not caplog.records and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.001)

        assert any(
            record.message == "Failed to generate Prometheus metrics"
            and record.exc_info is not None
            and record.exc_info[0] is RuntimeError
            for record in caplog.records
        )

    asyncio.run(run())
