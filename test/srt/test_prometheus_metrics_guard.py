import asyncio
import sys
import threading

import pytest

if sys.platform == "win32":
    pytest.skip("sglang imports resource on Unix platforms", allow_module_level=True)

from sglang.srt.utils.common import _make_guarded_metrics_app


async def _call_asgi(app):
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await app(
        {
            "type": "http",
            "method": "GET",
            "path": "/metrics",
            "headers": [],
            "query_string": b"",
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


def test_metrics_guard_rejects_concurrent_scrape():
    async def run():
        started = threading.Event()
        finish = threading.Event()

        def slow_generate():
            started.set()
            finish.wait(timeout=1)
            return b"ok\n"

        app = _make_guarded_metrics_app(
            slow_generate, "text/plain", scrape_timeout_s=1.0
        )
        first = asyncio.create_task(_call_asgi(app))
        await asyncio.to_thread(started.wait)

        second_messages = await _call_asgi(app)
        finish.set()
        first_messages = await first

        assert _status(second_messages) == 503
        assert _status(first_messages) == 200

    asyncio.run(run())


def test_metrics_guard_times_out_slow_scrape():
    async def run():
        finish = threading.Event()

        def stuck_generate():
            finish.wait(timeout=1)
            return b"late\n"

        app = _make_guarded_metrics_app(
            stuck_generate, "text/plain", scrape_timeout_s=0.01
        )
        messages = await _call_asgi(app)
        finish.set()

        assert _status(messages) == 504

    asyncio.run(run())
