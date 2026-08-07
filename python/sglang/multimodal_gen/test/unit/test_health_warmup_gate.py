"""Unit tests for the /health and /health_generate warmup-awareness fix.

Regression coverage for: `/health` and `/health_generate` used to always
return 200 regardless of `app.state.server_warmup_done`, so orchestrators
saw a green health check while the backend scheduler was still mid-warmup
and real requests were blocked behind `wait_for_server_warmup`. Both
endpoints now check the warmup event directly and report 503 until it
fires.

Plus, `_wait_until_http_ready` probes /health inside the warmup task
itself, so it must treat 503 as "HTTP is up" to prevent warmup from
deadlocking on health check.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.runtime.entrypoints.http_server import (
    health,
    health_generate,
)


def _make_request(warmup_done=None) -> SimpleNamespace:
    state = (
        SimpleNamespace()
        if warmup_done is None
        else SimpleNamespace(server_warmup_done=warmup_done)
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


class TestHealthWarmupGate(unittest.IsolatedAsyncioTestCase):
    async def test_health_returns_503_before_warmup(self):
        warmup_done = asyncio.Event()
        resp = await health(_make_request(warmup_done))
        self.assertEqual(resp.status_code, 503)

    async def test_health_returns_200_after_warmup(self):
        warmup_done = asyncio.Event()
        warmup_done.set()
        resp = await health(_make_request(warmup_done))
        self.assertEqual(resp, {"status": "ok"})

    async def test_health_generate_returns_503_before_warmup(self):
        warmup_done = asyncio.Event()
        resp = await health_generate(_make_request(warmup_done))
        self.assertEqual(resp.status_code, 503)

    async def test_health_generate_returns_200_after_warmup(self):
        warmup_done = asyncio.Event()
        warmup_done.set()
        resp = await health_generate(_make_request(warmup_done))
        self.assertEqual(resp, {"status": "ok"})

    async def test_health_returns_200_when_state_missing(self):
        # No server_warmup_done on app.state at all -- should not raise, and
        # should behave like "ready".
        resp = await health(_make_request())
        self.assertEqual(resp, {"status": "ok"})


class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _FakeAsyncClient:
    """Stands in for `httpx.AsyncClient` inside `_wait_until_http_ready`."""

    def __init__(self, status_code: int):
        self._status_code = status_code
        self.get_calls = 0

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, timeout=None):
        self.get_calls += 1
        return _FakeResponse(self._status_code)


class TestWaitUntilHttpReady(unittest.IsolatedAsyncioTestCase):
    async def test_503_counts_as_http_ready(self):
        # _wait_until_http_ready now accepts status code 503 to prevent the
        # warmup task, which awaits this probe, from deadlocking on itself.
        fake_client = _FakeAsyncClient(503)
        server_args = SimpleNamespace(url=lambda: "http://127.0.0.1:11000")
        with mock.patch.object(http_server.httpx, "AsyncClient", fake_client):
            await asyncio.wait_for(
                http_server._wait_until_http_ready(server_args), timeout=5.0
            )
        self.assertEqual(fake_client.get_calls, 1)

    async def test_200_counts_as_http_ready(self):
        fake_client = _FakeAsyncClient(200)
        server_args = SimpleNamespace(url=lambda: "http://127.0.0.1:11000")
        with mock.patch.object(http_server.httpx, "AsyncClient", fake_client):
            await asyncio.wait_for(
                http_server._wait_until_http_ready(server_args), timeout=5.0
            )
        self.assertEqual(fake_client.get_calls, 1)


if __name__ == "__main__":
    unittest.main()
