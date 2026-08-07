"""Unit tests for diffusion server liveness and readiness endpoints.

`/liveness` reports HTTP availability independently of model warmup.
`/health` and `/health_generate` report readiness for inference traffic.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.runtime.entrypoints.http_server import (
    health,
    health_generate,
    liveness,
)


def _make_request(warmup_done) -> SimpleNamespace:
    state = SimpleNamespace(server_warmup_done=warmup_done)
    return SimpleNamespace(app=SimpleNamespace(state=state))


class TestHealthWarmupGate(unittest.IsolatedAsyncioTestCase):
    async def test_liveness_returns_200_before_warmup(self):
        self.assertEqual(await liveness(), {"status": "ok"})

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


class _FakeResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _FakeAsyncClient:
    def __init__(self, status_codes: list[int]):
        self._status_codes = iter(status_codes)
        self.get_calls = 0
        self.urls = []

    def __call__(self, *args, **kwargs):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def get(self, url, timeout=None):
        self.get_calls += 1
        self.urls.append(url)
        return _FakeResponse(next(self._status_codes))


class TestWaitUntilHttpLive(unittest.IsolatedAsyncioTestCase):
    async def test_waits_for_liveness_200(self):
        fake_client = _FakeAsyncClient([503, 200])
        server_args = SimpleNamespace(url=lambda: "http://127.0.0.1:11000")
        with (
            mock.patch.object(http_server.httpx, "AsyncClient", fake_client),
            mock.patch.object(http_server.asyncio, "sleep", mock.AsyncMock()),
        ):
            await asyncio.wait_for(
                http_server._wait_until_http_live(server_args), timeout=5.0
            )
        self.assertEqual(fake_client.get_calls, 2)
        self.assertEqual(fake_client.urls, ["http://127.0.0.1:11000/liveness"] * 2)


if __name__ == "__main__":
    unittest.main()
