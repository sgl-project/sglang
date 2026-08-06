"""Unit tests for the /health and /health_generate warmup-awareness fix.

Regression coverage for: `/health` and `/health_generate` used to always
return 200 regardless of `app.state.server_warmup_done`, so orchestrators
saw a green health check while the backend scheduler was still mid-warmup
and real requests were blocked behind `wait_for_server_warmup`. Both
endpoints now check the warmup event directly and report 503 until it
fires.

All tests are CPU-only; no model loading, no distributed init, no HTTP
server -- the endpoint functions are called directly with a minimal fake
`Request` shaped just enough for `request.app.state.server_warmup_done`.
"""

import asyncio
import unittest
from types import SimpleNamespace

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
        # No server_warmup_done set at all (e.g. warmup disabled) -- should
        # not raise, and should behave like "ready".
        resp = await health(_make_request())
        self.assertEqual(resp, {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
