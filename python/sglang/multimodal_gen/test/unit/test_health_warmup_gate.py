"""Unit tests for diffusion server liveness and readiness endpoints.

`/liveness` reports HTTP availability independently of model warmup.
`/health` and `/health_generate` report readiness for inference traffic.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.error import URLError

from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.runtime.entrypoints.http_server import (
    health,
    health_generate,
    liveness,
)
from sglang.multimodal_gen.test.server.test_server_utils import ServerManager


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


class _ReadyResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _RunningProcess:
    returncode = None

    def poll(self):
        return None


class TestServerManagerReadiness(unittest.TestCase):
    def test_waits_for_health_after_http_startup(self):
        manager = ServerManager("test-model", port=11000, wait_deadline=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            stdout_path = Path(temp_dir) / "server.log"
            stdout_path.write_text("Application startup complete.\n", encoding="utf-8")
            with (
                mock.patch(
                    "sglang.multimodal_gen.test.server.test_server_utils.urlopen",
                    side_effect=[URLError("warming up"), _ReadyResponse()],
                ) as health_request,
                mock.patch(
                    "sglang.multimodal_gen.test.server.test_server_utils.time.sleep"
                ),
            ):
                manager._wait_for_ready(_RunningProcess(), stdout_path)

        self.assertEqual(health_request.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in health_request.call_args_list],
            ["http://127.0.0.1:11000/health"] * 2,
        )

    def test_start_cleans_up_process_when_readiness_fails(self):
        manager = ServerManager("test-model", port=11000, wait_deadline=1)
        process = SimpleNamespace(pid=123, stdout=None)

        with tempfile.TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir)
            with (
                mock.patch(
                    "sglang.multimodal_gen.test.server.test_server_utils.prepare_perf_log",
                    return_value=(log_dir, log_dir / "perf.jsonl"),
                ),
                mock.patch(
                    "sglang.multimodal_gen.test.server.test_server_utils.subprocess.Popen",
                    return_value=process,
                ),
                mock.patch.object(
                    manager,
                    "_wait_for_ready",
                    side_effect=TimeoutError("startup timed out"),
                ),
                mock.patch(
                    "sglang.multimodal_gen.test.server.test_server_utils.kill_process_tree"
                ) as kill_process,
                mock.patch(
                    "sglang.multimodal_gen.test.server.test_server_utils.time.sleep"
                ),
            ):
                with self.assertRaisesRegex(TimeoutError, "startup timed out"):
                    manager.start()

        kill_process.assert_called_once_with(123)


if __name__ == "__main__":
    unittest.main()
