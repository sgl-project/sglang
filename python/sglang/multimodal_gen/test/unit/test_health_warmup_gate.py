"""Unit tests for diffusion server liveness and readiness endpoints.

`/liveness` reports HTTP availability independently of model warmup.
`/health` and `/health_generate` report readiness for inference traffic.
"""

import asyncio
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from urllib.error import URLError

import psutil
import zmq.asyncio
from fastapi import Response

from sglang.multimodal_gen.runtime.disaggregation.orchestrator import DiffusionServer
from sglang.multimodal_gen.runtime.entrypoints import http_server
from sglang.multimodal_gen.runtime.entrypoints.http_server import (
    health,
    health_generate,
    liveness,
)
from sglang.multimodal_gen.runtime.scheduler_client import AsyncSchedulerClient
from sglang.multimodal_gen.test.server.test_server_utils import ServerManager


def _make_request(warmup_done) -> SimpleNamespace:
    state = SimpleNamespace(
        server_warmup_done=warmup_done,
        worker_processes=[],
        disagg_server=None,
        broker_task=None,
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


class TestHealthWarmupGate(unittest.IsolatedAsyncioTestCase):
    async def test_router_exit_fails_pending_requests_without_rpc_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            endpoint = f"ipc://{directory}/frontend"
            router = DiffusionServer(
                frontend_endpoint=endpoint,
                encoder_work_endpoints=[],
                denoiser_work_endpoints=[f"ipc://{directory}/denoiser-work"],
                decoder_work_endpoints=[],
                encoder_result_endpoint=f"ipc://{directory}/encoder",
                denoiser_result_endpoint=f"ipc://{directory}/denoiser",
                decoder_result_endpoint=f"ipc://{directory}/decoder",
            )
            client = AsyncSchedulerClient()
            client.context = zmq.asyncio.Context()
            client.server_args = SimpleNamespace(scheduler_rpc_timeout=None)
            request = _make_request(asyncio.Event())
            request.app.state.server_warmup_done.set()
            request.app.state.disagg_server = router
            task = None
            try:
                router.start()
                self.assertTrue(await asyncio.to_thread(router.wait_ready, 5))
                with mock.patch.object(http_server, "async_scheduler_client", client):
                    self.assertEqual(await health(request), {"status": "ok"})
                    with mock.patch.object(
                        router,
                        "_handle_client_request",
                        side_effect=RuntimeError("injected routing failure"),
                    ):
                        task = asyncio.create_task(
                            client._forward_one(endpoint, {"method": "ping"}, None)
                        )
                        await asyncio.to_thread(router._thread.join, 5)
                    self.assertFalse(router._thread.is_alive())
                    self.assertEqual((await health(request)).status_code, 503)
                    await asyncio.wait_for(http_server._monitor_backend(request.app), 5)
                    with self.assertRaisesRegex(RuntimeError, "restart the server"):
                        await asyncio.wait_for(task, 5)
                    self.assertFalse(client._pending_requests)
                    self.assertEqual(await liveness(), {"status": "ok"})
            finally:
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                router.stop()
                client.close()

    async def test_exited_broker_is_unready(self):
        client = AsyncSchedulerClient()
        with zmq.Context() as context, context.socket(zmq.REP) as occupied:
            port = occupied.bind_to_random_port("tcp://127.0.0.1")
            args = SimpleNamespace(
                enable_metrics=False,
                warmup_mode="off",
                broker_port=port,
                prompt_enhancer_config=None,
                log_requests=False,
                log_requests_level=0,
                log_requests_format="text",
                log_requests_target=None,
                pipeline_config=mock.Mock(
                    supports_action_endpoint=lambda: False,
                    supports_openpi_endpoint=lambda: False,
                ),
            )
            with mock.patch.object(http_server, "async_scheduler_client", client):
                app = http_server.create_app(args)
                async with http_server.lifespan(app):
                    deadline = asyncio.get_running_loop().time() + 5
                    while client.failure_reason is None:
                        self.assertLess(asyncio.get_running_loop().time(), deadline)
                        await asyncio.sleep(0.01)
                    self.assertIn("broker", client.failure_reason)
                    async with http_server.httpx.AsyncClient(
                        transport=http_server.httpx.ASGITransport(app=app),
                        base_url="http://test",
                    ) as http:
                        self.assertEqual((await http.get("/health")).status_code, 503)
                        self.assertEqual((await http.get("/liveness")).status_code, 200)
                        self.assertEqual(
                            (await http.post("/v1/videos")).status_code, 503
                        )
                self.assertIsNone(client.context)

    async def test_worker_monitor_rejects_posts_but_preserves_completed_results(self):
        client = AsyncSchedulerClient()
        args = SimpleNamespace(
            enable_metrics=False,
            pipeline_config=mock.Mock(
                supports_action_endpoint=lambda: False,
                supports_openpi_endpoint=lambda: False,
            ),
        )
        with (
            subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"]
            ) as worker,
            mock.patch.object(http_server, "async_scheduler_client", client),
        ):
            app = http_server.create_app(args, worker_pids=(worker.pid,))
            app.state.server_warmup_done = asyncio.Event()
            app.state.server_warmup_done.set()
            app.add_api_route("/completed", lambda: {"result": "saved"})
            worker.kill()
            worker.wait(timeout=5)
            await asyncio.wait_for(http_server._monitor_backend(app), timeout=5)
            async with http_server.httpx.AsyncClient(
                transport=http_server.httpx.ASGITransport(app=app),
                base_url="http://test",
            ) as http:
                self.assertEqual((await http.get("/health")).status_code, 503)
                self.assertEqual((await http.get("/liveness")).status_code, 200)
                self.assertEqual((await http.post("/v1/videos")).status_code, 503)
                self.assertEqual(
                    (await http.get("/completed")).json(), {"result": "saved"}
                )

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

    @unittest.skipUnless(sys.platform == "linux", "requires waitid WNOWAIT")
    async def test_exited_worker_is_unready_before_and_after_reaping(self):
        """A dead replica must not remain ready just because warmup completed."""
        warmup_done = asyncio.Event()
        warmup_done.set()
        request = _make_request(warmup_done)
        with subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        ) as worker:
            # Another live replica must not hide the dead one.
            request.app.state.worker_processes = [
                psutil.Process(os.getpid()),
                psutil.Process(worker.pid),
            ]
            self.assertEqual(await health(request), {"status": "ok"})
            worker.kill()
            await asyncio.to_thread(
                os.waitid, os.P_PID, worker.pid, os.WEXITED | os.WNOWAIT
            )
            for reaped in (False, True):
                if reaped:
                    worker.wait(timeout=5)
                for endpoint in (health, health_generate):
                    response = await endpoint(request)
                    self.assertIsInstance(response, Response)
                    self.assertEqual(response.status_code, 503)
                self.assertEqual(await liveness(), {"status": "ok"})


class TestRouteModelPrebuild(unittest.TestCase):
    def _app(self):
        from fastapi import APIRouter, FastAPI
        from pydantic import BaseModel

        class Body(BaseModel):
            value: int

        router = APIRouter()

        @router.post("/thing")
        async def thing(body: Body) -> Body:
            return body

        app = FastAPI()
        app.include_router(router)
        return app

    def test_prebuild_resolves_route_models_before_the_first_request(self):
        app = self._app()
        self.assertIsNone(app.openapi_schema)
        http_server.build_route_request_models(app)
        self.assertIn("/thing", app.openapi_schema["paths"])

    def test_prebuild_failure_does_not_break_startup(self):
        app = self._app()
        with mock.patch.object(app, "openapi", side_effect=RuntimeError("bad schema")):
            http_server.build_route_request_models(app)
        self.assertIsNone(app.openapi_schema)


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
