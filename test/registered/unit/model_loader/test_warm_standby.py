# SPDX-License-Identifier: Apache-2.0
"""Warm standby routing, readiness, streaming, and failed-request semantics."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from sglang.srt.weight_cache.standby import StandbyGateway, forward_headers
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestWarmStandby(CustomTestCase):
    def make_gateway(self):
        return StandbyGateway(
            {
                "workers": [
                    {
                        "name": "a",
                        "url": "http://127.0.0.1:10001",
                        "command": ["python", "a"],
                    },
                    {
                        "name": "b",
                        "url": "http://127.0.0.1:10002",
                        "command": ["python", "b"],
                    },
                ]
            }
        )

    def test_readiness_and_failed_standby(self):
        gateway = self.make_gateway()
        a, b = gateway.workers
        self.assertIsNone(gateway.choose())
        a.state = "ready"
        self.assertIs(gateway.choose(), a)
        gateway.fail(a, "crash")
        self.assertIsNone(gateway.choose())
        b.state = "ready"
        self.assertIs(gateway.choose(), b)
        self.assertEqual(gateway.epoch, 2)
        gateway.fail(a, "late error")
        self.assertIs(gateway.choose(), b)

    def test_exited_launcher_is_not_routed(self):
        gateway = self.make_gateway()
        a, b = gateway.workers
        a.state = b.state = "ready"
        self.assertIs(gateway.choose(), a)
        a.process = SimpleNamespace(returncode=-9)
        self.assertIs(gateway.choose(), b)
        self.assertEqual(a.state, "failed")

    def test_configuration_validation(self):
        for workers in (
            [],
            [{"name": "a", "url": "http://example.com", "command": ["python"]}] * 2,
        ):
            with self.assertRaises(ValueError):
                StandbyGateway({"workers": workers})
        config = self.make_gateway().config
        config["warmup"] = {"stream": True}
        with self.assertRaises(ValueError):
            StandbyGateway(config)

    def test_hop_headers(self):
        self.assertEqual(
            forward_headers(
                {
                    "Connection": "custom",
                    "custom": "secret",
                    "Transfer-Encoding": "chunked",
                    "Authorization": "token",
                }
            ),
            {"Authorization": "token"},
        )

    def test_warmup_failure_never_ready(self):
        async def run():
            gateway = self.make_gateway()
            worker = gateway.workers[0]
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as directory:
                gateway.log_dir = Path(directory)
                response = SimpleNamespace(
                    status=500, read=AsyncMock(return_value=b"failed")
                )
                manager = AsyncMock()
                manager.__aenter__.return_value = response
                gateway.session = SimpleNamespace(post=lambda *a, **kw: manager)
                process = SimpleNamespace(pid=99999, returncode=None)

                async def stop(w):
                    w.process = None
                    w.log_file.close()
                    w.log_file = None

                with (
                    patch(
                        "asyncio.create_subprocess_exec",
                        AsyncMock(return_value=process),
                    ),
                    patch.object(gateway, "probe", AsyncMock(return_value=True)),
                    patch.object(gateway, "stop_worker", stop),
                ):
                    await gateway.prepare(worker)
                self.assertEqual(worker.state, "failed")
                self.assertIsNone(gateway.active)
                self.assertFalse(any(e["event"] == "ready" for e in gateway.events))

        asyncio.run(run())

    async def routing_fixture(self, operation):
        hits = {"a": 0, "b": 0}

        async def handler(request):
            name = request.app[name_key]
            hits[name] += 1
            if request.query.get("fail") == "1":
                request.transport.close()
                return web.Response()
            if request.query.get("stream") == "1":
                response = web.StreamResponse(
                    headers={"Content-Type": "text/event-stream"}
                )
                await response.prepare(request)
                await response.write(b'data: {"text":"hello"}\n\n')
                await asyncio.sleep(0.01)
                if request.query.get("truncate") == "1":
                    request.transport.close()
                    return response
                await response.write(b"data: [DONE]\n\n")
                return response
            return web.json_response({"text": name})

        name_key = web.AppKey("name", str)
        servers = []
        gateway = self.make_gateway()
        for worker in gateway.workers:
            app = web.Application()
            app[name_key] = worker.name
            app.router.add_route("*", "/{path:.*}", handler)
            server = TestServer(app)
            await server.start_server()
            worker.url = str(server.make_url("/")).rstrip("/")
            worker.state = "ready"
            worker.generation = 1
            servers.append(server)
        gateway.session = aiohttp.ClientSession(auto_decompress=False)
        app = gateway.app()
        app.on_startup.clear()
        app.on_cleanup.clear()
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            await operation(gateway, client, hits)
        finally:
            await client.close()
            await gateway.session.close()
            for server in servers:
                await server.close()

    def test_switch_and_stream(self):
        async def operation(gateway, client, hits):
            response = await client.post("/generate", json={"text": "hi"})
            self.assertEqual(await response.json(), {"text": "a"})
            self.assertEqual(response.headers["X-SGLang-Recovery-Worker"], "a")
            gateway.fail(gateway.workers[0], "crash")
            response = await client.post("/generate?stream=1", json={"text": "hi"})
            self.assertEqual(response.headers["X-SGLang-Recovery-Worker"], "b")
            self.assertEqual(
                await response.read(), b'data: {"text":"hello"}\n\ndata: [DONE]\n\n'
            )
            self.assertEqual(hits, {"a": 1, "b": 1})

        asyncio.run(self.routing_fixture(operation))

    def test_failed_post_is_not_replayed(self):
        async def operation(gateway, client, hits):
            response = await client.post("/generate?fail=1", json={"text": "hi"})
            self.assertEqual(response.status, 503)
            self.assertEqual(hits, {"a": 1, "b": 0})
            response = await client.post("/generate", json={"text": "next"})
            self.assertEqual(await response.json(), {"text": "b"})
            self.assertEqual(hits, {"a": 1, "b": 1})

        asyncio.run(self.routing_fixture(operation))

    def test_truncated_stream_is_not_replayed(self):
        async def operation(gateway, client, hits):
            response = await client.post(
                "/generate?stream=1&truncate=1", json={"text": "hi"}
            )
            with self.assertRaises(aiohttp.ClientPayloadError):
                await response.read()
            self.assertEqual(hits, {"a": 1, "b": 0})
            self.assertEqual(gateway.choose().name, "b")

        asyncio.run(self.routing_fixture(operation))

    def test_stateful_and_admin_requests_are_rejected(self):
        async def operation(gateway, client, hits):
            response = await client.post("/generate", json={"session_params": {}})
            self.assertEqual(response.status, 409)
            response = await client.post("/update_weights_from_disk", json={})
            self.assertEqual(response.status, 404)
            self.assertEqual(hits, {"a": 0, "b": 0})

        asyncio.run(self.routing_fixture(operation))


if __name__ == "__main__":
    unittest.main()
