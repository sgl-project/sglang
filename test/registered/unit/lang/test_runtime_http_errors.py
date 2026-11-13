import asyncio
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer

from sglang.lang.backend.runtime_endpoint import Runtime
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@asynccontextmanager
async def runtime_server(handler):
    app = web.Application()
    app.router.add_post("/generate", handler)
    async with TestServer(app) as server:
        runtime = Runtime.__new__(Runtime)
        runtime.pid = None
        runtime.server_args = SimpleNamespace(skip_tokenizer_init=False)
        runtime.generate_url = str(server.make_url("/generate"))
        yield runtime


class TestRuntimeHttpErrors(CustomTestCase):
    def test_http_errors_are_not_empty_generations(self):
        """Rejected streaming requests must raise with their HTTP status."""

        async def check(status, generate):
            async def handler(request):
                body = await request.json()
                if status == 503:
                    return web.Response(status=status, text="Service unavailable")
                return web.json_response({"error": body["text"]}, status=status)

            async with runtime_server(handler) as runtime:
                with self.assertRaises(aiohttp.ClientResponseError) as raised:
                    async for _ in generate(runtime, "private prompt"):
                        self.fail("HTTP error response yielded a generation")
                self.assertEqual(raised.exception.status, status)
                self.assertNotIn("private prompt", str(raised.exception))

        for status in (400, 422, 503):
            for method_name, generate in (
                ("async_generate", Runtime.async_generate),
                ("add_request", Runtime.add_request),
            ):
                with self.subTest(status=status, method=method_name):
                    asyncio.run(check(status, generate))

    def test_successful_stream_preserves_text_deltas(self):
        async def check():
            async def handler(request):
                response = web.StreamResponse(
                    headers={"Content-Type": "text/event-stream"}
                )
                await response.prepare(request)
                await response.write(b'data: {"text": "hello"}\n\n')
                await response.write(b'data: {"text": "hello world"}\n\n')
                await response.write(b"data: [DONE]\n\n")
                return response

            async with runtime_server(handler) as runtime:
                self.assertEqual(
                    [part async for part in runtime.async_generate("prompt")],
                    ["hello", " world"],
                )

        asyncio.run(check())

    def test_successful_response_preserves_error_events(self):
        async def check():
            async def handler(request):
                response = web.StreamResponse(
                    headers={"Content-Type": "text/event-stream"}
                )
                await response.prepare(request)
                await response.write(
                    b'data: {"error": {"message": "invalid input"}}\n\n'
                )
                await response.write(b"data: [DONE]\n\n")
                return response

            async with runtime_server(handler) as runtime:
                self.assertEqual(
                    [part async for part in runtime.async_generate("prompt")],
                    [{"error": {"message": "invalid input"}}],
                )

        asyncio.run(check())


if __name__ == "__main__":
    unittest.main()
