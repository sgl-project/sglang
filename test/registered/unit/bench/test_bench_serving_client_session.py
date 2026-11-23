"""Unit tests for the client session that bench_serving requests share."""

import asyncio
import json
import threading
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sglang.benchmark.serving import (
    RequestFuncInput,
    _shared_bench_client_session,
    async_request_openai_completions,
    set_global_args,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _CompletionsHandler(BaseHTTPRequestHandler):
    """Streams one token, keeps the connection open, and sets a cookie."""

    protocol_version = "HTTP/1.1"

    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers["Content-Length"]))
        self.server.seen.append((self.client_address[1], self.headers.get("Cookie")))
        if self.server.barrier is not None:
            self.server.barrier.wait()  # all requests must be open at once
        chunk = {"choices": [{"index": 0, "text": "hi"}]}
        payload = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Set-Cookie", "route=a")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


class _Server(ThreadingHTTPServer):
    request_queue_size = 128  # room for every connection of the concurrency test


class TestBenchServingClientSession(CustomTestCase):
    def setUp(self):
        self.server = _Server(("127.0.0.1", 0), _CompletionsHandler)
        self.server.seen = []
        self.server.barrier = None
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        set_global_args(
            Namespace(
                disable_stream=False,
                disable_ignore_eos=True,
                print_requests=False,
                tokenizer="",
                header=None,
                cache_report=False,
                return_logprob=False,
            )
        )

    def _request(self):
        return RequestFuncInput(
            prompt="hello",
            api_url=f"http://127.0.0.1:{self.server.server_port}/v1/completions",
            prompt_len=1,
            output_len=1,
            model="dummy-model",
            lora_name="",
            image_data=None,
            extra_request_body={},
        )

    def _run(self, shared, n=3):
        async def run():
            for _ in range(n):
                out = await async_request_openai_completions(self._request())
                self.assertTrue(out.success, out.error)

        async def main():
            if shared:
                async with _shared_bench_client_session():
                    await run()
            else:
                await run()

        asyncio.run(main())
        return self.server.seen

    def test_benchmark_requests_reuse_one_connection(self):
        ports = {port for port, _ in self._run(shared=True)}
        self.assertEqual(len(ports), 1)

    def test_shared_session_sends_no_cookies(self):
        cookies = [cookie for _, cookie in self._run(shared=True)]
        self.assertEqual(cookies, [None, None, None])

    def test_shared_session_has_no_connection_limit(self):
        # aiohttp's default connector holds at most 100 connections.
        n = 101
        self.server.barrier = threading.Barrier(n, timeout=10)

        async def main():
            async with _shared_bench_client_session():
                outs = await asyncio.gather(
                    *(
                        async_request_openai_completions(self._request())
                        for _ in range(n)
                    )
                )
            for out in outs:
                self.assertTrue(out.success, out.error)

        asyncio.run(main())
        self.assertEqual(len({port for port, _ in self.server.seen}), n)

    def test_standalone_requests_open_their_own_connection(self):
        ports = {port for port, _ in self._run(shared=False)}
        self.assertEqual(len(ports), 3)


if __name__ == "__main__":
    unittest.main()
