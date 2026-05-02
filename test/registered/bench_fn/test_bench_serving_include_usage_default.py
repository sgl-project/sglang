"""Unit tests: bench_serving defaults stream_options.include_usage=true.

Without ``include_usage`` in the request payload, OpenAI-compatible servers
(including sglang and vLLM) never emit a trailing ``usage`` chunk on
streaming responses. As a result, bench_serving's
``async_request_openai_completions`` and
``async_request_openai_chat_completions`` could never observe
``completion_tokens`` from the server and silently fell back to the
requested ``max_tokens`` for ``output_len`` — poisoning TPOT, output
token throughput, and retokenized-length metrics whenever generation
stopped before ``max_tokens``.

This change sets ``stream_options.include_usage=true`` by default for both
OpenAI-compatible request builders (streaming only), while letting users
override via ``--extra-request-body``.
"""

import asyncio
import json
import threading
import time
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer

from sglang.bench_serving import (
    RequestFuncInput,
    async_request_openai_chat_completions,
    async_request_openai_completions,
    set_global_args,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


class _PayloadCaptureHandler(BaseHTTPRequestHandler):
    """Captures the POST JSON body, then streams a trivial content chunk.

    The stream intentionally does NOT emit a ``usage`` chunk even though the
    caller will now be requesting one — this isolates the test to "did the
    caller SET the flag?" and does not cross-test the response parser. That
    parser fix is covered by the sibling test suites.
    """

    captured_bodies: list = []
    mode: str = "completions"  # or "chat"

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        try:
            self.captured_bodies.append(json.loads(raw.decode("utf-8")))
        except Exception:
            self.captured_bodies.append({"__raw__": raw.decode("utf-8", "replace")})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        if self.mode == "completions":
            chunks = [
                {"choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}]},
            ]
        else:
            chunks = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            ]
        for c in chunks:
            self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(0.01)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, fmt, *args):  # silence access logs
        return


class TestBenchServingIncludeUsageDefault(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        set_global_args(
            Namespace(
                disable_stream=False,
                disable_ignore_eos=True,
                print_requests=False,
                tokenizer="",
                header=None,
                return_logprob=False,
                top_logprobs_num=0,
            )
        )

    def _serve(self, mode: str):
        class Handler(_PayloadCaptureHandler):
            pass

        Handler.captured_bodies = []
        Handler.mode = mode
        # Bind to port 0 so the kernel picks an available port atomically;
        # avoids a probe-then-bind race where another process could grab it
        # between _free_port() and HTTPServer().
        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread, port, Handler

    def _req(self, port: int, path: str, extra: dict):
        return RequestFuncInput(
            prompt="hi",
            api_url=f"http://127.0.0.1:{port}{path}",
            prompt_len=1,
            output_len=64,
            model="dummy",
            lora_name="",
            image_data=None,
            extra_request_body=extra,
        )

    # --- completions ---

    def test_completions_stream_default_sets_include_usage(self):
        server, _, port, Handler = self._serve("completions")
        try:
            asyncio.run(
                async_request_openai_completions(self._req(port, "/v1/completions", {}))
            )
        finally:
            server.shutdown()
            server.server_close()

        body = Handler.captured_bodies[0]
        self.assertIn("stream_options", body)
        self.assertEqual(body["stream_options"], {"include_usage": True})

    def test_completions_user_override_wins(self):
        server, _, port, Handler = self._serve("completions")
        try:
            asyncio.run(
                async_request_openai_completions(
                    self._req(
                        port,
                        "/v1/completions",
                        {"stream_options": {"include_usage": False}},
                    )
                )
            )
        finally:
            server.shutdown()
            server.server_close()

        body = Handler.captured_bodies[0]
        self.assertEqual(body["stream_options"], {"include_usage": False})

    # --- chat completions ---

    def test_chat_stream_default_sets_include_usage(self):
        server, _, port, Handler = self._serve("chat")
        try:
            asyncio.run(
                async_request_openai_chat_completions(
                    self._req(port, "/v1/chat/completions", {})
                )
            )
        finally:
            server.shutdown()
            server.server_close()

        body = Handler.captured_bodies[0]
        self.assertIn("stream_options", body)
        self.assertEqual(body["stream_options"], {"include_usage": True})

    def test_chat_user_override_wins(self):
        server, _, port, Handler = self._serve("chat")
        try:
            asyncio.run(
                async_request_openai_chat_completions(
                    self._req(
                        port,
                        "/v1/chat/completions",
                        {"stream_options": {"include_usage": False}},
                    )
                )
            )
        finally:
            server.shutdown()
            server.server_close()

        body = Handler.captured_bodies[0]
        self.assertEqual(body["stream_options"], {"include_usage": False})


if __name__ == "__main__":
    unittest.main()
