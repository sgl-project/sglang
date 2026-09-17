"""Unit tests for the prompt length bench_serving attributes to each request.

``RequestFuncOutput.prompt_len`` is seeded from the dataset row. For a
single-turn row that is correct, because one row is one request. A multi-turn
row is replayed as one request per round, and every round's output inherits the
row's single value -- so summing ``prompt_len`` across outputs counts one number
once per round instead of adding up each request's own prompt.

Two consumers divide by that sum: the ``--cache-report`` hit rate and
``input_lens`` in the JSON output. On an ``agentic-trace`` run of 16
conversations x 10 turns the denominator came out 6.5x too large, turning a real
83.8% cache hit rate into a reported 12.4%.

The fix prefers the length the server reports -- ``usage.prompt_tokens`` on the
OpenAI-compatible routes, ``meta_info.prompt_tokens`` on the native one -- which
is the same quantity for a single-turn row and therefore needs no per-dataset
branch.
"""

import asyncio
import json
import socket
import threading
import time
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer

from sglang.benchmark.serving import (
    RequestFuncInput,
    async_request_openai_chat_completions,
    set_global_args,
    wrap_multi_turn_request_func,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

# What the dataset row claims. Deliberately not equal to anything the stub
# server reports, so a test can only pass by reading the server's figure.
ROW_PROMPT_LEN = 9999


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _SSEHandler(BaseHTTPRequestHandler):
    """Streams one round's chunks, then advances to the next round's script."""

    rounds: list = []
    call_count: int = 0

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler interface)
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        chunks = type(self).rounds[min(type(self).call_count, len(type(self).rounds) - 1)]
        type(self).call_count += 1
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(0.01)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, fmt, *args):  # silence access logs
        return


class _JSONHandler(BaseHTTPRequestHandler):
    response_body: dict = {}

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(type(self).response_body).encode())
        self.wfile.flush()

    def log_message(self, fmt, *args):
        return


def _content_chunk(text):
    return {"choices": [{"index": 0, "delta": {"content": text}}]}


def _usage_chunk(prompt_tokens=None, completion_tokens=1):
    """A usage-only trailer, as OpenAI-compatible servers emit (choices=[])."""
    usage = {"completion_tokens": completion_tokens}
    if prompt_tokens is not None:
        usage["prompt_tokens"] = prompt_tokens
    return {"choices": [], "usage": usage}


class TestBenchServingPromptLen(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        set_global_args(
            Namespace(
                disable_stream=False,
                disable_ignore_eos=True,
                print_requests=False,
                tokenizer="",
                header=None,
                cache_report=False,
            )
        )

    def _serve(self, handler_cls):
        port = _free_port()
        server = HTTPServer(("127.0.0.1", port), handler_cls)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return port, server

    def _request(self, port, prompt="hello"):
        return RequestFuncInput(
            prompt=prompt,
            api_url=f"http://127.0.0.1:{port}/v1/chat/completions",
            prompt_len=ROW_PROMPT_LEN,
            output_len=64,
            model="dummy-model",
            lora_name="",
            image_data=None,
            extra_request_body={},
        )

    def test_streaming_prefers_server_prompt_tokens(self):
        class Handler(_SSEHandler):
            rounds = [[_content_chunk("hi"), _usage_chunk(prompt_tokens=123)]]
            call_count = 0

        port, server = self._serve(Handler)
        try:
            out = asyncio.run(async_request_openai_chat_completions(self._request(port)))
        finally:
            server.shutdown()
        self.assertTrue(out.success, out.error)
        self.assertEqual(out.prompt_len, 123)

    def test_streaming_without_usage_keeps_dataset_value(self):
        """No server figure, no change -- backends that report nothing regress nothing."""

        class Handler(_SSEHandler):
            rounds = [[_content_chunk("hi")]]
            call_count = 0

        port, server = self._serve(Handler)
        try:
            out = asyncio.run(async_request_openai_chat_completions(self._request(port)))
        finally:
            server.shutdown()
        self.assertTrue(out.success, out.error)
        self.assertEqual(out.prompt_len, ROW_PROMPT_LEN)

    def test_non_streaming_prefers_server_prompt_tokens(self):
        class Handler(_JSONHandler):
            response_body = {
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {"completion_tokens": 1, "prompt_tokens": 456},
            }

        set_global_args(
            Namespace(
                disable_stream=True,
                disable_ignore_eos=True,
                print_requests=False,
                tokenizer="",
                header=None,
                cache_report=False,
            )
        )
        try:
            port, server = self._serve(Handler)
            try:
                out = asyncio.run(
                    async_request_openai_chat_completions(self._request(port))
                )
            finally:
                server.shutdown()
        finally:
            self.setUpClass()
        self.assertTrue(out.success, out.error)
        self.assertEqual(out.prompt_len, 456)

    def test_multi_turn_sums_each_round_not_the_row_value(self):
        """The bug itself: three rounds of a growing conversation.

        Before the fix every round reported ROW_PROMPT_LEN, so the sum was
        3 x 9999. It must instead be the three lengths the server reported.
        """
        per_round = [100, 250, 400]

        class Handler(_SSEHandler):
            rounds = [
                [_content_chunk("reply"), _usage_chunk(prompt_tokens=n)]
                for n in per_round
            ]
            call_count = 0

        port, server = self._serve(Handler)
        try:
            multi_turn = wrap_multi_turn_request_func(
                async_request_openai_chat_completions, backend="sglang-oai-chat"
            )
            req = self._request(port, prompt=["turn one", "turn two", "turn three"])
            outputs = asyncio.run(multi_turn(req))
        finally:
            server.shutdown()

        self.assertEqual(len(outputs), len(per_round))
        self.assertTrue(all(o.success for o in outputs))
        self.assertEqual([o.prompt_len for o in outputs], per_round)
        self.assertEqual(sum(o.prompt_len for o in outputs), sum(per_round))
        self.assertNotEqual(
            sum(o.prompt_len for o in outputs), ROW_PROMPT_LEN * len(per_round)
        )


if __name__ == "__main__":
    unittest.main()
