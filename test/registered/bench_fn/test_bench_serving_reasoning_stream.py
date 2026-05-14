"""Unit tests for bench_serving streaming with reasoning_content chunks.

Reasoning models (DeepSeek-R1, MiMo, Qwen3 reasoning, Kimi-K2, ...) stream their
chain-of-thought via OpenAI's `delta.reasoning_content` field. Without explicit
support, bench_serving only inspects `delta.content` and silently reports zero
TTFT / ITL and an empty `generated_text`, which then retokenizes to 0 tokens
even though the backend completed real work.
"""

import asyncio
import json
import socket
import threading
import time
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer

from sglang.bench_serving import (
    RequestFuncInput,
    async_request_openai_chat_completions,
    set_global_args,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="stage-a-test-cpu")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _SSEHandler(BaseHTTPRequestHandler):
    """Streams a fixed sequence of OpenAI-compatible SSE chunks per test."""

    chunks: list = []
    chunk_delay_s: float = 0.02

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler interface)
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk in self.chunks:
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(self.chunk_delay_s)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, fmt, *args):  # silence access logs
        return


def _make_chunk(content=None, reasoning_content=None, completion_tokens=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    chunk = {"choices": [{"index": 0, "delta": delta}]}
    if completion_tokens is not None:
        chunk["usage"] = {"completion_tokens": completion_tokens}
    return chunk


class TestBenchServingReasoningStream(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        set_global_args(
            Namespace(
                disable_stream=False,
                disable_ignore_eos=True,
                print_requests=False,
                tokenizer="",
                header=None,
            )
        )

    def _run(self, chunks):
        port = _free_port()

        class Handler(_SSEHandler):
            pass

        Handler.chunks = list(chunks)
        server = HTTPServer(("127.0.0.1", port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            req = RequestFuncInput(
                prompt="hello",
                api_url=f"http://127.0.0.1:{port}/v1/chat/completions",
                prompt_len=1,
                output_len=64,
                model="dummy-model",
                lora_name="",
                image_data=None,
                extra_request_body={},
            )
            return asyncio.run(async_request_openai_chat_completions(req))
        finally:
            server.shutdown()
            server.server_close()

    def test_reasoning_only_stream_populates_metrics(self):
        chunks = [
            _make_chunk(reasoning_content="Let "),
            _make_chunk(reasoning_content="me "),
            _make_chunk(reasoning_content="think."),
            _make_chunk(completion_tokens=3),
        ]
        out = self._run(chunks)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "Let me think.")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(len(out.itl), 2, msg="should record ITL for chunks 2..N")
        for v in out.itl:
            self.assertGreater(v, 0.0)
        self.assertEqual(out.text_chunks, ["me ", "think."])
        self.assertEqual(out.output_len, 3)

    def test_reasoning_then_content_accounts_both(self):
        chunks = [
            _make_chunk(reasoning_content="step1 "),
            _make_chunk(reasoning_content="step2 "),
            _make_chunk(content="answer "),
            _make_chunk(content="here"),
            _make_chunk(completion_tokens=4),
        ]
        out = self._run(chunks)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "step1 step2 answer here")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(len(out.itl), 3)
        self.assertEqual(out.text_chunks, ["step2 ", "answer ", "here"])
        self.assertEqual(out.output_len, 4)

    def test_single_delta_preserves_reasoning_before_content(self):
        chunks = [
            _make_chunk(content="answer", reasoning_content="thought "),
            _make_chunk(completion_tokens=2),
        ]
        out = self._run(chunks)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "thought answer")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(out.output_len, 2)

    def test_usage_only_stream_chunk_does_not_break(self):
        chunks = [
            _make_chunk(reasoning_content="thinking"),
            {"choices": [], "usage": {"completion_tokens": 1}},
        ]
        out = self._run(chunks)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "thinking")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(out.output_len, 1)

    def test_content_only_stream_unchanged(self):
        chunks = [
            _make_chunk(content="hi "),
            _make_chunk(content="there"),
            _make_chunk(completion_tokens=2),
        ]
        out = self._run(chunks)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "hi there")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(len(out.itl), 1)
        self.assertEqual(out.text_chunks, ["there"])
        self.assertEqual(out.output_len, 2)

    def test_null_reasoning_field_does_not_break(self):
        # Mirrors sglang's _StreamDelta: reasoning_content is always emitted,
        # serialized as null when only content is present.
        chunks = [
            {
                "choices": [
                    {"index": 0, "delta": {"content": "ok", "reasoning_content": None}}
                ]
            },
            {
                "choices": [
                    {"index": 0, "delta": {"content": None, "reasoning_content": None}}
                ],
                "usage": {"completion_tokens": 1},
            },
        ]
        out = self._run(chunks)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "ok")
        self.assertGreater(out.ttft, 0.0)


if __name__ == "__main__":
    unittest.main()
