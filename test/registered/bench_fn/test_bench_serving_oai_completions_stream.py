"""Unit tests for bench_serving streaming with the /v1/completions backend.

OpenAI-compatible servers (sglang, vLLM with ``include_usage``) can emit a
trailing usage-only chunk where ``choices`` is either ``[]`` or
``[{"text": ""}]`` carrying the final ``completion_tokens``. Prior to this
change, ``async_request_openai_completions`` only updated ``output_len`` from
``usage`` inside ``if data["choices"][0]["text"]:``, so that trailing chunk
was skipped and ``output_len`` fell back to the request's ``max_tokens`` —
making TPOT, output throughput, and retokenized length metrics incorrect any
time generation stopped before ``max_tokens`` (EOS, length stop, abort).

The symmetric chat-completions path was already fixed by #23954; this suite
mirrors its structure for the completions path.
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
    async_request_openai_completions,
    set_global_args,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


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


def _make_chunk(text=None, completion_tokens=None, empty_choices=False):
    """Build an OpenAI /v1/completions stream chunk.

    - ``text``: populated in ``choices[0].text`` when not None.
    - ``completion_tokens``: attaches a trailing ``usage`` block.
    - ``empty_choices``: emit ``choices=[]`` instead of a text choice (some
      OpenAI-compatible servers do this for usage-only tail chunks).
    """
    chunk: dict = {}
    if empty_choices:
        chunk["choices"] = []
    else:
        choice: dict = {"index": 0, "finish_reason": None}
        if text is not None:
            choice["text"] = text
        chunk["choices"] = [choice]
    if completion_tokens is not None:
        chunk["usage"] = {"completion_tokens": completion_tokens}
    return chunk


class TestBenchServingOaiCompletionsStream(CustomTestCase):
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

    def _run(self, chunks, requested_output_len=64):
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
                api_url=f"http://127.0.0.1:{port}/v1/completions",
                prompt_len=1,
                output_len=requested_output_len,
                model="dummy-model",
                lora_name="",
                image_data=None,
                extra_request_body={},
            )
            return asyncio.run(async_request_openai_completions(req))
        finally:
            server.shutdown()
            server.server_close()

    # --- regressions the scope fix explicitly addresses ---

    def test_output_len_from_trailing_usage_chunk_with_empty_text(self):
        """sglang shape: tail chunk carries empty text + usage.

        The pre-fix code's `if data["choices"][0]["text"]:` evaluates False on
        the tail, so the usage update was skipped and `output_len` stayed at
        the requested `max_tokens` (64), not the real 3.
        """
        chunks = [
            _make_chunk(text="Hel"),
            _make_chunk(text="lo"),
            _make_chunk(text="!"),
            _make_chunk(text="", completion_tokens=3),  # tail: empty text + usage
        ]
        out = self._run(chunks, requested_output_len=64)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "Hello!")
        self.assertEqual(out.output_len, 3)
        self.assertGreater(out.ttft, 0.0)

    def test_output_len_from_trailing_usage_chunk_with_empty_choices(self):
        """vLLM (include_usage=true) shape: tail chunk has choices=[] + usage.

        The pre-fix code raised IndexError on `data["choices"][0]` — this is
        the regression reported in #5451. The fix guards with
        ``data.get("choices") or []`` and ``continue``.
        """
        chunks = [
            _make_chunk(text="Hi"),
            _make_chunk(text=" there"),
            _make_chunk(empty_choices=True, completion_tokens=2),
        ]
        out = self._run(chunks, requested_output_len=64)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "Hi there")
        self.assertEqual(out.output_len, 2)
        self.assertGreater(out.ttft, 0.0)

    def test_null_text_field_does_not_break(self):
        """Some backends serialize ``text`` as ``null`` rather than ``""``.

        Pre-fix, ``data["choices"][0]["text"]`` was None which fails the
        truthiness check without crashing; the fix explicitly normalizes via
        ``choices[0].get("text") or ""`` so the behavior is documented.
        """
        chunks = [
            {"choices": [{"index": 0, "text": None}]},
            _make_chunk(text="ok"),
            _make_chunk(text="", completion_tokens=1),
        ]
        out = self._run(chunks, requested_output_len=64)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "ok")
        self.assertEqual(out.output_len, 1)
        self.assertGreater(out.ttft, 0.0)

    # --- sanity: non-regression for the happy path ---

    def test_content_only_stream_unchanged(self):
        chunks = [
            _make_chunk(text="hi "),
            _make_chunk(text="there"),
            _make_chunk(text="", completion_tokens=2),
        ]
        out = self._run(chunks, requested_output_len=64)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "hi there")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(len(out.itl), 1)
        self.assertEqual(out.text_chunks, ["there"])
        self.assertEqual(out.output_len, 2)

    def test_no_usage_reported_keeps_request_output_len(self):
        """If the server never emits ``usage``, we must keep the request's
        ``max_tokens`` as a fallback (legacy behaviour)."""
        chunks = [
            _make_chunk(text="a"),
            _make_chunk(text="b"),
        ]
        out = self._run(chunks, requested_output_len=7)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "ab")
        self.assertEqual(out.output_len, 7)  # fell back to requested


if __name__ == "__main__":
    unittest.main()
