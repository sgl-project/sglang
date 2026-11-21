"""Unit tests for bench_serving streaming with reasoning chunks.

Reasoning models (DeepSeek-R1, MiMo, Qwen3 reasoning, Kimi-K2, ...) stream their
chain-of-thought via fields such as OpenAI's `delta.reasoning_content` and
vLLM Kimi's `delta.reasoning`. Without explicit support, bench_serving only
inspects `delta.content` and silently reports zero TTFT / ITL and an empty
`generated_text`, which then retokenizes to 0 tokens even though the backend
completed real work.
"""

import asyncio
import io
import json
import socket
import threading
import time
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer

from aiohttp import web
from tqdm import tqdm

from sglang.benchmark.serving import (
    RequestFuncInput,
    async_request_openai_chat_completions,
    async_request_openai_completions,
    calculate_metrics,
    set_global_args,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=16, suite="base-a-test-cpu")


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


class _JSONHandler(BaseHTTPRequestHandler):
    response_body: dict = {}
    request_bodies: list = []

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler interface)
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.request_bodies.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.response_body).encode())
        self.wfile.flush()

    def log_message(self, fmt, *args):
        return


def _make_chunk(
    content=None, reasoning_content=None, reasoning=None, completion_tokens=None
):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    if reasoning is not None:
        delta["reasoning"] = reasoning
    chunk = {"choices": [{"index": 0, "delta": delta}]}
    if completion_tokens is not None:
        chunk["usage"] = {"completion_tokens": completion_tokens}
    return chunk


def _make_response(content=None, reasoning_content=None, completion_tokens=1):
    message = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return {
        "choices": [{"index": 0, "message": message, "finish_reason": "length"}],
        "usage": {"completion_tokens": completion_tokens},
    }


class _StrictStringTokenizer:
    def encode(self, text, add_special_tokens=False):
        if not isinstance(text, str):
            raise ValueError("text input must be of type `str`")
        return text.split()


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

    def test_vllm_kimi_reasoning_stream_populates_metrics(self):
        chunks = [
            _make_chunk(reasoning="Let "),
            _make_chunk(reasoning="me "),
            _make_chunk(reasoning="think."),
            _make_chunk(completion_tokens=3),
        ]
        out = self._run(chunks)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "Let me think.")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(len(out.itl), 2, msg="should record ITL for chunks 2..N")
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


class TestBenchServingReasoningNonStream(CustomTestCase):
    def _run(self, response_body):
        set_global_args(
            Namespace(
                disable_stream=True,
                disable_ignore_eos=False,
                print_requests=False,
                tokenizer="",
                header=None,
            )
        )
        port = _free_port()

        class Handler(_JSONHandler):
            pass

        Handler.response_body = response_body
        Handler.request_bodies = []
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
            return (
                asyncio.run(async_request_openai_chat_completions(req)),
                Handler.request_bodies,
            )
        finally:
            server.shutdown()
            server.server_close()

    def test_reasoning_only_non_stream_metrics_retokenize_text(self):
        out, request_bodies = self._run(
            _make_response(
                content=None,
                reasoning_content="Let me think.",
                completion_tokens=3,
            )
        )

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "Let me think.")
        self.assertEqual(out.output_len, 3)
        self.assertFalse(request_bodies[0]["stream"])

        metrics, output_lens = calculate_metrics(
            input_requests=None,
            outputs=[out],
            dur_s=1.0,
            tokenizer=_StrictStringTokenizer(),
            backend="sglang-oai-chat",
        )
        self.assertEqual(metrics.completed, 1)
        self.assertEqual(output_lens, [3])
        self.assertEqual(metrics.total_output_retokenized, 3)

    def test_reasoning_then_content_non_stream_accounts_both(self):
        out, _ = self._run(
            _make_response(
                content="answer",
                reasoning_content="thought ",
                completion_tokens=2,
            )
        )

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "thought answer")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(out.output_len, 2)

    def test_content_only_non_stream_unchanged(self):
        out, _ = self._run(_make_response(content="answer", completion_tokens=1))

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.generated_text, "answer")
        self.assertGreater(out.ttft, 0.0)
        self.assertEqual(out.output_len, 1)


class TestBenchServingIncompleteStream(CustomTestCase):
    """HTTP 200 with a clean HTTP EOF must not hide an unfinished SSE response."""

    @staticmethod
    def _chunk(kind, text="", index=0, **choice_fields):
        choice = {"delta": {"content": text}} if kind == "chat" else {"text": text}
        return {"choices": [{"index": index, **choice, **choice_fields}]}

    @staticmethod
    def _sse(chunks, done=False):
        body = b"".join(
            b"data: " + json.dumps(chunk, ensure_ascii=False).encode() + b"\n\n"
            for chunk in chunks
        )
        return body + (b"data: [DONE]\n\n" if done else b"")

    async def _request(
        self, kind, body, *, stream=True, extra=None, width=None, truncated=False
    ):
        # Exercise the real aiohttp client, including split UTF-8/SSE lines and
        # the distinction between clean application EOF and broken HTTP framing.
        received = []

        async def handler(request):
            received.append(await request.json())
            response = web.StreamResponse(
                headers={
                    "Content-Type": (
                        "text/event-stream"
                        if received[-1]["stream"]
                        else "application/json"
                    ),
                    "Content-Length": str(len(body) + int(truncated)),
                }
            )
            await response.prepare(request)
            step = width or max(len(body), 1)
            for offset in range(0, len(body), step):
                await response.write(body[offset : offset + step])
                await asyncio.sleep(0)
            if truncated:
                request.transport.close()
            else:
                await response.write_eof()
            return response

        app = web.Application()
        endpoint = "/v1/chat/completions" if kind == "chat" else "/v1/completions"
        app.router.add_post(endpoint, handler)
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            site = web.TCPSite(runner, "127.0.0.1", 0)
            await site.start()
            port = runner.addresses[0][1]
            set_global_args(
                Namespace(
                    disable_stream=not stream,
                    disable_ignore_eos=True,
                    return_logprob=False,
                    print_requests=False,
                    tokenizer="",
                    header=None,
                )
            )
            request = RequestFuncInput(
                prompt="hello",
                api_url=f"http://127.0.0.1:{port}{endpoint}",
                prompt_len=1,
                output_len=64,
                model="dummy-model",
                lora_name="",
                image_data=None,
                extra_request_body=extra or {},
            )
            request_func = (
                async_request_openai_chat_completions
                if kind == "chat"
                else async_request_openai_completions
            )
            with tqdm(total=1, file=io.StringIO()) as progress:
                output = await request_func(request, progress)
                self.assertEqual(progress.n, 1)
            self.assertEqual(received[0]["stream"], (extra or {}).get("stream", stream))
            return output
        finally:
            await runner.cleanup()

    def test_clean_eof_is_not_success(self):
        async def run():
            for kind in ("chat", "completions"):
                partial = self._chunk(kind, "你")
                empty = self._chunk(kind)
                if kind == "chat":
                    empty["choices"][0]["delta"] = {"role": "assistant"}
                cases = {
                    "empty body": b"",
                    "blank lines": b"\n\n",
                    "partial content": self._sse([partial]),
                    "role or empty delta": self._sse([empty]),
                    "null finish": self._sse([self._chunk(kind, finish_reason=None)]),
                    "empty finish": self._sse([self._chunk(kind, finish_reason="")]),
                    "provisional usage": self._sse(
                        [{**partial, "usage": {"completion_tokens": 1}}]
                    ),
                    "usage only": self._sse(
                        [{"choices": [], "usage": {"completion_tokens": 0}}]
                    ),
                }
                for name, body in cases.items():
                    for width in (1, None):
                        with self.subTest(kind=kind, case=name, width=width):
                            output = await self._request(kind, body, width=width)
                            self.assertFalse(output.success)
                            self.assertIn("Incomplete streaming response", output.error)

        asyncio.run(run())

    def test_terminal_markers_and_usage_tail(self):
        async def run():
            for kind in ("chat", "completions"):
                partial = self._chunk(kind, "你")
                finish = self._chunk(kind, finish_reason="stop")
                usage = {"choices": [], "usage": {"completion_tokens": 1}}
                cases = [
                    ([partial], True, "你", 64),
                    ([partial, finish, usage], False, "你", 1),
                    ([partial, finish, self._chunk(kind), usage], True, "你", 1),
                    ([{**finish, "usage": {"completion_tokens": 0}}], False, "", 0),
                ]
                if kind == "chat":
                    tool = self._chunk(kind, finish_reason="tool_calls")
                    tool["choices"][0]["delta"] = {
                        "tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]
                    }
                    cases.append(([tool, usage], False, "", 1))
                for chunks, done, text, tokens in cases:
                    with self.subTest(kind=kind, chunks=chunks, done=done):
                        output = await self._request(
                            kind, self._sse(chunks, done), width=1
                        )
                        self.assertTrue(output.success, output.error)
                        self.assertEqual(output.generated_text, text)
                        self.assertEqual(output.output_len, tokens)

        asyncio.run(run())

    def test_all_requested_and_observed_choices_must_finish(self):
        async def run():
            for kind in ("chat", "completions"):
                finished = self._chunk(kind, finish_reason="length")
                second = self._chunk(kind, "partial", index=1)
                second_finished = self._chunk(kind, index=1, finish_reason="stop")
                cases = [
                    ([finished], {"n": 2}, False),
                    ([finished, second], {"n": 2}, False),
                    ([finished, second], {}, False),
                    ([second_finished], {"n": 2}, False),
                    ([finished, second_finished], {"n": 2}, True),
                    # A trailing metadata delta must not clear earlier finish state.
                    ([finished, self._chunk(kind, finish_reason=None)], {}, True),
                    (
                        [{"choices": finished["choices"] + second["choices"]}],
                        {"n": 2},
                        False,
                    ),
                    (
                        [{"choices": finished["choices"] + second_finished["choices"]}],
                        {"n": 2},
                        True,
                    ),
                ]
                if kind == "completions":
                    cases.extend(
                        [
                            ([finished], {"prompt": ["one", "two"]}, False),
                            ([finished], {"prompt": [[1], [2]]}, False),
                            ([finished], {"prompt": [1, 2]}, True),
                            (
                                [finished, second_finished],
                                {"prompt": ["one", "two"]},
                                True,
                            ),
                        ]
                    )
                for chunks, extra, expected in cases:
                    with self.subTest(kind=kind, chunks=chunks, extra=extra):
                        output = await self._request(
                            kind, self._sse(chunks), extra=extra
                        )
                        self.assertEqual(output.success, expected, output.error)

        asyncio.run(run())

    def test_effective_stream_flag_and_non_stream_json(self):
        async def run():
            for kind in ("chat", "completions"):
                choice = (
                    {"message": {"content": "answer"}}
                    if kind == "chat"
                    else {"text": "answer"}
                )
                # A JSON response does not need an SSE marker or finish_reason.
                body = json.dumps(
                    {"choices": [choice], "usage": {"completion_tokens": 1}}
                ).encode()
                for stream, extra in ((False, {}), (True, {"stream": False})):
                    with self.subTest(kind=kind, stream=stream, extra=extra):
                        output = await self._request(
                            kind, body, stream=stream, extra=extra
                        )
                        self.assertTrue(output.success, output.error)
                        self.assertEqual(output.generated_text, "answer")
                        self.assertEqual(output.output_len, 1)
                with self.subTest(kind=kind, empty_non_stream_choices=True):
                    output = await self._request(
                        kind,
                        b'{"choices": [], "usage": {"completion_tokens": 0}}',
                        stream=False,
                    )
                    self.assertFalse(output.success)
                for done in (False, True):
                    with self.subTest(kind=kind, override_stream=True, done=done):
                        output = await self._request(
                            kind,
                            self._sse([self._chunk(kind, "partial")], done),
                            stream=False,
                            extra={"stream": True},
                        )
                        self.assertEqual(output.success, done, output.error)
                        if not done:
                            self.assertIn("Incomplete streaming response", output.error)

        asyncio.run(run())

    def test_broken_http_framing_stays_a_transport_error(self):
        async def run():
            for kind in ("chat", "completions"):
                for done in (False, True):
                    with self.subTest(kind=kind, done=done):
                        output = await self._request(
                            kind,
                            self._sse(
                                [self._chunk(kind, "partial", finish_reason="stop")],
                                done,
                            ),
                            truncated=True,
                        )
                        self.assertFalse(output.success)
                        self.assertIn("ClientPayloadError", output.error)
                        self.assertIn("ContentLengthError", output.error)

        asyncio.run(run())

    def test_incomplete_requests_are_excluded_from_metrics(self):
        async def run():
            for kind in ("chat", "completions"):
                partial = await self._request(
                    kind, self._sse([self._chunk(kind, "partial")])
                )
                complete = await self._request(
                    kind,
                    self._sse(
                        [
                            {
                                **self._chunk(kind, "answer", finish_reason="stop"),
                                "usage": {"completion_tokens": 1},
                            }
                        ]
                    ),
                )
                metrics, lengths = calculate_metrics(
                    input_requests=None,
                    outputs=[partial, complete],
                    dur_s=1.0,
                    tokenizer=_StrictStringTokenizer(),
                    backend="sglang-oai-chat" if kind == "chat" else "sglang-oai",
                )
                with self.subTest(kind=kind):
                    self.assertEqual(metrics.completed, 1)
                    self.assertEqual(metrics.total_output, 1)
                    self.assertEqual(lengths, [0, 1])

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
