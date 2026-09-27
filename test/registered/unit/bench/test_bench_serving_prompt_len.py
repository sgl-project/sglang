"""Unit tests for the per-request prompt length bench_serving records."""

import asyncio
import json
import threading
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

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# The dataset row's value; never what the stub reports.
ROW_PROMPT_LEN = 9999


class _ChatHandler(BaseHTTPRequestHandler):
    """Reports server.prompt_tokens[i] for the i-th request. Like a default SGLang
    server, a streamed response carries usage only when include_usage is set."""

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.bodies.append(body)
        usage = {
            "completion_tokens": 1,
            "prompt_tokens": self.server.prompt_tokens[len(self.server.bodies) - 1],
        }
        if body.get("stream"):
            chunks = [{"choices": [{"index": 0, "delta": {"content": "hi"}}]}]
            if (body.get("stream_options") or {}).get("include_usage"):
                chunks.append({"choices": [], "usage": usage})
            content_type = "text/event-stream"
            payload = b"".join(f"data: {json.dumps(c)}\n\n".encode() for c in chunks)
            payload += b"data: [DONE]\n\n"
        else:
            message = {"role": "assistant", "content": "hi"}
            choice = {"index": 0, "message": message, "finish_reason": "length"}
            content_type = "application/json"
            payload = json.dumps({"choices": [choice], "usage": usage}).encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass


class TestBenchServingPromptLen(CustomTestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), _ChatHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def _run(self, request_func, prompt, prompt_tokens, stream=True, extra=None):
        self.server.bodies = []
        self.server.prompt_tokens = prompt_tokens
        set_global_args(
            Namespace(
                disable_stream=not stream,
                disable_ignore_eos=True,
                print_requests=False,
                tokenizer="",
                header=None,
                cache_report=False,
            )
        )
        request = RequestFuncInput(
            prompt=prompt,
            api_url=f"http://127.0.0.1:{self.server.server_port}/v1/chat/completions",
            prompt_len=ROW_PROMPT_LEN,
            output_len=8,
            model="dummy-model",
            lora_name="",
            image_data=None,
            extra_request_body=extra or {},
        )
        out = asyncio.run(request_func(request))
        outputs = out if isinstance(out, list) else [out]
        for o in outputs:
            self.assertTrue(o.success, o.error)
        return [o.prompt_len for o in outputs]

    def test_single_request(self):
        usage_on = {"stream_options": {"include_usage": True}}
        cases = [
            ("streaming with usage", True, usage_on, 123),
            ("non-streaming", False, None, 123),
            ("no usage keeps row value", True, None, ROW_PROMPT_LEN),
        ]
        for name, stream, extra, expected in cases:
            with self.subTest(name):
                lens = self._run(
                    async_request_openai_chat_completions,
                    "hello",
                    [123],
                    stream=stream,
                    extra=extra,
                )
                self.assertEqual(lens, [expected])

    def test_multi_turn_requests_usage_per_round(self):
        per_round = [100, 250, 400]
        multi_turn = wrap_multi_turn_request_func(
            async_request_openai_chat_completions, backend="sglang-oai-chat"
        )
        on = {"include_usage": True}
        other = {"continuous_usage_stats": False}
        off = {"include_usage": False}
        # (user's extra_request_body, stream_options sent, prompt_len per round)
        cases = [
            (None, on, per_round),
            ({"stream_options": {}}, on, per_round),
            ({"stream_options": None}, on, per_round),
            ({"stream_options": other}, {**other, **on}, per_round),
            ({"stream_options": off}, off, [ROW_PROMPT_LEN] * 3),
        ]
        for extra, sent, expected in cases:
            with self.subTest(extra=extra):
                lens = self._run(
                    multi_turn, ["one", "two", "three"], per_round, extra=extra
                )
                self.assertEqual(lens, expected)
                self.assertEqual(
                    [b["stream_options"] for b in self.server.bodies], [sent] * 3
                )


if __name__ == "__main__":
    unittest.main()
