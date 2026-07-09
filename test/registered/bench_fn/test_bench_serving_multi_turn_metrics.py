"""Unit tests for the multi-turn metrics contract in sglang.benchmark.serving:
per-round prompt lengths, ``usage.prompt_tokens`` parsing with client-side
fallback, multi-turn input accounting, and the cache-report denominator."""

import asyncio
import json
import socket
import threading
import time
import unittest
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, HTTPServer

from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.serving import (
    RequestFuncInput,
    RequestFuncOutput,
    aggregate_cache_report,
    async_request_openai_chat_completions,
    calculate_metrics,
    set_global_args,
    wrap_multi_turn_request_func,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _SSEHandler(BaseHTTPRequestHandler):
    """Streams a fixed sequence of OpenAI-compatible SSE chunks per test."""

    chunks: list = []
    request_bodies: list = []

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler interface)
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.request_bodies.append(json.loads(self.rfile.read(length)))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in self.chunks:
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(0.01)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, fmt, *args):  # silence access logs
        return


class _JSONHandler(BaseHTTPRequestHandler):
    response_body: dict = {}

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler interface)
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(self.response_body).encode())
        self.wfile.flush()

    def log_message(self, fmt, *args):
        return


class _StrictStringTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def _make_request_input(prompt, prompt_len=100, prompt_lens=None, api_url="http://x"):
    return RequestFuncInput(
        prompt=prompt,
        api_url=api_url,
        prompt_len=prompt_len,
        output_len=8,
        model="dummy-model",
        lora_name="",
        image_data=None,
        extra_request_body={},
        prompt_lens=prompt_lens,
    )


def _make_output(prompt_len, server_prompt_len=None, cached_tokens=0, success=True):
    out = RequestFuncOutput()
    out.success = success
    out.prompt_len = prompt_len
    out.server_prompt_len = server_prompt_len
    out.cached_tokens = cached_tokens
    out.output_len = 3
    out.generated_text = "a b c"
    out.latency = 1.0
    out.ttft = 0.5
    return out


class TestMultiTurnWrapperPromptLens(CustomTestCase):
    """Per-round prompt lengths flow through wrap_multi_turn_request_func."""

    def _run_wrapper(self, prompts, prompt_lens):
        calls = []

        async def fake_request_func(request_func_input, pbar=None):
            calls.append(request_func_input)
            out = RequestFuncOutput.init_new(request_func_input)
            out.success = True
            out.generated_text = f"reply{len(calls)}"
            return out

        wrapped = wrap_multi_turn_request_func(
            fake_request_func, backend="sglang-oai-chat"
        )
        req = _make_request_input(prompts, prompt_len=100, prompt_lens=prompt_lens)
        return asyncio.run(wrapped(req)), calls

    def test_per_round_prompt_lens_recorded(self):
        outputs, calls = self._run_wrapper(["q1", "q2", "q3"], [7, 11, 13])

        self.assertEqual([o.prompt_len for o in outputs], [7, 11, 13])
        # History still accumulates: round N sends 2N-1 messages
        # (N user rounds + N-1 real assistant replies).
        self.assertEqual([len(c.prompt) for c in calls], [1, 3, 5])
        self.assertEqual(calls[1].prompt[1], {"role": "assistant", "content": "reply1"})

    def test_mismatched_prompt_lens_rejected_before_any_request(self):
        for lens in ([7, 11], [7, 11, 13, 17]):
            with self.assertRaisesRegex(ValueError, "rounds"):
                self._run_wrapper(["q1", "q2", "q3"], lens)

    def test_no_prompt_lens_preserves_existing_behavior(self):
        outputs, _ = self._run_wrapper(["q1", "q2", "q3"], None)
        self.assertEqual([o.prompt_len for o in outputs], [100, 100, 100])


class TestChatUsagePromptTokens(CustomTestCase):
    """usage.prompt_tokens parsing on the OpenAI chat path."""

    def _run_against(self, handler_cls):
        """Serve one chat request against ``handler_cls`` and return the output."""
        port = _free_port()
        server = HTTPServer(("127.0.0.1", port), handler_cls)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            req = _make_request_input(
                "hello",
                prompt_len=9,
                api_url=f"http://127.0.0.1:{port}/v1/chat/completions",
            )
            return asyncio.run(async_request_openai_chat_completions(req))
        finally:
            server.shutdown()
            server.server_close()

    def _run_stream(self, chunks):
        set_global_args(
            Namespace(
                disable_stream=False,
                disable_ignore_eos=True,
                print_requests=False,
                tokenizer="",
                header=None,
            )
        )

        class Handler(_SSEHandler):
            pass

        Handler.chunks = list(chunks)
        Handler.request_bodies = []
        return self._run_against(Handler), Handler

    def _run_non_stream(self, response_body):
        set_global_args(
            Namespace(
                disable_stream=True,
                disable_ignore_eos=False,
                print_requests=False,
                tokenizer="",
                header=None,
            )
        )

        class Handler(_JSONHandler):
            pass

        Handler.response_body = response_body
        return self._run_against(Handler)

    def test_streaming_records_server_prompt_tokens_and_requests_usage(self):
        chunks = [
            {"choices": [{"index": 0, "delta": {"content": "hi"}}]},
            {"choices": [], "usage": {"completion_tokens": 1, "prompt_tokens": 57}},
        ]
        out, handler = self._run_stream(chunks)

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.server_prompt_len, 57)
        self.assertEqual(out.effective_prompt_len(), 57)
        self.assertEqual(out.output_len, 1)
        # Servers only emit usage on the streaming path when asked.
        self.assertEqual(
            handler.request_bodies[0]["stream_options"], {"include_usage": True}
        )

    def test_streaming_without_usage_falls_back_to_client_len(self):
        out, _ = self._run_stream(
            [{"choices": [{"index": 0, "delta": {"content": "hi"}}]}]
        )

        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertIsNone(out.server_prompt_len)
        self.assertEqual(out.effective_prompt_len(), 9)

    def test_non_streaming_usage_and_fallback(self):
        response = {
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "length",
                }
            ],
            "usage": {"completion_tokens": 1, "prompt_tokens": 41},
        }
        out = self._run_non_stream(response)
        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertEqual(out.effective_prompt_len(), 41)

        response["usage"] = None
        out = self._run_non_stream(response)
        self.assertTrue(out.success, msg=f"request failed: {out.error}")
        self.assertIsNone(out.server_prompt_len)
        self.assertEqual(out.effective_prompt_len(), 9)


class TestMultiTurnInputAccounting(CustomTestCase):
    """calculate_metrics derives multi-turn input from the outputs."""

    def test_multi_turn_total_input_prefers_server_values(self):
        outputs = [
            _make_output(prompt_len=100, server_prompt_len=120),
            _make_output(prompt_len=200, server_prompt_len=None),
            _make_output(prompt_len=999, success=False),
        ]
        metrics, _ = calculate_metrics(
            input_requests=None,
            outputs=outputs,
            dur_s=2.0,
            tokenizer=_StrictStringTokenizer(),
            backend="sglang-oai-chat",
        )

        self.assertEqual(metrics.total_input, 320)
        self.assertAlmostEqual(metrics.input_throughput, 160.0)

    def test_single_turn_accounting_unchanged(self):
        metrics, _ = calculate_metrics(
            input_requests=[DatasetRow(prompt="p", prompt_len=10, output_len=3)],
            outputs=[_make_output(prompt_len=999, server_prompt_len=888)],
            dur_s=1.0,
            tokenizer=_StrictStringTokenizer(),
            backend="sglang-oai-chat",
        )
        self.assertEqual(metrics.total_input, 10)


class TestCacheReportDenominator(CustomTestCase):
    def test_denominator_prefers_server_tokens_and_skips_failures(self):
        out_with_details = _make_output(
            prompt_len=100, server_prompt_len=120, cached_tokens=60
        )
        out_with_details.cached_tokens_details = {"device": 40, "host": 20}
        agg = aggregate_cache_report(
            [
                out_with_details,
                _make_output(prompt_len=200, cached_tokens=50),
                _make_output(prompt_len=500, cached_tokens=400, success=False),
            ]
        )

        self.assertEqual(agg["total_prompt_tokens"], 320)
        self.assertEqual(agg["total_cached"], 110)
        self.assertAlmostEqual(agg["hit_rate"], 110 / 320 * 100)
        self.assertTrue(agg["has_details"])
        self.assertEqual((agg["total_device"], agg["total_host"]), (40, 20))

        self.assertEqual(aggregate_cache_report([])["hit_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
