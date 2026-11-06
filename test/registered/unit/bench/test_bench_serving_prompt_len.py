"""Unit tests for bench_serving response accounting."""

import asyncio
import copy
import json
import threading
import unittest
from argparse import Namespace
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.serving import (
    RequestFuncInput,
    async_request_openai_chat_completions,
    async_request_openai_completions,
    async_request_sglang_generate,
    calculate_metrics,
    set_global_args,
    wrap_multi_turn_request_func,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

# The dataset row's value; never what the stub reports.
ROW_PROMPT_LEN = 9999


class _ServingHandler(BaseHTTPRequestHandler):
    """Serve native events or chat responses with opt-in streaming usage."""

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.bodies.append(body)
        usage = {
            "completion_tokens": 1,
            "prompt_tokens": self.server.prompt_tokens[len(self.server.bodies) - 1],
        }
        if body.get("stream"):
            if self.path == "/generate":
                chunks = self.server.native_chunks
            else:
                choice = (
                    {"index": 0, "text": "hi"}
                    if self.path == "/v1/completions"
                    else {"index": 0, "delta": {"content": "hi"}}
                )
                chunks = [{"choices": [choice]}]
                if (body.get("stream_options") or {}).get(
                    "include_usage"
                ) and not self.server.omit_usage:
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


class _BenchServingHTTPTestCase(CustomTestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), _ServingHandler)
        self.server.omit_usage = False
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def _run(
        self, request_func, prompt, prompt_tokens, stream=True, extra=None, success=True
    ):
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
                temperature=0.0,
                top_p=1.0,
                return_logprob=False,
                return_routed_experts=False,
                logprob_start_len=-1,
                top_logprobs_num=0,
                token_ids_logprob=None,
            )
        )
        endpoint = {
            async_request_sglang_generate: "/generate",
            async_request_openai_completions: "/v1/completions",
        }.get(request_func, "/v1/chat/completions")
        request = RequestFuncInput(
            prompt=prompt,
            api_url=f"http://127.0.0.1:{self.server.server_port}{endpoint}",
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
            self.assertEqual(o.success, success, o.error)
        return outputs


class TestBenchServingNativeStream(_BenchServingHTTPTestCase):
    def test_native_streamed_failures_are_excluded_from_metrics(self):
        """HTTP 200 must not make an aborted stream count as successful work."""
        message = "Input length exceeds the maximum allowed length."
        normal = {
            "text": "hi",
            "meta_info": {"prompt_tokens": 7, "completion_tokens": 1},
        }
        self.server.native_chunks = [normal]
        good = self._run(async_request_sglang_generate, "hello", [7])[0]
        row = DatasetRow(prompt="hello", prompt_len=ROW_PROMPT_LEN, output_len=8)
        cases = [
            (
                "abort",
                {"meta_info": {"finish_reason": {"type": "abort", "message": message}}},
                message,
            ),
            (
                "abort without message",
                {"meta_info": {"finish_reason": {"type": "abort"}}},
                "Request aborted",
            ),
            ("error object", {"error": {"message": message}}, message),
            ("error string", {"error": message}, message),
            ("empty error", {"error": {}}, "Server returned an error"),
        ]
        for name, terminal, error in cases:
            for prefix in ([], [normal]):
                with self.subTest(name=name, partial_output=bool(prefix)):
                    self.server.native_chunks = prefix + [terminal]
                    failed = self._run(
                        async_request_sglang_generate, "hello", [7], success=False
                    )[0]
                    self.assertIn(error, failed.error)
                    self.assertEqual(failed.output_len, 0)
                    metrics, output_lens = calculate_metrics(
                        input_requests=[row, row],
                        outputs=[good, failed],
                        dur_s=2.0,
                        tokenizer=SimpleNamespace(
                            encode=lambda text, **kwargs: list(text)
                        ),
                        backend="sglang",
                    )
                    self.assertEqual(metrics.completed, 1)
                    self.assertEqual(metrics.total_input, 7)
                    self.assertEqual(metrics.total_output, 1)
                    self.assertEqual(metrics.total_throughput, 4.0)
                    self.assertEqual(output_lens, [1, 0])

    def test_native_completion_count_does_not_require_visible_text(self):
        """An empty or metadata-only chunk can still report a token count."""
        for completion_tokens, prefix in ((0, []), (3, []), (3, ["hi"])):
            with self.subTest(completion_tokens=completion_tokens, prefix=prefix):
                self.server.native_chunks = [
                    {"text": text, "meta_info": {"completion_tokens": 1}}
                    for text in prefix
                ] + [
                    {"text": "", "meta_info": {"completion_tokens": completion_tokens}}
                ]
                output = self._run(async_request_sglang_generate, "hello", [7])[0]
                self.assertEqual(output.output_len, completion_tokens)
                self.assertEqual(output.generated_text, "".join(prefix))
                self.assertEqual(len(output.itl), max(0, completion_tokens - 1))

    def test_native_metadata_only_token_deltas_update_itl(self):
        self.server.native_chunks = [
            {"meta_info": {"prompt_tokens": 7}},
            {"text": "", "meta_info": {"completion_tokens": 0}},
            {"text": "a", "meta_info": {"completion_tokens": 1}},
            {"text": "abcdefgh", "meta_info": {"completion_tokens": 8}},
            {"meta_info": {"completion_tokens": 10}},
            {"text": "", "meta_info": {"completion_tokens": 10}},
        ]
        output = self._run(async_request_sglang_generate, "hello", [7])[0]
        self.assertEqual(output.output_len, 10)
        self.assertEqual(output.generated_text, "abcdefgh")
        self.assertEqual(len(output.itl), 9)
        self.assertGreater(output.ttft, 0.0)
        self.assertGreater(sum(output.itl), 0.0)


class TestBenchServingPromptLen(_BenchServingHTTPTestCase):
    def test_single_turn_stream_usage_defaults_and_fallback(self):
        cases = [
            None,
            {"stream_options": {}},
            {"stream_options": None},
            {"stream_options": {"continuous_usage_stats": False}},
            {"stream_options": {"include_usage": False}},
        ]
        for request_func in (
            async_request_openai_chat_completions,
            async_request_openai_completions,
        ):
            for extra in cases:
                with self.subTest(backend=request_func.__name__, extra=extra):
                    before = copy.deepcopy(extra)
                    output = self._run(request_func, "hello", [123], extra=extra)[0]
                    options = {
                        "include_usage": True,
                        **((extra or {}).get("stream_options") or {}),
                    }
                    self.assertEqual(self.server.bodies[0]["stream_options"], options)
                    self.assertEqual(extra, before)
                    self.assertEqual(
                        output.prompt_len,
                        123 if options["include_usage"] else ROW_PROMPT_LEN,
                    )
                    self.assertEqual(
                        output.prompt_len_source,
                        "server" if options["include_usage"] else "dataset",
                    )

            with self.subTest(backend=request_func.__name__, server_omits_usage=True):
                self.server.omit_usage = True
                output = self._run(request_func, "hello", [123])[0]
                self.assertEqual(
                    self.server.bodies[0]["stream_options"], {"include_usage": True}
                )
                self.assertEqual(output.prompt_len, ROW_PROMPT_LEN)
                self.assertEqual(output.prompt_len_source, "dataset")
                self.server.omit_usage = False

    def test_input_count_sources_follow_successful_requests(self):
        server = self._run(async_request_openai_chat_completions, "hello", [7])[0]
        fallback = self._run(
            async_request_openai_chat_completions,
            "hello",
            [7],
            extra={"stream_options": {"include_usage": False}},
        )[0]
        row = DatasetRow(prompt="hello", prompt_len=ROW_PROMPT_LEN, output_len=8)
        for second, expected_source, sources, total in (
            (fallback, "mixed", ["server", "dataset"], 7 + ROW_PROMPT_LEN),
            (replace(fallback, success=False), "server", ["server", None], 7),
        ):
            with self.subTest(source=expected_source):
                metrics, _ = calculate_metrics(
                    input_requests=[row, row],
                    outputs=[server, second],
                    dur_s=2.0,
                    tokenizer=SimpleNamespace(encode=lambda text, **kwargs: list(text)),
                    backend="sglang-oai-chat",
                )
                self.assertEqual(metrics.total_input, total)
                self.assertEqual(metrics.input_token_count_source, expected_source)
                self.assertEqual(metrics.input_token_count_sources, sources)

    def test_single_request(self):
        usage_on = {"stream_options": {"include_usage": True}}
        cases = [
            ("streaming with usage", True, usage_on, 123, 1),
            ("default streaming usage", True, None, 123, 1),
            ("non-streaming", False, None, 123, 1),
            (
                "no usage keeps row value",
                True,
                {"stream_options": {"include_usage": False}},
                ROW_PROMPT_LEN,
                8,
            ),
        ]
        for name, stream, extra, expected, output_tokens in cases:
            with self.subTest(name):
                outputs = self._run(
                    async_request_openai_chat_completions,
                    "hello",
                    [123],
                    stream=stream,
                    extra=extra,
                )
                self.assertEqual([o.prompt_len for o in outputs], [expected])
                # The response may report a truncated or differently tokenized
                # prompt. Summary throughput must agree with the recorded length.
                for vision_tokens in (0, 100):
                    with self.subTest(vision_tokens=vision_tokens):
                        row = DatasetRow(
                            prompt="hello",
                            prompt_len=ROW_PROMPT_LEN,
                            output_len=8,
                            text_prompt_len=ROW_PROMPT_LEN - vision_tokens,
                            vision_prompt_len=vision_tokens,
                        )
                        metrics, output_lens = calculate_metrics(
                            input_requests=[row, row],
                            outputs=[outputs[0], replace(outputs[0], success=False)],
                            dur_s=2.0,
                            tokenizer=SimpleNamespace(
                                encode=lambda text, **kwargs: list(text)
                            ),
                            backend="sglang-oai-chat",
                        )
                        self.assertEqual(metrics.completed, 1)
                        expected_total = ROW_PROMPT_LEN if vision_tokens else expected
                        self.assertEqual(metrics.total_input, expected_total)
                        self.assertEqual(
                            metrics.total_input_text,
                            ROW_PROMPT_LEN - vision_tokens
                            if vision_tokens
                            else expected,
                        )
                        self.assertEqual(metrics.total_input_vision, vision_tokens)
                        self.assertEqual(
                            metrics.total_input,
                            metrics.total_input_text + metrics.total_input_vision,
                        )
                        source = (
                            "dataset"
                            if vision_tokens or expected == ROW_PROMPT_LEN
                            else "server"
                        )
                        self.assertEqual(metrics.input_token_count_source, source)
                        self.assertEqual(
                            metrics.input_token_count_sources, [source, None]
                        )
                        self.assertEqual(output_lens, [output_tokens, 0])
                        self.assertEqual(metrics.input_throughput, expected_total / 2.0)
                        self.assertEqual(
                            metrics.total_throughput,
                            (expected_total + output_tokens) / 2.0,
                        )
                        self.assertEqual(
                            metrics.total_throughput_retokenized,
                            (expected_total + 2) / 2.0,
                        )

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
                outputs = self._run(
                    multi_turn, ["one", "two", "three"], per_round, extra=extra
                )
                self.assertEqual([o.prompt_len for o in outputs], expected)
                self.assertEqual(
                    [b["stream_options"] for b in self.server.bodies], [sent] * 3
                )


if __name__ == "__main__":
    unittest.main()
