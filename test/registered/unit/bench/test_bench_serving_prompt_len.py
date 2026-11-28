"""Unit tests for prompt length and usage accounting in bench_serving."""

import copy
import unittest
from dataclasses import replace
from types import SimpleNamespace

from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.serving import (
    async_request_openai_chat_completions,
    async_request_openai_completions,
    calculate_metrics,
    wrap_multi_turn_request_func,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.bench_serving_http import (
    ROW_PROMPT_LEN,
    _BenchServingHTTPTestCase,
)

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


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
