"""Unit tests for native bench_serving stream failures and token timing."""

import unittest
from types import SimpleNamespace

from sglang.benchmark.datasets.common import DatasetRow
from sglang.benchmark.serving import async_request_sglang_generate, calculate_metrics
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.bench_serving_http import (
    ROW_PROMPT_LEN,
    _BenchServingHTTPTestCase,
)

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


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


if __name__ == "__main__":
    unittest.main()
