"""Unit tests for peak output token throughput accounting in bench_serving.

Under speculative decoding (or non-streaming / batched SSE responses), each
streaming chunk can carry multiple output tokens, so `len(output.itl) + 1`
counts chunks rather than tokens. `calculate_metrics` must weight each chunk
by the request's mean tokens per chunk (`output.output_len / len(token_times)`)
so `max_output_tokens_per_s` measures tokens/s instead of chunks/s.
"""

import unittest

from sglang.benchmark.serving import RequestFuncOutput, calculate_metrics
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _WordTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


def _make_output(
    *,
    start_time: float,
    ttft: float,
    itl: list[float],
    output_len: int,
    success: bool = True,
) -> RequestFuncOutput:
    latency = ttft + sum(itl)
    return RequestFuncOutput(
        generated_text=" ".join(["tok"] * output_len) if success else "",
        prompt_len=16,
        output_len=output_len if success else 0,
        latency=latency if success else 0.0,
        ttft=ttft if success else 0.0,
        itl=list(itl) if success else [],
        success=success,
        error="" if success else "simulated failure",
        start_time=start_time,
    )


class TestBenchServingPeakOutputThroughput(CustomTestCase):
    def test_bundled_chunks_count_tokens_not_chunks(self):
        # 40 output tokens delivered in 10 chunks (4 tokens/chunk) evenly across
        # two 1-second buckets: 5 chunks in [0, 1) and 5 chunks in [1, 2).
        out = _make_output(
            start_time=10.0,
            ttft=0.1,
            itl=[0.2] * 9,
            output_len=40,
        )
        metrics, output_lens = calculate_metrics(
            input_requests=None,
            outputs=[out],
            dur_s=2.0,
            tokenizer=_WordTokenizer(),
            backend="sglang-oai",
        )
        self.assertEqual(output_lens, [40])
        self.assertAlmostEqual(metrics.output_throughput, 20.0)
        # Before the fix this reported 5.0 (chunks/s) instead of 20.0 (tok/s).
        self.assertAlmostEqual(metrics.max_output_tokens_per_s, 20.0)
        self.assertGreaterEqual(
            metrics.max_output_tokens_per_s, metrics.output_throughput
        )

    def test_one_token_per_chunk_unchanged(self):
        out = _make_output(
            start_time=0.0,
            ttft=0.1,
            itl=[0.1] * 4,
            output_len=5,
        )
        metrics, _ = calculate_metrics(
            input_requests=None,
            outputs=[out],
            dur_s=1.0,
            tokenizer=_WordTokenizer(),
            backend="sglang",
        )
        self.assertAlmostEqual(metrics.max_output_tokens_per_s, 5.0)

    def test_non_streaming_single_chunk_counts_all_tokens(self):
        out = _make_output(
            start_time=5.0,
            ttft=0.3,
            itl=[],
            output_len=12,
        )
        metrics, _ = calculate_metrics(
            input_requests=None,
            outputs=[out],
            dur_s=1.0,
            tokenizer=_WordTokenizer(),
            backend="sglang-oai-chat",
        )
        self.assertAlmostEqual(metrics.max_output_tokens_per_s, 12.0)

    def test_zero_output_tokens_contributes_zero(self):
        out = _make_output(
            start_time=1.0,
            ttft=0.05,
            itl=[],
            output_len=0,
        )
        metrics, _ = calculate_metrics(
            input_requests=None,
            outputs=[out],
            dur_s=1.0,
            tokenizer=_WordTokenizer(),
            backend="sglang-oai-embed",
        )
        self.assertAlmostEqual(metrics.max_output_tokens_per_s, 0.0)

    def test_mixed_failed_and_concurrent_requests(self):
        failed = _make_output(
            start_time=0.0, ttft=0.0, itl=[], output_len=0, success=False
        )
        # 2 chunks in bucket 0, 2 chunks in bucket 1 -> 12 tokens/chunk => 24 tok/s
        req_a = _make_output(
            start_time=100.0,
            ttft=0.2,
            itl=[0.4, 0.5, 0.4],
            output_len=48,
        )
        # 2 chunks in bucket 1 only -> 5 tokens/chunk => 10 tok/s in bucket 1
        req_b = _make_output(
            start_time=101.0,
            ttft=0.1,
            itl=[0.2],
            output_len=10,
        )
        metrics, output_lens = calculate_metrics(
            input_requests=None,
            outputs=[failed, req_a, req_b],
            dur_s=2.0,
            tokenizer=_WordTokenizer(),
            backend="sglang-oai",
        )
        self.assertEqual(output_lens, [0, 48, 10])
        # Bucket 0: 24 tok/s (req_a); Bucket 1: 24 + 10 = 34 tok/s (req_a + req_b).
        self.assertAlmostEqual(metrics.max_output_tokens_per_s, 34.0)


if __name__ == "__main__":
    unittest.main()
