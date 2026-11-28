"""Unit tests for bench_serving metrics when every request fails."""

import unittest
import warnings

from sglang.benchmark.serving import RequestFuncOutput, calculate_metrics
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class TestBenchServingAllFailed(CustomTestCase):
    def test_all_failed_requests_report_zero_metrics_instead_of_raising(self):
        outputs = [
            RequestFuncOutput(success=False, error="Connection refused")
            for _ in range(4)
        ]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            metrics, output_lens = calculate_metrics(
                input_requests=None,
                outputs=outputs,
                dur_s=1.0,
                tokenizer=_WhitespaceTokenizer(),
                backend="sglang-oai",
            )
        self.assertTrue(any("All requests failed" in str(w.message) for w in caught))
        self.assertEqual(metrics.completed, 0)
        self.assertEqual(output_lens, [0, 0, 0, 0])
        for name in ("mean", "median", "std", "p90", "p95", "p99"):
            with self.subTest(stat=name):
                self.assertEqual(getattr(metrics, f"{name}_e2e_latency_ms"), 0.0)

    def test_successful_requests_still_report_e2e_latency(self):
        outputs = [
            RequestFuncOutput(
                success=True, generated_text="a b", ttft=0.1, latency=0.5, output_len=2
            ),
            RequestFuncOutput(success=False, error="Connection refused"),
        ]
        metrics, _ = calculate_metrics(
            input_requests=None,
            outputs=outputs,
            dur_s=1.0,
            tokenizer=_WhitespaceTokenizer(),
            backend="sglang-oai",
        )
        self.assertEqual(metrics.completed, 1)
        self.assertAlmostEqual(metrics.mean_e2e_latency_ms, 500.0)
        self.assertAlmostEqual(metrics.p99_e2e_latency_ms, 500.0)


if __name__ == "__main__":
    unittest.main()
