import unittest

from sglang.benchmark.serving import RequestFuncOutput, calculate_metrics
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _UnexpectedTokenizer:
    def encode(self, text, add_special_tokens=False):
        raise AssertionError("failed requests must not be tokenized")


class TestBenchServingMetrics(CustomTestCase):
    def test_all_failed_requests_return_zero_e2e_metrics(self):
        outputs = [RequestFuncOutput(success=False, error="request failed")]

        with self.assertWarnsRegex(UserWarning, "All requests failed"):
            metrics, output_lens = calculate_metrics(
                input_requests=None,
                outputs=outputs,
                dur_s=1.0,
                tokenizer=_UnexpectedTokenizer(),
                backend="sglang",
            )

        self.assertEqual(output_lens, [0])
        self.assertEqual(metrics.completed, 0)
        self.assertEqual(metrics.mean_e2e_latency_ms, 0.0)
        self.assertEqual(metrics.median_e2e_latency_ms, 0.0)
        self.assertEqual(metrics.std_e2e_latency_ms, 0.0)
        self.assertEqual(metrics.p90_e2e_latency_ms, 0.0)
        self.assertEqual(metrics.p95_e2e_latency_ms, 0.0)
        self.assertEqual(metrics.p99_e2e_latency_ms, 0.0)
        self.assertEqual(metrics.concurrency, 0.0)


if __name__ == "__main__":
    unittest.main()
