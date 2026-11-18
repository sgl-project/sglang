import unittest
from types import SimpleNamespace

from sglang.benchmark.serving import RequestFuncOutput, calculate_metrics
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestAllFailedBenchmarkMetrics(CustomTestCase):
    def test_all_failed_requests_still_produce_a_metric_report(self):
        """A failed benchmark must retain its zero-success report for diagnosis."""
        tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: [])
        with self.assertWarns(UserWarning):
            metrics, lengths = calculate_metrics(
                None, [RequestFuncOutput(success=False)], 1.0, tokenizer, "sglang"
            )
        self.assertEqual(metrics.completed, 0)
        self.assertEqual(metrics.request_throughput, 0)
        self.assertEqual(lengths, [0])
        self.assertEqual(metrics.p99_e2e_latency_ms, 0)


if __name__ == "__main__":
    unittest.main()
