"""Latency and throughput of the /v1/score endpoint."""

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.perf_bench_kit import (
    at_least,
    at_most,
    check_batch_scaling,
    check_perf,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_SCORE,
    CustomTestCase,
    run_score_benchmark,
    run_score_benchmark_multi,
)

register_cuda_ci(est_time=215, stage="extra-a", runner_config="1-gpu-large")
register_amd_ci(est_time=210, suite="stage-b-test-1-gpu-large-amd")


class TestScoreAPI(CustomTestCase):
    def test_score_api_latency_throughput(self):
        res = run_score_benchmark(
            model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST_SCORE,
            num_requests=1000,
            batch_size=10,
            other_server_args=[],
            need_warmup=True,
        )

        self.assertEqual(res["successful_requests"], res["total_requests"])
        check_perf(
            self,
            at_most("avg_latency_ms", res["avg_latency_ms"], 31, amd=60, unit="ms"),
            at_most("p95_latency_ms", res["p95_latency_ms"], 37, amd=65, unit="ms"),
            at_least("throughput", res["throughput"], 32, amd=16, unit="req/s"),
        )

    def test_score_api_batch_scaling(self):
        check_batch_scaling(
            self,
            lambda batch_sizes: run_score_benchmark_multi(
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST_SCORE,
                batch_sizes,
                num_requests=500,
            ),
            # batch size, avg ms, p95 ms, then the same two relaxed for mi300x
            [(10, 32, 40, 60, 65), (25, 37, 42, 70, 80), (50, 54, 64, 80, 90)],
        )


if __name__ == "__main__":
    unittest.main()
