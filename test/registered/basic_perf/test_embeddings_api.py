"""Latency and throughput of the /v1/embeddings endpoint."""

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.perf_bench_kit import (
    at_least,
    at_most,
    check_batch_scaling,
    check_perf,
)
from sglang.test.test_utils import (
    DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    run_embeddings_benchmark,
    run_embeddings_benchmark_multi,
)

register_cuda_ci(est_time=245, stage="extra-a", runner_config="1-gpu-large")
register_amd_ci(est_time=240, suite="stage-b-test-1-gpu-large-amd")


class TestEmbeddingsAPI(CustomTestCase):
    def test_embeddings_api_latency_throughput(self):
        res = run_embeddings_benchmark(
            model=DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
            num_requests=1000,
            batch_size=1,
            input_tokens=500,
            other_server_args=[],
            need_warmup=True,
        )

        self.assertEqual(res["successful_requests"], res["total_requests"])
        check_perf(
            self,
            at_most("avg_latency_ms", res["avg_latency_ms"], 23, amd=35, unit="ms"),
            at_most("p95_latency_ms", res["p95_latency_ms"], 34, amd=40, unit="ms"),
            at_least("throughput", res["throughput"], 45, amd=30, unit="req/s"),
        )

    def test_embeddings_api_batch_scaling(self):
        check_batch_scaling(
            self,
            lambda batch_sizes: run_embeddings_benchmark_multi(
                DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
                batch_sizes,
                num_requests=500,
                input_tokens=500,
            ),
            # batch size, avg ms, p95 ms, then the same two relaxed for mi300x
            [
                (10, 44, 52, 80, 90),
                (25, 72, 101, 140, 150),
                (50, 126, 200, 230, 240),
            ],
        )


if __name__ == "__main__":
    unittest.main()
