"""
Performance tests for single GPU - VLM, Score API, and Embeddings API tests.
Works on 5090 (32GB).
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_SCORE,
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    run_bench_serving,
    run_embeddings_benchmark,
    run_score_benchmark,
    write_github_step_summary,
)

register_cuda_ci(est_time=900, suite="stage-b-test-large-1-gpu-performance")


class TestBenchServing1GPUPart2(CustomTestCase):
    def test_vlm_offline_throughput(self):
        res = run_bench_serving(
            model=DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            num_prompts=200,
            request_rate=float("inf"),
            other_server_args=[
                "--mem-fraction-static",
                "0.7",
            ],
            dataset_name="mmmu",
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_vlm_offline_throughput\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["output_throughput"], 2000)
            else:
                self.assertGreater(res["output_throughput"], 2500)

    def test_vlm_online_latency(self):
        res = run_bench_serving(
            model=DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
            num_prompts=250,
            request_rate=1,
            other_server_args=[
                "--mem-fraction-static",
                "0.7",
            ],
            dataset_name="mmmu",
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_vlm_online_latency\n"
                f"median_e2e_latency_ms: {res['median_e2e_latency_ms']:.2f} ms\n"
            )
            self.assertLess(res["median_e2e_latency_ms"], 16500)
            if is_in_amd_ci():
                self.assertLess(res["median_ttft_ms"], 150)
            else:
                self.assertLess(res["median_ttft_ms"], 100)
            self.assertLess(res["median_itl_ms"], 8)

    def test_score_api_latency_throughput(self):
        """Test score API latency and throughput performance"""
        res = run_score_benchmark(
            model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST_SCORE,
            num_requests=1000,
            batch_size=10,
            other_server_args=[],
            need_warmup=True,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_score_api_throughput\n"
                f"Average latency: {res['avg_latency_ms']:.2f} ms\n"
                f"P95 latency: {res['p95_latency_ms']:.2f} ms\n"
                f"Score API throughput: {res['throughput']:.2f} req/s\n"
                f"Successful requests: {res['successful_requests']}/{res['total_requests']}\n"
            )

        self.assertEqual(res["successful_requests"], res["total_requests"])
        self.assertLess(res["avg_latency_ms"], 48)
        self.assertLess(res["p95_latency_ms"], 50)
        self.assertGreater(res["throughput"], 20)

    def test_score_api_batch_scaling(self):
        """Test score API performance with different batch sizes"""
        batch_sizes = [10, 25, 50]

        for batch_size in batch_sizes:
            res = run_score_benchmark(
                model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST_SCORE,
                num_requests=500,
                batch_size=batch_size,
            )

            if is_in_ci():
                write_github_step_summary(
                    f"### test_score_api_batch_scaling_size_{batch_size}\n"
                    f"Batch size: {batch_size}\n"
                    f"Average latency: {res['avg_latency_ms']:.2f} ms\n"
                    f"P95 latency: {res['p95_latency_ms']:.2f} ms\n"
                    f"Throughput: {res['throughput']:.2f} req/s\n"
                    f"Successful requests: {res['successful_requests']}/{res['total_requests']}\n"
                )

            self.assertEqual(res["successful_requests"], res["total_requests"])
            bounds = {
                10: (45, 50),
                25: (50, 60),
                50: (60, 65),
            }
            avg_latency_bound, p95_latency_bound = bounds.get(batch_size, (60, 65))
            self.assertLess(res["avg_latency_ms"], avg_latency_bound)
            self.assertLess(res["p95_latency_ms"], p95_latency_bound)

    def test_embeddings_api_latency_throughput(self):
        """Test embeddings API latency and throughput performance"""
        res = run_embeddings_benchmark(
            model=DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
            num_requests=1000,
            batch_size=1,
            input_tokens=500,
            other_server_args=[],
            need_warmup=True,
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_embeddings_api_throughput\n"
                f"Average latency: {res['avg_latency_ms']:.2f} ms\n"
                f"P95 latency: {res['p95_latency_ms']:.2f} ms\n"
                f"Embeddings API throughput: {res['throughput']:.2f} req/s\n"
                f"Successful requests: {res['successful_requests']}/{res['total_requests']}\n"
            )

        self.assertEqual(res["successful_requests"], res["total_requests"])
        self.assertLess(res["avg_latency_ms"], 20)
        self.assertLess(res["p95_latency_ms"], 25)
        self.assertGreater(res["throughput"], 60)

    def test_embeddings_api_batch_scaling(self):
        """Test embeddings API performance with different batch sizes"""
        batch_sizes = [10, 25, 50]

        for batch_size in batch_sizes:
            res = run_embeddings_benchmark(
                model=DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
                num_requests=500,
                batch_size=batch_size,
                input_tokens=500,
            )

            if is_in_ci():
                write_github_step_summary(
                    f"### test_embeddings_api_batch_scaling_size_{batch_size}\n"
                    f"Batch size: {batch_size}\n"
                    f"Average latency: {res['avg_latency_ms']:.2f} ms\n"
                    f"P95 latency: {res['p95_latency_ms']:.2f} ms\n"
                    f"Throughput: {res['throughput']:.2f} req/s\n"
                    f"Successful requests: {res['successful_requests']}/{res['total_requests']}\n"
                )

            self.assertEqual(res["successful_requests"], res["total_requests"])
            bounds = {
                10: (60, 65),
                25: (115, 120),
                50: (190, 195),
            }
            avg_latency_bound, p95_latency_bound = bounds.get(batch_size, (250, 250))
            self.assertLess(res["avg_latency_ms"], avg_latency_bound)
            self.assertLess(res["p95_latency_ms"], p95_latency_bound)


if __name__ == "__main__":
    unittest.main()
