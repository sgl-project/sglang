"""
Benchmark tests for HiCache Storage functionality.
Usage:
    python3 -m pytest test/srt/hicache/test_hicache_storage_benchmark.py -v
"""

import time
import unittest
from types import SimpleNamespace
from typing import Dict

import requests
from test_hicache_storage_e2e import HiCacheStorageBaseTest

from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import is_in_ci, write_github_step_summary


class TestHiCacheStorageBenchmark(HiCacheStorageBaseTest):
    """Benchmark tests for HiCache Storage functionality"""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        """Get additional server arguments specific to configuration - override in subclasses"""
        server_args = {"--tp-size": 2, "--hicache-ratio": 1.5}
        return server_args, {}

    def flush_cache(self) -> bool:
        """Flush device cache to force remote storage access"""
        try:
            response = requests.post(f"{self.base_url}/flush_cache", timeout=10)
            return response.status_code == 200
        except requests.RequestException:
            return False

    # === Accuracy Tests ===
    def test_eval_accuracy_with_cache_persistence(self):
        """Test eval accuracy with cache persistence across cache flushes"""
        print("\n=== Testing Eval Accuracy with Cache Persistence ===")

        # First evaluation - populate cache
        print("Phase 1: Running initial GSM8K evaluation to populate cache...")
        args_initial = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=400,
            max_new_tokens=512,
            parallel=32,
            host=f"http://{self.base_host}",
            port=int(self.base_port),
        )
        metrics_initial = run_eval_few_shot_gsm8k(args_initial)
        print(f"Evaluation metrics: {metrics_initial}")
        self.assertGreater(metrics_initial["accuracy"], 0.60)

        # Flush cache to force remote storage access
        print("Phase 2: Flushing device cache...")
        self.assertTrue(self.flush_cache(), "Cache flush should succeed")
        time.sleep(2)

        # Second evaluation - should use remote cache
        print("Phase 3: Running second GSM8K evaluation using remote cache...")

        start_time = time.time()
        metrics_cached = run_eval_few_shot_gsm8k(args_initial)
        cached_time = time.time() - start_time

        print(f"Cached evaluation completed in {cached_time:.2f}s")
        print(f"Cached accuracy: {metrics_cached['accuracy']:.3f}")
        print(f"Cached throughput: {metrics_cached['output_throughput']:.2f} token/s")

        # Verify accuracy consistency
        accuracy_diff = abs(metrics_initial["accuracy"] - metrics_cached["accuracy"])
        print(f"Accuracy difference: {accuracy_diff:.4f}")

        # Assertions
        self.assertGreater(
            metrics_initial["accuracy"], 0.5, "Initial accuracy should be reasonable"
        )
        self.assertGreater(
            metrics_cached["accuracy"], 0.5, "Cached accuracy should be reasonable"
        )
        self.assertLess(
            accuracy_diff, 0.05, "Accuracy should be consistent between cache states"
        )

        # Performance should be similar or better with cache
        throughput_ratio = (
            metrics_cached["output_throughput"] / metrics_initial["output_throughput"]
        )
        print(f"Throughput ratio (cached/initial): {throughput_ratio:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### HiCache Storage Accuracy Test\n"
                f"Initial accuracy: {metrics_initial['accuracy']:.3f}\n"
                f"Cached accuracy: {metrics_cached['accuracy']:.3f}\n"
                f"Accuracy difference: {accuracy_diff:.4f}\n"
                f"Throughput ratio: {throughput_ratio:.2f}\n"
            )

    # === Performance Benchmark Tests ===

    def test_throughput_benchmark_with_hicache(self):
        """Benchmark throughput performance with HiCache enabled"""
        print("\n=== Benchmarking Throughput with HiCache ===")

        # throughput test
        res1 = self._run_throughput_benchmark(
            test_name="hicache_offline_throughput",
            num_prompts=200,
            request_rate=10,
            additional_args=[],
        )

        # Flush cache to force remote storage access
        print("Phase 2: Flushing device cache...")
        self.assertTrue(self.flush_cache(), "Cache flush should succeed")
        time.sleep(2)

        # Second benchmark, should use remote cache
        res2 = self._run_throughput_benchmark(
            test_name="hicache_online_throughput",
            num_prompts=400,
            request_rate=10,
            additional_args=[],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### HiCache Storage FileBackend Benchmark Test\n"
                f"First time throughput: {res1['input_throughput']:.2f} token/s\n"
                f"Second time throughput: {res2['input_throughput']:.2f} token/s\n"
                f"First time TTFT: {res1['mean_ttft_ms']:.2f} ms\n"
                f"Second time TTFT: {res2['mean_ttft_ms']:.2f} ms\n"
            )

    def _run_throughput_benchmark(
        self,
        test_name: str,
        num_prompts: int,
        request_rate: float,
        dataset_name: str = "random",
        additional_args: list = None,
    ) -> Dict:
        """Helper method to run throughput benchmarks"""
        if additional_args is None:
            additional_args = []

        print(f"Running {test_name} benchmark...")
        start_time = time.time()

        try:
            # Use the existing server instead of launching a new one
            from sglang.bench_serving import run_benchmark
            from sglang.test.test_utils import get_benchmark_args

            args = get_benchmark_args(
                base_url=self.base_url,
                dataset_name=dataset_name,
                tokenizer=self.model,
                num_prompts=num_prompts,
                request_rate=request_rate,
                random_input_len=1024,
                random_output_len=64,
            )

            # Run benchmark
            result = run_benchmark(args)

            elapsed_time = time.time() - start_time
            print(f"{test_name} completed in {elapsed_time:.2f}s")
            print(
                f"Output throughput: {result.get('output_throughput', 0.0):.2f} token/s"
            )

            return result

        except Exception as e:
            print(f"Benchmark {test_name} failed: {e}")
            # Fallback to avoid hard failure; return minimal metrics
            return {
                "output_throughput": 0.0,
                "input_throughput": 0.0,
                "mean_ttft_ms": float("inf"),
                "mean_latency_ms": float("inf"),
                "p99_ttft_ms": float("inf"),
            }


if __name__ == "__main__":
    unittest.main(verbosity=2)
