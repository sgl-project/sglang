"""
Benchmark tests for HiCache Storage functionality.
Usage:
    python3 -m pytest test/srt/hicache/test_hicache_storage_benchmark.py -v
"""

import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from typing import Dict
from urllib.parse import urlparse

import requests

from sglang.bench_serving import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    run_bench_serving,
    write_github_step_summary,
)


class TestHiCacheStorageBenchmark(CustomTestCase):
    """Benchmark tests for HiCache Storage functionality"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment and launch server once for all tests"""
        cls.temp_dir = tempfile.mkdtemp()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        parsed_url = urlparse(DEFAULT_URL_FOR_TEST)
        cls.base_host = parsed_url.hostname
        cls.base_port = str(parsed_url.port)

        # Prepare tokenizer for prompt generation
        cls.tokenizer = get_tokenizer(cls.model)

        # Launch server with HiCache enabled
        cls.process = cls._launch_server_with_hicache()
        cls._wait_for_server_ready()

        print(f"Benchmark server launched successfully at {cls.base_url}")
        print(f"Cache directory: {cls.temp_dir}")

    @classmethod
    def _launch_server_with_hicache(cls, storage_backend="file"):
        """Launch server with HiCache enabled for benchmarking"""
        server_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            "0.8",
            "--hicache-ratio",
            "1.2",
            "--page-size",
            "64",
            "--hicache-storage-backend",
            storage_backend,
            "--enable-cache-report",
            "--hicache-write-policy",
            "write_through",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--log-level",
            "info",
        ]

        if storage_backend == "file":
            env_vars = {
                **os.environ,
                "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
            }
        else:
            env_vars = os.environ

        process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
            env=env_vars,
        )
        return process

    @classmethod
    def _wait_for_server_ready(cls, timeout: int = 60) -> bool:
        """Wait for server to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{cls.base_url}/health", timeout=5)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError("Server failed to start within timeout")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        kill_process_tree(cls.process.pid)

        import shutil

        shutil.rmtree(cls.temp_dir, ignore_errors=True)

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
            num_questions=200,
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

        # Offline throughput test
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

        # Second time benchmark, should use remote cache
        res2 = self._run_throughput_benchmark(
            test_name="hicache_online_throughput",
            num_prompts=200,
            request_rate=10,
            additional_args=[],
        )

        self.assertGreater(
            res2["input_throughput"],
            res1["input_throughput"],
            "Second time throughput should be greater than first time throughput",
        )
        self.assertLess(
            res2["mean_ttft_ms"],
            res1["mean_ttft_ms"],
            "Second time ttft should be less than first time ttft",
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
