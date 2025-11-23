"""
Performance benchmark test for HTTP server tokenization under high concurrency.

This test verifies that the optimized tokenization (dynamic batch + threading + reduced lock contention)
significantly improves performance compared to the old sequential tokenization approach.
"""

import asyncio
import os
import time
import unittest
from typing import List

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)


class TestTokenizerConcurrencyPerformance(CustomTestCase):
    """Test suite for tokenization performance under high concurrency."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = "http://127.0.0.1"
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

    def _launch_server(self, disable_optimized_tokenization: bool, port: int):
        """Launch sglang server with or without optimized tokenization."""
        env = os.environ.copy()
        if disable_optimized_tokenization:
            env["SGLANG_DISABLE_OPTIMIZED_TOKENIZATION"] = "1"

        process = popen_launch_server(
            self.model,
            self.base_url,
            port,
            timeout=300,
            other_args=[
                "--log-level",
                "error",
            ],
            env=env,
        )
        return process

    def _send_single_request(self, port: int, prompt: str):
        """Send a single completion request and measure time."""
        url = f"{self.base_url}:{port}/v1/completions"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 10,
            "temperature": 0,
        }

        start = time.time()
        response = requests.post(url, json=payload, timeout=30)
        latency = time.time() - start

        if response.status_code != 200:
            raise RuntimeError(f"Request failed: {response.text}")

        return latency

    async def _send_concurrent_requests(
        self, port: int, num_requests: int, prompt: str
    ) -> List[float]:
        """Send multiple requests concurrently and measure latencies."""

        async def send_request():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self._send_single_request, port, prompt
            )

        tasks = [send_request() for _ in range(num_requests)]
        latencies = await asyncio.gather(*tasks)
        return latencies

    def _benchmark_server(
        self, disable_optimized_tokenization: bool, port: int, num_requests: int = 50
    ):
        """Benchmark server with given configuration."""
        process = None
        try:
            # Launch server
            process = self._launch_server(disable_optimized_tokenization, port)

            # Wait for server to be ready
            time.sleep(5)

            # Warm up
            for _ in range(3):
                self._send_single_request(port, "Hello, world!")

            # Benchmark with high concurrency
            prompt = "Tell me a short story about artificial intelligence."
            latencies = asyncio.run(
                self._send_concurrent_requests(port, num_requests, prompt)
            )

            # Calculate metrics
            avg_latency = sum(latencies) / len(latencies)
            p50_latency = sorted(latencies)[len(latencies) // 2]
            p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]
            max_latency = max(latencies)
            total_time = max(latencies)  # Total time for all concurrent requests

            return {
                "avg_latency": avg_latency,
                "p50_latency": p50_latency,
                "p95_latency": p95_latency,
                "max_latency": max_latency,
                "total_time": total_time,
                "throughput": num_requests / total_time,
            }

        finally:
            if process:
                kill_process_tree(process.pid)

    def test_optimized_vs_old_tokenization(self):
        """Compare optimized tokenization vs old sequential tokenization."""
        num_requests = 50

        # Test with optimized tokenization (new default)
        print("\n=== Testing OPTIMIZED tokenization ===")
        optimized_metrics = self._benchmark_server(
            disable_optimized_tokenization=False, port=18401, num_requests=num_requests
        )

        print(f"Optimized - Avg latency: {optimized_metrics['avg_latency']:.3f}s")
        print(f"Optimized - P50 latency: {optimized_metrics['p50_latency']:.3f}s")
        print(f"Optimized - P95 latency: {optimized_metrics['p95_latency']:.3f}s")
        print(f"Optimized - Max latency: {optimized_metrics['max_latency']:.3f}s")
        print(f"Optimized - Throughput: {optimized_metrics['throughput']:.2f} req/s")

        # Test with old sequential tokenization
        print("\n=== Testing OLD sequential tokenization ===")
        old_metrics = self._benchmark_server(
            disable_optimized_tokenization=True, port=18402, num_requests=num_requests
        )

        print(f"Old - Avg latency: {old_metrics['avg_latency']:.3f}s")
        print(f"Old - P50 latency: {old_metrics['p50_latency']:.3f}s")
        print(f"Old - P95 latency: {old_metrics['p95_latency']:.3f}s")
        print(f"Old - Max latency: {old_metrics['max_latency']:.3f}s")
        print(f"Old - Throughput: {old_metrics['throughput']:.2f} req/s")

        # Calculate improvement
        speedup = old_metrics["avg_latency"] / optimized_metrics["avg_latency"]
        throughput_improvement = (
            optimized_metrics["throughput"] / old_metrics["throughput"]
        )

        print(f"\n=== Performance Improvement ===")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Throughput improvement: {throughput_improvement:.2f}x")

        # Verify optimized version is faster
        # We expect at least 1.5x improvement (conservative, likely much more)
        self.assertGreater(
            speedup,
            1.5,
            f"Optimized tokenization should be at least 1.5x faster, got {speedup:.2f}x",
        )

        # Verify throughput improvement
        self.assertGreater(
            throughput_improvement,
            1.5,
            f"Optimized tokenization should have at least 1.5x better throughput, got {throughput_improvement:.2f}x",
        )

        # In CI, verify we meet minimum performance targets
        if is_in_ci():
            # Optimized version should handle at least 5 req/s with 50 concurrent requests
            self.assertGreater(
                optimized_metrics["throughput"],
                5.0,
                f"Optimized throughput should be >5 req/s, got {optimized_metrics['throughput']:.2f}",
            )

    def test_optimized_tokenization_single_request(self):
        """Verify optimized tokenization doesn't slow down single requests."""
        process = None
        try:
            # Launch server with optimized tokenization
            process = self._launch_server(
                disable_optimized_tokenization=False, port=18403
            )

            # Wait for server to be ready
            time.sleep(5)

            # Test single request latency (should be fast, no batching overhead)
            prompt = "Hello, world!"
            latencies = []
            for _ in range(10):
                latency = self._send_single_request(18403, prompt)
                latencies.append(latency)

            avg_latency = sum(latencies) / len(latencies)
            print(f"\nSingle request avg latency: {avg_latency:.3f}s")

            # Single requests should still be fast (<2s for small model + short prompt)
            self.assertLess(
                avg_latency,
                2.0,
                f"Single request should be fast, got {avg_latency:.3f}s",
            )

        finally:
            if process:
                kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
