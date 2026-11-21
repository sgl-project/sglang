"""CPU tests for timing arithmetic and cache eviction placement in the tuner."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import List

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeDevice:
    """Device clock with expensive cache eviction and a fixed-cost matmul."""

    def __init__(self):
        self.time_ms = 0.0
        self.cache_buffer = object()
        self.cache_clears = 0
        self.weights_cached = False
        self.cached_at_launch = []
        self.intervals = []

    def synchronize(self):
        pass

    def Event(self, enable_timing):
        device = self

        class Event:
            def record(self):
                self.time_ms = device.time_ms

            def synchronize(self):
                pass

            def elapsed_time(self, end):
                elapsed = end.time_ms - self.time_ms
                device.intervals.append(elapsed)
                return elapsed

        return Event()

    def get_empty_cache_for_benchmark(self):
        return self.cache_buffer

    def clear_cache(self, buffer):
        assert buffer is self.cache_buffer
        self.cache_clears += 1
        self.time_ms += 100.0
        self.weights_cached = False

    def matmul(self, *args):
        self.cached_at_launch.append(self.weights_cached)
        self.weights_cached = True
        self.time_ms += 0.25


def _load_benchmark(device):
    # Execute the actual timing function without importing CUDA kernels or
    # triggering the benchmark script's multiprocessing setup.
    source = (
        Path(__file__).resolve().parents[4]
        / "benchmark/kernels/quantization/tuning_block_wise_kernel.py"
    )
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "benchmark_config"
    )
    namespace = {
        "torch": SimpleNamespace(float16=object(), get_device_module=lambda: device),
        "triton": SimpleNamespace(
            runtime=SimpleNamespace(driver=SimpleNamespace(active=device))
        ),
        "w8a8_block_matmul": device.matmul,
        "List": List,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["benchmark_config"]


class TestBlockWiseTunerTiming(CustomTestCase):
    def test_reports_mean_microseconds_without_cache_eviction_cost(self):
        for num_iters in (1, 3, 10):
            with self.subTest(num_iters=num_iters):
                device = _FakeDevice()
                benchmark = _load_benchmark(device)
                latency_us = benchmark(
                    None, None, None, None, [128, 128], {}, num_iters=num_iters
                )
                self.assertAlmostEqual(latency_us, 250.0)
                self.assertEqual(device.intervals, [0.25] * num_iters)

    def test_every_timed_launch_starts_after_cache_eviction(self):
        device = _FakeDevice()
        benchmark = _load_benchmark(device)
        benchmark(None, None, None, None, [128, 128], {}, num_iters=3)
        self.assertEqual(device.cache_clears, 3)
        self.assertEqual(device.cached_at_launch[-3:], [False, False, False])

    def test_rejects_nonpositive_sample_count_before_launching(self):
        for num_iters in (0, -1):
            with self.subTest(num_iters=num_iters):
                device = _FakeDevice()
                benchmark = _load_benchmark(device)
                with self.assertRaisesRegex(ValueError, "num_iters"):
                    benchmark(
                        None, None, None, None, [128, 128], {}, num_iters=num_iters
                    )
                self.assertEqual(device.cached_at_launch, [])


if __name__ == "__main__":
    unittest.main()
