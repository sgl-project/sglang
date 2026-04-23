"""
Usage:
python3 -m unittest test_intel_xpu_backend.TestIntelXPUBackend.test_latency_qwen_model
"""

import unittest
from functools import wraps

from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
    is_in_ci,
    run_bench_one_batch,
)


def intel_xpu_benchmark(
    extra_args=None, min_throughput=None, mem_fraction_static="0.4"
):
    def decorator(test_func):
        @wraps(test_func)
        def wrapper(self):
            common_args = [
                "--disable-radix",
                "--trust-remote-code",
                "--mem-fraction-static",
                str(mem_fraction_static),
                "--batch-size",
                "1",
                "--device",
                "xpu",
            ]
            ci_args = ["--input", "64", "--output", "4"] if is_in_ci() else []
            full_args = common_args + ci_args + (extra_args or [])

            model = test_func(self)
            prefill_latency, decode_throughput, decode_latency = run_bench_one_batch(
                model, full_args
            )

            print(f"{model=}")
            print(f"{prefill_latency=}")
            print(f"{decode_throughput=}")
            print(f"{decode_latency=}")

            if is_in_ci() and min_throughput is not None:
                self.assertGreater(decode_throughput, min_throughput)

        return wrapper

    return decorator


class TestIntelXPUBackend(CustomTestCase):

    @intel_xpu_benchmark(min_throughput=10, mem_fraction_static="0.3")
    def test_latency_qwen_model(self):
        return DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN

    @intel_xpu_benchmark(
        ["--attention-backend", "intel_xpu", "--page-size", "128"],
        mem_fraction_static="0.5",
    )
    def test_attention_backend(self):
        return DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE


if __name__ == "__main__":
    unittest.main()
