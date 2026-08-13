"""
Basic XPU test: verifies the server starts and produces a non-empty
response on Intel XPU with the default attention backend.

Assigned to stage-a so it gates stage-b before the heavier tests run.

Usage:
python3 -m unittest test_xpu_basic.TestXPUBasic.test_basic_generation
"""

import unittest

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
    is_in_ci,
    run_bench_one_batch,
)

register_xpu_ci(est_time=300, suite="stage-a-test-1-gpu-xpu")


class TestXPUBasic(CustomTestCase):

    def test_basic_generation(self):
        """Server starts on XPU and completes at least one decode step."""
        args = [
            "--device",
            "xpu",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.6",
            "--batch-size",
            "1",
        ]
        if is_in_ci():
            args += ["--input", "64", "--output", "4"]

        _, decode_throughput, _ = run_bench_one_batch(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN, args
        )
        self.assertGreater(decode_throughput, 0, "XPU decode throughput must be > 0")


if __name__ == "__main__":
    unittest.main()
