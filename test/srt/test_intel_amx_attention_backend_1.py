"""
For intel_amx attention backend FP8 tests
Usage:
python3 -m unittest test_intel_amx_attention_backend_1.TestIntelAMXAttnBackendQuant.test_latency_fp8_qwen
"""

import unittest

from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE,
    DEFAULT_MODEL_NAME_FOR_TEST_QWEN_FP8,
    CustomTestCase,
    intel_amx_benchmark,
)


class TestIntelAMXAttnBackendQuant(CustomTestCase):

    @intel_amx_benchmark(
        extra_args=["--batch-size", "4", "--mem-fraction-static", "0.1"],
        min_throughput=150,
    )
    def test_latency_fp8_qwen(self):
        return DEFAULT_MODEL_NAME_FOR_TEST_QWEN_FP8

    @intel_amx_benchmark(
        extra_args=["--batch-size", "4", "--mem-fraction-static", "0.1"],
        min_throughput=50,
    )
    def test_latency_fp8_moe_model(self):
        return DEFAULT_MODEL_NAME_FOR_TEST_FP8_WITH_MOE


if __name__ == "__main__":
    unittest.main()
