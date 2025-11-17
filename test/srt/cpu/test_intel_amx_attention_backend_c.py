"""
For intel_amx attention backend w8a8 tests
Usage:
python3 -m unittest test_intel_amx_attention_backend_2.TestIntelAMXAttnBackendQuant.test_latency_w8a8_default_model
"""

import unittest

from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_W8A8,
    DEFAULT_MODEL_NAME_FOR_TEST_W8A8_WITH_MOE,
    CustomTestCase,
    intel_amx_benchmark,
)


class TestIntelAMXAttnBackendQuant(CustomTestCase):

    @intel_amx_benchmark(
        extra_args=[
            "--batch-size",
            "4",
            "--quantization",
            "w8a8_int8",
            "--mem-fraction-static",
            "0.1",
        ],
        min_throughput=100,
    )
    def test_latency_w8a8_default_model(self):
        return DEFAULT_MODEL_NAME_FOR_TEST_W8A8

    @intel_amx_benchmark(
        extra_args=[
            "--batch-size",
            "4",
            "--quantization",
            "w8a8_int8",
            "--mem-fraction-static",
            "0.9",
            "--max-total-tokens",
            "65536",
            "--tp",
            "6",
        ],
        min_throughput=100,
    )
    def test_latency_w8a8_moe_model(self):
        return DEFAULT_MODEL_NAME_FOR_TEST_W8A8_WITH_MOE


if __name__ == "__main__":
    unittest.main()
