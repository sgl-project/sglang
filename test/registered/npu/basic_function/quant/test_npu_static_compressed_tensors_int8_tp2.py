"""Real static-INT8 TP2 loading and decoding against a TP1 reference."""

import unittest

import test_npu_static_compressed_tensors_int8 as base_test

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=600, suite="base-b-test-2-npu-a3")


class TestStaticCompressedTensorsInt8TP2(base_test.TestStaticCompressedTensorsInt8):
    test_tp_size = 2


if __name__ == "__main__":
    unittest.main()
