"""CPU regression tests for Ascend MXFP4 W4A8 Linear helpers."""

import unittest

import torch

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    _prepare_mxfp4_w4a8_bias,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNPUMXFP4W4A8Linear(CustomTestCase):
    def test_preserves_no_bias_path(self):
        self.assertIsNone(_prepare_mxfp4_w4a8_bias(None))

    def test_formats_vector_bias_for_a8w4_matmul(self):
        bias = torch.arange(8, dtype=torch.float32)

        quant_bias = _prepare_mxfp4_w4a8_bias(bias)

        self.assertEqual(quant_bias.dtype, torch.bfloat16)
        self.assertEqual(quant_bias.shape, (1, 8))
        self.assertTrue(torch.equal(quant_bias, bias.to(torch.bfloat16).unsqueeze(0)))

    def test_preserves_matrix_shape(self):
        bias = torch.arange(8, dtype=torch.bfloat16).reshape(1, 8)

        quant_bias = _prepare_mxfp4_w4a8_bias(bias)

        self.assertEqual(quant_bias.dtype, torch.bfloat16)
        self.assertEqual(quant_bias.shape, (1, 8))
        self.assertIs(quant_bias, bias)


if __name__ == "__main__":
    unittest.main()
