import unittest

import torch

from sglang.srt.layers.quantization.fp8_utils import (
    quantize_block_fp8_weight_to_mxfp4,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestFp8UtilsMxfp4(unittest.TestCase):
    def test_quantize_block_fp8_weight_to_mxfp4_shapes_and_dtype(self):
        fp8_weight = (
            torch.linspace(-2.0, 2.0, 32 * 32, dtype=torch.float32)
            .reshape(32, 32)
            .to(torch.float8_e4m3fn)
        )
        fp8_scale = torch.ones(1, 1, dtype=torch.float8_e8m0fnu)

        fp4_weight, fp4_scale = quantize_block_fp8_weight_to_mxfp4(
            fp8_weight, fp8_scale, [128, 128]
        )

        self.assertEqual(fp4_weight.dtype, torch.int8)
        self.assertEqual(fp4_weight.shape, torch.Size([32, 16]))
        self.assertEqual(fp4_scale.dtype, torch.float8_e8m0fnu)
        self.assertEqual(fp4_scale.shape, torch.Size([32, 1]))

    def test_quantize_block_fp8_weight_to_mxfp4_grouped_weight(self):
        fp8_weight = (
            torch.linspace(-2.0, 2.0, 2 * 32 * 32, dtype=torch.float32)
            .reshape(2, 32, 32)
            .to(torch.float8_e4m3fn)
        )
        fp8_scale = torch.ones(2, 1, 1, dtype=torch.float8_e8m0fnu)

        fp4_weight, fp4_scale = quantize_block_fp8_weight_to_mxfp4(
            fp8_weight, fp8_scale, [128, 128]
        )

        self.assertEqual(fp4_weight.dtype, torch.int8)
        self.assertEqual(fp4_weight.shape, torch.Size([2, 32, 16]))
        self.assertEqual(fp4_scale.dtype, torch.float8_e8m0fnu)
        self.assertEqual(fp4_scale.shape, torch.Size([2, 32, 1]))


if __name__ == "__main__":
    unittest.main()
