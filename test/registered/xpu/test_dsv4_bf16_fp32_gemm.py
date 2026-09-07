import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.gemm import (
    _linear_bf16_fp32_cublas,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=10, suite="stage-b-test-1-gpu-xpu")


class TestDsv4Bf16Fp32GemmXpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.xpu.is_available():
            raise unittest.SkipTest("XPU required")
        cls.device = torch.device("xpu")

    def test_matches_fp32_reference_without_materializing_inputs(self):
        torch.manual_seed(0)
        x = torch.randn(1, 256, device=self.device, dtype=torch.bfloat16)
        weight = torch.randn(128, 256, device=self.device, dtype=torch.bfloat16)
        original_mm = torch.mm
        with patch.object(torch, "mm", wraps=original_mm) as mm:
            actual = _linear_bf16_fp32_cublas(x, weight)
        expected = original_mm(x.float(), weight.float().t())

        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
        self.assertEqual(len(mm.call_args_list), 1)
        self.assertIs(mm.call_args.kwargs["out_dtype"], torch.float32)


if __name__ == "__main__":
    unittest.main()
