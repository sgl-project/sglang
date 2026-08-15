import unittest

import torch

from sglang.kernels.ops.quantization.fp8_kernel import (
    fp8_dtype,
    fp8_max,
    sglang_per_token_group_quant_fp8,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=5, stage="jit-kernel-unit", runner_config="amd")


@unittest.skipUnless(is_hip(), "ROCm-only regression")
class TestRocmPerTokenGroupQuantFp8(unittest.TestCase):
    def test_row_major_group_128(self):
        torch.manual_seed(34296)
        value = torch.randn((4, 7168), device="cuda", dtype=torch.bfloat16)

        actual, actual_scale = sglang_per_token_group_quant_fp8(value, 128)

        grouped = value.float().reshape(4, 56, 128)
        expected_scale = grouped.abs().amax(dim=-1).clamp_min(1e-10) / fp8_max
        torch.cuda.synchronize()

        self.assertEqual(actual.dtype, fp8_dtype)
        self.assertTrue(actual_scale.is_contiguous())
        torch.testing.assert_close(actual_scale, expected_scale, rtol=2e-7, atol=0)

        dequantized = actual.float().reshape_as(grouped) * actual_scale[..., None]
        relative_error = (
            (grouped - dequantized).abs() / (grouped.abs() + 1e-6)
        ).mean()
        self.assertLess(relative_error.item(), 0.05)


if __name__ == "__main__":
    unittest.main()