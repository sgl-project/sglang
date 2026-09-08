"""Numerical and parameter-lifetime coverage for online rowwise FP8 weights."""

import unittest
import weakref

import torch

from sglang.kernels.ops.gemm import sm120_online_fp8 as fp8
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class TestOnlineFP8(CustomTestCase):
    def test_rowwise_quantization_handles_zero_and_signed_rows(self):
        weight = torch.tensor([[0, 0, 0, 0], [1, -2, 3, -4]], dtype=torch.bfloat16)
        quant, scale = fp8._quantize_rowwise_fp8(weight)
        self.assertEqual(quant.dtype, torch.float8_e4m3fn)
        self.assertTrue(torch.isfinite(scale).all())
        torch.testing.assert_close(
            quant.float() * scale[:, None], weight.float(), rtol=0.05, atol=0.01
        )

    def test_replaced_head_releases_original_and_preserves_logits(self):
        torch.manual_seed(0)
        linear = torch.nn.Linear(
            128, 64, bias=False, device="cuda", dtype=torch.bfloat16
        )
        old_weight = weakref.ref(linear.weight)
        original = linear.weight.detach().clone()
        freed = fp8.replace_linears_with_fp8_copies([linear])
        self.assertEqual(freed, original.numel() * original.element_size())
        self.assertIsNone(old_weight())
        self.assertIsNotNone(fp8.rowwise_scale_of(linear.weight))
        for rows in (1, 8, 32, 33):
            with self.subTest(rows=rows):
                hidden = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
                actual = fp8.sm120_fp8_lm_head_logits(hidden, linear.weight)
                # Decode scales the accumulated dot; prefill reconstructs BF16 weights.
                if rows <= 32:
                    reference = (
                        (hidden.float() @ linear.weight.float().t())
                        * fp8.rowwise_scale_of(linear.weight)[None, :]
                    ).to(torch.bfloat16)
                else:
                    reference = hidden @ fp8.dequant_rowwise_weight(linear.weight).t()
                torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)
                baseline = (hidden @ original.t()).float()
                relative_error = (actual.float() - baseline).norm() / baseline.norm()
                self.assertLess(relative_error.item(), 0.05)


if __name__ == "__main__":
    unittest.main()
