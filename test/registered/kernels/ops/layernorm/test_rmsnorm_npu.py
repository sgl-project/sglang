import itertools
import math
import unittest

import torch

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

if is_npu():
    import torch_npu

register_npu_ci(est_time=30, suite="base-a-test-npu")


@unittest.skipUnless(is_npu(), "Requires an Ascend NPU")
class TestRMSNormNPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch.npu.set_device(0)

    @torch.inference_mode()
    def test_cast_before_weight(self):
        """An explicit cast flag must round normalization before the learned multiply."""
        for width, dtype, fp32_weight, mode in itertools.product(
            (128, 1024),
            (torch.bfloat16, torch.float16),
            (False, True),
            ("plain", "residual", "post"),
        ):
            with self.subTest(
                width=width, dtype=dtype, fp32_weight=fp32_weight, mode=mode
            ):
                weight_dtype = torch.float32 if fp32_weight else dtype
                weight = (1 + torch.arange(width, dtype=torch.float64) % 129 / 128).to(
                    weight_dtype
                )
                norm = RMSNorm(width, eps=2**-20, cast_x_before_out_mul=True).to(
                    device="npu", dtype=weight_dtype
                )
                norm.weight.copy_(weight.to("npu"))
                exponent = {"plain": -10, "residual": -11, "post": -12}[mode]
                x = torch.full((2, width), 2**exponent, device="npu", dtype=dtype)
                residual = x.clone() if mode != "plain" else None
                post = torch.full_like(x, 2**-11) if mode == "post" else None
                # Each input sum is 2**-10, so the normalized value is 1/sqrt(2).
                normalized = torch.full(
                    (2, width), 1 / math.sqrt(2), dtype=torch.float64
                )
                expected = normalized.to(dtype).to(weight_dtype) * weight
                wrong_order = (normalized * weight.double()).to(weight_dtype)
                self.assertFalse(torch.equal(expected, wrong_order))
                actual = norm(x, residual, post)
                if residual is not None:
                    actual, residual_out = actual
                    self.assertEqual(residual_out.dtype, dtype)
                    torch.testing.assert_close(
                        residual_out, torch.full_like(x, 2**-10), rtol=0, atol=0
                    )
                self.assertEqual(actual.dtype, weight_dtype)
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_default_keeps_fused_npu_results(self):
        """The opt-in cast fix must preserve the default fused NPU behavior."""
        for dtype, with_residual, with_post in itertools.product(
            (torch.bfloat16, torch.float16, torch.float32), (False, True), (False, True)
        ):
            with self.subTest(dtype=dtype, residual=with_residual, post=with_post):
                torch.manual_seed(23)
                norm = RMSNorm(128).to(device="npu", dtype=dtype)
                norm.weight.copy_(torch.linspace(0.5, 1.5, 128).to(norm.weight))
                x = torch.randn(3, 128, dtype=dtype).to("npu")
                residual = torch.randn_like(x) if with_residual else None
                post = torch.randn_like(x) if with_post else None
                actual = norm(
                    x.clone(), residual.clone() if with_residual else None, post
                )
                if with_residual:
                    summed = residual + post if with_post else residual
                    expected, _, expected_residual = torch_npu.npu_add_rms_norm(
                        summed, x, norm.weight, norm.variance_epsilon
                    )
                    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        actual[1], expected_residual, rtol=0, atol=0
                    )
                else:
                    expected = torch_npu.npu_rms_norm(
                        x, norm.weight, norm.variance_epsilon
                    )[0]
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
