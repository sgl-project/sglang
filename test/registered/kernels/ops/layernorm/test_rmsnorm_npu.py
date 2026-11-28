"""Regression coverage for the explicit NPU RMSNorm cast order (issue #50715)."""

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

register_npu_ci(est_time=10, suite="extra-b-test-1-npu-a3")


@unittest.skipUnless(is_npu(), "Requires an Ascend NPU")
class TestRMSNormNPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        torch.npu.set_device(0)

    @torch.inference_mode()
    def test_cast_before_weight(self):
        # The analytical example and mixed-weight coverage extend the regression
        # cases from the earlier proposal in #50274.
        for width, dtype, weight_fp32, mode in itertools.product(
            (128, 1024),
            (torch.float16, torch.bfloat16),
            (False, True),
            ("plain", "residual", "post"),
        ):
            with self.subTest(
                width=width, dtype=dtype, weight_fp32=weight_fp32, mode=mode
            ):
                weight_dtype = torch.float32 if weight_fp32 else dtype
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

                # Every input sum is 2**-10 and eps == (2**-10)**2, so
                # normalization gives 1/sqrt(2) before the activation cast.
                normalized = torch.full(
                    (2, width), 1 / math.sqrt(2), dtype=torch.float64
                )
                expected = normalized.to(dtype).to(weight_dtype) * weight
                late_cast = (normalized * weight.double()).to(weight_dtype)
                self.assertFalse(torch.equal(expected, late_cast))

                actual = norm(x, residual, post)
                self.assertEqual(norm._forward_method, norm.forward_npu)
                if residual is not None:
                    actual, residual_out = actual
                    self.assertEqual(residual_out.dtype, dtype)
                    torch.testing.assert_close(
                        residual_out, torch.full_like(x, 2**-10), rtol=0, atol=0
                    )
                self.assertEqual(actual.dtype, weight_dtype)
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)

    @torch.inference_mode()
    def test_post_residual_is_added_in_float32(self):
        for dtype, magnitude in ((torch.float16, 4096), (torch.bfloat16, 256)):
            with self.subTest(dtype=dtype):
                norm = RMSNorm(128, eps=0, cast_x_before_out_mul=True).to(
                    device="npu", dtype=dtype
                )
                x = torch.full((2, 128), -magnitude, device="npu", dtype=dtype)
                residual = -x
                post = torch.ones_like(x)
                # Adding post to residual in the input dtype loses the 1.
                # The native order instead computes (x.float() + residual) + post.
                self.assertTrue(torch.equal(residual + post, residual))
                out, residual_out = norm(x, residual, post)
                torch.testing.assert_close(out, post, rtol=0, atol=0)
                torch.testing.assert_close(residual_out, post, rtol=0, atol=0)

    @torch.inference_mode()
    def test_cast_options_match_native(self):
        for fp32_residual, override_dtype in itertools.product(
            (False, True), (None, torch.float32)
        ):
            with self.subTest(
                fp32_residual=fp32_residual, override_dtype=override_dtype
            ):
                norm = RMSNorm(
                    128,
                    cast_x_before_out_mul=True,
                    fp32_residual=fp32_residual,
                    override_orig_dtype=override_dtype,
                ).to(device="npu", dtype=torch.bfloat16)
                x = torch.full((2, 128), 0.125, device="npu", dtype=torch.bfloat16)
                residual = torch.full_like(x, 2**-12)
                expected = norm.forward_native(x, residual)
                actual = norm(x, residual)
                for result, reference in zip(actual, expected):
                    self.assertEqual(result.dtype, reference.dtype)
                    torch.testing.assert_close(result, reference, rtol=0, atol=0)

    @torch.inference_mode()
    def test_default_keeps_fused_npu_results(self):
        for dtype, with_residual, with_post in itertools.product(
            (torch.float16, torch.bfloat16, torch.float32), (False, True), (False, True)
        ):
            with self.subTest(dtype=dtype, residual=with_residual, post=with_post):
                torch.manual_seed(23)
                norm = RMSNorm(128).to(device="npu", dtype=dtype)
                norm.weight.copy_(torch.linspace(0.5, 1.5, 128).to(norm.weight))
                x = torch.randn(3, 128, dtype=dtype, device="npu")
                residual = torch.randn_like(x) if with_residual else None
                post = torch.randn_like(x) if with_post else None
                actual = norm(
                    x.clone(), residual.clone() if with_residual else None, post
                )
                self.assertEqual(norm._forward_method, norm.forward_npu)
                if with_residual:
                    fused_residual = residual + post if with_post else residual
                    out, _, residual_out = torch_npu.npu_add_rms_norm(
                        fused_residual, x, norm.weight, norm.variance_epsilon
                    )
                    expected = out, residual_out
                else:
                    expected = torch_npu.npu_rms_norm(
                        x, norm.weight, norm.variance_epsilon
                    )[0]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
