"""Tests for RMSNorm forward_mlu results."""

import unittest

import torch
import torch_mlu  # noqa: F401 -- registers the MLU device backend

from sglang.test.ci.ci_register import register_mlu_ci
from sglang.test.test_utils import CustomTestCase

register_mlu_ci(est_time=60, suite="pr-test-1-mlu")


class TestRMSNormMLU(CustomTestCase):
    def setUp(self):
        self.assertTrue(
            torch.mlu.is_available(), "MLU device is required for this suite"
        )
        torch.manual_seed(42)
        torch.mlu.set_device(0)
        from sglang.srt.layers.layernorm import RMSNorm

        self.norm = RMSNorm(128, eps=1e-6).to(device="mlu", dtype=torch.bfloat16)
        with torch.no_grad():
            self.norm.weight.normal_(mean=1.0, std=0.1)

    def test_2d_input(self):
        x = torch.randn(4, 128, dtype=torch.bfloat16, device="mlu")
        out = self.norm(x.clone())
        self.assertEqual(self.norm._forward_method, self.norm.forward_mlu)
        ref = self.norm.forward_native(x.clone())
        self.assertEqual(out.shape, x.shape)
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    def test_residual(self):
        x = torch.randn(4, 128, dtype=torch.bfloat16, device="mlu")
        residual = torch.randn_like(x)
        out, residual_out = self.norm(x.clone(), residual.clone())
        ref, residual_ref = self.norm.forward_native(x.clone(), residual.clone())
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(residual_out, residual_ref, rtol=1e-2, atol=1e-2)

    def test_post_residual_addition(self):
        x = torch.randn(4, 128, dtype=torch.bfloat16, device="mlu")
        residual = torch.randn_like(x)
        post = torch.randn_like(x)
        out, residual_out = self.norm(x.clone(), residual.clone(), post.clone())
        ref, residual_ref = self.norm.forward_native(
            x.clone(), residual.clone(), post.clone()
        )
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(residual_out, residual_ref, rtol=1e-2, atol=1e-2)

    def test_empty_input(self):
        x = torch.empty(0, 128, dtype=torch.bfloat16, device="mlu")
        self.assertEqual(self.norm(x).shape, x.shape)
        for post in (None, x.clone()):
            out, residual = self.norm(x, x.clone(), post)
            self.assertEqual(out.shape, x.shape)
            self.assertEqual(residual.shape, x.shape)

    def test_mixed_dtypes(self):
        """Mixed weights or residual additions must not fail dtype checks."""
        for weight_dtype, residual_dtype, post_dtype in (
            (torch.float32, None, None),
            (torch.bfloat16, torch.float32, None),
            (torch.bfloat16, torch.bfloat16, torch.float32),
        ):
            with self.subTest(
                weight_dtype=weight_dtype,
                residual_dtype=residual_dtype,
                post_dtype=post_dtype,
            ):
                norm = self.norm.to(dtype=weight_dtype)
                x = torch.randn(4, 128, dtype=torch.bfloat16, device="mlu")
                residual = (
                    torch.randn_like(x, dtype=residual_dtype)
                    if residual_dtype is not None
                    else None
                )
                post = (
                    torch.randn_like(x, dtype=post_dtype)
                    if post_dtype is not None
                    else None
                )
                ref = norm.forward_native(x, residual, post)
                out = norm(
                    x.clone(),
                    residual.clone() if residual is not None else None,
                    post.clone() if post is not None else None,
                )
                torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_input_shape_preservation(self):
        """Verify shape preservation and numerical agreement with native RMSNorm.

        Cover 1D, higher-rank, and transposed inputs that need reshaping before
        the MLU kernel call, including residual and post-addition paths.
        """
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            norm = self.norm.to(dtype=dtype)
            for shape in ((128,), (2, 3, 128), (2, 3, 4, 128)):
                x = torch.randn(shape, device="mlu", dtype=dtype)
                residual = torch.randn_like(x)
                if len(shape) == 4:
                    # Last dimension stays contiguous, but flattening the leading
                    # dimensions requires a copy for both input and residual.
                    x = x.transpose(0, 1)
                    residual = residual.transpose(0, 1)
                post = torch.randn_like(x)
                for additions in ((), (residual,), (residual, post)):
                    with self.subTest(dtype=dtype, shape=x.shape, adds=len(additions)):
                        result = norm(x, *additions)
                        reference = norm.forward_native(x, *additions)
                        if not additions:
                            result, reference = (result,), (reference,)
                        for actual, expected in zip(result, reference):
                            self.assertEqual(actual.shape, x.shape)
                            torch.testing.assert_close(
                                actual, expected, rtol=1e-2, atol=1e-2
                            )

    def test_noncontiguous_input(self):
        x = torch.randn(128, 4, dtype=torch.bfloat16, device="mlu").T
        residual = torch.randn(128, 4, dtype=torch.bfloat16, device="mlu").T
        self.assertFalse(x.is_contiguous())
        row_strided = torch.randn(8, 128, dtype=torch.bfloat16, device="mlu")[::2]
        for inp in (x, x.contiguous(), row_strided):
            for res in (None, residual, row_strided):
                with self.subTest(
                    input_stride=inp.stride(),
                    residual_stride=res.stride() if res is not None else None,
                ):
                    out = self.norm(inp, res)
                    ref = self.norm.forward_native(inp, res)
                    if res is None:
                        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
                    else:
                        for actual, expected in zip(out, ref):
                            torch.testing.assert_close(
                                actual, expected, rtol=1e-2, atol=1e-2
                            )


if __name__ == "__main__":
    unittest.main()
