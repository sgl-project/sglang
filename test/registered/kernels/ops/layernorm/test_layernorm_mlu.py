"""Tests for RMSNorm forward_mlu results."""

import unittest
from unittest.mock import patch

import torch
import torch_mlu  # noqa: F401 -- registers the MLU device backend
import torch_mlu_ops

from sglang.kernels.fused_op import get_fused_op_backend, set_fused_op_backend
from sglang.srt.environ import envs
from sglang.srt.layers.layernorm import (
    Gemma3RMSNorm,
    Gemma4RMSNorm,
    GemmaRMSNorm,
    LayerNorm,
    RMSNormWithoutScale,
)
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

    def _check_gemma(self, norm, x, *args):
        expected = norm.forward_native(x, *args)
        with patch.object(
            torch_mlu_ops, "fused_rms_norm", wraps=torch_mlu_ops.fused_rms_norm
        ) as op:
            actual = norm(x, *args)
        if x.dtype == torch.bfloat16 and not args:
            op.assert_called()
        tol = 1e-5 if x.dtype == torch.float32 else 0.03
        torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)

    def test_gemma_numerics(self):
        for cls in (GemmaRMSNorm, Gemma3RMSNorm, Gemma4RMSNorm):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                with self.subTest(cls=cls.__name__, dtype=dtype):
                    kwargs = {"scale_shift": 1.0} if cls is Gemma4RMSNorm else {}
                    norm = cls(128, **kwargs).to(device="mlu", dtype=dtype)
                    with torch.no_grad():
                        norm.weight.uniform_(-0.5, 0.5)
                    x = torch.randn(2, 3, 128, device="mlu", dtype=dtype)
                    self._check_gemma(norm, x)
                    # Low-precision addition order is a separate contract;
                    # it does not need repeating for every dtype.
                    if dtype == torch.bfloat16 and cls is not Gemma4RMSNorm:
                        residual, post = torch.randn_like(x), torch.randn_like(x)
                        self._check_gemma(norm, x, residual)
                        if cls is GemmaRMSNorm:
                            self._check_gemma(norm, x, residual, post)
                            self._check_gemma(norm, x, None, post)

    def test_gemma_native_fallback(self):
        for cls in (GemmaRMSNorm, Gemma3RMSNorm, Gemma4RMSNorm):
            for x in (
                torch.randn(2, 128, device="mlu", dtype=torch.bfloat16),
                torch.randn(128, 2, device="mlu", dtype=torch.float32).T,
                torch.empty(0, 128, device="mlu", dtype=torch.float32),
            ):
                norm = cls(128).to(device="mlu", dtype=torch.float32)
                with patch.object(
                    torch_mlu_ops, "fused_rms_norm", wraps=torch_mlu_ops.fused_rms_norm
                ) as op:
                    actual = norm(x)
                op.assert_not_called()
                torch.testing.assert_close(
                    actual, norm.forward_native(x), rtol=0, atol=0
                )
            if cls is not Gemma4RMSNorm:
                norm = cls(128).to(device="mlu", dtype=torch.bfloat16)
                x = torch.randn(2, 128, device="mlu", dtype=torch.bfloat16)
                additions = [(torch.randn_like(x, dtype=torch.float32),)]
                if cls is GemmaRMSNorm:
                    additions.append(
                        (torch.randn_like(x), torch.randn_like(x, dtype=torch.float32))
                    )
                for args in additions:
                    with patch.object(
                        torch_mlu_ops,
                        "fused_rms_norm",
                        wraps=torch_mlu_ops.fused_rms_norm,
                    ) as op:
                        actual = norm(x, *args)
                    op.assert_not_called()
                    torch.testing.assert_close(
                        actual, norm.forward_native(x, *args), rtol=0, atol=0
                    )
            with envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(True):
                with self.assertRaisesRegex(NotImplementedError, "Deterministic"):
                    norm.forward_mlu(x)

    def test_gemma4_scale_options(self):
        for with_scale, shift in ((True, 0.25), (False, 3.0)):
            with self.subTest(with_scale=with_scale, shift=shift):
                norm = Gemma4RMSNorm(128, scale_shift=shift, with_scale=with_scale).to(
                    device="mlu", dtype=torch.bfloat16
                )
                x = torch.randn(2, 128, device="mlu", dtype=torch.bfloat16)
                self._check_gemma(norm, x)


class TestSimpleNormMLU(CustomTestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(42)
        backend = get_fused_op_backend()
        self.addCleanup(set_fused_op_backend, backend)
        set_fused_op_backend(None)

    def _check(self, norm, x, native=False):
        original = x.clone()
        expected = norm.forward_native(x)
        op = "fused_layer_norm" if isinstance(norm, LayerNorm) else "fused_rms_norm"
        with patch.object(
            torch_mlu_ops, op, wraps=getattr(torch_mlu_ops, op)
        ) as kernel:
            actual = norm(x)
        if native:
            kernel.assert_not_called()
        elif x.dtype == torch.bfloat16:
            kernel.assert_called()
        tolerance = 0 if native else (1e-5 if x.dtype == torch.float32 else 0.03)
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
        torch.testing.assert_close(x, original, rtol=0, atol=0)

    def test_numerics(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            x = torch.randn(2, 3, 128, device="mlu", dtype=dtype)
            for affine, bias in ((True, True), (True, False), (False, True)):
                with self.subTest(dtype=dtype, affine=affine, bias=bias):
                    norm = LayerNorm(
                        128, elementwise_affine=affine, bias=bias, dtype=dtype
                    ).to("mlu")
                    with torch.no_grad():
                        norm.weight.uniform_(0.5, 1.5)
                        norm.bias.uniform_(-0.5, 0.5)
                    self._check(norm, x)
            self._check(RMSNormWithoutScale(128).to("mlu"), x)

    def test_native_fallback(self):
        for norm in (LayerNorm(128).to("mlu"), RMSNormWithoutScale(128).to("mlu")):
            for x in (
                torch.empty(0, 128, device="mlu"),
                torch.randn(128, 3, device="mlu").T,
                torch.randn(3, 128, device="mlu", dtype=torch.float64),
            ):
                with self.subTest(
                    norm=type(norm).__name__, shape=x.shape, dtype=x.dtype
                ):
                    self._check(norm, x, native=True)
        self._check(
            LayerNorm(128).to("mlu"),
            torch.randn(3, 128, device="mlu", dtype=torch.bfloat16),
            native=True,
        )


if __name__ == "__main__":
    unittest.main()
