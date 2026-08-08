"""Unit tests for RVV activation kernels."""

import itertools
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

from ..utils import GeluAndMul, SiluAndMul
from .rvv_utils import has_sgl_kernel_op, helper_non_contiguous, precision

torch.manual_seed(1234)


@unittest.skipUnless(
    has_sgl_kernel_op("silu_and_mul_cpu"),
    "sgl_kernel activation not available (non-RISC-V build)",
)
class TestRVVActivation(CustomTestCase):
    """Test suite for RVV activation kernels."""

    M = [1, 128, 129, 257]
    N = [32, 64, 80, 128, 22016, 22018]
    # Cover sub-VL, exact-VL, unrolled, and tail-heavy shapes.
    dtype = [torch.float16, torch.bfloat16]

    def _run_silu_mul(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        out = torch.ops.sgl_kernel.silu_and_mul_cpu(x)
        ref = SiluAndMul(x)
        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def _run_gelu_tanh_mul(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        out = torch.ops.sgl_kernel.gelu_tanh_and_mul_cpu(x)
        ref = GeluAndMul(x, approximate="tanh")
        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def _run_gelu_mul(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        out = torch.ops.sgl_kernel.gelu_and_mul_cpu(x)
        ref = GeluAndMul(x, approximate="none")
        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_silu_mul(self):
        """Shape matrix includes sub-VL, exact-VL, unrolled, and tail-heavy sizes."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._run_silu_mul(m, n, dt)

    def test_gelu_tanh_mul(self):
        """Shape matrix includes sub-VL, exact-VL, unrolled, and tail-heavy sizes."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._run_gelu_tanh_mul(m, n, dt)

    def test_gelu_mul(self):
        """Shape matrix includes sub-VL, exact-VL, unrolled, and tail-heavy sizes."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._run_gelu_mul(m, n, dt)

    def test_non_contiguous(self):
        """Non-contiguous input must not silently read wrong elements due to stride errors."""
        for dtype in self.dtype:
            x = helper_non_contiguous(torch.randn(128, 64, dtype=dtype))
            self.assertFalse(x.is_contiguous())
            atol = rtol = precision["pointwise_default"][dtype]
            with self.subTest(dtype=dtype, op="silu_and_mul"):
                torch.testing.assert_close(
                    SiluAndMul(x),
                    torch.ops.sgl_kernel.silu_and_mul_cpu(x),
                    atol=atol,
                    rtol=rtol,
                )
            with self.subTest(dtype=dtype, op="gelu_tanh_and_mul"):
                torch.testing.assert_close(
                    GeluAndMul(x, approximate="tanh"),
                    torch.ops.sgl_kernel.gelu_tanh_and_mul_cpu(x),
                    atol=atol,
                    rtol=rtol,
                )
            with self.subTest(dtype=dtype, op="gelu_and_mul"):
                torch.testing.assert_close(
                    GeluAndMul(x, approximate="none"),
                    torch.ops.sgl_kernel.gelu_and_mul_cpu(x),
                    atol=atol,
                    rtol=rtol,
                )


if __name__ == "__main__":
    unittest.main()
