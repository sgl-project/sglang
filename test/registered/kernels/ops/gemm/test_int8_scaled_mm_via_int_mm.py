"""int8_scaled_mm_via_int_mm: torch._int_mm + one fused Triton epilogue (the XPU int8 GEMM).

Checks against an exact integer reference, including the epilogue's masked M/N tails and
bias in a dtype other than the output's. Zero rows go through the layer test
(xpu/test_w8a8_int8_linear_xpu.py).
"""

import itertools
import unittest

import torch

from sglang.kernels.ops.gemm.int8_kernel import int8_scaled_mm_via_int_mm
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=20, suite="stage-b-test-1-gpu-xpu")

DEVICE = "xpu"


def _reference(mat_a, mat_b, scales_a, scales_b, bias, out_dtype):
    acc = mat_a.cpu().long() @ mat_b.cpu().long()  # exact integer GEMM
    out = (
        acc.double()
        * scales_a.cpu().double().reshape(-1, 1)
        * scales_b.cpu().double().reshape(1, -1)
    )
    if bias is not None:
        out = out + bias.cpu().double()
    return out.to(out_dtype)


@unittest.skipUnless(
    hasattr(torch, "xpu") and torch.xpu.is_available(), "requires an XPU device"
)
class TestInt8ScaledMmViaIntMm(CustomTestCase):
    def test_matches_exact_reference(self):
        torch.manual_seed(0)
        k = 512  # |acc| <= 512 * 127 * 127 < 2**24, so the int32 GEMM result is exact
        # M=7/300 and N=1344 are not multiples of the epilogue's 32 x 128 tile.
        for m, n, bias_dtype, out_dtype in itertools.product(
            (1, 7, 300),
            (1280, 1344),
            (None, torch.bfloat16, torch.float32),
            (torch.bfloat16, torch.float16),
        ):
            with self.subTest(m=m, n=n, bias=bias_dtype, out_dtype=out_dtype):
                a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device=DEVICE)
                w = torch.randint(-127, 128, (n, k), dtype=torch.int8, device=DEVICE)
                sa = torch.rand(m, 1, device=DEVICE) * 1e-2 + 1e-3
                sb = torch.rand(n, 1, device=DEVICE) * 1e-2 + 1e-3
                bias = (
                    None
                    if bias_dtype is None
                    else torch.randn(n, device=DEVICE).to(bias_dtype)
                )
                out = int8_scaled_mm_via_int_mm(a, w.t(), sa, sb, out_dtype, bias)
                self.assertEqual(out.dtype, out_dtype)
                self.assertEqual(tuple(out.shape), (m, n))
                ref = _reference(a, w.t(), sa, sb, bias, out_dtype)
                torch.testing.assert_close(
                    out.float().cpu(), ref.float(), rtol=1e-2, atol=1e-2
                )


if __name__ == "__main__":
    unittest.main()
