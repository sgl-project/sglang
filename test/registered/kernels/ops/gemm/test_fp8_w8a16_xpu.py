"""Block-FP8 W8A16 linear on XPU must match the dequantized-weight reference on both of its paths.

Inputs up to 512 rows run sgl_kernel's fused W8A16 GEMM; larger inputs dequantize the weight with a
Triton kernel that maps one program to one 128x128 scale block, including partial edge blocks.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=30, suite="stage-b-test-1-gpu-xpu")


def _make_weight(n: int, k: int):
    weight = (torch.randn(n, k, device="xpu") * 64).to(torch.float8_e4m3fn)
    scale = torch.rand(-(n // -128), -(k // -128), device="xpu") * 1e-2 + 1e-3
    dense = (
        weight.float()
        * scale.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    )
    return weight, scale, dense


class TestFp8W8A16LinearXPU(CustomTestCase):
    def setUp(self):
        if not torch.xpu.is_available():
            self.skipTest("XPU not available")
        torch.manual_seed(0)

    def test_dequant_handles_partial_scale_blocks(self):
        from sglang.kernels.ops.gemm.fp8_w8a16_xpu import dequant_block_fp8_weight

        weight, scale, dense = _make_weight(n=320, k=448)
        out = dequant_block_fp8_weight(weight, scale, torch.bfloat16)
        torch.testing.assert_close(out, dense.to(torch.bfloat16), rtol=0, atol=0)

    def test_fused_and_dequant_paths_match_reference(self):
        from sglang.kernels.ops.gemm.fp8_w8a16_xpu import (
            has_xpu_w8a16_gemm,
            xpu_w8a16_block_fp8_linear,
        )

        if not has_xpu_w8a16_gemm():
            self.skipTest("sgl_kernel lacks fp8_w8a16_mm_xe20")
        weight, scale, dense = _make_weight(n=2560, k=5120)
        bias = torch.randn(2560, device="xpu", dtype=torch.bfloat16)
        for rows in (1, 24, 512, 513):
            x = torch.randn(rows, 5120, device="xpu", dtype=torch.bfloat16)
            out = xpu_w8a16_block_fp8_linear(
                input=x.view(1, rows, 5120),
                weight=weight,
                weight_scale=scale,
                bias=bias,
            )
            ref = x.float() @ dense.t() + bias.float()
            self.assertEqual(out.shape, (1, rows, 2560))
            torch.testing.assert_close(
                out.view(rows, 2560).float(),
                ref,
                rtol=1e-2,
                atol=1e-2 * ref.abs().max().item(),
            )


if __name__ == "__main__":
    unittest.main()
