# SPDX-License-Identifier: Apache-2.0
"""gfx1250 BF16 linears routed to aiter's FlyDSL GEMM (gfx1250 + AITER only)."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers import linear


@unittest.skipUnless(
    linear.USE_AITER_GFX1250, "needs a gfx1250 GPU with SGLANG_USE_AITER=1"
)
class TestGfx1250FlyDSLLinear(unittest.TestCase):
    def _inputs(self, m: int, n: int, k: int):
        torch.manual_seed(0)
        x = torch.randn(1, m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        return x, w / k**0.5

    def _assert_matches_fp32(self, out: torch.Tensor, x: torch.Tensor, w: torch.Tensor):
        ref = F.linear(x.float(), w.float())
        torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=2e-2)

    def test_large_bf16_linear_runs_flydsl_and_matches_fp32(self):
        x, w = self._inputs(m=512, n=1024, k=512)
        with patch.object(linear, "_flydsl_gemm", wraps=linear._flydsl_gemm) as gemm:
            out = linear.apply_unquantized_linear(x, w)
        self.assertEqual(gemm.call_count, 1)
        self.assertEqual(out.shape, (1, 512, 1024))
        self._assert_matches_fp32(out, x, w)

    def test_small_m_linear_keeps_the_default_gemm(self):
        x, w = self._inputs(m=64, n=1024, k=512)
        with patch.object(linear, "_flydsl_gemm") as gemm:
            out = linear.apply_unquantized_linear(x, w)
        gemm.assert_not_called()
        self._assert_matches_fp32(out, x, w)


if __name__ == "__main__":
    unittest.main()
