"""gfx1250 per-token/per-channel FP8 GEMM tile.

The tile indexes its operands modulo M and N instead of masking the loads, and
caps BLOCK_SIZE_K at 64 because FP8 ``tl.dot`` miscompiles above that on this
target. Guards: ragged M and small N still match a dequantized reference,
``gfx1250_fp8_scaled_mm_supported`` rejects the shapes the K loop cannot cover,
and an all-zero activation row quantizes to exact zeros.
"""

import unittest

import torch

from sglang.kernels.ops.gemm.fp8_gfx1250 import (
    gfx1250_fp8_scaled_mm,
    gfx1250_fp8_scaled_mm_supported,
    per_token_quant_fp8_gfx1250,
)
from sglang.srt.utils.common import is_gfx1250_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")

_FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def _quantize_weight(w: torch.Tensor):
    """Mirror Fp8LinearMethod's online path: [N, K] bf16 -> fp8 [K, N] + per-channel scale."""
    scale = (w.float().abs().amax(dim=1) / _FP8_MAX).clamp(min=1e-12)
    q = (w.float() / scale[:, None]).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale


@unittest.skipUnless(is_gfx1250_supported(), "requires an AMD gfx1250 GPU")
class TestFp8GemmGfx1250(CustomTestCase):
    def _check(self, M: int, N: int, K: int, with_bias: bool):
        torch.manual_seed(0)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        bias = (
            torch.randn(N, device="cuda", dtype=torch.bfloat16) if with_bias else None
        )

        wq, w_scale = _quantize_weight(w)
        xq, x_scale = per_token_quant_fp8_gfx1250(x)

        out = gfx1250_fp8_scaled_mm(
            xq, wq.t(), x_scale, w_scale, out_dtype=torch.bfloat16, bias=bias
        )
        # Reference dequantizes the same FP8 operands, so this isolates the
        # tile from the quantization error it inherits.
        ref = (xq.float() * x_scale[:, None]) @ (wq.float() * w_scale[:, None]).t()
        if bias is not None:
            ref = ref + bias.float()

        self.assertTrue(torch.isfinite(out).all())
        rel = ((out.float() - ref).abs().mean() / ref.abs().mean()).item()
        self.assertLess(rel, 5e-3, f"M={M} N={N} K={K} bias={with_bias} rel={rel}")

    def test_flux2_720p_shapes(self):
        for M, N, K in (
            (4112, 55296, 6144),
            (4112, 6144, 24576),
            (3600, 36864, 6144),
            (512, 6144, 15360),
        ):
            with self.subTest(M=M, N=N, K=K):
                self._check(M, N, K, with_bias=False)

    def test_ragged_and_small(self):
        for M, N, K in ((65, 6144, 6144), (4111, 6144, 6144), (4113, 320, 6144)):
            with self.subTest(M=M, N=N, K=K):
                self._check(M, N, K, with_bias=True)

    def test_unsupported_shapes_are_rejected(self):
        self.assertFalse(gfx1250_fp8_scaled_mm_supported(M=32, N=6144, K=6144))
        self.assertFalse(gfx1250_fp8_scaled_mm_supported(M=512, N=6144, K=6176))
        self.assertFalse(gfx1250_fp8_scaled_mm_supported(M=512, N=100, K=6144))
        self.assertTrue(gfx1250_fp8_scaled_mm_supported(M=512, N=6144, K=6144))

    def test_zero_row_quantizes_to_zero(self):
        x = torch.randn(8, 6144, device="cuda", dtype=torch.bfloat16)
        x[3] = 0
        q, scale = per_token_quant_fp8_gfx1250(x)
        self.assertTrue(torch.isfinite(scale).all())
        self.assertTrue((q[3].float() == 0).all())
        roundtrip = q.float() * scale[:, None]
        rel = ((roundtrip - x.float()).abs().mean() / x.float().abs().mean()).item()
        self.assertLess(rel, 0.05)


if __name__ == "__main__":
    unittest.main()
