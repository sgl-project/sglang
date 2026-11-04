"""gfx1250 per-token/per-channel FP8 GEMM tile, and the linear method that uses it.

The tile indexes its operands modulo M and N instead of masking the loads, and
caps BLOCK_SIZE_K at 64 because FP8 ``tl.dot`` miscompiles above that on this
target. Guards: ragged M and small N still match a dequantized reference,
``gfx1250_fp8_scaled_mm_supported`` rejects the shapes the K loop cannot cover,
an all-zero activation row quantizes to exact zeros, and ``Fp8LinearMethod``
both reaches the tile for an online-quantized checkpoint, lays its weights out
row-major, and stays off the tile when the kill switch is set.
"""

import unittest

import torch

from sglang.kernels.ops.gemm.fp8_gfx1250 import (
    gfx1250_fp8_scaled_mm,
    gfx1250_fp8_scaled_mm_supported,
    per_token_quant_fp8_gfx1250,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang.srt.environ import envs
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


@unittest.skipUnless(is_gfx1250_supported(), "requires an AMD gfx1250 GPU")
class TestFp8LinearMethodDispatchGfx1250(CustomTestCase):
    """The seam this change moves: which GEMM an online-FP8 linear ends up on."""

    def test_online_checkpoint_selects_the_tile_and_matches_bf16(self):
        method = Fp8LinearMethod(Fp8Config())
        self.assertTrue(method.use_gfx1250_triton_fp8)

        M, N, K = 512, 6144, 6144
        torch.manual_seed(0)
        layer = torch.nn.Module()
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02
        layer.weight = torch.nn.Parameter(w, requires_grad=False)
        layer.input_scale = None
        method.process_weights_after_loading(layer)

        # The tile wants [K, N] fp8 weights with one scale per output channel;
        # a per-tensor scale here would mean the dispatch picked another path.
        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(tuple(layer.weight.shape), (K, N))
        self.assertEqual(layer.weight_scale.numel(), N)
        # Row-major, not the transposed view this would otherwise be. The
        # tile's configs are tuned for this layout and it is worth 1.05-1.33x,
        # so a refactor that drops the transpose is a silent regression.
        self.assertTrue(
            layer.weight.is_contiguous(),
            f"weight must be row-major [K, N]; strides {layer.weight.stride()}",
        )

        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        out = method.apply(layer, x)
        self.assertEqual(tuple(out.shape), (M, N))
        self.assertEqual(out.dtype, torch.bfloat16)

        ref = x.float() @ w.float().t()
        rel = ((out.float() - ref).abs().mean() / ref.abs().mean()).item()
        self.assertLess(rel, 0.05, f"FP8 linear drifted from the BF16 result: {rel}")

    def test_kill_switch_leaves_the_tile_unselected(self):
        envs.SGLANG_DISABLE_GFX1250_FP8_GEMM.set(True)
        try:
            self.assertFalse(Fp8LinearMethod(Fp8Config()).use_gfx1250_triton_fp8)
        finally:
            envs.SGLANG_DISABLE_GFX1250_FP8_GEMM.clear()


if __name__ == "__main__":
    unittest.main()
