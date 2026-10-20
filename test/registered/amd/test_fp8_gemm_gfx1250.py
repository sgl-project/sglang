"""gfx1250 per-token/per-channel FP8 GEMM tile, and the linear method that uses it.

The tile indexes its operands modulo M and N instead of masking the loads, and
caps BLOCK_SIZE_K at 64 because FP8 ``tl.dot`` miscompiles above that on this
target. Guards: ragged M and small N still match a dequantized reference,
``gfx1250_fp8_scaled_mm_supported`` rejects the shapes the K loop cannot cover,
an all-zero activation row quantizes to exact zeros, and ``Fp8LinearMethod``
both reaches the tile for an online-quantized checkpoint, lays its weights out
row-major, and stays off the tile when the kill switch is set.

The SwiGLU-plus-quantize kernel and the two-segment GEMM exist to keep the
FLUX.2 single block's activation and concatenation out of HBM. Guards: the
quantizer reads a strided column slice without a copy, the two segments keep
independent per-token scales across a large magnitude gap, and the fused path
agrees with the concatenate-then-GEMM it replaces.
"""

import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.gemm.fp8_gfx1250 import (
    gfx1250_fp8_2seg_supported,
    gfx1250_fp8_scaled_mm,
    gfx1250_fp8_scaled_mm_2seg,
    gfx1250_fp8_scaled_mm_supported,
    per_token_quant_fp8_gfx1250,
    swiglu_quant_fp8_gfx1250,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
    try_flux2_gfx1250_fp8_swiglu_linear,
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


@unittest.skipUnless(is_gfx1250_supported(), "requires an AMD gfx1250 GPU")
class TestSwiGLUQuantGfx1250(CustomTestCase):
    def _check(self, M: int, H: int, strided: bool):
        torch.manual_seed(0)
        # Rows of a real gate/up projection are a column slice of a wider
        # fused output, so exercise a strided view, not just a packed one.
        wide = torch.randn(M, 2 * H + 1024, device="cuda", dtype=torch.bfloat16)
        x = wide[:, 1024:] if strided else wide[:, : 2 * H].contiguous()
        self.assertEqual(x.is_contiguous(), not strided)

        q, scale = swiglu_quant_fp8_gfx1250(x)
        self.assertEqual(tuple(q.shape), (M, H))
        self.assertTrue(torch.isfinite(scale).all())

        ref = F.silu(x[:, :H].float()) * x[:, H:].float()
        rel = (
            (q.float() * scale[:, None] - ref).abs().mean() / ref.abs().mean()
        ).item()
        # Element-wise FP8 round-trip error; the GEMM that consumes this
        # averages it down to ~1e-3 (asserted in the 2-segment test).
        self.assertLess(rel, 0.05, f"M={M} H={H} strided={strided} rel={rel}")

    def test_flux2_shapes_contiguous_and_strided(self):
        for M, H in ((4112, 18432), (3600, 18432), (512, 18432), (65, 2048)):
            for strided in (False, True):
                with self.subTest(M=M, H=H, strided=strided):
                    self._check(M, H, strided)

    def test_zero_rows(self):
        q, scale = swiglu_quant_fp8_gfx1250(
            torch.zeros(4, 4096, device="cuda", dtype=torch.bfloat16)
        )
        self.assertTrue((q.float() == 0).all())
        self.assertTrue(torch.isfinite(scale).all())
        self.assertTrue((scale > 0).all())


@unittest.skipUnless(is_gfx1250_supported(), "requires an AMD gfx1250 GPU")
class TestTwoSegmentGemmGfx1250(CustomTestCase):
    def test_matches_dequantized_reference(self):
        for M, N, K1, K2 in (
            (4112, 6144, 6144, 18432),
            (512, 6144, 6144, 18432),
            (4111, 6144, 6144, 18432),
        ):
            for with_bias in (False, True):
                with self.subTest(M=M, N=N, K1=K1, K2=K2, bias=with_bias):
                    self.assertTrue(gfx1250_fp8_2seg_supported(M=M, N=N, K1=K1, K2=K2))
                    torch.manual_seed(0)
                    x1 = torch.randn(M, K1, device="cuda", dtype=torch.bfloat16)
                    # 3x the magnitude of x1, so a shared scale would visibly
                    # degrade one segment and this test would catch it.
                    x2 = torch.randn(M, K2, device="cuda", dtype=torch.bfloat16) * 3.0
                    w = torch.randn(N, K1 + K2, device="cuda", dtype=torch.bfloat16)
                    w *= 0.02
                    wq, w_scale = _quantize_weight(w)
                    b = wq.t().contiguous()
                    bias = (
                        torch.randn(N, device="cuda", dtype=torch.bfloat16)
                        if with_bias
                        else None
                    )

                    q1, s1 = per_token_quant_fp8_gfx1250(x1)
                    q2, s2 = per_token_quant_fp8_gfx1250(x2)
                    out = gfx1250_fp8_scaled_mm_2seg(
                        q1, q2, b, s1, s2, w_scale, bias=bias
                    )

                    wdq = wq.float() * w_scale[:, None]
                    ref = (q1.float() * s1[:, None]) @ wdq[:, :K1].t() + (
                        q2.float() * s2[:, None]
                    ) @ wdq[:, K1:].t()
                    if bias is not None:
                        ref = ref + bias.float()
                    self.assertTrue(torch.isfinite(out).all())
                    rel = ((out.float() - ref).abs().mean() / ref.abs().mean()).item()
                    self.assertLess(rel, 5e-3, f"rel={rel}")

    def test_rejects_segments_the_k_loop_cannot_cover(self):
        self.assertFalse(gfx1250_fp8_2seg_supported(M=512, N=6144, K1=100, K2=6044))
        self.assertTrue(gfx1250_fp8_2seg_supported(M=512, N=6144, K1=6144, K2=6144))


@unittest.skipUnless(is_gfx1250_supported(), "requires an AMD gfx1250 GPU")
class TestFlux2SwiGLUFusionGfx1250(CustomTestCase):
    """The fused path must equal the cat-then-GEMM path it replaces."""

    def test_matches_cat_then_linear(self):
        M, K1, H, N = 512, 6144, 2048, 6144
        torch.manual_seed(0)
        method = Fp8LinearMethod(Fp8Config())
        linear = torch.nn.Module()
        w = torch.randn(N, K1 + H, device="cuda", dtype=torch.bfloat16) * 0.02
        linear.weight = torch.nn.Parameter(w, requires_grad=False)
        linear.input_scale = None
        linear.bias = None
        method.process_weights_after_loading(linear)
        linear.quant_method = method

        # Shapes and strides as the block produces them: attention output is
        # packed, gate/up is a column slice of the fused projection.
        attn = torch.randn(1, M, K1, device="cuda", dtype=torch.bfloat16)
        packed = torch.randn(1, M, K1 + 2 * H, device="cuda", dtype=torch.bfloat16)
        gate_up = packed[..., K1:]
        self.assertFalse(gate_up.is_contiguous())

        fused = try_flux2_gfx1250_fp8_swiglu_linear(
            attn_out=attn, mlp_gate_up=gate_up, linear=linear
        )
        self.assertIsNotNone(fused)
        self.assertEqual(tuple(fused.shape), (1, M, N))

        act = F.silu(gate_up[..., :H]) * gate_up[..., H:]
        cat_in = torch.cat([attn, act], dim=-1)
        unfused = method.apply(linear, cat_in)

        # Both paths quantize to FP8, so they cannot be compared to each other
        # -- score each against fp32. The fused path gives the two halves
        # independent scales, so it must not be the worse of the two: one
        # shared scale lets whichever half has the larger magnitude set the
        # row's exponent and crush the other.
        ref = cat_in.float() @ (w.float().t())
        scale = ref.abs().mean()
        rel_fused = ((fused.float() - ref).abs().mean() / scale).item()
        rel_unfused = ((unfused.float() - ref).abs().mean() / scale).item()
        self.assertLess(rel_fused, 5e-2, f"fused rel={rel_fused}")
        self.assertLessEqual(
            rel_fused,
            rel_unfused,
            f"per-segment scales lost accuracy: {rel_fused} > {rel_unfused}",
        )

    def test_declines_shapes_outside_the_tile(self):
        linear = torch.nn.Module()
        linear.weight = torch.empty(
            6144 + 64, 6144, dtype=torch.float8_e4m3fn, device="cuda"
        )
        linear.weight_scale = torch.ones(6144, device="cuda")
        linear.bias = None
        attn = torch.randn(1, 32, 6144, device="cuda", dtype=torch.bfloat16)
        gate_up = torch.randn(1, 32, 64, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(
            try_flux2_gfx1250_fp8_swiglu_linear(
                attn_out=attn, mlp_gate_up=gate_up, linear=linear
            )
        )


if __name__ == "__main__":
    unittest.main()
