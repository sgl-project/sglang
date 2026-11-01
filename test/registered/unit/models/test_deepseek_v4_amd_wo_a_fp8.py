"""DeepSeek-V4 / V4.1 fp8 wo_a on gfx950: the 32x32-block (V4.1) weight keeps its checkpoint fp8
codes in the preshuffled layout, its GEMM matches an fp64 product of the quantized operands, the
fused inverse-RoPE front end agrees with the standalone one, and the 128x128 weight stays row-major."""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

R, D, HEAD_DIM, ROPE_DIM = 1024, 4096, 512, 64


def _block_fp8(groups: int, block: int, seed: int):
    """A [G*R, D] fp8 weight and its fp32 power-of-two [G*R/block, D/block] scales."""
    torch.manual_seed(seed)
    w = torch.randn(groups * R, D, device="cuda") * 0.02
    blocks = w.view(groups * R // block, block, D // block, block)
    amax = blocks.abs().amax(dim=(1, 3)).clamp(min=2.0**-100)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    q = (blocks / scale[:, None, :, None]).view(groups * R, D).to(torch.float8_e4m3fn)
    return q, scale


def _dequant(weight: torch.Tensor, scale: torch.Tensor, block: int) -> torch.Tensor:
    n = weight.shape[0]
    blocks = weight.float().view(n // block, block, D // block, block)
    return (blocks * scale[:, None, :, None]).view(n // R, R, D)


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 wo_a fp8")
class TestWoAFp8Gfx950(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.models.deepseek_common.amd import deepseek_v4_wo_a_fp8 as m

        if not m.is_wo_a_fp8_block32_supported():
            raise unittest.SkipTest("aiter lacks the preshuffled mxscale BMM")
        cls.m = m

    def _assert_matches_quantized_product(self, out, o, weight_deq, block, ctx):
        """Within one bf16 rounding plus fp32 summation of the fp64 product of the
        operands the kernel actually multiplies: o quantized per block, the weight."""
        T, G, _ = o.shape
        oq, os_ = self.m.quant_wo_a_act_mxfp8(o, block)
        o_deq = oq.float().view(T, G, D // block, block) * torch.exp2(
            os_.float() - 127
        ).unsqueeze(-1)
        o_deq = o_deq.view(T, G, D).double()
        exact = torch.einsum("tgd,grd->tgr", o_deq, weight_deq.double())
        absprod = torch.einsum("tgd,grd->tgr", o_deq.abs(), weight_deq.abs().double())
        bound = exact.abs() * 2.0**-8 + absprod * 2.0**-17 + 1e-6
        err = (out.double() - exact).abs()
        self.assertTrue(bool((err <= bound).all()), (*ctx, (err / bound).max().item()))

    def test_block32_weight_keeps_the_checkpoint_codes(self):
        """The load-time conversion only re-lays the fp8 codes out for the BMM and turns
        the fp32 scales into their e8m0 exponents."""
        from aiter.ops.shuffle import shuffle_weight

        for g in (1, 8):
            w, s = _block_fp8(g, 32, seed=g)
            wq, e = self.m.wo_a_weight_scale_to_e8m0(w, s, g, R)
            self.assertEqual(self.m.wo_a_fp8_block(wq, e), 32)
            self.assertTrue(
                torch.equal(
                    wq.view(torch.uint8),
                    shuffle_weight(w.view(g, R, D), layout=(16, 16)).view(torch.uint8),
                ),
                g,
            )
            self.assertTrue(
                torch.equal(e.float(), (torch.log2(s) + 127).view(g, R // 32, D // 32)),
                g,
            )

    def test_block32_gemm(self):
        """Every group count and decode / verify / prefill row count."""
        for g in (1, 2, 8):
            w, s = _block_fp8(g, 32, seed=10 + g)
            wq, e = self.m.wo_a_weight_scale_to_e8m0(w, s, g, R)
            w_deq = _dequant(w, s, 32)
            for t in (1, 16, 80, 300):
                o = (torch.randn(t, g, D, device="cuda") * 1.5).bfloat16()
                out = self.m.apply_wo_a_fp8_mxscale(o, wq, e)
                self.assertEqual(out.shape, (t, g, R))
                self._assert_matches_quantized_product(out, o, w_deq, 32, (g, t))

    def test_block32_fused_inverse_rope(self):
        """The fused inverse-RoPE + per-32 quant front end gives the standalone inverse
        RoPE + quant result up to fp8 rounding of the rotated (bf16 vs fp32) rope dims."""
        from sglang.kernels.ops.attention.dsv4.norm_rope import fused_rope_inplace

        if not self.m.is_wo_a_fp8_fused_invrope_supported():
            self.skipTest("aiter lacks inverse_rope_group_quant")
        g = 8
        w, s = _block_fp8(g, 32, seed=3)
        wq, e = self.m.wo_a_weight_scale_to_e8m0(w, s, g, R)
        torch.manual_seed(4)
        freqs = torch.polar(
            torch.ones(8192, ROPE_DIM // 2), torch.rand(8192, ROPE_DIM // 2) * 6.28
        ).cuda()
        real = torch.view_as_real(freqs)
        cos, sin = (
            real[..., 0].bfloat16().contiguous(),
            real[..., 1].bfloat16().contiguous(),
        )
        heads = g * D // HEAD_DIM
        for t in (1, 80):
            o = torch.randn(t, heads, HEAD_DIM, device="cuda").bfloat16()
            pos = torch.randint(0, 8192, (t,), device="cuda")
            fused = self.m.apply_wo_a_fp8_mxscale_fused_invrope(
                o, pos, cos, sin, g, wq, e
            ).float()
            rotated = o.clone()
            fused_rope_inplace(
                rotated[..., -ROPE_DIM:], None, freqs, positions=pos, inverse=True
            )
            ref = self.m.apply_wo_a_fp8_mxscale(rotated.view(t, g, D), wq, e).float()
            self.assertLess(((fused - ref).norm() / ref.norm()).item(), 2e-2, t)

    def test_block128_weight_stays_row_major(self):
        """DeepSeek-V4's 128x128 weight passes through untouched to the row-major BMM."""
        g = 2
        w, s = _block_fp8(g, 128, seed=7)
        wq, e = self.m.wo_a_weight_scale_to_e8m0(w, s, g, R)
        self.assertEqual(self.m.wo_a_fp8_block(wq, e), 128)
        self.assertTrue(
            torch.equal(wq.view(torch.uint8), w.view(g, R, D).view(torch.uint8))
        )
        o = torch.randn(16, g, D, device="cuda").bfloat16()
        out = self.m.apply_wo_a_fp8_mxscale(o, wq, e)
        self._assert_matches_quantized_product(out, o, _dequant(w, s, 128), 128, (g,))


if __name__ == "__main__":
    unittest.main()
