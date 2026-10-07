"""The gfx950 fused fp8-grid producers (RMSNorm + fake-quant, clamp + silu * mul, wo_a GEMM epilogue) must match the unfused launches bitwise on the quant step."""

import unittest
from typing import Optional

import torch

from sglang.kernels.ops.activation.silu_and_mul_clamp_hip import (
    silu_and_mul_clamp_triton,
)
from sglang.kernels.ops.gemm.gfx95_batched_gemm_bf16_fp8_grid import (
    batched_gemm_bf16_fp8_grid,
)
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
    _mxfp8_e4m3_quantize_torch,
    dequant_mxfp8_to_bf16,
    fake_quant_fp8_activation,
)
from sglang.kernels.ops.quantization.rmsnorm_fake_quant_amd_gfx95 import (
    rmsnorm_fake_quant_fp8,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd-mi35x")


EPS = 1e-6


def _reference_norm(
    x: torch.Tensor, weight: torch.Tensor, residual: Optional[torch.Tensor] = None
):
    """The unfused RMSNorm: fp32 math, bf16 output (and bf16-rounded residual sum)."""
    xf = x.float()
    if residual is not None:
        xf = xf + residual.float()
        residual = xf.to(x.dtype)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    y = (xf * torch.rsqrt(var + EPS) * weight.float()).to(x.dtype)
    return y, residual


def _reference_fake_quant(y: torch.Tensor) -> torch.Tensor:
    """Per-32 ue8m0 scale, e4m3 round trip, back to bf16 (torch only)."""
    q, scales = _mxfp8_e4m3_quantize_torch(y)
    return dequant_mxfp8_to_bf16(q, scales)


def _ulp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "requires gfx950")
class TestRmsnormFakeQuantFp8(CustomTestCase):
    def _make(self, m, k):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 2
        w = torch.rand(k, device="cuda", dtype=torch.bfloat16) * 2
        return x, w

    def test_matches_norm_then_fake_quant(self):
        """The fused norm is within 1 ulp of the unfused one, its quant is exact on its own
        norm output, and the fp8 output dequantizes to the fp8-grid output."""
        torch.manual_seed(0)
        x, w = self._make(33, 5120)
        fq, y = rmsnorm_fake_quant_fp8(x, w, EPS)
        y_ref, _ = _reference_norm(x, w)
        ulp = _ulp(y, y_ref)
        self.assertLessEqual(ulp.max().item(), 1)
        self.assertLessEqual((ulp > 0).float().mean().item(), 1e-4)
        self.assertTrue(torch.equal(fq.x, _reference_fake_quant(y)))
        q8, y8 = rmsnorm_fake_quant_fp8(x, w, EPS, emit_fp8=True)
        self.assertTrue(torch.equal(y8, y))
        self.assertTrue(torch.equal(dequant_mxfp8_to_bf16(q8.q, q8.scale), fq.x))

    def test_residual_add(self):
        """The residual is updated in place with the bf16 sum and the norm reads that sum,
        the fused_add_rmsnorm contract."""
        torch.manual_seed(1)
        x, w = self._make(17, 5120)
        residual = torch.randn(17, 5120, device="cuda", dtype=torch.bfloat16)
        y_ref, res_ref = _reference_norm(x, w, residual.clone())
        res = residual.clone()
        _, y = rmsnorm_fake_quant_fp8(x, w, EPS, residual=res)
        self.assertTrue(torch.equal(res, res_ref))
        self.assertLessEqual(_ulp(y, y_ref).max().item(), 1)


def _silu_mul_clamp_reference(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    g, u = gate_up.float().chunk(2, dim=-1)
    g = g.clamp(max=limit)
    u = u.clamp(min=-limit, max=limit)
    return (torch.nn.functional.silu(g) * u).to(gate_up.dtype)


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "requires gfx950")
class TestSiluAndMulClampTriton(CustomTestCase):
    def test_matches_torch_form(self):
        torch.manual_seed(0)
        x = torch.randn(33, 2 * 576, device="cuda", dtype=torch.bfloat16) * 6
        # the clamps engage: some gate values exceed the limit
        self.assertTrue((x[:, :576] > 10.0).any())
        ref = _silu_mul_clamp_reference(x, 10.0)
        out = silu_and_mul_clamp_triton(x, 10.0)
        torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)

    def test_quantized_outputs_match_separate_fake_quant(self):
        """The fp8-grid epilogue equals a separate fake-quant of the plain output, and the
        fp8 output dequantizes to it."""
        torch.manual_seed(1)
        x = torch.randn(33, 2 * 576, device="cuda", dtype=torch.bfloat16) * 6
        grid = silu_and_mul_clamp_triton(x, 10.0, fp8_grid=True)
        q8 = silu_and_mul_clamp_triton(x, 10.0, emit_fp8=True)
        plain = silu_and_mul_clamp_triton(x, 10.0)
        self.assertTrue(torch.equal(grid.x, fake_quant_fp8_activation(plain)))
        self.assertTrue(torch.equal(dequant_mxfp8_to_bf16(q8.q, q8.scale), grid.x))


G, R, D = 2, 1024, 4096


def _aiter_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """aiter batched_gemm_bf16 on the [G, T, D] copy, transposed back and flattened to [T, G * R]."""
    from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16

    xq = x.transpose(0, 1).contiguous()
    y = batched_gemm_bf16(xq, w, dtype=torch.bfloat16)
    return y.transpose(0, 1).contiguous().flatten(1)


def _weights(seed: int, r: int = R) -> torch.Tensor:
    torch.manual_seed(seed)
    return (torch.randn(G, r, D, device="cuda") * 0.02).bfloat16()


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 wo_a GEMM")
class TestBatchedGemmBf16Fp8Grid(CustomTestCase):
    def _assert_within_bf16_of_exact(self, out, x, w, ctx):
        """One bf16 rounding (half an ulp of the result) of an fp32 sum whose
        association differs from aiter's: the fp32 error is bounded by the sum of
        absolute products."""
        exact = torch.einsum("tgd,grd->tgr", x.double(), w.double()).flatten(1)
        absprod = torch.einsum(
            "tgd,grd->tgr", x.abs().float(), w.abs().float()
        ).flatten(1)
        err = (out.double() - exact).abs()
        bound = exact.abs() * 2.0**-8 + absprod.double() * 2.0**-20 + 1e-6
        self.assertTrue(bool((err <= bound).all()), (*ctx, err.max().item()))

    def test_bitwise_against_aiter_and_separate_fake_quant(self):
        """The single-launch regime is bitwise aiter's, and its epilogue equals a separate
        fake-quant."""
        w = _weights(1)
        x = (torch.randn(64, G, D, device="cuda") * 0.5).bfloat16()
        ref = _aiter_reference(x, w)
        plain = batched_gemm_bf16_fp8_grid(x, w, fp8_grid=False, split_k=False)
        self.assertTrue(torch.equal(plain, ref))
        grid = batched_gemm_bf16_fp8_grid(x, w, split_k=False)
        self.assertTrue(torch.equal(grid, fake_quant_fp8_activation(ref)))

    def test_split_k_regime(self):
        """T <= 64 takes the split-K launches by default: within one bf16 rounding of the
        fp32 product (the reassociated sum), and on the grid."""
        w = _weights(5)
        x = (torch.randn(64, G, D, device="cuda") * 1.5).bfloat16()
        full_plain = batched_gemm_bf16_fp8_grid(x, w, fp8_grid=False)
        self.assertTrue(
            torch.equal(
                full_plain,
                batched_gemm_bf16_fp8_grid(x, w, fp8_grid=False, split_k=True),
            )
        )
        self._assert_within_bf16_of_exact(full_plain, x, w, ("split",))
        full_grid = batched_gemm_bf16_fp8_grid(x, w)
        self.assertTrue(torch.equal(full_grid, fake_quant_fp8_activation(full_plain)))

    def test_odd_r_takes_the_single_launch(self):
        """R that is not a 32 multiple never takes split-K (its partial kernel stores
        whole N tiles), so the default regime is the single launch."""
        w = _weights(7, r=1000)
        x = torch.randn(8, G, D, device="cuda").bfloat16()
        out = batched_gemm_bf16_fp8_grid(x, w, fp8_grid=False)
        self.assertTrue(
            torch.equal(
                out, batched_gemm_bf16_fp8_grid(x, w, fp8_grid=False, split_k=False)
            )
        )
        self.assertTrue(torch.equal(out, _aiter_reference(x, w)))


if __name__ == "__main__":
    unittest.main()
