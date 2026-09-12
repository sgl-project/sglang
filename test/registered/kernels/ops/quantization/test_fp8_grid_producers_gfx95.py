"""The gfx950 fused fp8-grid producers (RMSNorm + fake-quant, clamp + silu * mul, wo_a GEMM epilogue) must match the unfused launches bitwise on the quant step."""

import unittest

import torch

from sglang.kernels.ops.activation.silu_and_mul_clamp_hip import (
    silu_and_mul_clamp_triton,
)
from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
    Fp8GridActivation,
    Mxfp8Activation,
    _mxfp8_e4m3_quantize_torch,
    dequant_mxfp8_to_bf16,
    fake_quant_fp8_activation,
)
from sglang.kernels.ops.quantization.rmsnorm_fake_quant_amd_gfx95 import (
    rmsnorm_fake_quant_fp8,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=35, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")


EPS = 1e-6


def _reference_norm(
    x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor = None
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


SHAPES = [(1, 5120), (33, 5120)]


class TestRmsnormFakeQuantFp8(CustomTestCase):
    def _make(self, m, k):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 2
        w = torch.rand(k, device="cuda", dtype=torch.bfloat16) * 2
        return x, w

    def test_matches_norm_then_fake_quant(self):
        torch.manual_seed(0)
        for m, k in SHAPES:
            x, w = self._make(m, k)
            fq, y = rmsnorm_fake_quant_fp8(x, w, EPS)
            self.assertIsInstance(fq, Fp8GridActivation)
            self.assertEqual(fq.x.shape, x.shape)
            self.assertEqual(y.dtype, torch.bfloat16)

            y_ref, _ = _reference_norm(x, w)
            ulp = _ulp(y, y_ref)
            self.assertLessEqual(ulp.max().item(), 1, (m, k))
            self.assertLessEqual((ulp > 0).float().mean().item(), 1e-4, (m, k))

            # The quant step is exact on the kernel's own bf16 norm output.
            self.assertTrue(torch.equal(fq.x, _reference_fake_quant(y)), (m, k))

            # identical to the unfused pair except where a 1-ulp pre-quant difference crossed an e4m3 midpoint
            fq_ref = _reference_fake_quant(y_ref)
            ne = fq.x != fq_ref
            self.assertLessEqual(ne.float().mean().item(), 1e-4, (m, k))
            if ne.any():
                rel = (
                    fq.x.float() - fq_ref.float()
                ).abs() / fq_ref.float().abs().clamp(min=1e-3)
                self.assertLessEqual(rel[ne].max().item(), 0.5, (m, k))

    def test_residual_add(self):
        torch.manual_seed(1)
        for m, k in [(17, 5120)]:
            x, w = self._make(m, k)
            residual = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            y_ref, res_ref = _reference_norm(x, w, residual.clone())
            res = residual.clone()
            fq, y = rmsnorm_fake_quant_fp8(x, w, EPS, residual=res)
            self.assertTrue(torch.equal(res, res_ref), (m, k))
            self.assertLessEqual(_ulp(y, y_ref).max().item(), 1, (m, k))
            self.assertTrue(torch.equal(fq.x, _reference_fake_quant(y)), (m, k))

    def test_rows_independent_and_repeatable(self):
        torch.manual_seed(2)
        for k in (5120,):
            x, w = self._make(64, k)
            fq, y = rmsnorm_fake_quant_fp8(x, w, EPS)
            fq2, y2 = rmsnorm_fake_quant_fp8(x, w, EPS)
            self.assertTrue(torch.equal(fq.x, fq2.x) and torch.equal(y, y2))
            for r in (0, 7, 63):
                fq1, y1 = rmsnorm_fake_quant_fp8(x[r : r + 1], w, EPS)
                self.assertTrue(torch.equal(fq1.x[0], fq.x[r]), (k, r))
                self.assertTrue(torch.equal(y1[0], y[r]), (k, r))
            fq_half, y_half = rmsnorm_fake_quant_fp8(x[:9], w, EPS)
            self.assertTrue(torch.equal(fq_half.x, fq.x[:9]))
            self.assertTrue(torch.equal(y_half, y[:9]))

    def test_emit_fp8_is_the_same_quantization(self):
        # the native-route fp8 codes + scales dequantize exactly to the same launch's fp8-grid bf16 activation
        torch.manual_seed(5)
        for m, k in SHAPES:
            x, w = self._make(m, k)
            fq, y = rmsnorm_fake_quant_fp8(x, w, EPS)
            q8, y8 = rmsnorm_fake_quant_fp8(x, w, EPS, emit_fp8=True)
            self.assertIsInstance(q8, Mxfp8Activation)
            self.assertEqual(q8.q.dtype, torch.float8_e4m3fn)
            self.assertEqual(tuple(q8.scale.shape), (m, k // 32))
            self.assertTrue(torch.equal(y8, y), (m, k))
            self.assertTrue(
                torch.equal(dequant_mxfp8_to_bf16(q8.q, q8.scale), fq.x), (m, k)
            )
            residual = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            r1, r2 = residual.clone(), residual.clone()
            fq_r, _ = rmsnorm_fake_quant_fp8(x, w, EPS, residual=r1)
            q8_r, _ = rmsnorm_fake_quant_fp8(x, w, EPS, residual=r2, emit_fp8=True)
            self.assertTrue(torch.equal(r1, r2))
            self.assertTrue(
                torch.equal(dequant_mxfp8_to_bf16(q8_r.q, q8_r.scale), fq_r.x)
            )


def _silu_mul_clamp_reference(gate_up: torch.Tensor, limit: float) -> torch.Tensor:
    g, u = gate_up.float().chunk(2, dim=-1)
    g = g.clamp(max=limit)
    u = u.clamp(min=-limit, max=limit)
    return (torch.nn.functional.silu(g) * u).to(gate_up.dtype)


class TestSiluAndMulClampTriton(CustomTestCase):
    def test_matches_torch_form(self):
        torch.manual_seed(0)
        for m, half, dtype in [(1, 576, torch.bfloat16), (33, 576, torch.bfloat16)]:
            x = torch.randn(m, 2 * half, device="cuda", dtype=dtype) * 6
            ref = _silu_mul_clamp_reference(x, 10.0)
            out = silu_and_mul_clamp_triton(x, 10.0)
            self.assertEqual(out.shape, ref.shape)
            self.assertEqual(out.dtype, dtype)
            # the clamps engage: some gate values exceed the limit, some up values sit at +-limit
            self.assertTrue((x[:, :half] > 10.0).any())
            tol = 1e-5 if dtype == torch.float32 else 2e-2
            torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=tol)

    def test_fp8_grid_epilogue_matches_separate_fake_quant(self):
        torch.manual_seed(1)
        for m, half in [(1, 576), (33, 576)]:
            x = torch.randn(m, 2 * half, device="cuda", dtype=torch.bfloat16) * 6
            plain = silu_and_mul_clamp_triton(x, 10.0)
            fused = silu_and_mul_clamp_triton(x, 10.0, fp8_grid=True)
            self.assertIsInstance(fused, Fp8GridActivation)
            ref = fake_quant_fp8_activation(plain)
            self.assertTrue(torch.equal(fused.x, ref), (m, half))
            # Idempotent: quantizing the fused output again changes nothing.
            self.assertTrue(torch.equal(fake_quant_fp8_activation(fused.x), fused.x))

    def test_emit_fp8_is_the_same_quantization(self):
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
            Mxfp8Activation,
            dequant_mxfp8_to_bf16,
        )

        torch.manual_seed(6)
        for m, inter in [(1, 576), (33, 576)]:
            gate_up = torch.randn(m, 2 * inter, device="cuda", dtype=torch.bfloat16) * 4
            grid = silu_and_mul_clamp_triton(gate_up, 7.0, fp8_grid=True)
            q8 = silu_and_mul_clamp_triton(gate_up, 7.0, emit_fp8=True)
            self.assertIsInstance(q8, Mxfp8Activation)
            self.assertEqual(q8.q.dtype, torch.float8_e4m3fn)
            self.assertTrue(
                torch.equal(dequant_mxfp8_to_bf16(q8.q, q8.scale), grid.x), (m, inter)
            )


GEMM_SHAPES = [(2, 1024, 4096)]
G, R, D = GEMM_SHAPES[0]


def _aiter_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """aiter batched_gemm_bf16 on the [G, T, D] copy, transposed back and flattened to [T, G * R]."""
    from aiter.ops.triton.gemm.batched.batched_gemm_bf16 import batched_gemm_bf16

    xq = x.transpose(0, 1).contiguous()
    y = batched_gemm_bf16(xq, w, dtype=torch.bfloat16)
    return y.transpose(0, 1).contiguous().flatten(1)


@unittest.skipUnless(
    is_hip() and is_gfx95_supported(), "gfx950 bf16-dequant dense route only"
)
class TestBatchedGemmBf16Fp8Grid(CustomTestCase):
    def setUp(self):
        from sglang.kernels.ops.gemm.gfx95_batched_gemm_bf16_fp8_grid import (
            batched_gemm_bf16_fp8_grid,
        )
        from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
            fake_quant_fp8_activation,
        )

        self.gemm = batched_gemm_bf16_fp8_grid
        self.fake_quant = fake_quant_fp8_activation

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
        """The single-launch regime (T above the split-K cap, or forced) is bitwise aiter's."""
        cases = [(1, 1.0), (64, 0.5)]
        for g, r, d in GEMM_SHAPES:
            for seed, (t, scale) in enumerate(cases):
                torch.manual_seed(seed)
                w = (torch.randn(g, r, d, device="cuda") * 0.02).bfloat16()
                x = (torch.randn(t, g, d, device="cuda") * scale).bfloat16()
                ref = _aiter_reference(x, w)
                plain = self.gemm(x, w, fp8_grid=False, split_k=False)
                self.assertEqual(plain.shape, (t, g * r))
                self.assertTrue(torch.equal(plain, ref), (g, r, d, t, scale))
                grid = self.gemm(x, w, split_k=False)
                self.assertTrue(
                    torch.equal(grid, self.fake_quant(ref)), (g, r, d, t, scale)
                )
                # Idempotent: the output is already on the grid.
                self.assertTrue(torch.equal(self.fake_quant(grid), grid))
            # Above the split-K cap the default regime is the single launch. aiter's own
            # kernel changes tile there, so the gate is the fp64 bound rather than bitwise.
            torch.manual_seed(99)
            w = (torch.randn(g, r, d, device="cuda") * 0.02).bfloat16()
            x = torch.randn(65, g, d, device="cuda").bfloat16()
            plain = self.gemm(x, w, fp8_grid=False)
            self._assert_within_bf16_of_exact(plain, x, w, (g, r, d, 65))
            self.assertTrue(torch.equal(self.gemm(x, w), self.fake_quant(plain)))

    def test_split_k_regime(self):
        """T <= 64 takes the split-K launches: within one bf16 ulp of the fp32 product
        (the reassociated sum), on the grid, batch-invariant and repeatable."""
        from sglang.kernels.ops.gemm.gfx95_batched_gemm_bf16_fp8_grid import (
            _split_k_applies,
        )

        for g, r, d in GEMM_SHAPES:
            if not _split_k_applies(1, d, r):
                continue
            torch.manual_seed(5)
            w = (torch.randn(g, r, d, device="cuda") * 0.02).bfloat16()
            x = (torch.randn(64, g, d, device="cuda") * 1.5).bfloat16()
            full_plain = self.gemm(x, w, fp8_grid=False)
            self.assertTrue(
                torch.equal(full_plain, self.gemm(x, w, fp8_grid=False, split_k=True))
            )
            self._assert_within_bf16_of_exact(full_plain, x, w, (g, r, d, "split"))
            single = self.gemm(x, w, fp8_grid=False, split_k=False)
            self._assert_within_bf16_of_exact(single, x, w, (g, r, d, "single"))
            full_grid = self.gemm(x, w)
            self.assertTrue(torch.equal(full_grid, self.fake_quant(full_plain)))
            for t in (1, 17):
                sub = self.gemm(x[:t], w)
                self.assertTrue(torch.equal(sub, full_grid[:t]), (g, r, d, t))
                self.assertTrue(
                    torch.equal(self.gemm(x[:t], w, fp8_grid=False), full_plain[:t])
                )
            for _ in range(3):
                self.assertTrue(torch.equal(self.gemm(x, w), full_grid))

    def test_odd_r_takes_the_single_launch(self):
        """R that is not a 32 multiple never takes split-K (its partial kernel stores
        whole N tiles): the default regime is the single launch and matches the
        reference."""
        from sglang.kernels.ops.gemm.gfx95_batched_gemm_bf16_fp8_grid import (
            _split_k_applies,
        )

        g, r, d = 2, 1000, 4096
        self.assertFalse(_split_k_applies(8, d, r))
        torch.manual_seed(7)
        w = (torch.randn(g, r, d, device="cuda") * 0.02).bfloat16()
        x = torch.randn(8, g, d, device="cuda").bfloat16()
        out = self.gemm(x, w, fp8_grid=False)
        self.assertEqual(out.shape, (8, g * r))
        self.assertTrue(
            torch.equal(out, self.gemm(x, w, fp8_grid=False, split_k=False))
        )
        self.assertTrue(torch.equal(out, _aiter_reference(x, w)))
        with self.assertRaises(AssertionError):
            self.gemm(x, w, fp8_grid=False, split_k=True)


if __name__ == "__main__":
    unittest.main()
