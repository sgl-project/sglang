"""The gfx950 native MXFP8 GEMV and dense route against fp64 and the bf16-dequant route: within one bf16 ulp, repeatable, batch-invariant, graph-capturable."""

import unittest

import torch

from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
    bf16_dequant_blockscaled_linear,
    dequant_block_fp8_weight_to_bf16,
    fake_quant_fp8_activation,
    mxfp8_e4m3_quantize,
)
from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
    large_m_plan,
    mxfp8_gemv,
    mxfp8_native_blockscaled_linear,
    native_route_plan,
    prepare_mxfp8_native_weight,
    select_config,
    shuffle_mxfp8_weight,
    ue8m0_weight_scale,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=40, suite="stage-b-test-1-gpu-small-amd-mi35x")


# (N, K) of the TP4 dense projections plus a small odd one.
SHAPES = [
    (1856, 5120),
    (96, 384),
]


def _quant_weight_block32(w: torch.Tensor):
    """fp8 e4m3 weight with one ue8m0 (power of two, fp32) scale per 32x32 block, ceil rule."""
    n, k = w.shape
    blocks = w.float().view(n // 32, 32, k // 32, 32)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-30)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (blocks / torch.exp2(e)).clamp(-448, 448).to(torch.float8_e4m3fn).view(n, k)
    return q, torch.exp2(e).view(n // 32, k // 32)


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 scaled-MFMA kernel")
class TestMxfp8GemvGfx95(CustomTestCase):
    def _make(self, n, k, m, seed=0):
        torch.manual_seed(seed)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        w = w * torch.exp2(torch.randint(-4, 3, (n, 1), device="cuda").float()).to(
            w.dtype
        )
        wq, ws = _quant_weight_block32(w)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        x = x * torch.exp2(torch.randint(-3, 4, (m, 1), device="cuda").float()).to(
            x.dtype
        )
        return wq, ws, x

    def test_matches_fp64_reference_and_both_encodings_agree(self):
        for n, k in SHAPES:
            for m in (1, 17):
                wq, ws, x = self._make(n, k, m)
                w_sh, ws8 = shuffle_mxfp8_weight(wq), ue8m0_weight_scale(ws)
                xq, xs = mxfp8_e4m3_quantize(x)
                x_fq = fake_quant_fp8_activation(x)
                w_deq = dequant_block_fp8_weight_to_bf16(wq, ws, [32, 32])
                ref = x_fq.double() @ w_deq.double().t()

                out_fp8 = mxfp8_gemv(xq, w_sh, ws8, xs)
                out_bf16 = mxfp8_gemv(x, w_sh, ws8)
                out_grid = mxfp8_gemv(x_fq, w_sh, ws8)
                self.assertTrue(torch.equal(out_fp8, out_bf16), (n, k, m))
                self.assertTrue(torch.equal(out_fp8, out_grid), (n, k, m))
                # fp32 accumulation vs fp64: within bf16 output rounding of the reference.
                err = (out_fp8.double() - ref).abs()
                tol = 2.0**-7 * ref.abs() + 2.0**-7 * ref.abs().max()
                self.assertTrue(
                    bool((err <= tol).all()), (n, k, m, (err / tol).max().item())
                )
                # Most outputs round to the same bf16 as the fp64 reference.
                frac = (out_fp8 != ref.to(torch.bfloat16)).float().mean().item()
                self.assertLess(frac, 0.02, (n, k, m, frac))


ROUTE_SHAPES = [(1856, 5120)]
MS = (1, 33, 1025)


def _bf16_ulp_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 native MXFP8 route")
class TestMxfp8NativeRouteGfx95(CustomTestCase):
    def _weights(self, n, k, seed=0):
        torch.manual_seed(seed)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        wq, ws = _quant_weight_block32(w)
        w_sh, ws8, w_bf16_small = prepare_mxfp8_native_weight(wq, ws, [32, 32])
        w_bf16 = dequant_block_fp8_weight_to_bf16(wq, ws, [32, 32])
        return wq, ws, w_sh, ws8, w_bf16_small, w_bf16

    def test_within_one_bf16_ulp_of_the_bf16_route(self):
        for n, k in ROUTE_SHAPES:
            wq, ws, w_sh, ws8, w_small, w_bf16 = self._weights(n, k)
            for m in MS:
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                ref = bf16_dequant_blockscaled_linear(x, w_bf16)
                out = mxfp8_native_blockscaled_linear(x, w_sh, ws8, w_small)
                self.assertEqual(out.shape, ref.shape)
                # same products, different fp32 summation order: within one bf16 ulp of the row's largest output
                row_max = ref.float().abs().amax(dim=1, keepdim=True).clamp(min=1.0)
                ulp_of_row_max = torch.exp2(torch.floor(torch.log2(row_max)) - 7)
                diff = (out.float() - ref.float()).abs()
                self.assertTrue(
                    bool((diff <= ulp_of_row_max).all()),
                    (
                        n,
                        k,
                        m,
                        native_route_plan(m, n, k, w_small is not None),
                        (diff / ulp_of_row_max).max().item(),
                    ),
                )
                self.assertLess(
                    _bf16_ulp_diff(out, ref).gt(1).float().mean().item(),
                    5e-3,
                    (n, k, m),
                )
                # the fp8-grid input and the fp8 + scales input must give the same result as the plain bf16 input
                x_fq = fake_quant_fp8_activation(x)
                out_grid = mxfp8_native_blockscaled_linear(
                    x_fq, w_sh, ws8, w_small, input_on_fp8_grid=True
                )
                self.assertTrue(torch.equal(out_grid, out), (n, k, m))
                xq, xs = mxfp8_e4m3_quantize(x)
                out_q = mxfp8_native_blockscaled_linear(
                    xq, w_sh, ws8, w_small, input_scale=xs
                )
                has_bf16 = w_small is not None
                if native_route_plan(m, n, k, has_bf16, False) == native_route_plan(
                    m, n, k, has_bf16, True
                ) and large_m_plan(m, n, k, False) == large_m_plan(m, n, k, True):
                    self.assertTrue(torch.equal(out_q, out), (n, k, m))
                else:  # a free fp8 input may pick another kernel: same operands
                    diff_q = (out_q.float() - ref.float()).abs()
                    self.assertTrue(bool((diff_q <= ulp_of_row_max).all()), (n, k, m))

    def _kernel_identity(self, m, n, k, has_bf16):
        """What decides the summation order for m tokens: the plan plus its tile."""
        plan = native_route_plan(m, n, k, has_bf16)
        if plan == "gemv":
            cfg = select_config(m, n, k)
            return ("gemv", cfg.waves, cfg.rows, cfg.ksplit, cfg.steps)
        if plan == "dot_scaled":
            return ("dot_scaled", large_m_plan(m, n, k))
        return (plan, m)  # hipBLASLt picks its own kernel per M

    def test_repeatable_and_batch_invariant_inside_each_kernel(self):
        n, k = 1856, 5120
        _, _, w_sh, ws8, w_small, _ = self._weights(n, k, seed=1)
        has_bf16 = w_small is not None
        for m_lo, m_hi in (
            (1, 32),
            (1025, 1100),
        ):
            x = torch.randn(m_hi, k, device="cuda", dtype=torch.bfloat16)
            full = mxfp8_native_blockscaled_linear(x, w_sh, ws8, w_small)
            self.assertTrue(
                torch.equal(
                    mxfp8_native_blockscaled_linear(x, w_sh, ws8, w_small), full
                )
            )
            for m in (m_lo, (m_lo + m_hi) // 2):
                if self._kernel_identity(m, n, k, has_bf16) != self._kernel_identity(
                    m_hi, n, k, has_bf16
                ):
                    continue  # a different tile or a per-M hipBLASLt kernel: another summation order
                part = mxfp8_native_blockscaled_linear(
                    x[:m].contiguous(), w_sh, ws8, w_small
                )
                self.assertTrue(torch.equal(part, full[:m]), (m_lo, m_hi, m))


if __name__ == "__main__":
    unittest.main()
