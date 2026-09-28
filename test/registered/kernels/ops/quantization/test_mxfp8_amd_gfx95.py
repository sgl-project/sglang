"""The gfx950 native MXFP8 GEMV and dense route against fp64 and the bf16-dequant route: within one bf16 ulp, repeatable, batch-invariant."""

import unittest

import torch

from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import (
    bf16_dequant_blockscaled_linear,
    dequant_block_fp8_weight_to_bf16,
    fake_quant_fp8_activation,
    fp8_grid_quantize,
)
from sglang.kernels.ops.quantization.mxfp8_native_amd_gfx95 import (
    mxfp8_gemv,
    mxfp8_native_blockscaled_linear,
    prepare_mxfp8_native_weight,
)
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=45, suite="stage-b-test-1-gpu-small-amd-mi35x")


# (N, K): a TP4 projection and the TP4 shared-expert down projection (K = 576, whose tail short
# of a 128-wide step is zero-padded in the weight and masked in the activation).
SHAPES = [
    (1792, 5120),
    (5120, 576),
]


def _quant_weight_block32(w: torch.Tensor):
    """fp8 e4m3 weight with one ue8m0 (power of two, fp32) scale per 32x32 block, ceil rule;
    a last row block short of 32 rows gets its own scale, as in a checkpoint."""
    n, k = w.shape
    row_blocks = -(-n // 32)
    padded = torch.nn.functional.pad(w.float(), (0, 0, 0, row_blocks * 32 - n))
    blocks = padded.view(row_blocks, 32, k // 32, 32)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-30)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (blocks / torch.exp2(e)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.view(row_blocks * 32, k)[:n], torch.exp2(e).view(row_blocks, k // 32)


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
                w_sh, ws8 = prepare_mxfp8_native_weight(wq, ws, [32, 32])
                xq, xs = fp8_grid_quantize(x)
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

    def test_non_finite_activations_encode_alike(self):
        """A NaN or +-inf in a bf16 activation quantizes in the GEMV as fp8_grid_quantize does
        (codes clamped to +-448, NaN to -448), so the input encoding cannot decide whether
        the row turns into NaN."""
        n, k = 96, 384
        wq, ws, _ = self._make(n, k, 1)
        w_sh, ws8 = prepare_mxfp8_native_weight(wq, ws, [32, 32])
        for bad in (float("nan"), float("inf"), float("-inf")):
            x = torch.randn(4, k, device="cuda", dtype=torch.bfloat16)
            x[0, 5] = bad
            xq, xs = fp8_grid_quantize(x)
            out_bf16 = mxfp8_gemv(x, w_sh, ws8)
            out_fp8 = mxfp8_gemv(xq, w_sh, ws8, xs)
            self.assertTrue(
                torch.equal(out_bf16.view(torch.int16), out_fp8.view(torch.int16)), bad
            )


ROUTE_SHAPES = [(1792, 5120), (5120, 576)]
# the dot_scaled tiles of the 64-row and 4096-row buckets (the gemv test covers M <= 32)
MS = (33, 1025)


@unittest.skipUnless(is_hip() and is_gfx95_supported(), "gfx950 native MXFP8 route")
class TestMxfp8NativeRouteGfx95(CustomTestCase):
    def _weights(self, n, k, seed=0):
        torch.manual_seed(seed)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        wq, ws = _quant_weight_block32(w)
        w_sh, ws8 = prepare_mxfp8_native_weight(wq, ws, [32, 32])
        w_bf16 = dequant_block_fp8_weight_to_bf16(wq, ws, [32, 32])
        return w_sh, ws8, w_bf16

    def test_within_one_bf16_ulp_of_the_bf16_route(self):
        for n, k in ROUTE_SHAPES:
            w_sh, ws8, w_bf16 = self._weights(n, k)
            for m in MS:
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                # groups whose amax is below the 1e-10 floor, where a quantizer off the CUDA rule diverges
                x[0] *= 1e-13
                ref = bf16_dequant_blockscaled_linear(x, w_bf16)
                out = mxfp8_native_blockscaled_linear(x, w_sh, ws8)
                # same products, different fp32 summation order: within one bf16 ulp of the row's largest output
                row_max = ref.float().abs().amax(dim=1, keepdim=True).clamp(min=1.0)
                ulp_of_row_max = torch.exp2(torch.floor(torch.log2(row_max)) - 7)
                diff = (out.float() - ref.float()).abs()
                self.assertTrue(
                    bool((diff <= ulp_of_row_max).all()),
                    (n, k, m, (diff / ulp_of_row_max).max().item()),
                )
                # the fp8-grid input and the fp8 + scales input must give the same result as the plain bf16 input
                out_grid = mxfp8_native_blockscaled_linear(
                    fake_quant_fp8_activation(x), w_sh, ws8
                )
                self.assertTrue(torch.equal(out_grid, out), (n, k, m))
                xq, xs = fp8_grid_quantize(x)
                out_q = mxfp8_native_blockscaled_linear(xq, w_sh, ws8, input_scale=xs)
                self.assertTrue(torch.equal(out_q, out), (n, k, m))


if __name__ == "__main__":
    unittest.main()
