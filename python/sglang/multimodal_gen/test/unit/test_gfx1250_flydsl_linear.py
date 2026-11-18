# SPDX-License-Identifier: Apache-2.0
"""gfx1250 BF16 linears routed to aiter's FlyDSL GEMM (gfx1250 + AITER only)."""

import time
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers import linear

_HAS_TDM_GEMM = linear.USE_AITER_GFX1250 and linear._flydsl_tdm_gemm is not None


def _inputs(m: int, n: int, k: int):
    torch.manual_seed(0)
    x = torch.randn(1, m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    return x, w / k**0.5


def _assert_matches_fp32(out, x, w, bias=None):
    ref = F.linear(x.float(), w.float(), None if bias is None else bias.float())
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=2e-2)


@unittest.skipUnless(
    linear.USE_AITER_GFX1250 and not linear._USE_FLYDSL_TDM_GEMM,
    "needs a gfx1250 GPU with SGLANG_USE_AITER=1 and the TDM GEMM switch off",
)
class TestGfx1250FlyDSLLinear(unittest.TestCase):
    def test_large_bf16_linear_runs_flydsl_and_matches_fp32(self):
        x, w = _inputs(m=512, n=1024, k=512)
        with patch.object(linear, "_flydsl_gemm", wraps=linear._flydsl_gemm) as gemm:
            out = linear.apply_unquantized_linear(x, w)
        self.assertEqual(gemm.call_count, 1)
        self.assertEqual(out.shape, (1, 512, 1024))
        _assert_matches_fp32(out, x, w)

    def test_small_m_linear_keeps_the_default_gemm(self):
        x, w = _inputs(m=64, n=1024, k=512)
        with patch.object(linear, "_flydsl_gemm") as gemm:
            out = linear.apply_unquantized_linear(x, w)
        gemm.assert_not_called()
        _assert_matches_fp32(out, x, w)


@unittest.skipUnless(
    linear._USE_FLYDSL_TDM_GEMM,
    "needs a gfx1250 GPU with SGLANG_USE_AITER=1 and an aiter with the TDM GEMM",
)
class TestGfx1250FlyDSLTdmLinear(unittest.TestCase):
    def _run(self, x, w, bias=None):
        with (
            patch.object(
                linear, "_flydsl_tdm_gemm", wraps=linear._flydsl_tdm_gemm
            ) as tdm,
            patch.object(linear, "_flydsl_gemm", wraps=linear._flydsl_gemm) as gemm,
        ):
            out = linear.apply_unquantized_linear(x, w, bias)
        return out, tdm.call_count, gemm.call_count

    def test_tdm_matches_fp32_on_both_tile_sizes(self):
        # 128-row tiles when 256x256 underfills the CUs, 256-row tiles otherwise;
        # M=4112 also ends on a partial tile.
        for m, n, k in ((512, 1024, 512), (4112, 6144, 6144)):
            with self.subTest(m=m, n=n, k=k):
                x, w = _inputs(m, n, k)
                bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
                out, tdm_calls, gemm_calls = self._run(x, w, bias)
                self.assertEqual((tdm_calls, gemm_calls), (1, 0))
                self.assertEqual(out.shape, (1, m, n))
                _assert_matches_fp32(out, x, w, bias)

    def test_tdm_reads_inputs_changed_between_calls(self):
        x, w = _inputs(m=512, n=1024, k=512)
        first, _, _ = self._run(x, w)
        x.neg_()
        second, _, _ = self._run(x, w)
        # Not bit-exact: the kernel's fp32 -> bf16 rounding is not sign-symmetric on ties.
        torch.testing.assert_close(second.float(), -first.float(), atol=1e-3, rtol=1e-2)

    def test_partial_k_tile_keeps_the_default_kernel(self):
        x, w = _inputs(m=512, n=1024, k=576)
        out, tdm_calls, gemm_calls = self._run(x, w)
        self.assertEqual((tdm_calls, gemm_calls), (0, 1))
        _assert_matches_fp32(out, x, w)


# FLUX.2-dev 720p single block: (M, K, N) of its two linears, 85% of the GEMM time.
_FLUX2_SINGLE_BLOCK_SHAPES = ((4112, 6144, 55296), (4112, 24576, 6144))
# Arbitrary; long enough to average over the package's ~1 s power-throttle cycle.
_SECONDS_PER_KERNEL_PER_ROUND = 3.0
_ROUNDS = 3
# About twice the largest gap measured between the two kernels (7%).
_MAX_KERNEL_GAP = 0.15


def _sustained_tflops(gemm_2d, operands, seconds: float) -> float:
    flops, start = 0.0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        for x_2d, w, bias in operands:
            for _ in range(8):
                gemm_2d(x_2d, w, bias)
            flops += 8 * 2.0 * x_2d.shape[0] * x_2d.shape[1] * w.shape[0]
        torch.cuda.synchronize()
    return flops / (time.perf_counter() - start) / 1e12


@unittest.skipUnless(
    _HAS_TDM_GEMM, "needs a gfx1250 GPU and an aiter with the TDM GEMM"
)
class TestGfx1250GemmKernelParity(unittest.TestCase):
    def test_default_and_tdm_kernels_stay_close_on_flux2_shapes(self):
        """Neither FlyDSL GEMM path may fall far behind the other on FLUX.2's linears.

        Which one wins depends on package load: measured with aiter 86f6a2b57 the
        TDM kernel leads by ~7% with the sibling partition idle and by under 1%
        with it loaded, so only a gap larger than that spread is a regression.
        """
        operands = []
        for m, k, n in _FLUX2_SINGLE_BLOCK_SHAPES:
            x, w = _inputs(m, n, k)
            bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
            operands.append((x[0], w, bias))
        kernels = {
            "default": linear._flydsl_default_gemm_2d,
            "tdm": linear._flydsl_tdm_gemm_2d,
        }
        for gemm_2d in kernels.values():
            _sustained_tflops(gemm_2d, operands, seconds=0.5)
        # Alternate the kernels so both see the same clock and sibling-partition load.
        samples = {name: [] for name in kernels}
        for _ in range(_ROUNDS):
            for name, gemm_2d in kernels.items():
                samples[name].append(
                    _sustained_tflops(gemm_2d, operands, _SECONDS_PER_KERNEL_PER_ROUND)
                )
        tflops = {name: sorted(vals)[len(vals) // 2] for name, vals in samples.items()}
        ratio = tflops["tdm"] / tflops["default"]
        print(
            f"\nFLUX.2 single-block GEMM mix: default {tflops['default']:.0f} TFLOPS, "
            f"tdm {tflops['tdm']:.0f} TFLOPS (tdm/default = {ratio:.3f}); "
            f"rounds default {[round(v) for v in samples['default']]}, "
            f"tdm {[round(v) for v in samples['tdm']]}"
        )
        self.assertGreater(ratio, 1 - _MAX_KERNEL_GAP)
        self.assertLess(ratio, 1 + _MAX_KERNEL_GAP)


if __name__ == "__main__":
    unittest.main()
