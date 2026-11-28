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
    "needs a gfx1250 GPU with SGLANG_USE_AITER=1 and an aiter without the TDM GEMM",
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
        # M=4112 ends on 16 leftover rows, which take a second 16-row-tile launch.
        for m, n, k, launches in ((512, 1024, 512, 1), (4112, 6144, 6144, 2)):
            with self.subTest(m=m, n=n, k=k):
                x, w = _inputs(m, n, k)
                bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
                out, tdm_calls, gemm_calls = self._run(x, w, bias)
                self.assertEqual((tdm_calls, gemm_calls), (launches, 0))
                self.assertEqual(out.shape, (1, m, n))
                _assert_matches_fp32(out, x, w, bias)

    def test_tdm_reads_inputs_changed_between_calls(self):
        x, w = _inputs(m=512, n=1024, k=512)
        first, _, _ = self._run(x, w)
        x.neg_()
        second, _, _ = self._run(x, w)
        # Not bit-exact: the kernel's fp32 -> bf16 rounding is not sign-symmetric on ties.
        torch.testing.assert_close(second.float(), -first.float(), atol=1e-3, rtol=1e-2)

    def test_offsets_past_32_bits_keep_the_default_kernel(self):
        """A shape whose operand offsets overflow 32 bits must not reach the TDM kernel.

        aiter indexes A, B and C with signed 32-bit byte offsets and raises past
        that instead of clamping, so a caller that routes such a shape there kills
        the request. Wan 2.2's QKV projection is over the line; FLUX.2 is not.
        """
        meta = dict(device="meta", dtype=torch.bfloat16)
        self.assertGreater(90000 * 30720, linear._TDM_MAX_LEADING_OFFSET)
        self.assertFalse(
            linear._tdm_offsets_fit(
                torch.empty(90000, 5120, **meta), torch.empty(30720, 5120, **meta)
            )
        )
        self.assertTrue(
            linear._tdm_offsets_fit(
                torch.empty(4112, 6144, **meta), torch.empty(55296, 6144, **meta)
            )
        )

    def test_partial_k_tile_keeps_the_default_kernel(self):
        x, w = _inputs(m=512, n=1024, k=576)
        out, tdm_calls, gemm_calls = self._run(x, w)
        self.assertEqual((tdm_calls, gemm_calls), (0, 1))
        _assert_matches_fp32(out, x, w)


@unittest.skipUnless(
    linear._USE_FLYDSL_SWIGLU_GEMM,
    "needs a gfx1250 GPU with SGLANG_USE_AITER=1 and an aiter whose TDM GEMM "
    "takes the SwiGLU epilogue",
)
class TestGfx1250FlyDSLSwigluLinear(unittest.TestCase):
    def test_epilogue_matches_unfused_swiglu_including_leftover_rows(self):
        # M=4112 covers both launches: 256-row tiles and the 16 leftover rows.
        m, k, plain, pairs, prefix = 4112, 512, 512, 1024, 256
        x, w = _inputs(m, plain + 2 * pairs, k)
        interleaved = w.clone()
        interleaved[plain:] = torch.stack(
            (w[plain : plain + pairs], w[plain + pairs :]), dim=1
        ).flatten(0, 1)
        self.assertTrue(
            linear.can_use_flydsl_swiglu_linear(x, interleaved, plain, prefix)
        )
        out, packed = linear.apply_flydsl_swiglu_linear(x, interleaved, plain, prefix)

        ref = linear.apply_unquantized_linear(x, w)
        # The reference's leftover rows use split-K, so only the tiled rows are exact.
        self.assertTrue(torch.equal(out[:, :4096], ref[:, :4096, :plain]))
        torch.testing.assert_close(
            out.float(), ref[..., :plain].float(), atol=3e-2, rtol=2e-2
        )
        gate, up = ref[..., plain:].float().chunk(2, dim=-1)
        self.assertEqual(packed.shape, (1, m, prefix + pairs))
        # swiglu_col = 0 activates every column, the double blocks' feed-forward.
        self.assertTrue(
            linear.can_use_flydsl_swiglu_linear(x, interleaved[plain:], 0, 0)
        )
        _, all_activated = linear.apply_flydsl_swiglu_linear(
            x, interleaved[plain:], 0, 0
        )
        self.assertTrue(torch.equal(all_activated, packed[..., prefix:]))

        activated = packed[..., prefix:].float()
        expected = F.silu(gate) * up
        # The epilogue's exp2/rcp sigmoid may differ from torch's by one BF16 ulp.
        torch.testing.assert_close(
            activated[:, :4096], expected[:, :4096], atol=1e-3, rtol=1e-2
        )
        # Leftover rows also carry the two launches' different BF16 rounding.
        torch.testing.assert_close(activated, expected, atol=1e-1, rtol=5e-2)


# FLUX.2-dev 720p single block: (M, K, N) of its two linears, 85% of the GEMM time.
_FLUX2_SINGLE_BLOCK_SHAPES = ((4112, 6144, 55296), (4112, 24576, 6144))
# Arbitrary; long enough to average over the package's ~1 s power-throttle cycle.
_SECONDS_PER_KERNEL_PER_ROUND = 3.0
_ROUNDS = 3
# Arbitrary margin under parity; the TDM path measures 20-23% ahead.
_MAX_TDM_DEFICIT = 0.05


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
    def test_tdm_kernel_does_not_fall_behind_default_on_flux2_shapes(self):
        """The TDM path, which is preferred where available, must not lose to the
        a16w16 one it displaces on FLUX.2's linears.

        Measured with aiter 86f6a2b57: the TDM path (2x2 waves, leftover rows in
        their own launch) leads by ~23% with the sibling partition idle and by
        ~20% with it loaded.
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
        self.assertGreater(ratio, 1 - _MAX_TDM_DEFICIT)


if __name__ == "__main__":
    unittest.main()
