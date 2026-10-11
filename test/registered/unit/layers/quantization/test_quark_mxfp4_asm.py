"""CPU-only tests for Quark MXFP4 AITER ASM layout helpers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
    _asm_fp4_prequantized_input,
    _asm_fp4_scale_swizzle_supported,
    _shuffle_asm_fp4_act_scale,
    _swizzle_asm_fp4_scale,
)
from sglang.test.test_utils import CustomTestCase


class TestQuarkMxfp4AsmScaleLayout(CustomTestCase):
    def test_supported_shape(self):
        scale = torch.empty((32, 8), dtype=torch.uint8)
        self.assertTrue(_asm_fp4_scale_swizzle_supported(scale))

    def test_rejects_unsupported_shapes(self):
        self.assertFalse(
            _asm_fp4_scale_swizzle_supported(torch.empty((16, 8), dtype=torch.uint8))
        )
        self.assertFalse(
            _asm_fp4_scale_swizzle_supported(torch.empty((32, 4), dtype=torch.uint8))
        )
        self.assertFalse(
            _asm_fp4_scale_swizzle_supported(torch.empty((1, 32, 8), dtype=torch.uint8))
        )

    def test_swizzle_matches_aiter_tile_permutation(self):
        scale = torch.arange(32 * 8, dtype=torch.int32).view(32, 8)
        expected = (
            scale.view(1, 2, 16, 1, 2, 4, 1)
            .permute(0, 3, 5, 2, 4, 1, 6)
            .contiguous()
            .view(32, 8)
        )
        actual = _swizzle_asm_fp4_scale(scale)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, scale.shape)
        self.assertTrue(actual.is_contiguous())


class TestQuarkMxfp4AsmTupleInput(CustomTestCase):
    def test_act_scale_shuffle_matches_aiter_padded_layout(self):
        """Row-major (M, K/32) activation scales from the Triton fused-quant
        kernels must land in the (pad256(M), pad8(K/32)) tile order that
        per_1x32_f4_quant(shuffle=True) emits, with E8M0 1.0 in the padding."""
        rows, cols = 3, 6
        scale = torch.randint(0, 0x7F, (rows, cols), dtype=torch.uint8)

        shuffled = _shuffle_asm_fp4_act_scale(scale)

        self.assertEqual(shuffled.dtype, torch.float8_e8m0fnu)
        self.assertEqual(tuple(shuffled.shape), (256, 8))
        # Invert the AITER tile permutation (0, 3, 5, 2, 4, 1, 6).
        unshuffled = (
            shuffled.view(torch.uint8)
            .view(256 // 32, 8 // 8, 4, 16, 2, 2, 1)
            .permute(0, 5, 3, 1, 4, 2, 6)
            .reshape(256, 8)
        )
        expected = torch.full((256, 8), 0x7F, dtype=torch.uint8)
        expected[:rows, :cols] = scale
        torch.testing.assert_close(unshuffled, expected, rtol=0, atol=0)

    def test_only_quantized_pair_is_accepted(self):
        rows, k = 4, 64
        x_q = torch.zeros((rows, k // 2), dtype=torch.uint8)
        x_scales = torch.full((rows, k // 32), 0x7F, dtype=torch.uint8)

        a, a_scales = _asm_fp4_prequantized_input((x_q, x_scales))
        self.assertEqual(a.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(a.data_ptr(), x_q.data_ptr())
        self.assertEqual(tuple(a_scales.shape), (256, 8))

        # The 3- and 5-tuples ask for Triton-only fused epilogues; they must not
        # be silently unpacked as (x, x_scales).
        for fused in ((x_q, x_scales, torch.empty(0)), (x_q,) * 5):
            with self.assertRaises(NotImplementedError):
                _asm_fp4_prequantized_input(fused)


class TestQuarkMxfp4AsmNarrowLayer(CustomTestCase):
    def test_narrow_layer_stays_on_triton(self):
        """With the ASM flag on, a layer with N <= 128 (Qwen3.8 GDN in_proj_ba,
        N=96) must keep the Triton GEMM: AITER's untuned ASM heuristic returns
        wrong rows 1..M-1 for it at small batch sizes."""
        n, k = 96, 5120
        layer = torch.nn.Module()
        weight = torch.nn.Parameter(
            torch.zeros((n, k // 2), dtype=torch.uint8), requires_grad=False
        )
        layer.weight = weight
        layer.weight_scale = torch.nn.Parameter(
            torch.zeros((n, k // 32), dtype=torch.uint8), requires_grad=False
        )
        scheme = quark_w4a4_mxfp4.QuarkW4A4MXFP4({}, {})

        with (
            mock.patch.object(quark_w4a4_mxfp4, "_use_aiter_asm_fp4_gemm", True),
            mock.patch.object(
                quark_w4a4_mxfp4, "is_gfx95_supported", return_value=True
            ),
        ):
            scheme.process_weights_after_loading(layer)

        self.assertFalse(layer.use_aiter_asm_fp4_gemm)
        self.assertIs(layer.weight, weight)


def _shuffle_weight_ref(x: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
    """AITER shuffle_weight for uint8 (fp4x2) weights, layout (16, 16)."""
    assert layout == (16, 16)
    n, k = x.shape
    return (
        x.view(1, n // 16, 16, k // 32, 2, 16)
        .permute(0, 1, 3, 4, 2, 5)
        .contiguous()
        .view(n, k)
    )


def _unshuffle_weight(x: torch.Tensor) -> torch.Tensor:
    n, k = x.shape
    return x.view(n // 16, k // 32, 2, 16, 16).permute(0, 3, 1, 2, 4).reshape(n, k)


def _unswizzle_scale(scale: torch.Tensor) -> torch.Tensor:
    rows, cols = scale.shape
    return (
        scale.view(torch.uint8)
        .view(rows // 32, cols // 8, 4, 16, 2, 2, 1)
        .permute(0, 5, 3, 1, 4, 2, 6)
        .reshape(rows, cols)
    )


_FP4_LUT = torch.tensor(quark_w4a4_mxfp4._MXFP4_VALUES, dtype=torch.float64)


def _dequant(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Row-major fp4x2 codes and E8M0 scales to float64, exactly."""
    rows, k_packed = codes.shape
    vals = torch.empty(rows, k_packed * 2, dtype=torch.float64)
    vals[:, 0::2] = _FP4_LUT[(codes & 0xF).long()]
    vals[:, 1::2] = _FP4_LUT[(codes >> 4).long()]
    scale = torch.exp2(scale.view(torch.uint8).to(torch.float64) - 127.0)
    return (vals.view(rows, -1, 32) * scale.unsqueeze(-1)).view(rows, -1)


def _asm_gemm_ref(x_q, x_scales, weight, weight_scale) -> torch.Tensor:
    """gemm_a4w4(bpreshuffle=True) semantics: shuffled weight, swizzled scales.
    Products and sums of these dyadic values are exact in float64, so the
    result does not depend on the accumulation order."""
    m = x_q.shape[0]
    x = _dequant(x_q, _unswizzle_scale(x_scales)[:m])
    w = _dequant(_unshuffle_weight(weight), _unswizzle_scale(weight_scale))
    return x @ w.t()


class TestQuarkMxfp4AsmPackedTail(CustomTestCase):
    def setUp(self):
        patcher = mock.patch.object(
            quark_w4a4_mxfp4, "shuffle_weight", _shuffle_weight_ref, create=True
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _rand_proj(n, k, gen):
        weight = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=gen)
        scale = torch.randint(120, 134, (n, k // 32), dtype=torch.uint8, generator=gen)
        return weight, scale

    def test_packed_rows_match_separate_gemms(self):
        """Qwen3.8 GDN in_proj packing: qkvz in ASM layout + in_proj_ba (N=96,
        raw layout) as a zero-padded tail. The shuffle is local to 16 rows and
        the swizzle to 32, so one GEMM over the packed buffer must reproduce
        both separate GEMMs exactly, and the padding rows must read zero."""
        gen = torch.Generator().manual_seed(0)
        k, n_qkvz, n_ba, m = 256, 512, 96, 5
        qkvz_w, qkvz_s = self._rand_proj(n_qkvz, k, gen)
        ba_w, ba_s = self._rand_proj(n_ba, k, gen)
        # process_weights_after_loading for the ASM layer.
        qkvz_w_asm = _shuffle_weight_ref(qkvz_w)
        qkvz_s_asm = _swizzle_asm_fp4_scale(qkvz_s)

        packed = quark_w4a4_mxfp4.pack_asm_fp4_weight_tail(
            qkvz_w_asm, qkvz_s_asm, ba_w, ba_s
        )
        self.assertIsNotNone(packed)
        weight, scale = packed
        self.assertEqual(tuple(weight.shape), (n_qkvz + 256, k // 2))
        self.assertEqual(tuple(scale.shape), (n_qkvz + 256, k // 32))
        torch.testing.assert_close(weight[:n_qkvz], qkvz_w_asm, rtol=0, atol=0)
        torch.testing.assert_close(scale[:n_qkvz], qkvz_s_asm, rtol=0, atol=0)

        x_q, x_s = self._rand_proj(m, k, gen)
        x_s_asm = _shuffle_asm_fp4_act_scale(x_s)
        out = _asm_gemm_ref(x_q, x_s_asm, weight, scale)
        x = _dequant(x_q, x_s)
        torch.testing.assert_close(
            out[:, :n_qkvz], x @ _dequant(qkvz_w, qkvz_s).t(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            out[:, n_qkvz : n_qkvz + n_ba],
            x @ _dequant(ba_w, ba_s).t(),
            rtol=0,
            atol=0,
        )
        self.assertTrue(bool((out[:, n_qkvz + n_ba :] == 0).all()))

    def test_unpackable_projections_are_rejected(self):
        """Packing must fall back (None) rather than build a buffer whose
        leading rows are not whole ASM tiles or whose K differs."""
        gen = torch.Generator().manual_seed(1)
        k = 256
        ba_w, ba_s = self._rand_proj(96, k, gen)
        pack = quark_w4a4_mxfp4.pack_asm_fp4_weight_tail
        # Leading rows not a multiple of 256 (whole gemm_a4w4 N tiles).
        w, s = self._rand_proj(288, k, gen)
        self.assertIsNone(pack(w, s, ba_w, ba_s))
        # K mismatch.
        w, s = self._rand_proj(256, 2 * k, gen)
        self.assertIsNone(pack(w, s, ba_w, ba_s))
        # K/32 not a multiple of 8: the scale swizzle does not apply.
        w, s = self._rand_proj(256, 128, gen)
        tail_w, tail_s = self._rand_proj(96, 128, gen)
        self.assertIsNone(pack(w, s, tail_w, tail_s))


if __name__ == "__main__":
    unittest.main()
