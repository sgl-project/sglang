"""CPU-only tests for Quark MXFP4 AITER layout and kernel-selection helpers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization.quark.schemes import quark_w4a4_mxfp4
from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
    _afp4wfp4_k_tail_safe_tiling,
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


class TestAfp4wfp4KTailSafeTiling(CustomTestCase):
    def test_narrows_a_tail_in_an_unfilled_stage(self):
        # Two K blocks in three stages: the tail lands in a never-filled stage.
        self.assertEqual(
            _afp4wfp4_k_tail_safe_tiling(1024, k=1536, num_stages=3), (512, 3)
        )
        # No power of two >= 128 divides 2880; six blocks of 512 put the tail in a filled stage.
        self.assertEqual(
            _afp4wfp4_k_tail_safe_tiling(1024, k=2880, num_stages=3), (512, 3)
        )
        # A block wider than K is shrunk to K's next power of two first, as the kernel wrapper does.
        self.assertEqual(
            _afp4wfp4_k_tail_safe_tiling(2048, k=768, num_stages=3), (256, 3)
        )

    def test_keeps_a_tail_in_a_filled_stage(self):
        # Kimi-K3 TP8 dense down_proj: five blocks in three stages keep AITER's block (128 is 1.7x slower).
        self.assertEqual(
            _afp4wfp4_k_tail_safe_tiling(1024, k=4224, num_stages=3), (1024, 3)
        )
        # The boundary: one block more than stages.
        self.assertEqual(
            _afp4wfp4_k_tail_safe_tiling(1024, k=3136, num_stages=3), (1024, 3)
        )
        self.assertEqual(
            _afp4wfp4_k_tail_safe_tiling(512, k=1088, num_stages=2), (512, 2)
        )

    def test_drops_stages_when_the_minimum_block_is_not_enough(self):
        # Three blocks of the minimum 128 in three stages.
        self.assertEqual(
            _afp4wfp4_k_tail_safe_tiling(256, k=320, num_stages=3), (128, 2)
        )
        # One masked block cannot share a stage with a full one.
        self.assertIsNone(_afp4wfp4_k_tail_safe_tiling(256, k=96, num_stages=3))


class TestAfp4wfp4KTailConfig(CustomTestCase):
    def _config(self, get_config, m=16, n=7168, k_packed=768):
        quark_w4a4_mxfp4._afp4wfp4_k_tail_config.cache_clear()
        try:
            with (
                mock.patch.object(
                    quark_w4a4_mxfp4, "_aiter_afp4wfp4_get_config", get_config
                ),
                mock.patch.object(
                    quark_w4a4_mxfp4, "is_gfx95_supported", return_value=True
                ),
            ):
                return quark_w4a4_mxfp4._afp4wfp4_k_tail_config(
                    m=m, n=n, k_packed=k_packed
                )
        finally:
            quark_w4a4_mxfp4._afp4wfp4_k_tail_config.cache_clear()

    # AITER's untuned decode config for 8 < M <= 32.
    _DECODE_CONFIG = {
        "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_K": 1024,
        "num_stages": 3,
        "NUM_KSPLIT": 1,
    }

    def test_narrows_the_kernels_own_config(self):
        get_config = mock.Mock(return_value=(dict(self._DECODE_CONFIG), False))

        config = self._config(get_config, k_packed=768)

        # gemm_afp4wfp4's _get_config takes packed K, as gemm_afp4wfp4 passes it.
        get_config.assert_called_once_with(16, 7168, 768)
        self.assertEqual(config, {**self._DECODE_CONFIG, "BLOCK_SIZE_K": 512})

    def test_tail_in_a_filled_stage_keeps_aiters_config(self):
        # Kimi-K3 TP8 dense down_proj, K=4224.
        get_config = mock.Mock(return_value=(dict(self._DECODE_CONFIG), False))
        self.assertIsNone(self._config(get_config, k_packed=2112))

    def test_failed_lookup_keeps_the_default_path(self):
        for error in (
            KeyError("no M bucket"),
            AssertionError("no DEFAULT.json"),
            # The private AITER getter changed its signature.
            TypeError("changed signature"),
        ):
            with self.subTest(error=type(error).__name__):
                self.assertIsNone(self._config(mock.Mock(side_effect=error)))
        self.assertIsNone(self._config(None))

    def test_split_k_config_is_left_to_aiter(self):
        split_k = {"BLOCK_SIZE_K": 512, "num_stages": 2, "NUM_KSPLIT": 4}
        self.assertIsNone(self._config(mock.Mock(return_value=(split_k, False))))


if __name__ == "__main__":
    unittest.main()
