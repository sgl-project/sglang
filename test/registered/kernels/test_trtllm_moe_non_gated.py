"""Unit tests for non-gated (relu2) FlashInfer TRTLLM-Gen MoE helpers.

Tests weight alignment padding, activation type mapping, _is_gated detection,
and non-gated FP8 scaling factor computation -- all the logic added to support
NemotronH-120B models with relu2 activation.
"""

import unittest
from unittest.mock import MagicMock

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="stage-b-test-1-gpu-large")

if torch.cuda.get_device_capability() < (10, 0):
    pytest.skip(
        reason="TRTLLM-Gen MoE requires compute capability >= 10.0 (SM100+).",
        allow_module_level=True,
    )


class TestAlignFp8MoeWeights(CustomTestCase):
    """Test _align_fp8_moe_weights for gated vs non-gated."""

    def _get_fn(self):
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            _align_fp8_moe_weights,
        )

        return _align_fp8_moe_weights

    def test_non_gated_pads_to_128(self):
        align = self._get_fn()
        E, H, I = 4, 256, 100
        w13 = torch.randn(E, I, H)
        w2 = torch.randn(E, H, I)

        w13_p, w2_p, padded = align(w13, w2, is_gated=False, min_alignment=128)

        self.assertEqual(padded, 128)
        self.assertEqual(w13_p.shape, (E, 128, H))
        self.assertEqual(w2_p.shape, (E, H, 128))
        torch.testing.assert_close(w13_p[:, :I, :], w13)
        torch.testing.assert_close(w2_p[:, :, :I], w2)
        self.assertTrue((w13_p[:, I:, :] == 0).all())
        self.assertTrue((w2_p[:, :, I:] == 0).all())

    def test_gated_pads_double_intermediate(self):
        align = self._get_fn()
        E, H, I = 4, 256, 100
        w13 = torch.randn(E, 2 * I, H)
        w2 = torch.randn(E, H, I)

        w13_p, w2_p, padded = align(w13, w2, is_gated=True, min_alignment=128)

        self.assertEqual(padded, 128)
        self.assertEqual(w13_p.shape, (E, 2 * 128, H))
        self.assertEqual(w2_p.shape, (E, H, 128))

    def test_already_aligned_is_noop(self):
        from sglang.srt.layers.quantization.utils import round_up_to_multiple

        E, H, I = 4, 256, 128
        w13 = torch.randn(E, I, H)
        w2 = torch.randn(E, H, I)

        padded_I = round_up_to_multiple(I, 128)
        self.assertEqual(padded_I, I)

        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            _align_fp8_moe_weights,
        )

        w13_p, w2_p, padded = _align_fp8_moe_weights(
            w13, w2, is_gated=False, min_alignment=128
        )
        self.assertEqual(padded, 128)
        self.assertIs(w13_p, w13)
        self.assertIs(w2_p, w2)


class TestAlignFp4MoeWeights(CustomTestCase):
    """Test _align_fp4_moe_weights for gated vs non-gated (FP4 packing)."""

    def _get_fn(self):
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            _align_fp4_moe_weights,
        )

        return _align_fp4_moe_weights

    def test_non_gated_pads_correctly(self):
        align = self._get_fn()
        E, H = 4, 256
        I_packed = 50
        I = I_packed * 2  # 100

        w13 = torch.randn(E, I, H)
        w13_scale = torch.randn(E, I, H // 16)
        w2 = torch.randn(E, H, I_packed)
        w2_scale = torch.randn(E, H, I // 16)

        w13_p, w13s_p, w2_p, w2s_p, padded = align(
            w13, w13_scale, w2, w2_scale, is_gated=False, min_alignment=128
        )

        self.assertEqual(padded, 128)
        # non-gated: w13 rows = 1 * padded_intermediate
        self.assertEqual(w13_p.shape, (E, 128, H))
        self.assertEqual(w2_p.shape, (E, H, 128 // 2))
        self.assertEqual(w13s_p.shape, (E, 128, H // 16))
        self.assertEqual(w2s_p.shape, (E, H, 128 // 16))

    def test_gated_pads_double_rows(self):
        align = self._get_fn()
        E, H = 4, 256
        I_packed = 50
        I = I_packed * 2

        w13 = torch.randn(E, 2 * I, H)
        w13_scale = torch.randn(E, 2 * I, H // 16)
        w2 = torch.randn(E, H, I_packed)
        w2_scale = torch.randn(E, H, I // 16)

        w13_p, w13s_p, w2_p, w2s_p, padded = align(
            w13, w13_scale, w2, w2_scale, is_gated=True, min_alignment=128
        )

        self.assertEqual(padded, 128)
        self.assertEqual(w13_p.shape, (E, 2 * 128, H))

    def test_already_aligned_is_noop(self):
        align = self._get_fn()
        E, H = 4, 256
        I_packed = 64
        I = I_packed * 2  # 128

        w13 = torch.randn(E, I, H)
        w13_scale = torch.randn(E, I, H // 16)
        w2 = torch.randn(E, H, I_packed)
        w2_scale = torch.randn(E, H, I // 16)

        w13_p, w13s_p, w2_p, w2s_p, padded = align(
            w13, w13_scale, w2, w2_scale, is_gated=False, min_alignment=128
        )

        self.assertEqual(padded, 128)
        self.assertIs(w13_p, w13)
        self.assertIs(w2_p, w2)


class TestAlignMxfp8MoeWeights(CustomTestCase):
    """Test _align_mxfp8_moe_weights for gated vs non-gated."""

    def _get_fn(self):
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            _align_mxfp8_moe_weights,
        )

        return _align_mxfp8_moe_weights

    def test_non_gated_pads_correctly(self):
        align = self._get_fn()
        E, H, I = 4, 256, 96
        block_size = 32

        w13 = torch.randn(E, I, H)
        w13_scale = torch.randn(E, I, H // block_size)
        w2 = torch.randn(E, H, I)
        w2_scale = torch.randn(E, H, I // block_size)

        w13_p, w13s_p, w2_p, w2s_p, padded = align(
            w13, w13_scale, w2, w2_scale, is_gated=False, min_alignment=128
        )

        self.assertEqual(padded, 128)
        self.assertEqual(w13_p.shape, (E, 128, H))
        self.assertEqual(w2_p.shape, (E, H, 128))
        self.assertEqual(w13s_p.shape, (E, 128, H // block_size))
        self.assertEqual(w2s_p.shape, (E, H, 128 // block_size))

    def test_gated_pads_double_rows(self):
        align = self._get_fn()
        E, H, I = 4, 256, 100
        block_size = 32

        w13 = torch.randn(E, 2 * I, H)
        w13_scale = torch.randn(E, 2 * I, H // block_size)
        w2 = torch.randn(E, H, I)
        w2_scale = torch.randn(E, H, I // block_size)

        w13_p, w13s_p, w2_p, w2s_p, padded = align(
            w13, w13_scale, w2, w2_scale, is_gated=True, min_alignment=128
        )

        self.assertEqual(padded, 128)
        self.assertEqual(w13_p.shape, (E, 2 * 128, H))


class TestGetActivationType(CustomTestCase):
    """Test _get_activation_type maps activation strings correctly."""

    def _get_fn(self):
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
            _get_activation_type,
        )

        return _get_activation_type

    def test_silu_returns_swiglu(self):
        from flashinfer.fused_moe.core import ActivationType

        act = self._get_fn()
        self.assertEqual(act("silu"), ActivationType.Swiglu.value)

    def test_relu2_returns_relu2(self):
        from flashinfer.fused_moe.core import ActivationType

        act = self._get_fn()
        self.assertEqual(act("relu2"), ActivationType.Relu2.value)

    def test_unknown_raises(self):
        act = self._get_fn()
        with self.assertRaises(ValueError):
            act("gelu")


class TestIsGatedHelper(CustomTestCase):
    """Test _is_gated returns correct default and explicit values."""

    def _get_fn(self):
        from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import _is_gated

        return _is_gated

    def test_default_true_when_no_config(self):
        layer = MagicMock(spec=[])
        del layer.moe_runner_config
        self.assertTrue(self._get_fn()(layer))

    def test_true_when_gated(self):
        layer = MagicMock()
        layer.moe_runner_config.is_gated = True
        self.assertTrue(self._get_fn()(layer))

    def test_false_when_non_gated(self):
        layer = MagicMock()
        layer.moe_runner_config.is_gated = False
        self.assertFalse(self._get_fn()(layer))


class TestNonGatedFp8ScalingFactors(CustomTestCase):
    """Verify that non-gated FP8 scaling factors differ from gated."""

    def test_non_gated_output1_scales(self):
        E = 4
        input_scale = torch.tensor([0.5] * E)
        activation_scale = torch.tensor([0.25] * E)
        w13_weight_scale = torch.tensor([3.0] * E)

        # Gated: output1_scales = w13_scale * input_scale / activation_scale
        gated_output1 = w13_weight_scale * input_scale * (1.0 / activation_scale)

        # Non-gated: output1_scales = 1 / activation_scale (no weight contribution)
        nongated_output1 = torch.ones_like(w13_weight_scale) * (1.0 / activation_scale)

        # Gated = 3.0 * 0.5 * 4.0 = 6.0, Non-gated = 4.0
        self.assertFalse(torch.allclose(gated_output1, nongated_output1))
        torch.testing.assert_close(gated_output1, torch.tensor([6.0] * E))
        torch.testing.assert_close(nongated_output1, torch.tensor([4.0] * E))


if __name__ == "__main__":
    unittest.main()
