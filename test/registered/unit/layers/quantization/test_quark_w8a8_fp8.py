"""Unit tests for srt/layers/quantization/quark/schemes/quark_w8a8_fp8 on ROCm."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8 import QuarkW8A8Fp8
from sglang.test.test_utils import CustomTestCase

_MODULE = "sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8"


def _per_channel_scheme() -> QuarkW8A8Fp8:
    return QuarkW8A8Fp8(
        weight_config={"qscheme": "per_channel"},
        input_config={"is_dynamic": True, "qscheme": "per_channel"},
    )


def _quantized_layer(out_features: int, in_features: int = 32):
    """A per-output-channel FP8 linear, the layout Quark serializes K3 with."""
    torch.manual_seed(0)
    dense = torch.randn(out_features, in_features, dtype=torch.float32)
    scale = dense.abs().amax(dim=1, keepdim=True) / 448.0
    weight = (dense / scale).to(torch.float8_e4m3fn)
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(weight, requires_grad=False),
        weight_scale=torch.nn.Parameter(scale.squeeze(1), requires_grad=False),
        input_scale=None,
        logical_widths=[out_features],
    )
    return layer, weight, scale


class TestNarrowOutputPartitionFp8(CustomTestCase):
    """hipBLASLt requires the GEMM's N to be a multiple of 16 and
    ``torch._scaled_mm`` raises rather than padding, so a per-channel FP8 linear
    whose output partition is narrower than that cannot be served quantized at
    any batch size. K3's KDA ``b_proj`` emits one beta channel per head (12 at
    TP=8) and hit exactly that. Such a layer is dequantized at load instead."""

    def setUp(self):
        self.scheme = _per_channel_scheme()
        # is_hip()/is_fp8_fnuz() are evaluated at import time; pin both so the
        # ROCm branch under test runs on any CI machine.
        patcher = patch.multiple(_MODULE, _is_hip=True, _is_fp8_fnuz=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_narrow_partition_is_dequantized_at_load(self):
        layer, weight, scale = _quantized_layer(out_features=12)

        self.scheme.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.dtype, self.scheme.out_dtype)
        # Dequantized weights fold the scale in, so keeping it would double-scale.
        self.assertIsNone(layer.weight_scale)
        self.assertIsNone(layer.input_scale)
        # apply_fp8_linear consumes [in, out]; the dense fallback must match it.
        self.assertEqual(tuple(layer.weight.shape), (32, 12))
        expected = (weight.to(torch.float32) * scale).to(self.scheme.out_dtype).t()
        torch.testing.assert_close(layer.weight, expected)

    def test_aligned_partition_stays_quantized(self):
        """The negative branch: widths hipBLASLt can serve keep their FP8
        weights, so the guard cannot degrade into dequantizing everything."""
        layer, _, _ = _quantized_layer(out_features=16)

        with patch(f"{_MODULE}.use_aiter_bpreshuffle_gemm", return_value=False):
            self.scheme.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertIsNotNone(layer.weight_scale)
        self.assertEqual(layer.weight_scale.numel(), 16)

    def test_dequantized_layer_is_served_by_a_dense_matmul(self):
        layer, weight, scale = _quantized_layer(out_features=12)
        self.scheme.process_weights_after_loading(layer)
        x = torch.randn(4, 32, dtype=self.scheme.out_dtype)

        out = self.scheme.apply_weights(layer, x)

        self.assertEqual(tuple(out.shape), (4, 12))
        torch.testing.assert_close(out, torch.matmul(x, layer.weight))

    def test_dense_fallback_applies_bias(self):
        layer, _, _ = _quantized_layer(out_features=12)
        self.scheme.process_weights_after_loading(layer)
        x = torch.randn(4, 32, dtype=self.scheme.out_dtype)
        bias = torch.randn(12, dtype=self.scheme.out_dtype)

        biased = self.scheme.apply_weights(layer, x, bias)

        torch.testing.assert_close(biased, self.scheme.apply_weights(layer, x) + bias)

    def test_non_rocm_keeps_the_narrow_partition_quantized(self):
        """The dequant is a ROCm workaround; other platforms must be untouched."""
        layer, _, _ = _quantized_layer(out_features=12)

        with patch.multiple(_MODULE, _is_hip=False, _is_fp8_fnuz=False):
            with patch(f"{_MODULE}.use_aiter_bpreshuffle_gemm", return_value=False):
                self.scheme.process_weights_after_loading(layer)

        self.assertEqual(layer.weight.dtype, torch.float8_e4m3fn)
        self.assertIsNotNone(layer.weight_scale)


if __name__ == "__main__":
    unittest.main()
