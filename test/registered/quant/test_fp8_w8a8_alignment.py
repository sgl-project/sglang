# Tests for FP8 W8A8 16-alignment padding logic in CompressedTensorsW8A8Fp8.
#
# Motivation: torch._scaled_mm requires both matrix dimensions to be divisible
# by 16.  Models whose intermediate_size / tp_size is not 16-aligned (e.g.
# GLM-4.5 with 10944 / tp=8 = 1368) fail CUDA Graph capture without padding.
# These tests verify the pad/unpad round-trip without launching a server.

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import (
    FP8_ALIGNMENT,
    CompressedTensorsW8A8Fp8,
    _pad_to_alignment,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, suite="stage-b-test-1-gpu-small")


def _make_fp8_weight(out_dim: int, in_dim: int) -> torch.Tensor:
    return torch.zeros(out_dim, in_dim, dtype=torch.float8_e4m3fn, device="cuda")


def _make_scheme(strategy_str: str) -> CompressedTensorsW8A8Fp8:
    from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy

    strategy = getattr(QuantizationStrategy, strategy_str)
    weight_quant = MagicMock(spec=QuantizationArgs)
    weight_quant.strategy = strategy
    weight_quant.block_structure = None
    return CompressedTensorsW8A8Fp8(
        weight_quant=weight_quant, is_static_input_scheme=False
    )


def _make_layer(weight: torch.Tensor, weight_scale: torch.Tensor) -> SimpleNamespace:
    """Build a minimal fake layer that mimics what process_weights_after_loading
    receives from the model loader."""
    layer = SimpleNamespace()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    layer.logical_widths = [weight.shape[0]]

    # register_buffer / register_parameter shims
    def _register_buffer(name, tensor, persistent=True):
        setattr(layer, name, tensor)

    layer.register_buffer = _register_buffer
    return layer


class TestPadToAlignment(CustomTestCase):
    """Unit tests for the _pad_to_alignment helper."""

    def test_already_aligned_is_noop(self):
        t = torch.zeros(16, 32)
        result = _pad_to_alignment(t, dim=0, alignment=FP8_ALIGNMENT)
        self.assertIs(result, t)

    def test_pads_dim0(self):
        t = torch.zeros(13, 32)
        result = _pad_to_alignment(t, dim=0, alignment=16)
        self.assertEqual(result.shape[0], 16)
        self.assertEqual(result.shape[1], 32)

    def test_pads_dim1(self):
        t = torch.zeros(32, 13)
        result = _pad_to_alignment(t, dim=1, alignment=16)
        self.assertEqual(result.shape[0], 32)
        self.assertEqual(result.shape[1], 16)

    def test_pads_to_correct_multiple(self):
        # 1368 = 85 * 16 + 8 → next multiple is 86 * 16 = 1376
        t = torch.zeros(1368, 64)
        result = _pad_to_alignment(t, dim=0, alignment=16)
        self.assertEqual(result.shape[0], 1376)

    def test_pad_values_are_zero(self):
        t = torch.ones(13, 16)
        result = _pad_to_alignment(t, dim=0, alignment=16)
        # padded rows should be zero
        self.assertTrue(torch.all(result[13:] == 0))
        # original rows should be unchanged
        self.assertTrue(torch.all(result[:13] == 1))


class TestTensorStrategyAlignment(CustomTestCase):
    """process_weights_after_loading (TENSOR) pads weight and apply_weights
    returns the correct unpadded output shape."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    def _run(self, out_dim: int, in_dim: int):
        scheme = _make_scheme("TENSOR")

        weight = _make_fp8_weight(out_dim, in_dim)
        # Per-tensor scale: single scalar
        weight_scale = torch.tensor(
            [torch.finfo(torch.float32).min] * 1, dtype=torch.float32, device="cuda"
        )

        layer = _make_layer(weight, weight_scale)
        # requantize_with_max_scale expects logical_widths matching weight rows
        layer.logical_widths = [out_dim]

        scheme.process_weights_after_loading(layer)

        # Weight stored transposed: (K_padded, N_padded)
        k_padded, n_padded = layer.weight.shape
        self.assertEqual(k_padded % FP8_ALIGNMENT, 0)
        self.assertEqual(n_padded % FP8_ALIGNMENT, 0)

        # _orig_output_dim must match the original out_dim
        self.assertEqual(layer._orig_output_dim, out_dim)

    def test_non_aligned(self):
        # 1368 is 10944/8 — the exact failing case from GLM-4.5 with tp=8
        self._run(out_dim=1368, in_dim=2048)

    def test_aligned(self):
        # Already-aligned — should be a no-op
        self._run(out_dim=1024, in_dim=2048)

    def test_output_shape_after_apply(self):
        out_dim, in_dim = 1368, 2048
        scheme = _make_scheme("TENSOR")

        weight = _make_fp8_weight(out_dim, in_dim)
        weight_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        layer = _make_layer(weight, weight_scale)
        layer.logical_widths = [out_dim]
        layer.input_scale = None

        scheme.process_weights_after_loading(layer)

        batch = 4
        x = torch.zeros(batch, in_dim, dtype=torch.bfloat16, device="cuda")
        output = scheme.apply_weights(layer, x)

        self.assertEqual(output.shape, (batch, out_dim))

    def test_output_shape_with_bias(self):
        """Bias (shape [out_dim]) must not cause a shape mismatch when N is padded."""
        out_dim, in_dim = 1368, 2048
        scheme = _make_scheme("TENSOR")

        weight = _make_fp8_weight(out_dim, in_dim)
        weight_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        layer = _make_layer(weight, weight_scale)
        layer.logical_widths = [out_dim]
        layer.input_scale = None

        scheme.process_weights_after_loading(layer)

        batch = 4
        x = torch.zeros(batch, in_dim, dtype=torch.bfloat16, device="cuda")
        bias = torch.ones(out_dim, dtype=torch.bfloat16, device="cuda")
        output = scheme.apply_weights(layer, x, bias=bias)

        self.assertEqual(output.shape, (batch, out_dim))


class TestChannelStrategyAlignment(CustomTestCase):
    """process_weights_after_loading (CHANNEL) pads weight + scale and
    apply_weights returns the correct unpadded output shape."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    def _run(self, out_dim: int, in_dim: int):
        scheme = _make_scheme("CHANNEL")

        weight = _make_fp8_weight(out_dim, in_dim)
        # Per-channel scale: shape (N, 1)
        weight_scale = torch.ones(out_dim, 1, dtype=torch.float32, device="cuda")

        layer = _make_layer(weight, weight_scale)

        scheme.process_weights_after_loading(layer)

        # Weight stored transposed: (K_padded, N_padded)
        k_padded, n_padded = layer.weight.shape
        self.assertEqual(k_padded % FP8_ALIGNMENT, 0)
        self.assertEqual(n_padded % FP8_ALIGNMENT, 0)

        # Scale dim 0 must also be padded to a multiple of FP8_ALIGNMENT
        self.assertEqual(layer.weight_scale.shape[0] % FP8_ALIGNMENT, 0)

        self.assertEqual(layer._orig_output_dim, out_dim)

    def test_non_aligned(self):
        self._run(out_dim=1368, in_dim=2048)

    def test_aligned(self):
        self._run(out_dim=1024, in_dim=2048)

    def test_output_shape_after_apply(self):
        out_dim, in_dim = 1368, 2048
        scheme = _make_scheme("CHANNEL")

        weight = _make_fp8_weight(out_dim, in_dim)
        weight_scale = torch.ones(out_dim, 1, dtype=torch.float32, device="cuda")
        layer = _make_layer(weight, weight_scale)
        layer.input_scale = None

        scheme.process_weights_after_loading(layer)

        batch = 4
        x = torch.zeros(batch, in_dim, dtype=torch.bfloat16, device="cuda")
        output = scheme.apply_weights(layer, x)

        self.assertEqual(output.shape, (batch, out_dim))

    def test_output_shape_with_bias(self):
        """Bias (shape [out_dim]) must not cause a shape mismatch when N is padded."""
        out_dim, in_dim = 1368, 2048
        scheme = _make_scheme("CHANNEL")

        weight = _make_fp8_weight(out_dim, in_dim)
        weight_scale = torch.ones(out_dim, 1, dtype=torch.float32, device="cuda")
        layer = _make_layer(weight, weight_scale)
        layer.input_scale = None

        scheme.process_weights_after_loading(layer)

        batch = 4
        x = torch.zeros(batch, in_dim, dtype=torch.bfloat16, device="cuda")
        bias = torch.ones(out_dim, dtype=torch.bfloat16, device="cuda")
        output = scheme.apply_weights(layer, x, bias=bias)

        self.assertEqual(output.shape, (batch, out_dim))


if __name__ == "__main__":
    unittest.main()
