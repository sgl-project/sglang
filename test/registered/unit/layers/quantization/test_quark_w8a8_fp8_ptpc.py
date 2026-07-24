"""CPU-only regression tests for the Quark PTPC-FP8 (W8A8 FP8) MLA attention fix.

These guard the hardware-independent pieces of the fix that enables loading and
running MLA models (e.g. GlmMoeDsaForCausalLM) with Quark-quantized PTPC FP8
attention on ROCm/gfx95.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.fp8_utils import channel_quant_to_tensor_quant
from sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
    QuarkW8A8Fp8,
    is_quark_w8a8_fp8_layer,
)
from sglang.test.test_utils import CustomTestCase


def _make_quark_w8a8_fp8_scheme() -> QuarkW8A8Fp8:
    """Per-channel (PTPC) dynamic-activation W8A8 FP8 scheme, as loaded from a
    Quark checkpoint's kv_b_proj / o_proj config."""
    return QuarkW8A8Fp8(
        weight_config={"qscheme": "per_channel"},
        input_config={"is_dynamic": True, "qscheme": "per_channel"},
    )


class TestChannelQuantToTensorQuantBroadcast(CustomTestCase):
    """Regression for the 1D per-channel scale failing to broadcast against a 2D
    weight.
    """

    def test_2d_weight_1d_scale_broadcasts_per_output_channel(self):
        n, k = 8, 4
        weight = torch.ones(n, k, dtype=torch.float8_e4m3fn)
        # Distinct per-channel scales so a wrong-axis broadcast would be visible.
        scale = torch.arange(1, n + 1, dtype=torch.float32)

        q_tensor, tensor_scale = channel_quant_to_tensor_quant(weight, scale)

        # Round-trip: dequantized tensor must equal weight * per-row scale. The
        # requant to a single tensor-wide FP8 scale is lossy, so tolerance is
        # loose — a wrong-axis broadcast would be off by whole factors (>>10%).
        dq = q_tensor.to(torch.float32) * tensor_scale
        expected = torch.ones(n, k, dtype=torch.float32) * scale.unsqueeze(-1)
        torch.testing.assert_close(dq, expected, rtol=0.1, atol=0.5)

    def test_mismatched_n_k_would_fail_without_unsqueeze(self):
        # N != K: the pre-fix ``weight * scale`` raises; the fix must not.
        n, k = 6, 3
        weight = torch.ones(n, k, dtype=torch.float8_e4m3fn)
        scale = torch.arange(1, n + 1, dtype=torch.float32)

        q_tensor, tensor_scale = channel_quant_to_tensor_quant(weight, scale)

        self.assertEqual(q_tensor.shape, torch.Size([n, k]))

    def test_scale_already_matching_rank_is_unchanged(self):
        # A [N, 1] scale must not be unsqueezed further (guards the while-loop
        # condition from over-reshaping).
        n, k = 5, 4
        weight = torch.ones(n, k, dtype=torch.float8_e4m3fn)
        scale = torch.arange(1, n + 1, dtype=torch.float32).unsqueeze(-1)

        q_tensor, tensor_scale = channel_quant_to_tensor_quant(weight, scale)

        dq = q_tensor.to(torch.float32) * tensor_scale
        expected = torch.ones(n, k, dtype=torch.float32) * scale
        torch.testing.assert_close(dq, expected, rtol=0.1, atol=0.5)


class TestIsQuarkW8A8Fp8Layer(CustomTestCase):
    """The predicate that routes Quark FP8 attention layers away from the aiter
    group-128 fused RMSNorm+quant kernel.

    A false negative re-enables the fused path and feeds the Quark linear a
    pre-quantized tuple it can't consume (the ``assert not isinstance(x, tuple)``
    in ``apply_weights`` fires); a false positive disables fusion for unrelated
    FP8 layers. Both branches are pinned here.
    """

    def test_true_for_quark_w8a8_fp8_scheme(self):
        layer = torch.nn.Linear(4, 4)
        layer.scheme = _make_quark_w8a8_fp8_scheme()
        self.assertTrue(is_quark_w8a8_fp8_layer(layer))

    def test_false_for_layer_without_scheme(self):
        # Plain FP8 linear (e.g. compressed_tensors) has no ``scheme`` attr.
        self.assertFalse(is_quark_w8a8_fp8_layer(torch.nn.Linear(4, 4)))

    def test_false_for_none_scheme(self):
        layer = torch.nn.Linear(4, 4)
        layer.scheme = None
        self.assertFalse(is_quark_w8a8_fp8_layer(layer))

    def test_false_for_unrelated_scheme_type(self):
        layer = torch.nn.Linear(4, 4)
        layer.scheme = object()
        self.assertFalse(is_quark_w8a8_fp8_layer(layer))


if __name__ == "__main__":
    unittest.main()
