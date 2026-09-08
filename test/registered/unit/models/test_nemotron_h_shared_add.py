"""
Unit tests for the NemotronHMoE latent-projection shared-expert add.

The fused path calls the projection's quant method directly,
not ``ReplicatedLinear.forward``.
These cases pin the gate that decides when the substitution is safe.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.lora.layers import ReplicatedLinearWithLoRA
from sglang.srt.models.nemotron_h import (
    NemotronHMoE,
    _latent_proj_fuses_shared_add,
)
from sglang.test.test_utils import CustomTestCase


# Stands in for a quantized linear method, which has no addend entry point.
class _DoubleMethod:
    def apply(self, layer, x, bias):
        return 2 * x


class _FakeLoRABackend:
    batch_info = object()
    skip_inactive_lora_batches = False

    def run_lora_a_sgemm(self, x, weights):
        return 2 * x

    def run_lora_b_sgemm(self, *, x, weights, output_offset, base_output):
        return base_output + x


def _apply(projection, routed, shared):
    moe = SimpleNamespace(fc2_latent_proj=projection)
    return NemotronHMoE._apply_latent_projection(moe, routed, shared)


class TestNemotronHSharedAdd(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.routed = torch.randn(4, 8)
        self.shared = torch.randn(4, 8)

    def test_gate_accepts_plain_bias_free_projection(self):
        """A bias-free unquantized ReplicatedLinear must stay eligible;
        a gate that degrades to always-false silently drops the fusion."""
        projection = ReplicatedLinear(8, 8, bias=False)

        self.assertTrue(_latent_proj_fuses_shared_add(projection))

    def test_gate_rejects_lora_wrapper(self):
        """LoRA swaps a wrapper module over fc2_latent_proj after model init.
        Calling the base quant method there would drop the adapter update."""
        projection = ReplicatedLinear(8, 8, bias=False)
        with torch.no_grad():
            projection.weight.zero_()
        wrapped = ReplicatedLinearWithLoRA(projection, _FakeLoRABackend())
        wrapped.set_lora_info(torch.empty(1, 8), torch.empty(8, 1))

        self.assertFalse(_latent_proj_fuses_shared_add(wrapped))
        torch.testing.assert_close(
            _apply(wrapped, self.routed, self.shared),
            2 * self.routed + self.shared,
            rtol=0,
            atol=0,
        )

    def test_gate_rejects_quantized_projection(self):
        """Only UnquantizedLinearMethod implements apply_with_addend."""
        projection = ReplicatedLinear(8, 8, bias=False)
        projection.quant_method = _DoubleMethod()

        self.assertFalse(_latent_proj_fuses_shared_add(projection))
        torch.testing.assert_close(
            _apply(projection, self.routed, self.shared),
            2 * self.routed + self.shared,
            rtol=0,
            atol=0,
        )

    def test_gate_rejects_projection_with_bias(self):
        """forward defers the bias under skip_bias_add and runs module hooks;
        both are lost if a bias-bearing projection takes the fused call."""
        projection = ReplicatedLinear(8, 8, bias=True, skip_bias_add=True)
        with torch.no_grad():
            projection.weight.copy_(torch.randn_like(projection.weight))
            projection.bias.copy_(torch.randn_like(projection.bias))
        hook_calls = []
        projection.register_forward_hook(lambda *args: hook_calls.append(True))

        self.assertFalse(_latent_proj_fuses_shared_add(projection))
        torch.testing.assert_close(
            _apply(projection, self.routed, self.shared),
            F.linear(self.routed, projection.weight) + self.shared,
            rtol=0,
            atol=0,
        )
        self.assertEqual(hook_calls, [True])

    def test_no_shared_output_keeps_plain_projection(self):
        """Layers without shared experts pass shared_output=None."""
        projection = ReplicatedLinear(8, 8, bias=False)
        with torch.no_grad():
            projection.weight.copy_(torch.randn_like(projection.weight))

        torch.testing.assert_close(
            _apply(projection, self.routed, None),
            F.linear(self.routed, projection.weight),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
