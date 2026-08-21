"""Unit tests for the Granite / GraniteMoe config gating and expert weight remap."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch
from transformers import GraniteConfig

from sglang.srt.models.granite import granite_layer_attn_params
from sglang.srt.models.granitemoe import (
    GraniteMoeForCausalLM,
    granitemoe_split_expert_weights,
)
from sglang.test.test_utils import CustomTestCase


class TestGraniteLayerAttnParams(CustomTestCase):
    def test_plain_granite_config_enables_nothing(self):
        """Granite checkpoints predate SWA, so every SWA feature must stay off."""
        config = GraniteConfig(num_hidden_layers=4)
        rope_theta = config.rope_parameters["rope_theta"]

        params = [granite_layer_attn_params(config, i) for i in range(4)]

        self.assertEqual(params, [(-1, rope_theta, False)] * 4)

    def test_swa_config_resolves_per_layer(self):
        """Full attention is -1, a sliding layer's window is exclusive
        (`sliding_window - 1`), and rope theta 0 selects NoPE."""
        config = SimpleNamespace(
            model_type="granite_swa",
            sliding_window=128,
            layer_types=["full_attention", "sliding_attention", "sliding_attention"],
            layer_rope_theta=[10000.0, 0.0, 1000000.0],
        )

        params = [granite_layer_attn_params(config, i) for i in range(3)]

        self.assertEqual(
            params,
            [(-1, 10000.0, True), (127, 0.0, True), (127, 1000000.0, True)],
        )


class TestGraniteMoeExpertWeights(CustomTestCase):
    def _remap(self, gate_up_name, down_name, router_name, tensors):
        gate_up, down, router = tensors
        weights = [
            (f"model.layers.0.block_sparse_moe.{gate_up_name}", gate_up),
            (f"model.layers.0.block_sparse_moe.{down_name}", down),
            (f"model.layers.0.block_sparse_moe.{router_name}", router),
            # Must pass through: the shared expert reuses the legacy expert names.
            ("model.layers.0.shared_mlp.input_linear.weight", down[0]),
        ]
        mapper = GraniteMoeForCausalLM.hf_to_sglang_mapper
        return dict(granitemoe_split_expert_weights(mapper.apply(weights)))

    def test_legacy_and_current_checkpoint_names_agree(self):
        """The two HF spellings share a layout, so both must remap identically."""
        num_experts, hidden, intermediate = 3, 8, 4
        tensors = (
            torch.randn(num_experts, 2 * intermediate, hidden),
            torch.randn(num_experts, hidden, intermediate),
            torch.randn(num_experts, hidden),
        )

        legacy = self._remap(
            "input_linear.weight",
            "output_linear.weight",
            "router.layer.weight",
            tensors,
        )
        current = self._remap(
            "experts.gate_up_proj", "experts.down_proj", "router.weight", tensors
        )

        self.assertEqual(legacy.keys(), current.keys())
        for name, weight in legacy.items():
            self.assertTrue(torch.equal(weight, current[name]), name)

        gate_up, down, router = tensors
        prefix = "model.layers.0.block_sparse_moe"
        self.assertTrue(torch.equal(legacy[f"{prefix}.gate.weight"], router))
        self.assertTrue(
            torch.equal(
                legacy["model.layers.0.shared_mlp.input_linear.weight"], down[0]
            )
        )
        for e in range(num_experts):
            self.assertTrue(
                torch.equal(
                    legacy[f"{prefix}.experts.{e}.w1.weight"], gate_up[e][:intermediate]
                )
            )
            self.assertTrue(
                torch.equal(
                    legacy[f"{prefix}.experts.{e}.w3.weight"], gate_up[e][intermediate:]
                )
            )
            self.assertTrue(
                torch.equal(legacy[f"{prefix}.experts.{e}.w2.weight"], down[e])
            )


if __name__ == "__main__":
    unittest.main()
