"""
Unit tests for Gemma3ForCausalLM.load_weights.

Regression coverage for ModelOpt FP8/NVFP4 checkpoints that store k/v scales
under k_proj/v_proj names.
"""

import unittest

import torch

from sglang.srt.models.gemma3_causal import Gemma3ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestGemma3WeightLoading(unittest.TestCase):
    def _make_minimal_model(self, named_parameters=()):
        model = object.__new__(Gemma3ForCausalLM)
        model.named_parameters = lambda: iter(named_parameters)
        return model

    def test_modelopt_kv_scales_are_remapped_before_qkv_stacking(self):
        k_scale = torch.nn.Parameter(torch.zeros(1))
        v_scale = torch.nn.Parameter(torch.zeros(1))
        model = self._make_minimal_model(
            [
                ("language_model.model.layers.0.self_attn.attn.k_scale", k_scale),
                ("language_model.model.layers.0.self_attn.attn.v_scale", v_scale),
            ]
        )
        weights = [
            (
                "language_model.model.layers.0.self_attn.k_proj.k_scale",
                torch.tensor([2.0]),
            ),
            (
                "language_model.model.layers.0.self_attn.v_proj.v_scale",
                torch.tensor([3.0]),
            ),
        ]

        loaded_params = model.load_weights(weights)

        self.assertEqual(
            loaded_params,
            {
                "language_model.model.layers.0.self_attn.attn.k_scale",
                "language_model.model.layers.0.self_attn.attn.v_scale",
            },
        )
        self.assertEqual(k_scale.item(), 2.0)
        self.assertEqual(v_scale.item(), 3.0)


if __name__ == "__main__":
    unittest.main()
