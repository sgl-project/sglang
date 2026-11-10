"""Official mtp.2.markov_head.{embed,head} maps to markov_w1/w2."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models.deepseek_v4_dspark import (
    DeepseekV4ForCausalLMDSpark,
    remap_dspark_markov_head_name,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Remapper:
    confidence_head = None
    uses_own_vocab_modules = False
    num_stages = 3


class TestDsv41DsparkMarkovRemap(CustomTestCase):
    def test_official_embed_head_map_to_w1_w2(self):
        self.assertEqual(
            remap_dspark_markov_head_name("markov_head.embed.weight"),
            "markov_head.markov_w1.weight",
        )
        self.assertEqual(
            remap_dspark_markov_head_name("markov_head.head.weight"),
            "markov_head.markov_w2.weight",
        )
        remapper = _Remapper()
        self.assertEqual(
            DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(
                remapper, "mtp.2.markov_head.embed.weight"
            ),
            "markov_head.markov_w1.weight",
        )
        self.assertEqual(
            DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(
                remapper, "mtp.2.markov_head.head.weight"
            ),
            "markov_head.markov_w2.weight",
        )

    def test_gate_bias_exact_and_vl_skip(self):
        remapper = _Remapper()
        self.assertEqual(
            DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(
                remapper, "mtp.0.ffn.gate.bias"
            ),
            "stages.0.mlp.gate.e_score_correction_bias",
        )
        self.assertIsNone(
            DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(
                remapper, "mtp.0.ffn.gate.bias_vl"
            )
        )

    def test_load_weights_copies_markov_tensors(self):
        w1 = nn.Parameter(torch.zeros(4, 2))
        w2 = nn.Parameter(torch.zeros(4, 2))
        loaded_w1 = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        loaded_w2 = torch.arange(8, 16, dtype=torch.float32).reshape(4, 2)

        model = SimpleNamespace(
            config=SimpleNamespace(n_routed_experts=1),
            num_fused_shared_experts=0,
            confidence_head=None,
            markov_head=object(),
            named_parameters=lambda: [
                ("markov_head.markov_w1.weight", w1),
                ("markov_head.markov_w2.weight", w2),
            ],
            _remap_dspark_weight_name=lambda name: (
                DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(_Remapper(), name)
            ),
            _assert_confidence_head_loaded=lambda **_kwargs: None,
            _assert_markov_head_loaded=lambda **kwargs: (
                DeepseekV4ForCausalLMDSpark._assert_markov_head_loaded(
                    SimpleNamespace(markov_head=object()), **kwargs
                )
            ),
        )
        DeepseekV4ForCausalLMDSpark.load_weights(
            model,
            [
                ("mtp.2.markov_head.embed.weight", loaded_w1),
                ("mtp.2.markov_head.head.weight", loaded_w2),
            ],
        )
        self.assertTrue(torch.equal(w1, loaded_w1))
        self.assertTrue(torch.equal(w2, loaded_w2))

    def test_missing_markov_weights_raise(self):
        w1 = nn.Parameter(torch.zeros(2, 2))
        model = SimpleNamespace(
            config=SimpleNamespace(n_routed_experts=1),
            num_fused_shared_experts=0,
            confidence_head=None,
            markov_head=object(),
            named_parameters=lambda: [("markov_head.markov_w1.weight", w1)],
            _remap_dspark_weight_name=lambda name: None,
            _assert_confidence_head_loaded=lambda **_kwargs: None,
            _assert_markov_head_loaded=lambda **kwargs: (
                DeepseekV4ForCausalLMDSpark._assert_markov_head_loaded(
                    SimpleNamespace(markov_head=object()), **kwargs
                )
            ),
        )
        with self.assertRaisesRegex(ValueError, "markov head"):
            DeepseekV4ForCausalLMDSpark.load_weights(
                model, [("mtp.2.markov_head.embed.weight", torch.ones(2, 2))]
            )


if __name__ == "__main__":
    unittest.main()
