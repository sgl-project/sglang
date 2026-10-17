"""Bailing target and NextN draft checkpoint selection across model versions."""

import unittest
from types import SimpleNamespace

from sglang.srt.models.bailing_moe import (
    BailingMoEForCausalLM,
    BailingMoeForCausalLM,
    BailingMoeV2ForCausalLM,
)
from sglang.srt.models.bailing_moe_linear import BailingMoeV2_5ForCausalLM
from sglang.srt.models.bailing_moe_nextn import BailingMoeForCausalLMNextN
from sglang.srt.models.bailing_moe_v3 import BailingMoeV3ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestBailingCheckpointSelection(unittest.TestCase):
    def test_each_target_version_excludes_nextn_layer(self):
        config = SimpleNamespace(
            num_hidden_layers=80,
            num_nextn_predict_layers=1,
            tie_word_embeddings=False,
        )
        for model_class in (
            BailingMoEForCausalLM,
            BailingMoeForCausalLM,
            BailingMoeV2ForCausalLM,
            BailingMoeV2_5ForCausalLM,
            BailingMoeV3ForCausalLM,
        ):
            model = object.__new__(model_class)
            model.config = config
            with self.subTest(model=model_class.__name__):
                self.assertFalse(
                    model.is_unused_checkpoint_weight(
                        "model.layers.79.attention.q_proj.weight"
                    )
                )
                self.assertTrue(
                    model.is_unused_checkpoint_weight(
                        "model.layers.80.attention.q_proj.weight"
                    )
                )
                self.assertTrue(
                    model.is_unused_checkpoint_weight("model.mtp.layers.0.weight")
                )

    def test_draft_keeps_only_its_layer_and_unshared_weights(self):
        model = object.__new__(BailingMoeForCausalLMNextN)
        model.config = SimpleNamespace(
            num_hidden_layers=80,
            num_nextn_predict_layers=1,
            tie_word_embeddings=False,
        )
        for name, unused in (
            ("model.layers.0.attention.q_proj.weight", True),
            ("model.layers.80.attention.q_proj.weight", False),
            ("model.layers.800.attention.q_proj.weight", True),
            ("model.layers.80.final_layernorm.weight", False),
            ("model.layers.80.shared_head.head.weight", True),
            ("model.layers.80.embed_tokens.weight", True),
            ("model.layers.80.attention.rotary_emb.inv_freq", True),
            ("model.mtp.layers.0.weight", True),
        ):
            with self.subTest(name=name):
                self.assertEqual(model.is_unused_checkpoint_weight(name), unused)


if __name__ == "__main__":
    unittest.main()
