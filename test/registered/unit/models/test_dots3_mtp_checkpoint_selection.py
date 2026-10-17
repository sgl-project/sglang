"""Dots3 target, Omni wrapper, and NextN checkpoint selection."""

import unittest
from types import SimpleNamespace

from sglang.srt.models.dots3_common.modeling import (
    Dots3LanguageModelForCausalLM,
    Dots3NoteForCausalLM,
)
from sglang.srt.models.dots3_common.nextn import Dots3NoteForCausalLMNextN
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDots3CheckpointSelection(unittest.TestCase):
    def test_language_target_excludes_nextn_layers(self):
        model = object.__new__(Dots3LanguageModelForCausalLM)
        model.config = SimpleNamespace(num_hidden_layers=30, num_nextn_predict_layers=1)
        for name, unused in (
            ("model.layers.29.self_attn.q_proj.weight", False),
            ("model.layers.30.self_attn.q_proj.weight", True),
            ("model.mtp.embed_tokens.weight", True),
            ("model.layers.0.self_attn.rotary_emb.inv_freq", True),
        ):
            with self.subTest(name=name):
                self.assertEqual(model.is_unused_checkpoint_weight(name), unused)

    def test_omni_wrapper_retains_tower_weights_when_present(self):
        model = object.__new__(Dots3NoteForCausalLM)
        language_model = object.__new__(Dots3LanguageModelForCausalLM)
        language_model.config = SimpleNamespace(
            num_hidden_layers=30, num_nextn_predict_layers=1
        )
        model.thinker = SimpleNamespace(visual=object(), language_model=language_model)
        self.assertFalse(
            model.is_unused_checkpoint_weight("vision_encoder.layers.0.weight")
        )
        self.assertTrue(
            model.is_unused_checkpoint_weight("model.layers.30.self_attn.q_proj.weight")
        )

    def test_draft_preserves_checkpoint_specific_embedding(self):
        model = object.__new__(Dots3NoteForCausalLMNextN)
        model.config = SimpleNamespace(num_hidden_layers=30, num_nextn_predict_layers=1)
        for name, unused in (
            ("model.layers.0.self_attn.q_proj.weight", True),
            ("model.layers.30.self_attn.q_proj.weight", False),
            ("model.mtp.embed_tokens.weight", False),
            ("model.mtp.embed_tokens.weight_scale_inv", False),
            ("model.embed_tokens.weight", True),
            ("model.layers.30.self_attn.rotary_emb.inv_freq", True),
        ):
            with self.subTest(name=name):
                self.assertEqual(model.is_unused_checkpoint_weight(name), unused)


if __name__ == "__main__":
    unittest.main()
