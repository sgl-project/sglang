"""MiMo-V2 LoRA module selection."""

import unittest

from sglang.srt.models.mimo_v2 import MiMoV2ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestMiMoV2ShouldApplyLoRA(CustomTestCase):
    def test_only_language_model_projections_are_adapted(self):
        model = MiMoV2ForCausalLM.__new__(MiMoV2ForCausalLM)
        adapted = [
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_up_proj",
            "model.layers.0.mlp.down_proj",
            "model.layers.1.mlp.experts",
        ]
        # The audio encoders reuse the decoder's `layers.<i>.*_proj` names.
        skipped = [
            "input_local_transformer.layers.0.self_attn.o_proj",
            "input_local_transformer.layers.0.mlp.down_proj",
            "audio_tokenizer.encoder.layers.0.self_attn.attn.qkv_proj",
            "visual.blocks.0.attn.qkv_proj",
            "model.layers.1.mlp.gate",
        ]
        for name in adapted:
            with self.subTest(name=name):
                self.assertTrue(model.should_apply_lora(name))
        for name in skipped:
            with self.subTest(name=name):
                self.assertFalse(model.should_apply_lora(name))


if __name__ == "__main__":
    unittest.main()
