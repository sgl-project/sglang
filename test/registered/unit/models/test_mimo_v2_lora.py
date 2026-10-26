"""MiMo-V2 LoRA module selection and LoRA dims."""

import unittest
from types import SimpleNamespace

from sglang.srt.models.mimo_v2 import MiMoV2ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# MiMo-V2.6-Flash: layer 0 is full attention with a dense MLP, layer 1 is
# sliding-window attention with MoE.
FLASH_CONFIG = SimpleNamespace(
    hidden_size=4096,
    num_attention_heads=64,
    num_key_value_heads=4,
    head_dim=192,
    v_head_dim=128,
    swa_num_attention_heads=64,
    swa_num_key_value_heads=8,
    swa_head_dim=192,
    swa_v_head_dim=128,
    hybrid_layer_pattern=[0, 1],
    intermediate_size=16384,
    moe_intermediate_size=2048,
)


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


class TestMiMoV2LoRAHiddenDim(CustomTestCase):
    def test_dims_follow_each_layer(self):
        model = MiMoV2ForCausalLM.__new__(MiMoV2ForCausalLM)
        model.config = FLASH_CONFIG
        expected = {
            ("qkv_proj", 0): (4096, 64 * 192 + 4 * 192 + 4 * 128),
            ("qkv_proj", 1): (4096, 64 * 192 + 8 * 192 + 8 * 128),
            ("o_proj", 0): (64 * 128, 4096),
            ("o_proj", 1): (64 * 128, 4096),
            ("gate_up_proj", 0): (4096, 2 * 16384),
            ("down_proj", 0): (16384, 4096),
            ("gate_up_proj_moe", 1): (4096, 2 * 2048),
            ("down_proj_moe", 1): (2048, 4096),
        }
        for (module, layer), dims in expected.items():
            with self.subTest(module=module, layer=layer):
                self.assertEqual(model.get_hidden_dim(module, layer), dims)


if __name__ == "__main__":
    unittest.main()
