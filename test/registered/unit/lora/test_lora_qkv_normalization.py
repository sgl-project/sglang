import unittest
from types import SimpleNamespace

import torch

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration


class TestLoRAQKVNormalization(unittest.TestCase):
    def test_missing_v_proj_is_zero_filled_from_k_shape(self):
        q_name = "base_model.model.model.language_model.layers.15.self_attn.q_proj.lora_A.weight"
        k_name = q_name.replace("q_proj", "k_proj")
        qkv_name = q_name.replace("q_proj", "qkv_proj")
        weights = {
            q_name: torch.ones(2, 3),
            k_name: torch.full((2, 3), 2.0),
        }

        LoRAAdapter.normalize_qkv_proj(None, list(weights), weights)

        self.assertEqual(set(weights), {qkv_name})
        self.assertTrue(torch.equal(weights[qkv_name][0:2], torch.ones(2, 3)))
        self.assertTrue(torch.equal(weights[qkv_name][2:4], torch.full((2, 3), 2.0)))
        self.assertTrue(torch.equal(weights[qkv_name][4:6], torch.zeros(2, 3)))

    def test_missing_k_proj_is_zero_filled_from_v_shape(self):
        q_name = "base_model.model.model.language_model.layers.7.self_attn.q_proj.lora_B.weight"
        v_name = q_name.replace("q_proj", "v_proj")
        qkv_name = q_name.replace("q_proj", "qkv_proj")
        weights = {
            q_name: torch.ones(8, 2),
            v_name: torch.full((3, 2), 4.0),
        }

        LoRAAdapter.normalize_qkv_proj(None, list(weights), weights)

        self.assertEqual(set(weights), {qkv_name})
        self.assertTrue(torch.equal(weights[qkv_name][0:8], torch.ones(8, 2)))
        self.assertTrue(torch.equal(weights[qkv_name][8:11], torch.zeros(3, 2)))
        self.assertTrue(torch.equal(weights[qkv_name][11:14], torch.full((3, 2), 4.0)))


class TestGemma4LoRADimensions(unittest.TestCase):
    def _make_model(self):
        text_config = SimpleNamespace(
            hidden_size=1536,
            head_dim=256,
            global_head_dim=512,
            swa_head_dim=256,
            num_attention_heads=8,
            num_key_value_heads=1,
            intermediate_size=6144,
            num_hidden_layers=2,
            num_kv_shared_layers=1,
            use_double_wide_mlp=True,
            layer_types=["sliding_attention", "full_attention"],
        )
        config = SimpleNamespace(get_text_config=lambda: text_config)
        model = Gemma4ForConditionalGeneration.__new__(Gemma4ForConditionalGeneration)
        model.config = config
        return model

    def test_sliding_attention_lora_dims_use_swa_head_dim(self):
        model = self._make_model()

        self.assertEqual(model.get_hidden_dim("qkv_proj", 0), (1536, 2560))
        self.assertEqual(model.get_hidden_dim("o_proj", 0), (2048, 1536))
        self.assertEqual(model.get_hidden_dim("gate_up_proj", 0), (1536, 12288))
        self.assertEqual(model.get_hidden_dim("down_proj", 0), (6144, 1536))

    def test_full_attention_lora_dims_use_global_head_dim_and_double_wide_mlp(self):
        model = self._make_model()

        self.assertEqual(model.get_hidden_dim("qkv_proj", 1), (1536, 5120))
        self.assertEqual(model.get_hidden_dim("o_proj", 1), (4096, 1536))
        self.assertEqual(model.get_hidden_dim("gate_up_proj", 1), (1536, 24576))
        self.assertEqual(model.get_hidden_dim("down_proj", 1), (12288, 1536))


if __name__ == "__main__":
    unittest.main()
