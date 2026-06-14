"""Unit tests for LoRA qkv normalization and Gemma 4 native-LoRA buffer dims.

Regression coverage for #25913: a Gemma 4 E2B PEFT adapter whose KV-sharing
layers carry q/k/o weights but omit v_proj used to crash LoRAAdapter loading,
and the Gemma 4 multimodal LoRA buffer dims were computed with a single
head_dim for every layer. CPU-only.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestLoRAQKVNormalization(CustomTestCase):
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

    def test_missing_k_and_v_use_kv_width_for_lora_b_under_gqa(self):
        # Adapter with only q_proj on a GQA layer: both K and V are zero-filled,
        # and a lora_B must use the KV output width (num_kv_heads*head_dim), not
        # q's wider output, or the stacked qkv_proj lora_B has the wrong shape.
        q_name = "base_model.model.model.language_model.layers.3.self_attn.q_proj.lora_B.weight"
        qkv_name = q_name.replace("q_proj", "qkv_proj")
        weights = {q_name: torch.ones(8, 2)}  # q_out = num_heads(4) * head_dim(2)
        fake_self = SimpleNamespace(
            base_hf_config=SimpleNamespace(num_attention_heads=4, num_key_value_heads=1)
        )

        LoRAAdapter.normalize_qkv_proj(fake_self, list(weights), weights)

        # kv_out = num_kv_heads(1) * head_dim(2) = 2; qkv = 8 + 2 + 2
        self.assertEqual(set(weights), {qkv_name})
        self.assertEqual(tuple(weights[qkv_name].shape), (12, 2))
        self.assertTrue(torch.equal(weights[qkv_name][0:8], torch.ones(8, 2)))
        self.assertTrue(torch.equal(weights[qkv_name][8:12], torch.zeros(4, 2)))

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


class TestGemma4LoRADimensions(CustomTestCase):
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
