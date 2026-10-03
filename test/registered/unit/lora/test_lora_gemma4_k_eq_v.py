"""CPU regression tests for Gemma 4 LoRA loading.

Gemma 4 dense checkpoints with ``attention_k_eq_v`` (e.g. gemma-4-31B-it) have
no ``v_proj`` on their full-attention layers, so PEFT adapters carry q/k LoRA
weights but no v LoRA there while the sliding-window layers still have all
three. Those full-attention layers also use a different head_dim and KV-head
count than the sliding layers. Covers #25913.
"""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.lora.lora import LoRAAdapter
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.mem_pool import LoRAMemoryPool
from sglang.srt.models.gemma4_mm import Gemma4ForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_LAYER = "base_model.model.model.language_model.layers.{}.self_attn.{}.lora_{}.weight"


def _name(layer, module, ab):
    return _LAYER.format(layer, module, ab)


class TestQKVNormalizationPerLayer(CustomTestCase):
    def _normalize(self, weights):
        LoRAAdapter.normalize_qkv_proj(None, list(weights), weights)
        return weights

    def test_v_proj_missing_on_some_layers_only(self):
        # Layer 0 (sliding) has q/k/v; layer 5 (attention_k_eq_v) has q/k only.
        weights = {
            _name(0, "q_proj", "B"): torch.full((8, 2), 1.0),
            _name(0, "k_proj", "B"): torch.full((4, 2), 2.0),
            _name(0, "v_proj", "B"): torch.full((4, 2), 3.0),
            _name(5, "q_proj", "B"): torch.full((16, 2), 1.0),
            _name(5, "k_proj", "B"): torch.full((2, 2), 2.0),
        }
        self._normalize(weights)

        self.assertEqual(
            set(weights), {_name(0, "qkv_proj", "B"), _name(5, "qkv_proj", "B")}
        )
        full = weights[_name(0, "qkv_proj", "B")]
        self.assertEqual(tuple(full.shape), (16, 2))
        self.assertTrue(torch.equal(full[8:12], torch.full((4, 2), 2.0)))
        self.assertTrue(torch.equal(full[12:16], torch.full((4, 2), 3.0)))

        k_eq_v = weights[_name(5, "qkv_proj", "B")]
        self.assertEqual(tuple(k_eq_v.shape), (20, 2))
        self.assertTrue(torch.equal(k_eq_v[16:18], torch.full((2, 2), 2.0)))
        self.assertTrue(torch.equal(k_eq_v[18:20], torch.zeros(2, 2)))

    def test_k_proj_missing_is_still_zero_filled(self):
        weights = {
            _name(0, "q_proj", "A"): torch.ones(2, 3),
            _name(0, "v_proj", "A"): torch.full((2, 3), 4.0),
        }
        self._normalize(weights)

        qkv = weights[_name(0, "qkv_proj", "A")]
        self.assertEqual(set(weights), {_name(0, "qkv_proj", "A")})
        self.assertTrue(torch.equal(qkv[2:4], torch.zeros(2, 3)))
        self.assertTrue(torch.equal(qkv[4:6], torch.full((2, 3), 4.0)))

    def test_both_k_and_v_missing_raises(self):
        weights = {_name(0, "q_proj", "B"): torch.ones(8, 2)}
        with self.assertRaises(ValueError):
            self._normalize(weights)


class TestGemma4LoRADimensions(CustomTestCase):
    @staticmethod
    def _model(text_config):
        model = Gemma4ForConditionalGeneration.__new__(Gemma4ForConditionalGeneration)
        model.config = SimpleNamespace(text_config=text_config)
        return model

    def test_gemma4_31b_shapes(self):
        # gemma-4-31B-it after SGLang's config normalization: base attributes
        # describe full-attention layers, swa_* the sliding-window layers.
        text_config = SimpleNamespace(
            hidden_size=5376,
            intermediate_size=21504,
            num_hidden_layers=6,
            num_attention_heads=32,
            head_dim=512,
            num_key_value_heads=4,
            swa_head_dim=256,
            swa_num_key_value_heads=16,
            num_kv_shared_layers=0,
            layer_types=["sliding_attention"] * 5 + ["full_attention"],
        )
        model = self._model(text_config)

        # Sliding layer: q 32*256, k/v 16*256 each -> matches the checkpoint's
        # q_proj [8192, 5376], k_proj/v_proj [4096, 5376].
        self.assertEqual(model.get_hidden_dim("qkv_proj", 0), (5376, 16384))
        self.assertEqual(model.get_hidden_dim("o_proj", 0), (8192, 5376))
        # Full-attention layer: q 32*512, k/v 4*512 each -> q_proj [16384, 5376],
        # k_proj [2048, 5376].
        self.assertEqual(model.get_hidden_dim("qkv_proj", 5), (5376, 20480))
        self.assertEqual(model.get_hidden_dim("o_proj", 5), (16384, 5376))
        self.assertEqual(model.get_hidden_dim("gate_up_proj", 0), (5376, 43008))
        self.assertEqual(model.get_hidden_dim("down_proj", 5), (21504, 5376))

    def test_double_wide_mlp_on_kv_shared_layers(self):
        # gemma-4-E2B-it style: the trailing num_kv_shared_layers use a 2x MLP.
        text_config = SimpleNamespace(
            hidden_size=1536,
            intermediate_size=6144,
            num_hidden_layers=4,
            num_attention_heads=8,
            head_dim=512,
            num_key_value_heads=1,
            swa_head_dim=256,
            swa_num_key_value_heads=1,
            num_kv_shared_layers=2,
            use_double_wide_mlp=True,
            layer_types=["sliding_attention", "full_attention"] * 2,
        )
        model = self._model(text_config)

        self.assertEqual(model.get_hidden_dim("gate_up_proj", 1), (1536, 12288))
        self.assertEqual(model.get_hidden_dim("down_proj", 1), (6144, 1536))
        self.assertEqual(model.get_hidden_dim("gate_up_proj", 2), (1536, 24576))
        self.assertEqual(model.get_hidden_dim("down_proj", 3), (12288, 1536))


class TestShouldApplyLoRAGate(CustomTestCase):
    def test_vision_tower_modules_are_not_wrapped(self):
        language_qkv = "language_model.layers.0.self_attn.qkv_proj"
        language_o = "language_model.layers.0.self_attn.o_proj"
        vision_o = "vision_tower.encoder.layers.0.self_attn.o_proj"
        vision_down = "vision_tower.encoder.layers.0.mlp.down_proj"
        modules = [
            (name, torch.nn.Identity())
            for name in (language_qkv, language_o, vision_o, vision_down)
        ]

        base_model = SimpleNamespace(
            named_modules=lambda: modules,
            should_apply_lora=lambda name: bool(
                Gemma4ForConditionalGeneration.lora_pattern.match(name)
            ),
        )
        manager = LoRAManager.__new__(LoRAManager)
        manager.base_model = base_model
        manager.base_hf_config = SimpleNamespace(num_hidden_layers=1)
        manager.target_modules = {"qkv_proj", "o_proj", "down_proj"}

        inkling_module = types.ModuleType("sglang.srt.models.inkling_common.dense_mlp")
        inkling_module.InklingBatchDenseMLP = type("InklingBatchDenseMLP", (), {})

        with (
            patch.object(
                manager, "set_lora_module", side_effect=lambda name, module: name
            ) as set_lora_module,
            patch.dict(
                sys.modules,
                {"sglang.srt.models.inkling_common.dense_mlp": inkling_module},
            ),
        ):
            manager.init_lora_modules()

        self.assertEqual(set(manager.lora_modules[0]), {language_qkv, language_o})
        self.assertEqual(set_lora_module.call_count, 2)

    def test_gemma4_pattern_matches_moe_experts(self):
        pattern = Gemma4ForConditionalGeneration.lora_pattern
        self.assertTrue(pattern.match("language_model.layers.3.mlp.experts"))
        self.assertTrue(pattern.match("language_model.layers.3.mlp.gate_up_proj"))
        self.assertFalse(pattern.match("vision_tower.encoder.layers.3.mlp.down_proj"))
        self.assertFalse(pattern.match("audio_tower.conformer.0.attention.attn.o_proj"))


class TestTowerWeightsDoNotOverwriteLanguageWeights(CustomTestCase):
    def test_colliding_vision_tower_tensors_are_dropped(self):
        # PEFT adapters trained on a VLM can carry tower LoRA weights that
        # share the layer index and module suffix with language weights.
        lang_a = _name(0, "o_proj", "A")
        lang_b = _name(0, "o_proj", "B")
        vis_a = "base_model.model.model.vision_tower.encoder.layers.0.self_attn.o_proj.lora_A.weight"
        vis_b = vis_a.replace("lora_A", "lora_B")
        qkv_a = _name(0, "qkv_proj", "A")
        layer_weights = {
            lang_a: torch.zeros(8, 8192),
            lang_b: torch.zeros(5376, 8),
            vis_a: torch.zeros(8, 1152),
            vis_b: torch.zeros(1152, 8),
            qkv_a: torch.zeros(24, 5376),
        }
        cur_layer_modules = {
            "language_model.layers.0.self_attn.qkv_proj": torch.nn.Identity(),
            "language_model.layers.0.self_attn.o_proj": torch.nn.Identity(),
        }
        pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
        pool.target_modules = {"qkv_proj", "o_proj"}
        dropped = []

        kept = pool._drop_colliding_weights(layer_weights, cur_layer_modules, dropped)

        self.assertEqual(set(kept), {lang_a, lang_b, qkv_a})
        self.assertEqual(dropped, sorted([vis_a, vis_b]))

    def test_no_collision_is_a_no_op(self):
        layer_weights = {_name(0, "o_proj", "A"): torch.zeros(8, 8192)}
        pool = LoRAMemoryPool.__new__(LoRAMemoryPool)
        pool.target_modules = {"o_proj"}
        dropped = []

        kept = pool._drop_colliding_weights(layer_weights, {}, dropped)

        self.assertIs(kept, layer_weights)
        self.assertEqual(dropped, [])


if __name__ == "__main__":
    unittest.main()
