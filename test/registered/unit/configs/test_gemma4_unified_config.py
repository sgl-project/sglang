"""Unit tests for srt/configs/gemma4_unified.py (Gemma 4 12B encoder-free)."""

import unittest

from sglang.srt.configs.gemma4_unified import (
    Gemma4UnifiedAudioConfig,
    Gemma4UnifiedConfig,
    Gemma4UnifiedTextConfig,
    Gemma4UnifiedVisionConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


# Authoritative subset of google/gemma-4-12B-it config.json.
_CONFIG_JSON = {
    "architectures": ["Gemma4UnifiedForConditionalGeneration"],
    "model_type": "gemma4_unified",
    "image_token_id": 258880,
    "video_token_id": 258884,
    "audio_token_id": 258881,
    "boi_token_id": 255999,
    "eoi_token_id": 258882,
    "boa_token_id": 256000,
    "eoa_token_index": 258883,
    "tie_word_embeddings": True,
    "text_config": {
        "model_type": "gemma4_unified_text",
        "hidden_size": 3840,
        "intermediate_size": 15360,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 256,
        "global_head_dim": 512,
        "num_global_key_value_heads": 1,
        "sliding_window": 1024,
        "enable_moe_block": False,
        "num_experts": None,
        "hidden_size_per_layer_input": 0,
        "attention_k_eq_v": True,
        "num_kv_shared_layers": 0,
        "final_logit_softcapping": 30.0,
        "tie_word_embeddings": True,
    },
    "vision_config": {
        "model_type": "gemma4_unified_vision",
        "mm_embed_dim": 3840,
        "mm_posemb_size": 1120,
        "model_patch_size": 48,
        "patch_size": 16,
        "num_soft_tokens": 280,
        "pooling_kernel_size": 3,
        "output_proj_dims": 3840,
    },
    "audio_config": {
        "model_type": "gemma4_unified_audio",
        "audio_embed_dim": 640,
        "audio_samples_per_token": 640,
        "hidden_size": 640,
        "output_proj_dims": 640,
    },
}


def _normalize_dual_attention(text_config):
    """Replicates the gemma4 normalization in hf_transformers/config.py.

    base attrs = full-attention; ``swa_*`` = sliding-window overrides.
    """
    global_head_dim = getattr(text_config, "global_head_dim", None)
    global_kv_heads = getattr(text_config, "num_global_key_value_heads", None)
    text_config.swa_head_dim = text_config.head_dim
    text_config.swa_num_key_value_heads = text_config.num_key_value_heads
    if global_head_dim is not None:
        text_config.head_dim = global_head_dim
    if global_kv_heads is not None:
        text_config.num_key_value_heads = global_kv_heads


class TestGemma4UnifiedConfig(CustomTestCase):
    def test_loads_from_config_json(self):
        cfg = Gemma4UnifiedConfig(**_CONFIG_JSON)
        self.assertEqual(cfg.model_type, "gemma4_unified")
        self.assertIsInstance(cfg.text_config, Gemma4UnifiedTextConfig)
        self.assertIsInstance(cfg.vision_config, Gemma4UnifiedVisionConfig)
        self.assertIsInstance(cfg.audio_config, Gemma4UnifiedAudioConfig)

    def test_text_backbone_is_dense_without_ple(self):
        cfg = Gemma4UnifiedConfig(**_CONFIG_JSON)
        tc = cfg.text_config
        self.assertEqual(tc.num_hidden_layers, 48)
        self.assertEqual(tc.hidden_size, 3840)
        # Dense (MoE disabled) and no per-layer embedding.
        self.assertFalse(tc.enable_moe_block)
        self.assertIn(tc.num_experts, (None, 0))
        self.assertEqual(tc.hidden_size_per_layer_input, 0)

    def test_layer_types_pattern(self):
        cfg = Gemma4UnifiedConfig(**_CONFIG_JSON)
        layer_types = cfg.text_config.layer_types
        self.assertEqual(len(layer_types), 48)
        # Every 6th layer is full attention -> 8 full layers.
        self.assertEqual(layer_types[5], "full_attention")
        self.assertEqual(layer_types[6], "sliding_attention")
        self.assertEqual(sum(t == "full_attention" for t in layer_types), 8)

    def test_dual_attention_normalization(self):
        cfg = Gemma4UnifiedConfig(**_CONFIG_JSON)
        _normalize_dual_attention(cfg.text_config)
        tc = cfg.text_config
        # full-attention layers: 512-dim heads, 1 KV head.
        self.assertEqual(tc.head_dim, 512)
        self.assertEqual(tc.num_key_value_heads, 1)
        # sliding-window layers: 256-dim heads, 8 KV heads.
        self.assertEqual(tc.swa_head_dim, 256)
        self.assertEqual(tc.swa_num_key_value_heads, 8)

    def test_multimodal_token_ids(self):
        cfg = Gemma4UnifiedConfig(**_CONFIG_JSON)
        self.assertEqual(cfg.image_token_id, 258880)
        self.assertEqual(cfg.audio_token_id, 258881)
        # eoa exposed under both the config.json spelling and the _id alias.
        self.assertEqual(cfg.eoa_token_index, 258883)
        self.assertEqual(cfg.eoa_token_id, 258883)

    def test_encoder_free_embedder_dims(self):
        cfg = Gemma4UnifiedConfig(**_CONFIG_JSON)
        self.assertEqual(cfg.vision_config.model_patch_size, 48)
        self.assertEqual(cfg.vision_config.num_soft_tokens, 280)
        self.assertEqual(cfg.audio_config.audio_samples_per_token, 640)

    def test_defaults_match_config_json(self):
        # Instantiating with no args should reproduce the 12B defaults.
        cfg = Gemma4UnifiedConfig()
        self.assertEqual(cfg.text_config.num_hidden_layers, 48)
        self.assertEqual(cfg.text_config.global_head_dim, 512)
        self.assertEqual(cfg.vision_config.num_soft_tokens, 280)

    def test_round_trip_dict(self):
        cfg = Gemma4UnifiedConfig(**_CONFIG_JSON)
        as_dict = cfg.to_dict()
        self.assertEqual(as_dict["model_type"], "gemma4_unified")


if __name__ == "__main__":
    unittest.main()
