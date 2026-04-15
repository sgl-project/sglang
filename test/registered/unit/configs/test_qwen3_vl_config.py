"""Unit tests for qwen3_vl and qwen3_5 config from_dict() handling.

This tests the fix for transformers 5.5.0 compatibility where nested
vision_config and text_config dicts need to be converted to config objects.
"""

import json
import unittest

from sglang.srt.configs.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from sglang.srt.configs.qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLMoeConfig,
    Qwen3VLMoeTextConfig,
    Qwen3VLMoeVisionConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="stage-a-test-cpu")


class TestQwen3VLConfigFromDict(CustomTestCase):
    """Test that Qwen3VLConfig.from_dict() converts nested dicts to config objects."""

    def test_qwen3vl_config_dict_conversion(self):
        """Test Qwen3VLConfig.from_dict() converts nested dicts to config objects."""
        config_dict = {
            "model_type": "qwen3_vl",
            "vision_config": {
                "hidden_size": 1152,
                "num_heads": 16,
                "depth": 27,
            },
            "text_config": {
                "hidden_size": 4096,
                "num_hidden_layers": 32,
            },
        }

        config = Qwen3VLConfig.from_dict(config_dict)

        # Verify types are converted
        self.assertIsNotNone(config.vision_config)
        self.assertIsNotNone(config.text_config)
        self.assertNotIsInstance(config.vision_config, dict)
        self.assertNotIsInstance(config.text_config, dict)

        # Verify values are preserved
        self.assertEqual(config.vision_config.hidden_size, 1152)
        self.assertEqual(config.vision_config.num_heads, 16)
        self.assertEqual(config.text_config.hidden_size, 4096)

    def test_qwen3vl_config_with_object(self):
        """Test Qwen3VLConfig handles existing config objects correctly."""
        vision_obj = Qwen3VLVisionConfig(hidden_size=2048)
        text_obj = Qwen3VLTextConfig(hidden_size=8192)

        config = Qwen3VLConfig(vision_config=vision_obj, text_config=text_obj)

        # Verify objects are preserved
        self.assertIs(config.vision_config, vision_obj)
        self.assertIs(config.text_config, text_obj)
        self.assertEqual(config.vision_config.hidden_size, 2048)

    def test_qwen3vl_moe_config_dict_conversion(self):
        """Test Qwen3VLMoeConfig.from_dict() converts nested dicts."""
        config_dict = {
            "model_type": "qwen3_vl_moe",
            "vision_config": {
                "hidden_size": 1152,
                "num_heads": 16,
            },
            "text_config": {
                "hidden_size": 2048,
                "num_experts": 60,
            },
        }

        config = Qwen3VLMoeConfig.from_dict(config_dict)

        self.assertIsInstance(config.vision_config, Qwen3VLMoeVisionConfig)
        self.assertIsInstance(config.text_config, Qwen3VLMoeTextConfig)
        self.assertEqual(config.vision_config.hidden_size, 1152)


class TestQwen3_5ConfigFromDict(CustomTestCase):
    """Test that Qwen3_5Config.from_dict() converts nested dicts to config objects."""

    def test_qwen3_5_config_dict_conversion(self):
        """Test Qwen3_5Config.from_dict() converts nested dicts."""
        config_dict = {
            "model_type": "qwen3_5",
            "vision_config": {
                "hidden_size": 1152,
                "num_heads": 16,
            },
            "text_config": {
                "hidden_size": 4096,
                "num_hidden_layers": 32,
            },
        }

        config = Qwen3_5Config.from_dict(config_dict)

        self.assertIsInstance(config.vision_config, Qwen3_5VisionConfig)
        self.assertIsInstance(config.text_config, Qwen3_5TextConfig)
        self.assertEqual(config.vision_config.hidden_size, 1152)

    def test_qwen3_5_moe_config_real_model(self):
        """Test with actual Qwen3.5-122B model config."""
        config_path = "/home/gm/projects/LLM/modelfile/qwen/Qwen3.5-122B-A10B-GPTQ-Int4/config.json"

        with open(config_path) as f:
            config_dict = json.load(f)

        config = Qwen3_5MoeConfig.from_dict(config_dict)

        # Verify nested configs are objects
        self.assertIsInstance(config.vision_config, Qwen3_5MoeVisionConfig)
        self.assertIsInstance(config.text_config, Qwen3_5MoeTextConfig)

        # Verify MoE attributes are preserved
        self.assertEqual(config.text_config.num_experts, 256)
        self.assertEqual(config.text_config.num_experts_per_tok, 8)
        self.assertEqual(config.text_config.moe_intermediate_size, 1024)
        self.assertTrue(config.text_config.norm_topk_prob)

        # Verify vision config
        self.assertEqual(config.vision_config.hidden_size, 1152)
        self.assertEqual(config.vision_config.num_heads, 16)

    def test_qwen3_5_moe_text_config_attributes(self):
        """Test Qwen3_5MoeTextConfig preserves all MoE attributes."""
        config_dict = {
            "model_type": "qwen3_5_moe_text",
            "hidden_size": 3072,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 1024,
            "norm_topk_prob": True,
            "decoder_sparse_step": 1,
            "layer_types": ["linear_attention", "full_attention"],
        }

        config = Qwen3_5MoeTextConfig(**config_dict)

        self.assertEqual(config.num_experts, 256)
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.moe_intermediate_size, 1024)
        self.assertTrue(config.norm_topk_prob)
        self.assertEqual(config.decoder_sparse_step, 1)
        self.assertEqual(config.layer_types, ["linear_attention", "full_attention"])


class TestQwen3VLConfigBackwardCompatibility(CustomTestCase):
    """Test backward compatibility with transformers 5.3.0."""

    def test_qwen3vl_config_init_with_dict(self):
        """Test __init__ with dict vision_config and text_config."""
        config = Qwen3VLConfig(
            vision_config={"hidden_size": 1152, "num_heads": 16},
            text_config={"hidden_size": 4096},
        )

        self.assertIsInstance(config.vision_config, Qwen3VLVisionConfig)
        self.assertIsInstance(config.text_config, Qwen3VLTextConfig)
        self.assertEqual(config.vision_config.hidden_size, 1152)

    def test_qwen3vl_config_init_with_none(self):
        """Test __init__ with None creates default configs."""
        config = Qwen3VLConfig(vision_config=None, text_config=None)

        self.assertIsInstance(config.vision_config, Qwen3VLVisionConfig)
        self.assertIsInstance(config.text_config, Qwen3VLTextConfig)

    def test_qwen3vl_config_init_with_object(self):
        """Test __init__ with config objects preserves them."""
        vision_obj = Qwen3VLVisionConfig(hidden_size=2048)
        text_obj = Qwen3VLTextConfig(hidden_size=8192)

        config = Qwen3VLConfig(vision_config=vision_obj, text_config=text_obj)

        self.assertIs(config.vision_config, vision_obj)
        self.assertIs(config.text_config, text_obj)


if __name__ == "__main__":
    unittest.main()
