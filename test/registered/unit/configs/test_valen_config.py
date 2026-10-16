# SPDX-License-Identifier: Apache-2.0
"""Native Valen dispatch must preserve Qwen3.5's hybrid layer configuration."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sglang.srt.configs.model_config import ModelConfig, is_qwen3_5
from sglang.srt.configs.valen import ValenQwenConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestValenConfig(CustomTestCase):
    def test_native_config_keeps_hybrid_and_multimodal_dispatch_without_remote_code(
        self,
    ):
        data = {
            "model_type": "valen_qwen",
            "architectures": ["ValenQwenForDecisionMaking"],
            "auto_map": {"AutoConfig": "configuration_valen.ValenQwenConfig"},
            "tie_word_embeddings": True,
            "text_config": {
                "model_type": "qwen3_5_text",
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 64,
                "vocab_size": 512,
                "max_position_embeddings": 32768,
                "full_attention_interval": 2,
                "layer_types": ["linear_attention", "full_attention"],
            },
            "valen_head": {"head_type": "mixer", "head_width": 16},
            "valen_max_length": 32768,
        }
        with TemporaryDirectory() as checkpoint:
            (Path(checkpoint) / "config.json").write_text(json.dumps(data))
            config = ModelConfig(checkpoint, trust_remote_code=False, dtype="bfloat16")
        self.assertTrue(config.is_valen)
        self.assertTrue(config.is_multimodal)
        self.assertTrue(is_qwen3_5(config.hf_config))
        self.assertFalse(config.is_generation)
        self.assertIsInstance(config.hf_config, ValenQwenConfig)
        self.assertEqual(config.hf_config.architectures, ["ValenQwenForDecisionMaking"])
        self.assertEqual(
            config.hf_text_config.layers_block_type, ["linear_attention", "attention"]
        )

    def test_incompatible_execution_and_head_fail_before_loading_weights(self):
        for kwargs in (
            {"qwen_execution": "question"},
            {"valen_head": {"head_type": "linear"}},
            {"valen_max_length": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ValenQwenConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
