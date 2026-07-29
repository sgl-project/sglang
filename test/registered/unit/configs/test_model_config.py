"""Unit tests for hybrid attention model configuration."""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import (
    ModelConfig,
    get_hybrid_layer_ids,
    is_embedding_gemma,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHybridLayerIds(CustomTestCase):
    def test_layer_type_architectures(self):
        config = SimpleNamespace(
            num_hidden_layers=4,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )

        for architecture in (
            "Gemma4ForCausalLM",
            "Gemma4ForConditionalGeneration",
            "LagunaForCausalLM",
            "MellumForCausalLM",
        ):
            with self.subTest(architecture=architecture):
                self.assertEqual(
                    get_hybrid_layer_ids([architecture], config),
                    ([0, 2], [1, 3]),
                )


class TestEmbeddingGemmaConfig(CustomTestCase):
    def test_detects_bidirectional_gemma3_text_config(self):
        config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=True
        )
        self.assertTrue(is_embedding_gemma(config))

    def test_does_not_misclassify_causal_gemma3(self):
        config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=False
        )
        self.assertFalse(is_embedding_gemma(config))


class TestGetMaxNumAttentionHeads(CustomTestCase):
    def _make_config(self, num_attention_heads, **hf_attrs):
        config = ModelConfig.__new__(ModelConfig)
        config.num_attention_heads = num_attention_heads
        config.hf_text_config = SimpleNamespace(**hf_attrs)
        return config

    def test_per_layer_heads_returns_max(self):
        config = self._make_config(48, num_attention_heads_per_layer=[48, 72, 60, 48])
        self.assertEqual(config.get_max_num_attention_heads(), 72)

    def test_absent_per_layer_heads_falls_back(self):
        config = self._make_config(48)
        self.assertEqual(config.get_max_num_attention_heads(), 48)


if __name__ == "__main__":
    unittest.main()
