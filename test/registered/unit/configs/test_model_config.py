"""Unit tests for hybrid attention model configuration."""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import (
    ModelConfig,
    get_hybrid_layer_ids,
    is_embedding_gemma,
    resolve_spec_hidden_size,
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


class TestDraftModelConfig(CustomTestCase):
    def test_qwen35_mtp_depth_is_synced_to_text_config(self):
        config = object.__new__(ModelConfig)
        config.is_draft_model = True
        config.speculative_algorithm = "EAGLE"
        config.hf_config = SimpleNamespace(
            architectures=["Qwen3_5MoeForConditionalGeneration"]
        )
        config.hf_text_config = SimpleNamespace()

        config._config_draft_model()

        self.assertEqual(config.hf_config.architectures, ["Qwen3_5ForCausalLMMTP"])
        self.assertEqual(config.hf_config.num_nextn_predict_layers, 1)
        self.assertEqual(config.hf_text_config.num_nextn_predict_layers, 1)


class TestSpecHiddenSize(CustomTestCase):
    def test_hc_flattening_applies_to_dsv4_only(self):
        dsv4 = SimpleNamespace(architectures=["DeepseekV4ForCausalLM"])
        self.assertEqual(resolve_spec_hidden_size(dsv4, 4096, 4), (16384, 16384))
        self.assertEqual(resolve_spec_hidden_size(dsv4, 4096, 1), (4096, None))

        hy4 = SimpleNamespace(architectures=["HYV4ForCausalLM"])
        self.assertEqual(resolve_spec_hidden_size(hy4, 6144, 4), (6144, None))
        self.assertEqual(
            resolve_spec_hidden_size(
                SimpleNamespace(architectures=["HYV4ForCausalLMNextN"]), 6144, 4
            ),
            (6144, None),
        )


if __name__ == "__main__":
    unittest.main()
