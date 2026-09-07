"""Unit tests for hybrid attention model configuration."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.configs.model_config import (
    ModelConfig,
    _get_and_verify_dtype,
    get_hybrid_layer_ids,
    is_embedding_gemma,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestGetAndVerifyDtype(CustomTestCase):
    def test_missing_config_dtype_warns_for_auto_downcast(self):
        """An implicit float32-to-float16 choice warns when dtype is absent."""
        with patch("sglang.srt.configs.model_config.logger.warning") as warning:
            dtype = _get_and_verify_dtype({}, "auto")

        self.assertIs(dtype, torch.float16)
        warning.assert_called_once()
        self.assertIn("declares no dtype/torch_dtype", warning.call_args.args[0])

    def test_auto_downcast_warning_requires_missing_config_dtype(self):
        """Declared or explicitly requested dtypes do not report a missing dtype."""
        cases = (
            ("declared-float32", {"dtype": "float32"}, "auto", torch.float16),
            ("explicit-dtype", {}, "bfloat16", torch.bfloat16),
            ("unknown-config-dtype", {"dtype": "unknown"}, "auto", torch.float16),
        )

        for name, config, requested_dtype, expected_dtype in cases:
            with self.subTest(name=name):
                with patch("sglang.srt.configs.model_config.logger.warning") as warning:
                    dtype = _get_and_verify_dtype(config, requested_dtype)

                self.assertIs(dtype, expected_dtype)
                warning.assert_not_called()


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


if __name__ == "__main__":
    unittest.main()
