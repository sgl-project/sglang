"""Unit tests for Nano Nemotron VL configuration compatibility."""

import unittest

from sglang.srt.configs.nano_nemotron_vl import (
    NemotronH_Omni_Reasoning_V3_Config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNemotronHOmniConfig(CustomTestCase):
    def test_uses_checkpoint_model_type(self):
        config = NemotronH_Omni_Reasoning_V3_Config(
            vision_config={"args": {"model": "radio"}},
            llm_config={},
            architectures=["NemotronH_Omni_Reasoning_V3"],
        )

        self.assertEqual(config.model_type, "nemotron_h_omni")

    def test_normalizes_current_nemotron_h_layer_names(self):
        llm_config = {
            "layers_block_type": ["linear_attention", "moe", "full_attention"],
            "num_nextn_predict_layers": 1,
            "mtp_layers_block_type": ["full_attention", "moe"],
        }

        config = NemotronH_Omni_Reasoning_V3_Config(
            vision_config={"args": {"model": "radio"}},
            llm_config=llm_config,
        )

        self.assertEqual(
            config.llm_config.layers_block_type,
            ["mamba", "moe", "attention"],
        )
        self.assertEqual(
            config.llm_config.mtp_layers_block_type,
            ["attention", "moe"],
        )
        self.assertEqual(
            llm_config["layers_block_type"],
            ["linear_attention", "moe", "full_attention"],
        )

    def test_text_config_resolves_to_the_language_model_config(self):
        """LoRAManager sizes its pools from base_hf_config.get_text_config();
        without the override that returns the VL config itself, which carries no
        num_hidden_layers, and LoRA init dies before serving a single request."""
        config = NemotronH_Omni_Reasoning_V3_Config(
            vision_config={"args": {"model": "radio"}},
            llm_config={},
        )

        self.assertIs(config.get_text_config(), config.llm_config)
        # Without the override this raises AttributeError rather than comparing
        # unequal, which is the reported failure.
        self.assertIsInstance(config.get_text_config().num_hidden_layers, int)


if __name__ == "__main__":
    unittest.main()
