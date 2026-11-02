"""Unit tests for Nano Nemotron VL configuration compatibility."""

import json
import unittest

from sglang.srt.configs.nano_nemotron_vl import (
    NemotronH_Nano_Omni_Reasoning_V3_Config,
    NemotronH_Nano_VL_V2_Config,
    NemotronH_Omni_Reasoning_V3_Config,
)
from sglang.srt.configs.nemotron_h import NemotronHConfig, NemotronHPuzzleConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


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
        """get_text_config() returns llm_config, which LoRAManager sizes from."""
        config = NemotronH_Omni_Reasoning_V3_Config(
            vision_config={"args": {"model": "radio"}},
            llm_config={},
        )

        self.assertIs(config.get_text_config(), config.llm_config)
        self.assertIs(config.get_text_config(encoder=True), config.llm_config)

    def test_puzzle_layer_settings_survive_serialization(self):
        """Nested Puzzle configs must retain layer shapes without default construction."""
        llm_config = {
            "model_type": "nemotron_h_puzzle",
            "layers_block_type": ["moe", "moe"],
            "block_configs": [
                {
                    "block_type": "moe",
                    "n_routed_experts": 240,
                    "moe_intermediate_size": 2048,
                },
                {
                    "block_type": "moe",
                    "n_routed_experts": 304,
                    "moe_intermediate_size": 2304,
                },
            ],
        }
        for config_class in (
            NemotronH_Nano_VL_V2_Config,
            NemotronH_Nano_Omni_Reasoning_V3_Config,
            NemotronH_Omni_Reasoning_V3_Config,
        ):
            with self.subTest(config_class=config_class.__name__):
                config = config_class(vision_config={}, llm_config=llm_config)
                restored = config_class.from_dict(json.loads(config.to_json_string()))
                for nested in (config.llm_config, restored.llm_config):
                    self.assertIsInstance(nested, NemotronHPuzzleConfig)
                    first = nested.get_nemotron_h_config_for_layer(0)
                    second = nested.get_nemotron_h_config_for_layer(1)
                    self.assertEqual(
                        (first.n_routed_experts, first.moe_intermediate_size),
                        (240, 2048),
                    )
                    self.assertEqual(
                        (second.n_routed_experts, second.moe_intermediate_size),
                        (304, 2304),
                    )

    def test_non_puzzle_configs_keep_existing_defaults(self):
        """Puzzle dispatch must not change ordinary or omitted nested configs."""
        for config_class in (
            NemotronH_Nano_VL_V2_Config,
            NemotronH_Nano_Omni_Reasoning_V3_Config,
            NemotronH_Omni_Reasoning_V3_Config,
        ):
            for llm_config in (None, {}, {"model_type": "nemotron_h"}):
                with self.subTest(
                    config_class=config_class.__name__, llm_config=llm_config
                ):
                    config = (
                        config_class()
                        if llm_config is None
                        else config_class(vision_config={}, llm_config=llm_config)
                    )
                    self.assertIs(type(config.llm_config), NemotronHConfig)
                    self.assertFalse(config.has_no_defaults_at_init)


if __name__ == "__main__":
    unittest.main()
