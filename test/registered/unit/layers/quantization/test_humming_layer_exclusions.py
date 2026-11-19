# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

import unittest
from unittest.mock import patch

from sglang.srt.layers.quantization.humming import HummingConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHummingLayerExclusions(CustomTestCase):
    def _config(self, ignored_layers: list[str]) -> HummingConfig:
        config = {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "ignore": ignored_layers,
        }
        with patch("sglang.srt.layers.quantization.humming._lazy_import_humming"):
            humming_config = HummingConfig(config)
        humming_config.update_packed_modules_mapping(
            {
                "fused_qkv_a_proj_with_mqa": [
                    "q_a_proj",
                    "kv_a_proj_with_mqa",
                ]
            }
        )
        return humming_config

    def test_fused_attention_layer_is_skipped_when_all_checkpoint_shards_are(self):
        ignored_layers = [
            "model.layers.0.self_attn.q_a_proj",
            "model.layers.0.self_attn.kv_a_proj_with_mqa",
        ]
        config = self._config(ignored_layers)
        layer_name = "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa"

        self.assertTrue(config.is_layer_skipped({"ignore": ignored_layers}, layer_name))
        self.assertIsNone(
            config.get_layer_weight_schema(config.full_config, layer_name)
        )
        self.assertIsNone(config.get_layer_input_schema(config.full_config, layer_name))

    def test_fused_attention_layer_rejects_mixed_exclusion_rules(self):
        config = self._config(["model.layers.0.self_attn.q_a_proj"])

        with self.assertRaisesRegex(ValueError, "some but not all shards"):
            config.is_layer_skipped(
                config.full_config,
                "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa",
            )

    def test_dynamic_exclusion_still_matches_layer_prefix(self):
        config = self._config([])

        self.assertTrue(
            config.is_layer_skipped(
                {"dynamic": {"-:.*self_attn\\.fused_qkv_a_proj_with_mqa": {}}},
                "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa",
            )
        )


if __name__ == "__main__":
    unittest.main()
