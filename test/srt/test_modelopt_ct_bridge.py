"""
Unit tests for ModelOpt-to-CompressedTensors bridge (FP8_PER_CHANNEL_PER_TOKEN_CFG).

E1: Config parsing, detection, infrastructure.
E2: Weight loading adapter (name mapping) and bridge integration.
"""

import unittest

from sglang.srt.layers.quantization.compressed_tensors.modelopt_ct_bridge import (
    FP8_PER_CHANNEL_PER_TOKEN_CFG,
    is_modelopt_fp8_per_channel_per_token_config,
    modelopt_config_to_compressed_tensors_config,
    remap_modelopt_state_dict_keys_for_ct,
)


class TestModelOptCTBridgeDetection(unittest.TestCase):
    """E1: Config detection tests."""

    def test_detect_by_quant_cfg_nested(self):
        """Detect when config has quantization.quant_cfg == FP8_PER_CHANNEL_PER_TOKEN_CFG."""
        config = {
            "producer": {"name": "modelopt"},
            "quantization": {
                "quant_cfg": FP8_PER_CHANNEL_PER_TOKEN_CFG,
                "quant_algo": "FP8",
            },
        }
        self.assertTrue(is_modelopt_fp8_per_channel_per_token_config(config))

    def test_detect_by_quant_cfg_flat(self):
        """Detect when config is flat (e.g. from config.json) with quant_cfg."""
        config = {
            "quant_cfg": FP8_PER_CHANNEL_PER_TOKEN_CFG,
            "quant_algo": "FP8",
            "ignore": [],
        }
        self.assertTrue(is_modelopt_fp8_per_channel_per_token_config(config))

    def test_detect_by_recipe_key(self):
        """Detect when recipe key is used instead of quant_cfg."""
        config = {
            "producer": {"name": "modelopt"},
            "quantization": {
                "recipe": FP8_PER_CHANNEL_PER_TOKEN_CFG,
            },
        }
        self.assertTrue(is_modelopt_fp8_per_channel_per_token_config(config))

    def test_detect_by_quant_algo_per_channel_per_token(self):
        """Detect when quant_algo contains PER_CHANNEL and PER_TOKEN."""
        config = {
            "producer": {"name": "modelopt"},
            "quantization": {
                "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN",
            },
        }
        self.assertTrue(is_modelopt_fp8_per_channel_per_token_config(config))

    def test_reject_non_modelopt_producer(self):
        """Do not detect when producer is not modelopt (and no explicit quant_cfg)."""
        config = {
            "producer": {"name": "other_tool"},
            "quantization": {
                "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN",
            },
        }
        self.assertFalse(is_modelopt_fp8_per_channel_per_token_config(config))

    def test_reject_fp4(self):
        """Do not detect FP4 config."""
        config = {
            "producer": {"name": "modelopt"},
            "quantization": {"quant_algo": "NVFP4"},
        }
        self.assertFalse(is_modelopt_fp8_per_channel_per_token_config(config))

    def test_reject_default_fp8(self):
        """Do not detect default FP8 (per-tensor) config."""
        config = {
            "producer": {"name": "modelopt"},
            "quantization": {"quant_algo": "FP8"},
        }
        self.assertFalse(is_modelopt_fp8_per_channel_per_token_config(config))


class TestModelOptCTBridgeConfigConversion(unittest.TestCase):
    """E1: Config conversion to CompressedTensorsConfig."""

    def test_conversion_produces_ct_config(self):
        """modelopt_config_to_compressed_tensors_config returns CompressedTensorsConfig."""
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )

        config = {
            "producer": {"name": "modelopt"},
            "quantization": {
                "quant_cfg": FP8_PER_CHANNEL_PER_TOKEN_CFG,
                "ignore": ["embed_tokens", "lm_head"],
            },
        }
        ct_config = modelopt_config_to_compressed_tensors_config(
            config,
            packed_modules_mapping={},
        )
        self.assertIsInstance(ct_config, CompressedTensorsConfig)
        self.assertEqual(ct_config.ignore, ["embed_tokens", "lm_head"])

    def test_conversion_rejects_non_bridge_config(self):
        """Conversion raises when config is not FP8 per-channel per-token."""
        config = {
            "producer": {"name": "modelopt"},
            "quantization": {"quant_algo": "FP8"},
        }
        with self.assertRaises(ValueError) as ctx:
            modelopt_config_to_compressed_tensors_config(config)
        self.assertIn("FP8 per-channel per-token", str(ctx.exception))

    def test_conversion_maps_to_w8a8_fp8_scheme(self):
        """Converted config yields CompressedTensorsW8A8Fp8 with channel strategy."""
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            CompressedTensorsW8A8Fp8,
        )
        from compressed_tensors.quantization import QuantizationStrategy

        config = {
            "producer": {"name": "modelopt"},
            "quantization": {"quant_cfg": FP8_PER_CHANNEL_PER_TOKEN_CFG},
        }
        ct_config = modelopt_config_to_compressed_tensors_config(config)
        # get_scheme is called with a layer and layer_name; we only check target_scheme_map
        self.assertIn("*", ct_config.target_scheme_map)
        scheme_dict = ct_config.target_scheme_map["*"]
        weights = scheme_dict["weights"]
        self.assertEqual(weights.strategy, QuantizationStrategy.CHANNEL.value)
        input_act = scheme_dict["input_activations"]
        self.assertIsNotNone(input_act)
        self.assertTrue(input_act.dynamic)
        self.assertEqual(input_act.strategy, QuantizationStrategy.TOKEN.value)


class TestModelOptCTBridgeWeightAdapter(unittest.TestCase):
    """E2: Weight loader adapter (name mapping)."""

    def test_remap_empty_map_returns_same(self):
        """With no suffix map, state dict is returned unchanged."""
        state = {"model.layers.0.weight": object(), "model.layers.0.weight_scale": object()}
        out = remap_modelopt_state_dict_keys_for_ct(state)
        self.assertEqual(out, state)

    def test_remap_applies_suffix_map(self):
        """Keys ending with mapped suffix are remapped."""
        state = {"a.b.scale": 1}
        out = remap_modelopt_state_dict_keys_for_ct(
            state,
            key_suffix_map={".scale": ".weight_scale"},
        )
        self.assertIn("a.b.weight_scale", out)
        self.assertEqual(out["a.b.weight_scale"], 1)
        self.assertNotIn("a.b.scale", out)


if __name__ == "__main__":
    unittest.main()
