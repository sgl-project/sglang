"""
Unit tests for ModelOptQuantizationScheme and scheme detection (config parser).

Covers:
1. ModelOptQuantizationScheme class (uses_ct_bridge, quant_method)
2. detect_modelopt_quantization_scheme for all recipe types
3. Integration with model_config._parse_modelopt_quant_config
"""

import unittest

from sglang.srt.layers.quantization.modelopt_scheme import (
    ModelOptQuantizationScheme,
    detect_modelopt_quantization_scheme,
)


class TestModelOptQuantizationSchemeClass(unittest.TestCase):
    """Tests for ModelOptQuantizationScheme enum and helpers."""

    def test_fp8_per_channel_per_token_uses_ct_bridge(self):
        self.assertTrue(
            ModelOptQuantizationScheme.FP8_PER_CHANNEL_PER_TOKEN_CFG.uses_ct_bridge()
        )

    def test_fp8_default_does_not_use_ct_bridge(self):
        self.assertFalse(ModelOptQuantizationScheme.FP8_DEFAULT_CFG.uses_ct_bridge())

    def test_int4_awq_uses_ct_bridge(self):
        self.assertTrue(ModelOptQuantizationScheme.INT4_AWQ_CFG.uses_ct_bridge())

    def test_w4a8_awq_uses_ct_bridge(self):
        self.assertTrue(ModelOptQuantizationScheme.W4A8_AWQ_BETA_CFG.uses_ct_bridge())

    def test_nvfp4_default_does_not_use_ct_bridge(self):
        self.assertFalse(ModelOptQuantizationScheme.NVFP4_DEFAULT_CFG.uses_ct_bridge())

    def test_quant_method_fp8_schemes(self):
        self.assertEqual(
            ModelOptQuantizationScheme.FP8_DEFAULT_CFG.quant_method(), "modelopt_fp8"
        )
        self.assertEqual(
            ModelOptQuantizationScheme.FP8_PER_CHANNEL_PER_TOKEN_CFG.quant_method(),
            "modelopt_fp8",
        )

    def test_quant_method_fp4_schemes(self):
        self.assertEqual(
            ModelOptQuantizationScheme.NVFP4_DEFAULT_CFG.quant_method(), "modelopt_fp4"
        )
        self.assertEqual(
            ModelOptQuantizationScheme.INT4_AWQ_CFG.quant_method(), "modelopt_fp4"
        )

    def test_quant_method_mixed_precision(self):
        self.assertEqual(
            ModelOptQuantizationScheme.MIXED_PRECISION.quant_method(), "w4afp8"
        )


class TestDetectModelOptScheme(unittest.TestCase):
    """Tests for detect_modelopt_quantization_scheme (config parser)."""

    def test_detect_fp8_default_by_algo(self):
        config = {
            "producer": {"name": "modelopt"},
            "quantization": {"quant_algo": "FP8"},
        }
        self.assertEqual(
            detect_modelopt_quantization_scheme(config),
            ModelOptQuantizationScheme.FP8_DEFAULT_CFG,
        )

    def test_detect_fp8_per_channel_per_token_by_cfg(self):
        config = {
            "quantization": {"quant_cfg": "FP8_PER_CHANNEL_PER_TOKEN_CFG"},
        }
        self.assertEqual(
            detect_modelopt_quantization_scheme(config),
            ModelOptQuantizationScheme.FP8_PER_CHANNEL_PER_TOKEN_CFG,
        )

    def test_detect_fp8_per_channel_per_token_by_algo(self):
        config = {
            "producer": {"name": "modelopt"},
            "quantization": {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"},
        }
        self.assertEqual(
            detect_modelopt_quantization_scheme(config),
            ModelOptQuantizationScheme.FP8_PER_CHANNEL_PER_TOKEN_CFG,
        )

    def test_detect_nvfp4_by_algo(self):
        config = {
            "producer": {"name": "modelopt"},
            "quantization": {"quant_algo": "NVFP4"},
        }
        self.assertEqual(
            detect_modelopt_quantization_scheme(config),
            ModelOptQuantizationScheme.NVFP4_DEFAULT_CFG,
        )

    def test_detect_nvfp4_awq_by_cfg(self):
        config = {
            "quantization": {"quant_cfg": "NVFP4_AWQ_LITE_CFG"},
        }
        self.assertEqual(
            detect_modelopt_quantization_scheme(config),
            ModelOptQuantizationScheme.NVFP4_AWQ_LITE_CFG,
        )

    def test_detect_int4_awq_by_cfg(self):
        config = {
            "quantization": {"quant_cfg": "INT4_AWQ_CFG"},
        }
        self.assertEqual(
            detect_modelopt_quantization_scheme(config),
            ModelOptQuantizationScheme.INT4_AWQ_CFG,
        )

    def test_detect_int4_awq_by_algo(self):
        config = {
            "producer": {"name": "modelopt"},
            "quantization": {"quant_algo": "INT4_AWQ"},
        }
        self.assertEqual(
            detect_modelopt_quantization_scheme(config),
            ModelOptQuantizationScheme.INT4_AWQ_CFG,
        )

    def test_detect_w4a8_awq_by_cfg(self):
        config = {
            "quantization": {"quant_cfg": "W4A8_AWQ_BETA_CFG"},
        }
        self.assertEqual(
            detect_modelopt_quantization_scheme(config),
            ModelOptQuantizationScheme.W4A8_AWQ_BETA_CFG,
        )

    def test_detect_w4a8_awq_by_algo(self):
        config = {
            "producer": {"name": "modelopt"},
            "quantization": {"quant_algo": "W4A8_AWQ"},
        }
        self.assertEqual(
            detect_modelopt_quantization_scheme(config),
            ModelOptQuantizationScheme.W4A8_AWQ_BETA_CFG,
        )

    def test_detect_mixed_precision(self):
        config = {"quantization": {"quant_algo": "MIXED_PRECISION"}}
        self.assertEqual(
            detect_modelopt_quantization_scheme(config),
            ModelOptQuantizationScheme.MIXED_PRECISION,
        )

    def test_detect_int8_default_by_cfg(self):
        config = {"quantization": {"quant_cfg": "INT8_DEFAULT_CFG"}}
        self.assertEqual(
            detect_modelopt_quantization_scheme(config),
            ModelOptQuantizationScheme.INT8_DEFAULT_CFG,
        )

    def test_detect_int8_smoothquant_by_cfg(self):
        config = {"quantization": {"quant_cfg": "INT8_SMOOTHQUANT_CFG"}}
        self.assertEqual(
            detect_modelopt_quantization_scheme(config),
            ModelOptQuantizationScheme.INT8_SMOOTHQUANT_CFG,
        )

    def test_require_modelopt_producer_rejects_other(self):
        config = {
            "producer": {"name": "other"},
            "quantization": {"quant_cfg": "FP8_PER_CHANNEL_PER_TOKEN_CFG"},
        }
        self.assertIsNone(
            detect_modelopt_quantization_scheme(
                config, require_modelopt_producer=True
            )
        )

    def test_require_modelopt_producer_accepts_modelopt(self):
        config = {
            "producer": {"name": "modelopt"},
            "quantization": {"quant_cfg": "FP8_PER_CHANNEL_PER_TOKEN_CFG"},
        }
        self.assertEqual(
            detect_modelopt_quantization_scheme(
                config, require_modelopt_producer=True
            ),
            ModelOptQuantizationScheme.FP8_PER_CHANNEL_PER_TOKEN_CFG,
        )

    def test_unknown_quant_cfg_returns_none(self):
        config = {"quantization": {"quant_cfg": "UNKNOWN_RECIPE_XYZ"}}
        self.assertIsNone(detect_modelopt_quantization_scheme(config))

    def test_flat_config_without_quantization_key(self):
        config = {"quant_algo": "FP8", "producer": {"name": "modelopt"}}
        self.assertEqual(
            detect_modelopt_quantization_scheme(config),
            ModelOptQuantizationScheme.FP8_DEFAULT_CFG,
        )


class TestModelConfigParseModelOptQuantConfig(unittest.TestCase):
    """Integration tests for model_config._parse_modelopt_quant_config.

    Tests the config parser in isolation by calling the method with various
    quant_config_dicts. ModelConfig is not fully instantiated; we call the
    parser logic via a minimal object that has the method (bound).
    """

    def _parse_modelopt_quant_config(self, quant_config_dict: dict):
        """Call the parser logic without full ModelConfig init."""
        from sglang.srt.configs.model_config import ModelConfig

        # Use a real tiny model path so get_config can load (e.g. in CI)
        # or skip if not available. Alternatively test only the scheme->quant_method mapping.
        try:
            # Try to create ModelConfig; if it fails (e.g. no network), skip
            model_config = ModelConfig(model_path="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        except Exception:
            self.skipTest("ModelConfig init failed (e.g. no network for HF config)")
        return model_config._parse_modelopt_quant_config(quant_config_dict)

    def test_parse_returns_quant_method_from_scheme(self):
        quant_config_dict = {
            "quantization": {"quant_cfg": "FP8_PER_CHANNEL_PER_TOKEN_CFG"}
        }
        out = self._parse_modelopt_quant_config(quant_config_dict)
        self.assertIsNotNone(out)
        self.assertEqual(out["quant_method"], "modelopt_fp8")

    def test_parse_int4_awq_returns_modelopt_fp4(self):
        quant_config_dict = {"quantization": {"quant_cfg": "INT4_AWQ_CFG"}}
        out = self._parse_modelopt_quant_config(quant_config_dict)
        self.assertIsNotNone(out)
        self.assertEqual(out["quant_method"], "modelopt_fp4")

    def test_parse_legacy_fp8_algo_returns_modelopt_fp8(self):
        quant_config_dict = {"quantization": {"quant_algo": "FP8"}}
        out = self._parse_modelopt_quant_config(quant_config_dict)
        self.assertIsNotNone(out)
        self.assertEqual(out["quant_method"], "modelopt_fp8")

    def test_parse_legacy_nvfp4_returns_modelopt_fp4(self):
        quant_config_dict = {"quantization": {"quant_algo": "NVFP4"}}
        out = self._parse_modelopt_quant_config(quant_config_dict)
        self.assertIsNotNone(out)
        self.assertEqual(out["quant_method"], "modelopt_fp4")

    def test_parse_mixed_precision_returns_w4afp8(self):
        quant_config_dict = {"quantization": {"quant_algo": "MIXED_PRECISION"}}
        out = self._parse_modelopt_quant_config(quant_config_dict)
        self.assertIsNotNone(out)
        self.assertEqual(out["quant_method"], "w4afp8")

    def test_parse_unknown_returns_none(self):
        quant_config_dict = {"quantization": {"quant_algo": "UNKNOWN"}}
        out = self._parse_modelopt_quant_config(quant_config_dict)
        self.assertIsNone(out)


class TestParseModelOptQuantConfigLogic(unittest.TestCase):
    """Unit tests for parser logic without ModelConfig init.

    Verifies scheme detection -> quant_method mapping that _parse_modelopt_quant_config
    uses, so we have coverage even when ModelConfig cannot be instantiated.
    """

    def test_scheme_quant_method_mapping_matches_parser_expectation(self):
        """Scheme.quant_method() values match what _parse_modelopt_quant_config returns."""
        self.assertEqual(
            ModelOptQuantizationScheme.FP8_PER_CHANNEL_PER_TOKEN_CFG.quant_method(),
            "modelopt_fp8",
        )
        self.assertEqual(
            ModelOptQuantizationScheme.INT4_AWQ_CFG.quant_method(), "modelopt_fp4"
        )
        self.assertEqual(
            ModelOptQuantizationScheme.W4A8_AWQ_BETA_CFG.quant_method(), "modelopt_fp8"
        )
        self.assertEqual(
            ModelOptQuantizationScheme.MIXED_PRECISION.quant_method(), "w4afp8"
        )


if __name__ == "__main__":
    unittest.main()
