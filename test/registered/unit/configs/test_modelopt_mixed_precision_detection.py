"""Unit tests for ModelOpt MIXED_PRECISION quant-method detection in
srt/configs/model_config.py (_parse_modelopt_quant_config)."""

import unittest

from sglang.srt.configs.model_config import ModelConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _parse(quantization: dict) -> dict:
    # _parse_modelopt_quant_config only reads the passed dict; bypass the
    # heavyweight ModelConfig constructor.
    model_config = ModelConfig.__new__(ModelConfig)
    return model_config._parse_modelopt_quant_config({"quantization": quantization})


class TestModelOptMixedPrecisionDetection(CustomTestCase):
    def test_mixed_precision_with_quantized_layers_is_modelopt_mixed(self):
        # ModelOpt per-layer mixed export (e.g. MiMo-V2.5 NVFP4: MXFP8/FP8
        # dense + NVFP4 experts; NemotronH mixed exports have the same shape).
        result = _parse(
            {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {
                    "model.layers.0.self_attn.qkv_proj": {"quant_algo": "FP8"},
                    "model.layers.1.mlp.experts.0.gate_proj": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    },
                },
            }
        )
        self.assertEqual(result["quant_method"], "modelopt_mixed")

    def test_mixed_precision_without_quantized_layers_is_w4afp8(self):
        # DeepSeek-style W4AFP8 export: bare MIXED_PRECISION config
        # (e.g. nvidia/DeepSeek-R1-0528-W4AFP8 hf_quant_config.json).
        result = _parse({"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": None})
        self.assertEqual(result["quant_method"], "w4afp8")

    def test_mixed_precision_with_empty_quantized_layers_is_w4afp8(self):
        # An empty map can't drive per-layer dispatch
        # (ModelOptMixedPrecisionConfig.from_config would reject it).
        result = _parse({"quant_algo": "MIXED_PRECISION", "quantized_layers": {}})
        self.assertEqual(result["quant_method"], "w4afp8")

    def test_nvfp4_is_modelopt_fp4(self):
        result = _parse({"quant_algo": "NVFP4"})
        self.assertEqual(result["quant_method"], "modelopt_fp4")

    def test_fp8_is_modelopt_fp8(self):
        result = _parse({"quant_algo": "FP8"})
        self.assertEqual(result["quant_method"], "modelopt_fp8")

    def test_unknown_algo_returns_none(self):
        self.assertIsNone(_parse({"quant_algo": "INT42"}))


if __name__ == "__main__":
    unittest.main()
