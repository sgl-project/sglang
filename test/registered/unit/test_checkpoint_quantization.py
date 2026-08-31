# SPDX-License-Identifier: Apache-2.0

import unittest

from transformers import PretrainedConfig

from sglang.srt.layers.modelopt_utils import canonicalize_modelopt_quant_algo
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_loader.checkpoint_quantization import (
    CheckpointQuantSpec,
    resolve_checkpoint_quant_spec,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestResolveCheckpointQuantSpec(CustomTestCase):
    def test_modelopt_quant_algo_canonicalization(self):
        cases = {
            "FP8": "modelopt_fp8",
            "mxfp8": "mxfp8",
            "NVFP4": "modelopt_fp4",
            "NVFP4_AWQ": "modelopt_fp4",
            "W4A16_NVFP4": "modelopt_fp4",
            "FP8_FAKE": None,
            "MIXED_PRECISION": None,
            None: None,
        }
        for quant_algo, expected in cases.items():
            with self.subTest(quant_algo=quant_algo):
                self.assertEqual(canonicalize_modelopt_quant_algo(quant_algo), expected)

    def test_srt_modelopt_override_uses_the_exact_algorithm_allowlist(self):
        self.assertEqual(
            QuantizationConfig._modelopt_override_quantization_method(
                {"quant_algo": "NVFP4"}, "modelopt"
            ),
            "modelopt_fp4",
        )
        self.assertIsNone(
            QuantizationConfig._modelopt_override_quantization_method(
                {"quant_algo": "NVFP4_FAKE"}, "modelopt"
            )
        )

    def test_top_level_quantization_config(self):
        config = {
            "quantization_config": {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
            }
        }

        spec = resolve_checkpoint_quant_spec(config)

        self.assertEqual(
            spec,
            CheckpointQuantSpec(
                declared_method="fp8",
                config={"quant_method": "fp8", "activation_scheme": "dynamic"},
                source="quantization_config",
            ),
        )

    def test_text_config_fallback_supports_pretrained_configs(self):
        config = PretrainedConfig(
            text_config=PretrainedConfig(
                quantization_config={"quant_method": "gptq", "bits": 4}
            ),
            compression_config={"quant_method": "compressed-tensors"},
        )

        spec = resolve_checkpoint_quant_spec(config)

        self.assertIsNotNone(spec)
        self.assertEqual(spec.declared_method, "gptq")
        self.assertEqual(spec.source, "text_config.quantization_config")

    def test_compression_config_fallback(self):
        config = PretrainedConfig(
            compression_config={"quant_method": "compressed-tensors"}
        )

        spec = resolve_checkpoint_quant_spec(config)

        self.assertIsNotNone(spec)
        self.assertEqual(spec.declared_method, "compressed-tensors")
        self.assertEqual(spec.source, "compression_config")

    def test_modelopt_quant_algo_does_not_infer_declared_method(self):
        config = {
            "quantization_config": {
                "quant_algo": "FP8",
                "exclude_modules": ["lm_head"],
            }
        }

        spec = resolve_checkpoint_quant_spec(config)

        self.assertIsNotNone(spec)
        self.assertIsNone(spec.declared_method)
        self.assertEqual(spec.config["quant_algo"], "FP8")

    def test_lookup_priority_matches_srt_loader(self):
        config = {
            "quantization_config": {},
            "text_config": {"quantization_config": {"quant_method": "gptq"}},
            "compression_config": {"quant_method": "compressed-tensors"},
        }

        spec = resolve_checkpoint_quant_spec(config)

        self.assertIsNotNone(spec)
        self.assertEqual(spec.config, {})
        self.assertEqual(spec.source, "quantization_config")

    def test_metadata_is_deep_copied(self):
        metadata = {"quant_method": "fp8", "modules_to_not_convert": ["lm_head"]}
        spec = resolve_checkpoint_quant_spec({"quantization_config": metadata})

        self.assertIsNotNone(spec)
        spec.config["modules_to_not_convert"].append("embed_tokens")

        self.assertEqual(metadata["modules_to_not_convert"], ["lm_head"])

    def test_missing_metadata_returns_none(self):
        self.assertIsNone(resolve_checkpoint_quant_spec({"model_type": "qwen3_vl"}))

    def test_invalid_metadata_type_has_clear_error(self):
        with self.assertRaisesRegex(TypeError, "quantization_config must be a mapping"):
            resolve_checkpoint_quant_spec({"quantization_config": "fp8"})


if __name__ == "__main__":
    unittest.main()
