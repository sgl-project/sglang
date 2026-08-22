import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from sglang.srt.layers.parameter import PerTensorScaleParameter
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestModelOptNvfp4(CustomTestCase):
    def _make_layer(self):
        return MergedColumnParallelLinear(
            input_size=16,
            output_sizes=[16, 16],
            bias=False,
            tp_rank=0,
            tp_size=1,
        )

    def _make_qkv_layer(self):
        return QKVParallelLinear(
            hidden_size=16,
            head_size=8,
            total_num_heads=2,
            total_num_kv_heads=2,
            bias=False,
            tp_rank=0,
            tp_size=1,
        )

    def test_fused_scalar_scale_load_fills_all_logical_slots(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.25, dtype=torch.float32))

        torch.testing.assert_close(scale, torch.tensor([0.25, 0.25]))

    def test_fused_scalar_scale_load_rejects_non_scalar(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        with self.assertRaisesRegex(ValueError, "Expected scalar scale"):
            layer.weight_loader_v2(scale, torch.tensor([0.25, 0.5]))

    def test_fused_qkv_scalar_scale_load_fills_all_logical_slots(self):
        layer = self._make_qkv_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(3, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.125, dtype=torch.float32))

        torch.testing.assert_close(scale, torch.tensor([0.125, 0.125, 0.125]))

    def test_explicit_shard_scale_loads_stay_independent(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.25, dtype=torch.float32), 0)
        layer.weight_loader_v2(scale, torch.tensor(0.5, dtype=torch.float32), 1)

        torch.testing.assert_close(scale, torch.tensor([0.25, 0.5]))

    def test_missing_input_scale_defaults_to_one_and_checkpoint_overwrites(self):
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            use_per_token_activation=False,
        )
        layer = nn.Module()
        ModelOptFp4LinearMethod(config).create_weights(
            layer,
            input_size_per_partition=16,
            output_partition_sizes=[16],
            input_size=16,
            output_size=16,
            params_dtype=torch.bfloat16,
            weight_loader=default_weight_loader,
        )

        torch.testing.assert_close(layer.input_scale, torch.ones(1))
        default_weight_loader(layer.input_scale, torch.tensor(0.25))
        torch.testing.assert_close(layer.input_scale, torch.tensor([0.25]))

    @patch(
        "sglang.srt.layers.quantization.modelopt_quant.envs."
        "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.get",
        return_value=True,
    )
    def test_modelopt_fp4_per_token_activation_contract(self, _):
        # Serialized ModelOpt FP4 retains the existing environment-controlled
        # per-token activation path.
        serialized_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )
        # Online modelopt_fp4 always uses per-tensor activation scaling, even
        # when the serialized-checkpoint environment switch is enabled.
        online_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=False,
            group_size=16,
        )

        self.assertTrue(serialized_config.use_per_token_activation)
        self.assertFalse(online_config.use_per_token_activation)
        # nvfp4_online is the public interface for online per-token scaling.
        with self.assertRaisesRegex(ValueError, "Use nvfp4_online"):
            ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=False,
                group_size=16,
                use_per_token_activation=True,
            )


class TestModelOptFp4Config(CustomTestCase):
    @staticmethod
    def _flat_config(input_group_size=16, weight_group_size=16):
        return {
            "config_groups": {
                "group_0": {
                    "input_activations": {"group_size": input_group_size},
                    "weights": {"group_size": weight_group_size},
                }
            },
            "ignore": ["lm_head"],
            "kv_cache_scheme": {"type": "float", "num_bits": 8},
            "quant_algo": "NVFP4",
        }

    def test_parses_flat_config(self):
        result = ModelOptFp4Config.from_config(self._flat_config())

        self.assertEqual(result.group_size, 16)
        self.assertEqual(result.kv_cache_quant_algo, "FP8")
        self.assertEqual(result.exclude_modules, ["lm_head"])

    def test_parses_legacy_config(self):
        result = ModelOptFp4Config.from_config(
            {
                "quantization": {
                    "exclude_modules": ["lm_head"],
                    "group_size": 16,
                    "kv_cache_quant_algo": "FP8",
                    "quant_algo": "NVFP4",
                }
            }
        )

        self.assertEqual(result.group_size, 16)
        self.assertEqual(result.kv_cache_quant_algo, "FP8")
        self.assertEqual(result.exclude_modules, ["lm_head"])

    def test_missing_group_size_defaults_to_nvfp4_block_size(self):
        config = self._flat_config()
        del config["config_groups"]["group_0"]["input_activations"]["group_size"]
        del config["config_groups"]["group_0"]["weights"]["group_size"]

        result = ModelOptFp4Config.from_config(config)

        self.assertEqual(result.group_size, 16)

    def test_accepts_equal_group_sizes_across_groups(self):
        config = self._flat_config()
        config["config_groups"]["group_1"] = {
            "group_size": 16,
            "weights": {"group_size": 16},
        }

        result = ModelOptFp4Config.from_config(config)

        self.assertEqual(result.group_size, 16)

    def test_rejects_inconsistent_group_sizes(self):
        across_groups = self._flat_config()
        across_groups["config_groups"]["group_1"] = {"weights": {"group_size": 32}}
        top_level = self._flat_config()
        top_level["group_size"] = 32

        for name, config in (
            ("within_group", self._flat_config(input_group_size=32)),
            ("across_groups", across_groups),
            ("top_level", top_level),
        ):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    ValueError, r"Inconsistent group_size values: \[16, 32\]"
                ):
                    ModelOptFp4Config.from_config(config)

    def test_rejects_unsupported_group_size(self):
        config = self._flat_config(input_group_size=32, weight_group_size=32)

        with self.assertRaisesRegex(
            ValueError, "ModelOpt NVFP4 requires group_size=16, got 32"
        ):
            ModelOptFp4Config.from_config(config)

    def test_rejects_non_integer_group_sizes(self):
        for value in ("32", 32.0, True):
            with self.subTest(value=value):
                config = self._flat_config(input_group_size=value)
                del config["config_groups"]["group_0"]["weights"]["group_size"]

                with self.assertRaisesRegex(
                    ValueError, "group_size must be an integer"
                ):
                    ModelOptFp4Config.from_config(config)


if __name__ == "__main__":
    unittest.main()
