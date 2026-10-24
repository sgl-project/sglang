import unittest
from types import SimpleNamespace
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

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestModelOptNvfp4(CustomTestCase):
    def _make_layer(self):
        return MergedColumnParallelLinear(
            input_size=16,
            output_sizes=[16, 16],
            bias=False,
            parallel_group="replicated",
        )

    def _make_qkv_layer(self):
        return QKVParallelLinear(
            hidden_size=16,
            head_size=8,
            total_num_heads=2,
            total_num_kv_heads=2,
            bias=False,
            parallel_group="replicated",
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

    def test_checkpoint_activation_metadata_overrides_environment(self):
        env = (
            "sglang.srt.layers.quantization.modelopt_quant.envs."
            "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.get"
        )
        for enabled in (False, True):
            with self.subTest(enabled=enabled), patch(env, return_value=not enabled):
                config = ModelOptFp4Config.from_config(
                    {"quant_algo": "NVFP4", "use_per_token_activation": enabled}
                )
                legacy = ModelOptFp4Config.from_config({"quant_algo": "NVFP4"})
                self.assertEqual(config.use_per_token_activation, enabled)
                self.assertEqual(legacy.use_per_token_activation, not enabled)

    def test_quantized_mtp_is_explicit_checkpoint_opt_in(self):
        legacy = ModelOptFp4Config.from_config({"quant_algo": "NVFP4"})
        packed = ModelOptFp4Config.from_config(
            {"quant_algo": "NVFP4", "quantized_mtp": True}
        )
        self.assertFalse(legacy.quantized_mtp)
        self.assertTrue(packed.quantized_mtp)
        self.assertTrue(packed.is_checkpoint_nvfp4_serialized)

    def test_nested_hf_quant_config_metadata_is_read_from_inner_block(self):
        env = (
            "sglang.srt.layers.quantization.modelopt_quant.envs."
            "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION.get"
        )
        with patch(env, return_value=False):
            config = ModelOptFp4Config.from_config(
                {
                    "quantization": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                        "exclude_modules": [],
                        "quantized_mtp": True,
                        "use_per_token_activation": True,
                    }
                }
            )
        self.assertTrue(config.quantized_mtp)
        self.assertTrue(config.use_per_token_activation)

    def test_per_token_metadata_rejects_cutedsl_deepep(self):
        runner = SimpleNamespace(is_flashinfer_cutedsl=lambda: True)
        a2a = SimpleNamespace(is_deepep=lambda: True)
        with (
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.get_moe_runner_backend",
                return_value=runner,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.get_moe_a2a_backend",
                return_value=a2a,
            ),
            self.assertRaisesRegex(
                ValueError, "moe_a2a_backend='none' or 'flashinfer'"
            ),
        ):
            ModelOptFp4Config.from_config(
                {"quant_algo": "NVFP4", "use_per_token_activation": True}
            )

    def test_checkpoint_metadata_rejects_incompatible_quantization(self):
        for algo in ("FP8", "W4A16_NVFP4"):
            with (
                self.subTest(algo=algo),
                self.assertRaisesRegex(
                    ValueError, "Per-token activation metadata requires serialized W4A4"
                ),
            ):
                ModelOptFp4Config.from_config(
                    {
                        "quant_algo": algo,
                        "group_size": 16,
                        "use_per_token_activation": True,
                    }
                )
        with self.assertRaisesRegex(ValueError, "quantized_mtp requires serialized"):
            ModelOptFp4Config.from_config(
                {"quant_algo": "FP8", "group_size": 16, "quantized_mtp": True}
            )

    def test_shared_expert_fusion_requires_matching_fp4_precision(self):
        quantized_shared = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )
        bf16_shared = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.*.mlp.shared_experts*"],
        )

        gate_only_bf16 = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["model.layers.*.mlp.shared_expert_gate"],
        )

        self.assertTrue(quantized_shared.can_fuse_shared_expert())
        self.assertFalse(bf16_shared.can_fuse_shared_expert())
        # Only the gate is BF16 (Qwen3-Next NVFP4): the FP4 body still fuses.
        self.assertTrue(gate_only_bf16.can_fuse_shared_expert())


if __name__ == "__main__":
    unittest.main()
