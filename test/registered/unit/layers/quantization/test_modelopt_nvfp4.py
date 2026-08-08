import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.parameter import PerTensorScaleParameter
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestModelOptNvfp4(CustomTestCase):
    def test_auto_backend_must_be_resolved_before_weight_setup(self):
        with patch(
            "sglang.srt.layers.quantization.modelopt_quant.get_moe_runner_backend",
            return_value=MoeRunnerBackend.AUTO,
        ), patch(
            "sglang.srt.layers.quantization.modelopt_quant.is_blackwell_supported",
            return_value=True,
        ):
            with self.assertRaisesRegex(ValueError, "must be resolved"):
                ModelOptNvFp4FusedMoEMethod(ModelOptFp4Config())

    def test_weight_setup_and_runner_use_cached_resolved_backend(self):
        runner_config = SimpleNamespace(is_gated=True)
        layer = torch.nn.Module()
        layer.num_experts = 1
        layer.num_local_experts = 1
        layer.moe_runner_config = runner_config

        with patch(
            "sglang.srt.layers.quantization.modelopt_quant.get_moe_runner_backend",
            return_value=MoeRunnerBackend.FLASHINFER_TRTLLM,
        ), patch(
            "sglang.srt.layers.quantization.modelopt_quant.is_blackwell_supported",
            return_value=True,
        ), patch(
            "sglang.srt.layers.quantization.modelopt_quant.MoeRunner"
        ) as runner_cls:
            method = ModelOptNvFp4FusedMoEMethod(
                ModelOptFp4Config(
                    is_checkpoint_nvfp4_serialized=True,
                    group_size=16,
                )
            )
            method.create_weights(
                layer,
                num_experts=1,
                hidden_size=16,
                intermediate_size_per_partition=16,
                params_dtype=torch.bfloat16,
            )
            method.create_moe_runner(layer, runner_config)

            self.assertEqual(
                method._moe_runner_backend,
                MoeRunnerBackend.FLASHINFER_TRTLLM,
            )
            self.assertTrue(method.enable_flashinfer_trtllm_moe)
            self.assertIsNone(layer.w13_blockscale_swizzled)
            self.assertIsNone(layer.w2_blockscale_swizzled)
            self.assertFalse(method.load_up_proj_weight_first)
            runner_cls.assert_called_once_with(
                MoeRunnerBackend.FLASHINFER_TRTLLM,
                runner_config,
            )

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


if __name__ == "__main__":
    unittest.main()
