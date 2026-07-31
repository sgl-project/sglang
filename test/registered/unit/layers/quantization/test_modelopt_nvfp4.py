import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from sglang.srt.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from sglang.srt.layers.parameter import PerTensorScaleParameter
from sglang.srt.layers.quantization import modelopt_quant, nvfp4_online
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.layers.quantization.nvfp4_online import (
    ModelOptNvFp4OnlineFusedMoEMethod,
    NvFp4OnlineConfig,
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
    def test_activation_scheme_matches_quantization_interface(self, _):
        serialized_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
        )
        online_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=False,
            group_size=16,
        )

        self.assertTrue(serialized_config.use_per_token_activation)
        self.assertFalse(online_config.use_per_token_activation)
        with self.assertRaisesRegex(ValueError, "Use nvfp4_online"):
            ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=False,
                group_size=16,
                use_per_token_activation=True,
            )

    @patch.object(
        modelopt_quant,
        "_make_per_tensor_scale_parameter",
        wraps=modelopt_quant._make_per_tensor_scale_parameter,
    )
    def test_moe_input_scale_fill_matches_quantization_interface(self, make_scale):
        methods_configs_and_fills = (
            (
                ModelOptNvFp4FusedMoEMethod,
                ModelOptFp4Config(group_size=16),
                1.0,
            ),
            (ModelOptNvFp4OnlineFusedMoEMethod, NvFp4OnlineConfig(), None),
        )

        for method_cls, config, expected_fill in methods_configs_and_fills:
            with self.subTest(config=config.get_name()):
                make_scale.reset_mock()
                method = object.__new__(method_cls)
                method.quant_config = config
                method.enable_flashinfer_trtllm_moe = True
                layer = nn.Module()
                layer.num_experts = 1
                layer.num_local_experts = 1
                layer.moe_runner_config = SimpleNamespace(is_gated=True)

                with patch.object(
                    method, "prepare_weight_loader", return_value=MagicMock()
                ):
                    method.create_weights(
                        layer,
                        num_experts=1,
                        hidden_size=16,
                        intermediate_size_per_partition=16,
                        params_dtype=torch.bfloat16,
                        weight_loader=MagicMock(),
                    )

                self.assertEqual(
                    [call.kwargs["fill_value"] for call in make_scale.call_args_list],
                    [expected_fill, expected_fill],
                )

    def test_non_gated_w1_is_quantized_without_pairing(self):
        layer = SimpleNamespace(
            moe_runner_config=SimpleNamespace(is_gated=False),
            w13_weight_scale=object(),
            w13_weight_scale_2=object(),
        )
        param = SimpleNamespace(device=torch.device("cpu"))
        original_weight_loader = MagicMock()
        fp4_weight = torch.empty(2, 8, dtype=torch.uint8)
        weight_scale = torch.empty(2, 1, dtype=torch.float8_e4m3fn)
        weight_scale_2 = torch.ones((), dtype=torch.float32)

        with patch.object(
            nvfp4_online,
            "quantize_nvfp4_weight",
            return_value=(fp4_weight, weight_scale, weight_scale_2),
        ) as quantize:
            weight_loader = nvfp4_online.make_nvfp4_online_weight_loader(
                layer=layer,
                original_weight_loader=original_weight_loader,
                layer_prefix="mtp.layers.0.mixer.experts",
            )
            weight_loader(
                param,
                torch.ones(2, 16, dtype=torch.bfloat16),
                "mtp.layers.0.mixer.experts.0.up_proj.weight",
                "w1",
                None,
            )

        quantize.assert_called_once()
        self.assertEqual(original_weight_loader.call_count, 3)
        self.assertIs(original_weight_loader.call_args_list[0].args[0], param)
        self.assertIs(
            original_weight_loader.call_args_list[1].args[0],
            layer.w13_weight_scale,
        )
        self.assertIs(
            original_weight_loader.call_args_list[2].args[0],
            layer.w13_weight_scale_2,
        )

    def test_online_loader_rejects_fp8_without_dequantizer(self):
        weight_loader = nvfp4_online.make_nvfp4_online_weight_loader(
            layer=SimpleNamespace(),
            original_weight_loader=MagicMock(),
            layer_prefix="mtp.layers.0.mlp.experts",
        )

        with self.assertRaisesRegex(ValueError, "does not declare serialized FP8"):
            weight_loader(
                SimpleNamespace(device=torch.device("cpu")),
                torch.ones(2, 16, dtype=torch.float8_e4m3fn),
                "mtp.layers.0.mlp.experts.0.down_proj.weight",
                "w2",
                None,
            )


if __name__ == "__main__":
    unittest.main()
