import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
)
from sglang.srt.layers.parameter import PerTensorScaleParameter
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.layers.quantization.nvfp4_online import (
    ModelOptNvFp4OnlineFusedMoEMethod,
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

    @patch(
        "sglang.srt.layers.quantization.nvfp4_online.envs."
        "SGLANG_FP4_IGNORED_LAYERS.get",
        return_value="model.layers.1.mlp.experts",
    )
    def test_online_fp8_source_preserves_source_quantization(self, _):
        config = ModelOptFp4Config.from_config(
            {
                "quant_method": "fp8",
                "quant_algo": "FP8",
                "activation_scheme": "dynamic",
            }
        )

        class FakeFusedMoE:
            pass

        dense_method = object()
        fp8_moe_method = object()
        self.assertFalse(config.use_per_token_activation)
        with (
            patch("sglang.srt.layers.moe.fused_moe_triton.FusedMoE", FakeFusedMoE),
            patch(
                "sglang.srt.layers.quantization.fp8.Fp8LinearMethod",
                return_value=dense_method,
            ),
            patch(
                "sglang.srt.layers.quantization.fp8.Fp8MoEMethod",
                return_value=fp8_moe_method,
            ),
        ):
            self.assertIs(
                config.get_quant_method(
                    ReplicatedLinear.__new__(ReplicatedLinear),
                    "model.layers.0.self_attn.q_proj",
                ),
                dense_method,
            )
            self.assertIs(
                config.get_quant_method(FakeFusedMoE(), "model.layers.1.mlp.experts"),
                fp8_moe_method,
            )

    def test_online_fp8_source_weight_loader_dequantizes_before_nvfp4(self):
        config = ModelOptFp4Config.from_config(
            {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
            }
        )

        class FakeFusedMoE:
            pass

        with (
            patch("sglang.srt.layers.moe.fused_moe_triton.FusedMoE", FakeFusedMoE),
            patch.object(
                ModelOptNvFp4FusedMoEMethod,
                "__init__",
                lambda method, quant_config: setattr(
                    method, "quant_config", quant_config
                ),
            ),
        ):
            method = config.get_quant_method(
                FakeFusedMoE(), "model.layers.0.mlp.experts"
            )
        layer = SimpleNamespace(
            moe_runner_config=SimpleNamespace(is_gated=False),
            w2_weight_scale=object(),
            w2_weight_scale_2=object(),
        )
        param = SimpleNamespace(device=torch.device("cpu"))
        fp8_weight = torch.tensor([[1.0, -2.0] * 8], dtype=torch.float8_e4m3fn)
        fp8_scale = torch.tensor(0.5, dtype=torch.float32)
        fp4_weight = torch.empty(1, 8, dtype=torch.uint8)
        weight_scale = torch.empty(1, 1, dtype=torch.float8_e4m3fn)
        weight_scale_2 = torch.ones((), dtype=torch.float32)

        for scale_first in (False, True):
            with self.subTest(scale_first=scale_first):
                original_weight_loader = MagicMock()
                with patch.object(
                    ModelOptNvFp4OnlineFusedMoEMethod,
                    "_quantize_weight_nvfp4",
                    return_value=(fp4_weight, weight_scale, weight_scale_2),
                ) as quantize:
                    weight_loader = method.prepare_weight_loader(
                        layer, original_weight_loader
                    )
                    weight_loader(
                        param,
                        torch.tensor(0.25),
                        "model.layers.0.mlp.experts.0.down_proj.input_scale",
                        "w2",
                        None,
                    )
                    original_weight_loader.assert_not_called()
                    args = (
                        (
                            param,
                            fp8_scale,
                            "model.layers.0.mlp.experts.0.down_proj.weight_scale_inv",
                            "w2",
                            None,
                        ),
                        (
                            param,
                            fp8_weight,
                            "model.layers.0.mlp.experts.0.down_proj.weight",
                            "w2",
                            None,
                        ),
                    )
                    if not scale_first:
                        args = tuple(reversed(args))
                    for loader_args in args:
                        weight_loader(*loader_args)

                quantized_input = quantize.call_args.args[0]
                self.assertEqual(quantized_input.dtype, torch.bfloat16)
                torch.testing.assert_close(
                    quantized_input,
                    (fp8_weight.to(torch.float16) * fp8_scale).to(torch.bfloat16),
                )
                self.assertEqual(original_weight_loader.call_count, 3)


if __name__ == "__main__":
    unittest.main()
