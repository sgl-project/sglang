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

    @staticmethod
    def _make_apply_fixture(*, is_awq=False):
        method = ModelOptFp4LinearMethod.__new__(ModelOptFp4LinearMethod)
        method.quant_config = SimpleNamespace(is_awq=is_awq)
        method.quant_mode = "w4a4"

        layer = nn.Module()
        layer.params_dtype = torch.bfloat16
        layer.output_size_per_partition = 5
        layer.input_size_per_partition = 64
        layer.weight = torch.zeros((5, 32), dtype=torch.uint8)
        layer.weight_scale_interleaved = torch.zeros((5, 4), dtype=torch.float8_e4m3fn)
        layer.alpha = torch.ones(1, dtype=torch.float32)
        layer.input_scale_inv = torch.ones(1, dtype=torch.float32)
        layer.weights_padding_cols = 0
        if is_awq:
            layer.pre_quant_scale = torch.linspace(
                0.5, 1.5, layer.input_size_per_partition
            )
        return method, layer

    def test_w4a4_apply_preserves_leading_dimensions(self):
        method, layer = self._make_apply_fixture()
        quantized_inputs = []

        def fake_quantize(x, _scale):
            quantized_inputs.append(x.detach().clone())
            return (
                torch.zeros((x.shape[0], x.shape[1] // 2), dtype=torch.uint8),
                torch.zeros((x.shape[0], 1), dtype=torch.float8_e4m3fn),
            )

        def fake_gemm(x_fp4, _weight, *_args):
            return torch.zeros(
                (x_fp4.shape[0], layer.output_size_per_partition),
                dtype=torch.bfloat16,
            )

        inputs = {
            "2d": torch.randn(6, 64, dtype=torch.bfloat16),
            "3d": torch.randn(2, 3, 64, dtype=torch.bfloat16),
            "4d_noncontiguous": torch.randn(
                2, 4, 3, 64, dtype=torch.bfloat16
            ).transpose(1, 2),
        }
        bias = torch.arange(5, dtype=torch.bfloat16)

        with (
            patch(
                "sglang.srt.layers.quantization.modelopt_quant."
                "get_fp4_gemm_runner_backend",
                return_value=Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.fp4_quantize",
                side_effect=fake_quantize,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.fp4_gemm",
                side_effect=fake_gemm,
            ),
        ):
            for name, x in inputs.items():
                with self.subTest(name=name):
                    out = method.apply(layer, x, bias)

                    self.assertEqual(
                        tuple(quantized_inputs[-1].shape),
                        (x.numel() // x.shape[-1], x.shape[-1]),
                    )
                    self.assertEqual(
                        tuple(out.shape),
                        (*x.shape[:-1], layer.output_size_per_partition),
                    )
                    torch.testing.assert_close(out.reshape(-1, out.shape[-1])[0], bias)

    def test_w4a4_apply_flattens_before_awq_prescale(self):
        method, layer = self._make_apply_fixture(is_awq=True)
        x = torch.randn(2, 3, 64, dtype=torch.bfloat16)
        quantized_inputs = []

        def fake_quantize(input_, _scale):
            quantized_inputs.append(input_.detach().clone())
            return (
                torch.zeros((input_.shape[0], input_.shape[1] // 2), dtype=torch.uint8),
                torch.zeros((input_.shape[0], 1), dtype=torch.float8_e4m3fn),
            )

        with (
            patch(
                "sglang.srt.layers.quantization.modelopt_quant."
                "get_fp4_gemm_runner_backend",
                return_value=Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.fp4_quantize",
                side_effect=fake_quantize,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.fp4_gemm",
                return_value=torch.zeros((6, 5), dtype=torch.bfloat16),
            ),
        ):
            out = method.apply(layer, x)

        torch.testing.assert_close(
            quantized_inputs[0],
            x.reshape(-1, x.shape[-1]) * layer.pre_quant_scale,
        )
        self.assertEqual(tuple(out.shape), (2, 3, 5))

    def test_w4a4_prequantized_tuple_keeps_flattened_contract(self):
        method, layer = self._make_apply_fixture()
        layer._accepts_prequantized_fp4 = True
        x_fp4 = torch.zeros((6, 32), dtype=torch.uint8)
        x_scale = torch.zeros((6, 1), dtype=torch.float8_e4m3fn)

        with (
            patch(
                "sglang.srt.layers.quantization.modelopt_quant."
                "get_fp4_gemm_runner_backend",
                return_value=Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.fp4_quantize"
            ) as quantize,
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.fp4_gemm",
                return_value=torch.zeros((6, 5), dtype=torch.bfloat16),
            ),
        ):
            out = method.apply(layer, (x_fp4, x_scale))

        quantize.assert_not_called()
        self.assertEqual(tuple(out.shape), (6, 5))


if __name__ == "__main__":
    unittest.main()
