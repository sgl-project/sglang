import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from sglang.srt.environ import envs
from sglang.srt.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from sglang.srt.layers.parameter import PerTensorScaleParameter
from sglang.srt.layers.quantization import fp4_utils, modelopt_quant
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.runtime_context import override_platform
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

    def test_global_w4a16_preserves_flashinfer_preparation_and_apply_contract(self):
        # SM100 uses uint8; other supported architectures require FlashInfer's
        # int32 preparation. Global W4A16 must preserve either returned layout.
        for dtype in (torch.uint8, torch.int32):
            with self.subTest(dtype=dtype):
                flashinfer = ModuleType("flashinfer")
                weight = torch.ones((16, 8 if dtype == torch.uint8 else 2), dtype=dtype)
                scales = torch.ones((16, 1), dtype=torch.float8_e4m3fn)
                alpha = torch.tensor([5.0])
                prepare = Mock(return_value=(weight, scales, alpha))
                flashinfer.prepare_bf16_fp4_weights = prepare
                output = torch.ones(6, 16, dtype=torch.bfloat16)
                with (
                    envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(True),
                    override_platform(is_sm100=dtype == torch.uint8),
                    patch.object(
                        fp4_utils,
                        "FP4_GEMM_RUNNER_BACKEND",
                        Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL,
                    ),
                    patch.dict(sys.modules, {"flashinfer": flashinfer}),
                    patch.object(
                        modelopt_quant, "swizzle_blockscale", return_value=scales
                    ),
                    patch.object(
                        modelopt_quant, "fp4_gemm", return_value=output
                    ) as gemm,
                ):
                    config = ModelOptFp4Config(
                        is_checkpoint_nvfp4_serialized=True,
                        group_size=16,
                        use_per_token_activation=False,
                    )
                    method = ModelOptFp4LinearMethod(config)
                    layer = nn.Module()
                    method.create_weights(
                        layer,
                        input_size_per_partition=16,
                        output_partition_sizes=[16],
                        input_size=16,
                        output_size=16,
                        params_dtype=torch.bfloat16,
                        weight_loader=default_weight_loader,
                    )
                    layer.input_scale.data.fill_(2.0)
                    layer.weight_scale_2.data.fill_(3.0)
                    method.process_weights_after_loading(layer)

                    self.assertEqual(method.quant_mode, "w4a16")
                    self.assertEqual(prepare.call_args.kwargs, {"backend": "cute-dsl"})
                    self.assertIs(prepare.call_args.args[1], scales)
                    torch.testing.assert_close(
                        prepare.call_args.args[2], torch.tensor([3.0])
                    )
                    torch.testing.assert_close(layer.weight, weight)
                    torch.testing.assert_close(layer.alpha, alpha)
                    self.assertFalse(hasattr(layer, "weight_scale_mma"))
                    self.assertFalse(hasattr(layer, "weight_global_scale"))

                    x = torch.zeros(2, 3, 16, dtype=torch.bfloat16)
                    bias = torch.arange(16, dtype=torch.bfloat16)
                    actual = method.apply(layer, x, bias)
                    args = gemm.call_args.args
                    self.assertEqual(args[0].shape, (6, 16))
                    self.assertIs(args[1], layer.weight)
                    self.assertIs(args[3], layer.weight_scale_interleaved)
                    self.assertIs(args[4], layer.alpha)
                    self.assertEqual(args[7], "w4a16")
                    torch.testing.assert_close(actual, (output + bias).view(2, 3, 16))

    def test_dense_w4a16_phase_requires_shared_layout_architecture(self):
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            use_per_token_activation=False,
        )
        for phase in ("prefill", "decode"):
            with (
                self.subTest(phase=phase),
                envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override(phase),
                override_platform(is_sm100=False),
                patch.object(
                    fp4_utils,
                    "FP4_GEMM_RUNNER_BACKEND",
                    Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL,
                ),
                self.assertRaisesRegex(ValueError, "requires SM100-family GPUs"),
            ):
                ModelOptFp4LinearMethod(config)

    def test_dense_w4a16_phase_is_noop_on_other_backends(self):
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            use_per_token_activation=False,
        )
        with (
            envs.SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16.override("decode"),
            override_platform(is_sm100=False),
            patch.object(
                fp4_utils,
                "FP4_GEMM_RUNNER_BACKEND",
                Fp4GemmRunnerBackend.FLASHINFER_CUTLASS,
            ),
        ):
            self.assertEqual(ModelOptFp4LinearMethod(config).quant_mode, "w4a4")

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


if __name__ == "__main__":
    unittest.main()
