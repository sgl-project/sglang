import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.parameter import PerTensorScaleParameter
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestModelOptNvfp4(CustomTestCase):
    def _check_visual_alias_mapping(self, *, excluded):
        mapper = Qwen3VLForConditionalGeneration.hf_to_sglang_mapper
        original_mappings = (
            dict(mapper.orig_to_new_substr),
            dict(mapper.orig_to_new_prefix),
            dict(mapper.orig_to_new_suffix),
        )
        runtime_prefix = "model.visual.blocks.0.attn.qkv_proj"
        mapped_prefix = "visual.blocks.0.attn.qkv_proj"
        for config_cls, quant_algo in (
            (ModelOptFp4Config, "NVFP4"),
            (ModelOptMixedPrecisionConfig, "MIXED_PRECISION"),
        ):
            for visual_prefix in ("model.visual", "visual"):
                for projection in ("qkv", "qkv_proj"):
                    with self.subTest(
                        config=config_cls.__name__,
                        visual_prefix=visual_prefix,
                        projection=projection,
                    ):
                        source_prefix = f"{visual_prefix}.blocks.0.attn.{projection}"
                        layer_info = {"quant_algo": "NVFP4", "group_size": 16}
                        cfg = config_cls.from_config(
                            {
                                "quant_algo": quant_algo,
                                "group_size": 16,
                                "ignore": [source_prefix] if excluded else [],
                                "packed_modules_mapping": {},
                                "quantized_layers": {source_prefix: layer_info},
                            }
                        )
                        cfg.apply_weight_name_mapper(mapper)
                        self.assertEqual(
                            cfg.quantized_layers, {mapped_prefix: layer_info}
                        )
                        self.assertEqual(
                            cfg.exclude_modules, [mapped_prefix] if excluded else []
                        )
                        self.assertEqual(
                            cfg.is_layer_excluded(runtime_prefix), excluded
                        )
                        if isinstance(cfg, ModelOptMixedPrecisionConfig):
                            self.assertEqual(
                                cfg.resolve_quant_algo(runtime_prefix), "NVFP4"
                            )
                        layer = ReplicatedLinear(
                            16,
                            16,
                            bias=False,
                            params_dtype=torch.bfloat16,
                            quant_config=cfg,
                            prefix=runtime_prefix,
                        )
                        self.assertIsInstance(
                            layer.quant_method,
                            UnquantizedLinearMethod
                            if excluded
                            else ModelOptFp4LinearMethod,
                        )
                        self.assertEqual(
                            layer.weight.dtype,
                            torch.bfloat16 if excluded else torch.uint8,
                        )
                        self.assertEqual(
                            layer.weight.shape, (16, 16 if excluded else 8)
                        )
                        self.assertEqual(
                            original_mappings,
                            (
                                mapper.orig_to_new_substr,
                                mapper.orig_to_new_prefix,
                                mapper.orig_to_new_suffix,
                            ),
                        )

    def test_visual_metadata_mapping_preserves_checkpoint_and_runtime_aliases(self):
        self._check_visual_alias_mapping(excluded=False)

    def test_visual_exclusion_mapping_preserves_checkpoint_and_runtime_aliases(self):
        self._check_visual_alias_mapping(excluded=True)

    def test_serialized_fp4_preserves_flat_and_nested_layer_metadata(self):
        layer_map = {"model.visual.blocks.0.mlp.linear_fc1": {"quant_algo": "NVFP4"}}
        for config in (
            {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "ignore": [],
                "quantized_layers": layer_map,
            },
            {
                "quantization": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                    "exclude_modules": [],
                    "quantized_layers": layer_map,
                }
            },
        ):
            parsed = ModelOptFp4Config.from_config(config)
            self.assertEqual(parsed.quantized_layers, layer_map)

    def test_visual_exclusions_match_checkpoint_and_runtime_qkv_names(self):
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=["visual.blocks.0.attn.qkv"],
        )
        self.assertTrue(cfg.is_layer_excluded("model.visual.blocks.0.attn.qkv_proj"))

    def test_uniform_fp4_partial_auxiliary_map_keeps_unlisted_layers_bf16(self):
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            packed_modules_mapping={},
            quantized_layers={
                "model.visual.blocks.0.mlp.linear_fc1": {"quant_algo": "NVFP4"},
                "mtp.fc": {"quant_algo": "NVFP4"},
            },
        )
        for prefix, quantized in (
            ("visual.blocks.0.mlp.linear_fc1", True),
            ("model.visual.blocks.0.mlp.linear_fc2", False),
            ("mtp.fc", True),
            ("mtp.layers.0.self_attn.o_proj", False),
            ("model.layers.0.self_attn.o_proj", True),
        ):
            with self.subTest(prefix=prefix):
                layer = ReplicatedLinear(
                    16,
                    16,
                    bias=False,
                    params_dtype=torch.bfloat16,
                    quant_config=cfg,
                    prefix=prefix,
                )
                self.assertIsInstance(
                    layer.quant_method,
                    ModelOptFp4LinearMethod if quantized else UnquantizedLinearMethod,
                )
                self.assertEqual(
                    layer.weight.dtype, torch.uint8 if quantized else torch.bfloat16
                )
                self.assertEqual(layer.weight.shape, (16, 8 if quantized else 16))

    def test_uniform_fp4_auxiliary_exclusion_overrides_layer_map(self):
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            packed_modules_mapping={},
            exclude_modules=["visual.blocks.0.attn.qkv"],
            quantized_layers={
                "model.visual.blocks.0.attn.qkv": {"quant_algo": "NVFP4"}
            },
        )
        layer = ReplicatedLinear(
            16,
            16,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=cfg,
            prefix="model.visual.blocks.0.attn.qkv_proj",
        )
        self.assertIsInstance(layer.quant_method, UnquantizedLinearMethod)
        self.assertEqual(layer.weight.dtype, torch.bfloat16)
        self.assertEqual(layer.weight.shape, (16, 16))

    def test_uniform_fp4_auxiliary_fused_shards_require_complete_metadata(self):
        prefix = "mtp.layers.0.self_attn"
        for shards, quantized in (((), False), (("q", "k", "v"), True)):
            cfg = ModelOptFp4Config(
                is_checkpoint_nvfp4_serialized=True,
                group_size=16,
                packed_modules_mapping={"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
                quantized_layers={
                    "mtp.fc": {"quant_algo": "NVFP4"},
                    **{
                        f"{prefix}.{shard}_proj": {"quant_algo": "NVFP4"}
                        for shard in shards
                    },
                },
            )
            layer = ReplicatedLinear(
                16,
                16,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=cfg,
                prefix=f"{prefix}.qkv_proj",
            )
            self.assertIsInstance(
                layer.quant_method,
                ModelOptFp4LinearMethod if quantized else UnquantizedLinearMethod,
            )

        cfg.quantized_layers.pop(f"{prefix}.v_proj")
        with self.assertRaisesRegex(ValueError, "Partial quantized_layers metadata"):
            ReplicatedLinear(
                16, 16, bias=False, quant_config=cfg, prefix=f"{prefix}.qkv_proj"
            )

    def test_uniform_fp4_auxiliary_moe_rejects_per_expert_metadata(self):
        """One expert projection cannot opt unlisted fused expert buffers into FP4."""
        # Only the config dispatch is needed; runtime/communication setup is
        # outside this test's boundary.
        layer = object.__new__(FusedMoE)
        nn.Module.__init__(layer)
        prefix = "mtp.layers.0.mlp.experts"
        cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            packed_modules_mapping={},
            quantized_layers={f"{prefix}.0.gate_proj": {"quant_algo": "NVFP4"}},
        )
        with self.assertRaisesRegex(ValueError, "Per-expert quantized_layers metadata"):
            cfg.get_quant_method(layer, prefix)

        # A descendant exclusion must not bypass the whitelist and quantize
        # the entire otherwise-unlisted MoE.
        cfg.quantized_layers = {"mtp.fc": {"quant_algo": "NVFP4"}}
        cfg.exclude_modules = [f"{prefix}.0.gate_proj"]
        self.assertIsNone(cfg.get_quant_method(layer, prefix))

    def test_mixed_precision_resolves_visual_aliases(self):
        self.assertEqual(
            ModelOptMixedPrecisionConfig.from_config(
                {
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {
                        "model.visual.blocks.0.attn.qkv": {
                            "quant_algo": "NVFP4",
                            "group_size": 16,
                        }
                    },
                }
            ).resolve_quant_algo("visual.blocks.0.attn.qkv_proj"),
            "NVFP4",
        )

    def test_w4a4_restores_noncontiguous_vision_activation_shape(self):
        """Vision rows retain their order when the matrix kernel flattens them."""
        method = object.__new__(ModelOptFp4LinearMethod)
        method.quant_mode = "w4a4"
        method.quant_config = SimpleNamespace(is_awq=False)
        layer = SimpleNamespace(
            weight=torch.zeros(4, 8, dtype=torch.uint8),
            weight_scale_interleaved=torch.zeros(4, 1, dtype=torch.float8_e4m3fn),
            alpha=torch.ones(1, dtype=torch.float32),
            input_scale_inv=torch.ones(1),
            output_size_per_partition=4,
        )
        x = torch.arange(96, dtype=torch.float32).reshape(2, 3, 16).transpose(0, 1)
        bias = torch.arange(4, dtype=torch.float32)
        matrix_inputs = []

        def quantize(matrix, scale):
            self.assertEqual(matrix.ndim, 2)
            matrix_inputs.append(matrix)
            return (
                torch.zeros(matrix.shape[0], 8, dtype=torch.uint8),
                torch.zeros(matrix.shape[0], 1, dtype=torch.float8_e4m3fn),
            )

        # Replace only accelerator kernels; the production apply path performs
        # flattening, output slicing, bias addition and shape restoration.
        with (
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.get_fp4_gemm_runner_backend",
                return_value=SimpleNamespace(is_marlin=lambda: False),
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.fp4_quantize",
                side_effect=quantize,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.fp4_gemm",
                side_effect=lambda *args: matrix_inputs[-1][:, :4].clone(),
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.enable_flashinfer_fp4_gemm",
                False,
            ),
        ):
            output = method.apply(layer, x, bias)

        torch.testing.assert_close(output, x[..., :4] + bias)
        self.assertEqual(output.shape, (3, 2, 4))

    def test_only_interleaving_backends_support_the_fusion(self):
        for backend, expected in (
            (Fp4GemmRunnerBackend.FLASHINFER_CUTEDSL, True),
            (Fp4GemmRunnerBackend.FLASHINFER_CUTLASS, True),
            (Fp4GemmRunnerBackend.FLASHINFER_CUDNN, True),
            (Fp4GemmRunnerBackend.FLASHINFER_TRTLLM, False),
            (Fp4GemmRunnerBackend.MARLIN, False),
        ):
            with self.subTest(backend=backend):
                self.assertEqual(backend.supports_swiglu_fusion(), expected)

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


if __name__ == "__main__":
    unittest.main()
