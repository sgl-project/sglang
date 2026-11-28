"""Unit coverage for Humming W4AFP8 packing and stacked block-FP8 scale shapes."""

from __future__ import annotations

import math
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.quantization.humming import (  # noqa: E402
    HummingConfig,
    HummingLayerQuantizationConfig,
    HummingLinearMethod,
    HummingMoEMethod,
    _convert_humming_tensors,
    _StackedBlockFp8CheckpointWeightSchema,
    _W4AFp8CheckpointWeightSchema,
)

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestW4AFp8CheckpointSchema(CustomTestCase):
    def test_packed_tensor_shapes(self):
        schema = _W4AFp8CheckpointWeightSchema(group_size=128)
        attrs = schema.get_tensors_attrs(
            shape_n=4096,
            shape_k=6144,
            param_dtype=torch.bfloat16,
            num_experts=8,
        )
        # int8 storage packs two int4 values per byte along K.
        self.assertEqual(attrs["weight"]["shape"], (8, 4096, 6144 // 2))
        self.assertEqual(attrs["weight"]["dtype"], torch.int8)
        self.assertEqual(attrs["weight_scale_inv"]["shape"], (8, 4096, 6144 // 128))
        self.assertEqual(attrs["weight_scale_inv"]["dtype"], torch.bfloat16)

    def test_group_size_validation(self):
        for bad in (None, 0, -128, True, "128", 12.8):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _W4AFp8CheckpointWeightSchema(group_size=bad)

    def test_shape_k_must_preserve_groups(self):
        schema = _W4AFp8CheckpointWeightSchema(group_size=128)
        with self.assertRaises(ValueError):
            schema.get_tensors_attrs(
                shape_n=64, shape_k=192, param_dtype=torch.bfloat16
            )

    def test_config_carries_declared_group_size(self):
        """HummingConfig must not silently fall back to the default group size.

        W4AFp8Config.from_config() ignores the checkpoint's group_size; the
        Humming wrapper is responsible for carrying the declared value into
        the per-layer weight config.
        """
        config = HummingConfig(
            {
                "quant_method": "w4afp8",
                "group_size": 64,
                "weight_block_size": [128, 128],
            }
        )
        weight_config, _ = config.get_checkpoint_configs_for_layer("moe")
        self.assertEqual(weight_config["group_size"], 64)

        default_config = HummingConfig(
            {"quant_method": "w4afp8", "weight_block_size": [128, 128]}
        )
        weight_config, _ = default_config.get_checkpoint_configs_for_layer("moe")
        self.assertEqual(weight_config["group_size"], 128)

        # An explicit null must be carried (and later rejected by schema
        # validation), not silently replaced with the default.
        null_config = HummingConfig(
            {
                "quant_method": "w4afp8",
                "group_size": None,
                "weight_block_size": [128, 128],
            }
        )
        weight_config, _ = null_config.get_checkpoint_configs_for_layer("moe")
        self.assertIsNone(weight_config["group_size"])
        with self.assertRaises(ValueError):
            _W4AFp8CheckpointWeightSchema(group_size=weight_config["group_size"])

    def test_config_carries_declared_weight_block_size(self):
        """Checkpoint-declared block geometry must survive config translation.

        Scale shapes and the MLA post-processing both read this value; losing a
        non-default declaration quantizes with the wrong block layout.
        """
        config = HummingConfig(
            {
                "quant_method": "w4afp8",
                "weight_block_size": [64, 64],
            }
        )
        self.assertEqual(config.weight_block_size, [64, 64])
        _, fp8_config = config.get_checkpoint_configs_for_layer("moe")
        self.assertEqual(fp8_config["weight_block_size"], [64, 64])

        default_config = HummingConfig({"quant_method": "w4afp8"})
        self.assertEqual(default_config.weight_block_size, [128, 128])


class TestStackedBlockFp8Schema(CustomTestCase):
    @staticmethod
    def _make_schema(weight_block_size=(128, 128)):
        base = SimpleNamespace(
            quant_method="fp8",
            weight_block_size=list(weight_block_size),
            weight_scale_key="weight_scale_inv",
            get_tensors_attrs=lambda **kwargs: {
                "weight": {
                    "shape": (kwargs["shape_n"], kwargs["shape_k"]),
                    "dtype": torch.float8_e4m3fn,
                    "extra_attrs": {},
                },
                "weight_scale_inv": {
                    "shape": (),
                    "dtype": torch.float32,
                    "extra_attrs": {},
                },
            },
        )
        return _StackedBlockFp8CheckpointWeightSchema(base)

    def test_block_size_validation(self):
        for bad in ([128], [128, 0], [128, -1], [128, True], [128.0, 128]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self._make_schema(bad)

    def test_stacked_scale_rows_use_per_stack_ceil(self):
        """Unequal output partitions each round up to their own block count.

        A gate of 96 rows and an up of 32 rows both occupy one 128-row scale
        block; the stacked checkpoint therefore stores 2 scale rows, not
        ceil((96+32)/128) == 1. Reading the naive shape would misalign every
        scale after the first stack.
        """
        schema = self._make_schema()
        attrs = schema.get_stacked_tensors_attrs(
            shape_n_stacks=[96, 32],
            shape_k=256,
            param_dtype=torch.bfloat16,
        )
        self.assertEqual(attrs["weight_scale_inv"]["shape"], (2, 2))

        # Equal partitions that align with the block size collapse to the
        # naive shape, so the distinction only shows up on unequal stacks.
        aligned = schema.get_stacked_tensors_attrs(
            shape_n_stacks=[128, 128],
            shape_k=256,
            param_dtype=torch.bfloat16,
        )
        self.assertEqual(aligned["weight_scale_inv"]["shape"], (2, 2))
        self.assertEqual(
            aligned["weight_scale_inv"]["shape"][0],
            sum(math.ceil(n / 128) for n in [128, 128]),
        )

    def test_single_tensor_scale_shape_uses_ceil(self):
        schema = self._make_schema()
        attrs = schema.get_tensors_attrs(
            shape_n=96, shape_k=384, param_dtype=torch.bfloat16, num_experts=4
        )
        self.assertEqual(attrs["weight_scale_inv"]["shape"], (4, 1, 3))


class TestHummingSchemaConversion(CustomTestCase):
    def setUp(self):
        # These cases exercise actual schema conversions on CPU tensors. Only
        # the CUDA device context is suppressed; no quantization kernels run.
        self.cuda_context = patch(
            "torch.cuda.device", side_effect=lambda _: nullcontext()
        )
        self.cuda_context.start()
        self.addCleanup(self.cuda_context.stop)

    def test_static_fp8_input_and_distinct_merged_weight_scales(self):
        from humming.schema.fp8 import Fp8InputSchema, Fp8WeightSchema

        tensors = {
            "weight": torch.ones((4, 8), dtype=torch.float8_e4m3fn),
            "weight_scale": torch.tensor([3.0, 5.0]),
            "input_scale": torch.tensor([0.5, 2.0]),
        }
        weight_schema, input_schema, converted = _convert_humming_tensors(
            Fp8WeightSchema(),
            Fp8InputSchema(activation_scheme="static"),
            tensors,
            shape_n_stacks=[2, 2],
            shape_k_stacks=[8],
            param_dtype=torch.bfloat16,
        )
        self.assertEqual(weight_schema.weight_scale_type.value, "channel")
        self.assertEqual(input_schema.input_quant_mode.value, "static_tensor")
        torch.testing.assert_close(converted["input_scale"], torch.tensor([2.0]))
        expected_scales = torch.tensor(
            [[3.0], [3.0], [5.0], [5.0]], dtype=torch.bfloat16
        )
        torch.testing.assert_close(converted["weight_scale"], expected_scales)
        self.assertEqual(converted["weight"].shape, (4, 2))

    def test_modelopt_nvfp4_preserves_secondary_input_scale(self):
        from humming.schema.modelopt import (
            ModeloptNvfp4InputSchema,
            ModeloptNvfp4WeightSchema,
        )

        tensors = {
            "weight": torch.zeros((4, 16), dtype=torch.uint8),
            "weight_scale": torch.ones((4, 2), dtype=torch.float8_e4m3fn),
            "weight_scale_2": torch.tensor([2.0, 4.0]),
            "input_scale": torch.tensor([0.5, 0.75]),
        }
        weight_schema, input_schema, converted = _convert_humming_tensors(
            ModeloptNvfp4WeightSchema(),
            ModeloptNvfp4InputSchema(quant_algo="nvfp4"),
            tensors,
            shape_n_stacks=[2, 2],
            shape_k_stacks=[32],
            param_dtype=torch.bfloat16,
        )
        self.assertEqual(weight_schema.weight_scale_2_type.value, "channel")
        self.assertEqual(
            input_schema.input_quant_mode.value, "static_tensor_dynamic_group"
        )
        self.assertNotIn("input_scale", converted)
        torch.testing.assert_close(converted["input_scale_2"], torch.tensor([0.75]))
        torch.testing.assert_close(
            converted["weight_scale_2"],
            torch.tensor([2.0, 2.0, 4.0, 4.0], dtype=torch.bfloat16),
        )

    def test_native_humming_scales_are_merged_for_each_expert(self):
        from humming.schema import HummingInputSchema, HummingWeightSchema

        tensors = {
            "weight": torch.zeros((2, 4, 2), dtype=torch.int32),
            "weight_scale": torch.tensor([[2.0, 3.0], [5.0, 7.0]]),
            "input_scale": torch.tensor([[0.5, 1.0], [2.0, 0.75]]),
        }
        weight_schema, _, converted = _convert_humming_tensors(
            HummingWeightSchema(b_dtype="float8e4m3", weight_scale_type="tensor"),
            HummingInputSchema(a_dtype="float8e4m3", input_quant_mode="static_tensor"),
            tensors,
            shape_n_stacks=[2, 2],
            shape_k_stacks=[8],
            param_dtype=torch.bfloat16,
            num_experts=2,
        )
        self.assertEqual(weight_schema.weight_scale_type.value, "channel")
        torch.testing.assert_close(converted["input_scale"], torch.tensor([2.0]))
        expected_scales = torch.tensor(
            [[[2.0], [2.0], [3.0], [3.0]], [[5.0], [5.0], [7.0], [7.0]]],
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(converted["weight_scale"], expected_scales)

    def test_dynamic_override_removes_checkpoint_static_scale(self):
        from humming.schema import HummingInputSchema
        from humming.schema.fp8 import Fp8InputSchema, Fp8WeightSchema

        _, input_schema, converted = _convert_humming_tensors(
            Fp8WeightSchema(),
            Fp8InputSchema(activation_scheme="static"),
            {
                "weight": torch.ones((2, 8), dtype=torch.float8_e4m3fn),
                "weight_scale": torch.tensor([3.0]),
                "input_scale": torch.tensor([0.5]),
            },
            shape_n_stacks=[2],
            shape_k_stacks=[8],
            param_dtype=torch.bfloat16,
            force_input_schema=HummingInputSchema(
                a_dtype="float8e4m3", input_quant_mode="dynamic_token"
            ),
        )
        self.assertEqual(input_schema.input_quant_mode.value, "dynamic_token")
        self.assertNotIn("input_scale", converted)
        self.assertNotIn("input_scale_2", converted)

    def test_moe_allocates_input_and_weight_scales_for_gate_and_up(self):
        from humming.schema import HummingInputSchema, HummingWeightSchema
        from humming.schema.fp8 import Fp8InputSchema, Fp8WeightSchema

        for weight_schema, input_schema in (
            (Fp8WeightSchema(), Fp8InputSchema(activation_scheme="static")),
            (
                HummingWeightSchema(b_dtype="float8e4m3", weight_scale_type="tensor"),
                HummingInputSchema(
                    a_dtype="float8e4m3", input_quant_mode="static_tensor"
                ),
            ),
        ):
            with self.subTest(schema=type(weight_schema).__name__):
                method = HummingMoEMethod(
                    HummingLayerQuantizationConfig(
                        weight_schema=weight_schema, input_schema=input_schema
                    )
                )
                layer = torch.nn.Module()
                method.create_weights(
                    layer,
                    num_experts=3,
                    hidden_size=128,
                    intermediate_size_per_partition=64,
                    params_dtype=torch.bfloat16,
                )
                self.assertEqual(layer.w13_weight_scale.shape, (3, 2))
                self.assertEqual(layer.w13_input_scale.shape, (3, 2))
                self.assertEqual(layer.w2_input_scale.shape, (3,))
                self.assertEqual(layer.w13_input_scale.quant_method, "tensor")
                self.assertTrue(layer.w13_input_scale.is_humming_input_scale)

    def test_online_tensor_scale_uses_weight_scale_name(self):
        from humming.schema import HummingWeightSchema

        schema = HummingWeightSchema(b_dtype="float4e2m1", weight_scale_type="tensor")
        layer = torch.nn.Module()
        layer.param_dtype = torch.bfloat16
        attrs = schema.get_tensors_attrs(2, 32, layer.param_dtype)
        for name, attr in attrs.items():
            param = torch.nn.Parameter(
                torch.zeros(attr["shape"], dtype=attr["dtype"]), requires_grad=False
            )
            param.param_name = name
            param.weight_loader = lambda target, loaded, shard: target.data.copy_(
                loaded
            )
            setattr(layer, name, param)
        method = HummingLinearMethod(
            HummingLayerQuantizationConfig(weight_schema=schema, is_online_quant=True)
        )
        load = method.prepare_weight_loader(layer, None)
        packed_weight = torch.full((2, 4), 17, dtype=torch.int32)
        tensor_scale = torch.tensor([2.0])
        with patch(
            "humming.schema.humming.quantize_weight",
            autospec=True,
            return_value=(packed_weight, None, None, tensor_scale),
        ):
            load(layer.weight, torch.ones((2, 32), dtype=torch.bfloat16))
        self.assertFalse(hasattr(layer, "global_scale"))
        torch.testing.assert_close(layer.weight_scale, tensor_scale)
        torch.testing.assert_close(layer.weight, packed_weight)

    def test_prepare_replacements_are_registered_before_transform(self):
        from humming.schema import HummingWeightSchema
        from humming.schema.fp8 import Fp8InputSchema

        from sglang.srt.layers.quantization import humming

        layer = torch.nn.Module()
        layer.with_bias = False
        method = HummingLinearMethod(
            HummingLayerQuantizationConfig(
                weight_schema=HummingWeightSchema(b_dtype="float4e2m1"),
                input_schema=Fp8InputSchema(activation_scheme="static"),
            )
        )
        method.create_weights(
            layer,
            input_size_per_partition=8,
            output_partition_sizes=[2],
            input_size=8,
            output_size=2,
            params_dtype=torch.bfloat16,
        )
        layer.weight.data.fill_(1)
        layer.weight_scale.data.fill_(2)
        layer.input_scale.data.fill_(0.5)
        replacement = torch.full((2, 2), 17, dtype=torch.int32)

        def prepare(**kwargs):
            # E0Mx preparation replaces packed tensor entries in-place.
            kwargs["tensors"]["weight"] = replacement

        def transform(transformed_layer):
            torch.testing.assert_close(transformed_layer.weight, replacement)
            torch.testing.assert_close(
                transformed_layer.input_scale, torch.tensor([0.5])
            )

        with (
            patch.object(
                humming.HummingMethod, "prepare_layer_meta", side_effect=prepare
            ),
            patch.object(
                humming.HummingMethod, "transform_humming_layer", side_effect=transform
            ),
        ):
            method.process_weights_after_loading(layer)


class TestHummingCheckpointConfig(CustomTestCase):
    def test_static_input_override_requires_checkpoint_calibration(self):
        from sglang.srt.environ import envs

        for quant_mode, extra_config in (
            ("static_tensor", {"dtype": "float8e4m3"}),
            ("static_tensor_dynamic_group", {"dtype": "float4e2m1", "group_size": 16}),
        ):
            with (
                self.subTest(quant_mode=quant_mode),
                patch.object(
                    envs.SGLANG_HUMMING_ONLINE_QUANT_CONFIG,
                    "get",
                    return_value={"dtype": "float4e2m1", "group_size": 32},
                ),
                patch.object(
                    envs.SGLANG_HUMMING_INPUT_QUANT_CONFIG,
                    "get",
                    return_value={"quant_mode": quant_mode, **extra_config},
                ),
            ):
                config = HummingConfig()
                with self.assertRaisesRegex(
                    ValueError, "calibrated checkpoint input scales"
                ):
                    config.get_quant_config_for_layer("model.layers.0.proj", "linear")

    def test_modelopt_without_config_groups(self):
        config = HummingConfig({"quant_method": "modelopt", "quant_algo": "NVFP4"})
        layer_config = config.get_quant_config_for_layer(
            "model.layers.0.proj", "linear"
        )
        self.assertEqual(layer_config.weight_schema.quant_algo, "NVFP4")
        self.assertEqual(layer_config.input_schema.get_activation_bits(), 4)


if __name__ == "__main__":
    unittest.main()
