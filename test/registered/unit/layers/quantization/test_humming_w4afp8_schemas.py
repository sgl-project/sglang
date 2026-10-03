"""Unit coverage for Humming schemas and input-quantization exclusions."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.quantization import humming_utils  # noqa: E402
from sglang.srt.layers.quantization.humming import (  # noqa: E402
    HummingConfig,
    _StackedBlockFp8CheckpointWeightSchema,
    _W4AFp8CheckpointWeightSchema,
)

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestHummingInputExclusions(CustomTestCase):
    def test_each_ignore_alias(self):
        for alias in ("ignored_layers", "ignore", "modules_to_not_convert"):
            with self.subTest(alias=alias):
                config = {alias: ["layers.1.mlp"]}
                self.assertTrue(
                    humming_utils.humming_is_layer_skipped(config, "model.layers.1.mlp")
                )
                self.assertFalse(
                    humming_utils.humming_is_layer_skipped(config, "model.layers.2.mlp")
                )

    def test_first_present_alias_takes_precedence(self):
        aliases = ("ignored_layers", "ignore", "modules_to_not_convert")
        for first in range(len(aliases) - 1):
            config = {alias: [f"layers.{i}.mlp"] for i, alias in enumerate(aliases)}
            for alias in aliases[:first]:
                del config[alias]
            for i in range(len(aliases)):
                with self.subTest(first=aliases[first], layer=i):
                    self.assertEqual(
                        humming_utils.humming_is_layer_skipped(
                            config, f"model.layers.{i}.mlp"
                        ),
                        i == first,
                    )

    def test_empty_or_null_alias_takes_precedence(self):
        for alias in ("ignored_layers", "ignore"):
            for value in ([], None):
                with self.subTest(alias=alias, value=value):
                    config = {alias: value, "modules_to_not_convert": ["mlp"]}
                    self.assertFalse(
                        humming_utils.humming_is_layer_skipped(config, "model.mlp")
                    )

    def test_missing_input_config_and_missing_exclusions(self):
        self.assertTrue(humming_utils.humming_is_layer_skipped({}, "model.mlp"))
        self.assertFalse(
            humming_utils.humming_is_layer_skipped(
                {"a_dtype": "float8e4m3"}, "model.mlp"
            )
        )

    def test_lm_head_and_dynamic_exclusions(self):
        self.assertTrue(
            humming_utils.humming_is_layer_skipped({"ignore": []}, "model.lm_head")
        )
        config = {"dynamic": {r"-:^model\.layers\.1\.": {}, r"+:.*": {}}}
        self.assertTrue(
            humming_utils.humming_is_layer_skipped(config, "model.layers.1.mlp")
        )
        self.assertFalse(
            humming_utils.humming_is_layer_skipped(config, "model.layers.2.mlp")
        )

    def test_moe_input_schemas_respect_exclusions(self):
        weight_schema = SimpleNamespace(
            convert_humming=lambda **kwargs: (weight_schema, kwargs["tensors"])
        )
        for alias in ("ignored_layers", "ignore", "modules_to_not_convert"):
            config = {
                "a_dtype": "float8e4m3",
                "input_scale_group_size": 128,
                alias: ["layers.1.mlp"],
            }
            for layer_index in (1, 2):
                with self.subTest(alias=alias, layer=layer_index):
                    layer = torch.nn.Module()
                    layer.layer_name = f"model.layers.{layer_index}.mlp"
                    layer.hidden_size = 128
                    layer.intermediate_size_per_partition = 128
                    layer.num_local_experts = 1
                    layer.params_dtype = torch.bfloat16
                    layer.with_bias = False
                    layer.w13_weight = torch.nn.Parameter(torch.empty(1, 256, 128))
                    layer.w2_weight = torch.nn.Parameter(torch.empty(1, 128, 128))
                    with (
                        patch.object(
                            humming_utils.envs.SGLANG_HUMMING_INPUT_QUANT_CONFIG,
                            "get",
                            return_value=config,
                        ),
                        patch.object(
                            humming_utils.BaseWeightSchema,
                            "from_config",
                            return_value=weight_schema,
                        ),
                        patch.object(
                            humming_utils,
                            "get_moe_a2a_backend",
                            return_value=SimpleNamespace(is_deepep=lambda: False),
                        ),
                        patch.object(humming_utils.HummingMethod, "prepare_layer_meta"),
                        patch.object(
                            humming_utils.HummingMethod, "transform_humming_layer"
                        ),
                    ):
                        humming_utils.prepare_humming_moe_layer(layer, {})

                    for sublayer in ("w13", "w2"):
                        schema = layer.input_schemas[sublayer]
                        if layer_index == 1:
                            self.assertIsNone(schema.a_dtype)
                            self.assertEqual(schema.input_scale_group_size, 0)
                        else:
                            self.assertEqual(str(schema.a_dtype), "float8e4m3")
                            self.assertEqual(schema.input_scale_group_size, 128)


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


if __name__ == "__main__":
    unittest.main()
