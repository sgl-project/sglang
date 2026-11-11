"""CPU coverage for Humming MoE schema conversion and dispatch contracts."""

import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from humming import dtypes  # noqa: E402
from humming.config import InputQuantizationMode  # noqa: E402
from humming.schema import HummingWeightSchema  # noqa: E402

from sglang.srt.layers.quantization import humming_utils  # noqa: E402

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestHummingMoePreparation(CustomTestCase):
    @staticmethod
    def _make_layer(scale_name="input_scale"):
        layer = torch.nn.Module()
        layer.layer_name = "model.layers.0.mlp.experts"
        layer.hidden_size = 256
        layer.intermediate_size_per_partition = 128
        layer.num_local_experts = 2
        layer.params_dtype = torch.bfloat16
        layer.with_bias = False
        for name, shape_n, shape_k in (("w13", 256, 256), ("w2", 256, 128)):
            tensors = {
                "weight": torch.zeros((2, shape_n, shape_k // 2), dtype=torch.uint8),
                "weight_scale": torch.full(
                    (2, shape_n, shape_k // 32), 127, dtype=torch.uint8
                ),
                scale_name: torch.tensor(
                    [[1.0, 2.0], [3.0, 4.0]] if name == "w13" else [[5.0], [6.0]]
                ),
            }
            for key, tensor in tensors.items():
                layer.register_parameter(
                    f"{name}_{key}", torch.nn.Parameter(tensor, requires_grad=False)
                )
        return layer

    def _prepare(self, layer, config, *, replace_weight=False, fp8_dispatch=False):
        transformed_tensors = {}

        def prepare_metadata(*, tensors, sublayer_name, **kwargs):
            # Humming's E0Mx -> integer conversion replaces dict entries. The
            # adapter must register these replacements before transformation.
            if replace_weight:
                tensors["weight"] = torch.full_like(tensors["weight"], 17)

        def transform_layer(layer, sublayer_name):
            transformed_tensors[sublayer_name] = {
                key.removeprefix(sublayer_name + "_"): value.clone()
                for key, value in layer.state_dict().items()
                if key.startswith(sublayer_name + "_")
            }

        with (
            patch.object(
                humming_utils.envs.SGLANG_HUMMING_INPUT_QUANT_CONFIG,
                "get",
                return_value=config,
            ),
            patch.object(
                humming_utils,
                "configure_humming_deepep_dispatch",
                return_value=fp8_dispatch,
            ),
            patch("torch.cuda.device", side_effect=lambda _: nullcontext()),
            patch(
                "humming.schema.humming.current_device",
                SimpleNamespace(sm_version=90),
            ),
            patch.object(
                humming_utils.HummingMethod,
                "prepare_layer_meta",
                side_effect=prepare_metadata,
            ),
            patch.object(
                humming_utils.HummingMethod,
                "transform_humming_layer",
                side_effect=transform_layer,
            ),
        ):
            humming_utils.prepare_humming_moe_layer(layer, {"quant_method": "mxfp4"})
        return transformed_tensors

    def test_preserves_normalized_static_scales_for_each_sublayer(self):
        for quant_mode, scale_name in (
            ("static_tensor", "input_scale"),
            ("static_tensor_dynamic_group", "input_scale_2"),
        ):
            with self.subTest(quant_mode=quant_mode):
                layer = self._make_layer(scale_name)
                config = {"dtype": "float8e4m3", "quant_mode": quant_mode}
                if scale_name == "input_scale_2":
                    config.update(
                        dtype="float4e2m1", group_size=32, scale_dtype="float8e8m0"
                    )
                tensors = self._prepare(layer, config)
                for name, expected in (("w13", 4.0), ("w2", 6.0)):
                    torch.testing.assert_close(
                        tensors[name][scale_name], torch.tensor([expected])
                    )
                    self.assertEqual(tensors[name]["weight"].dtype, torch.int32)
                    self.assertEqual(
                        tensors[name]["weight_scale"].dtype, torch.float8_e8m0fnu
                    )
                self.assertIsNot(layer.input_schemas["w13"], layer.input_schemas["w2"])

    def test_registers_weight_replaced_during_metadata_preparation(self):
        layer = self._make_layer()
        tensors = self._prepare(
            layer, {"dtype": "int8", "group_size": 128}, replace_weight=True
        )
        for name in ("w13", "w2"):
            self.assertTrue(torch.all(tensors[name]["weight"] == 17).item())
            self.assertNotIn("input_scale", tensors[name])

    def test_fp8_dispatch_replaces_static_input_schema_and_scales(self):
        layer = self._make_layer()
        tensors = self._prepare(
            layer,
            {"dtype": "float8e4m3", "quant_mode": "static_tensor"},
            fp8_dispatch=True,
        )
        for name in ("w13", "w2"):
            schema = layer.input_schemas[name]
            self.assertEqual(
                schema.input_quant_mode, InputQuantizationMode.DynamicGroup
            )
            self.assertEqual(schema.input_scale_dtype, dtypes.float32)
            self.assertEqual(schema.input_scale_group_size, 128)
            self.assertNotIn("input_scale", tensors[name])
            self.assertNotIn("input_scale_2", tensors[name])

    def test_input_quantization_honors_ignore_aliases(self):
        for key in ("ignored_layers", "ignore", "modules_to_not_convert"):
            with self.subTest(key=key):
                self.assertTrue(
                    humming_utils.humming_is_layer_skipped(
                        {key: ["experts"]}, "mlp.experts"
                    )
                )

    def test_first_present_ignore_alias_takes_precedence(self):
        for ignored_layers in ([], None):
            with self.subTest(ignored_layers=ignored_layers):
                config = {"ignored_layers": ignored_layers, "ignore": ["experts"]}
                self.assertFalse(
                    humming_utils.humming_is_layer_skipped(config, "mlp.experts")
                )


class TestHummingDeepEpSchema(CustomTestCase):
    def test_rejects_unaligned_input_groups(self):
        with self.assertRaisesRegex(ValueError, "divisible by 128"):
            humming_utils.make_humming_deepep_input_schema("w2", 192)

    def test_rejects_incompatible_weight_schema(self):
        weight_schema = HummingWeightSchema(
            b_dtype="float4e2m1",
            bs_dtype="float8e8m0",
            weight_scale_group_size=32,
        )
        with patch.object(
            humming_utils.HummingInputSchema, "is_compatible_with", return_value=False
        ) as is_compatible_with:
            with self.assertRaisesRegex(ValueError, "bf16"):
                humming_utils.make_humming_deepep_input_schema(
                    "w13",
                    256,
                    weight_schema=weight_schema,
                    param_dtype=torch.bfloat16,
                )
        is_compatible_with.assert_called_once_with(weight_schema, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
