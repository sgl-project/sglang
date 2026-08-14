"""CPU contract tests for ModelSlim MXFP MoE and FP8 KV scale loading."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.npu.moe import finalize_routing
from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    MXFP_E8M0_NOT_LOADED as MXFP8_E8M0_NOT_LOADED,
    NPUMXFP8LinearMethod,
)
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUW4A4MXFP4MoEMethod,
    NPUW4A8MXFPMoEMethod,
    _normalize_mxfp_input_scale,
    _pack_mxfp_weight_scale,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.moe.utils import DispatcherOutputDtype
from sglang.srt.layers.quantization.modelslim.modelslim import (
    ModelSlimConfig,
    ModelSlimQFP8DynamicKVFP8Method,
)
from sglang.srt.layers.quantization.modelslim.schemes import (
    ModelSlimMXFP4MoEScheme,
    ModelSlimMXFP4W4A8MoEScheme,
    ModelSlimMXFP8Scheme,
)
from sglang.srt.layers.quantization.modelslim.schemes import (
    modelslim_q_fp8_dynamic_kv_fp8 as kv_scheme,
)
from sglang.srt.layers.quantization.modelslim.schemes.modelslim_mxfp4_moe import (
    MXFP_E8M0_NOT_LOADED,
    _mxfp4_moe_weight_shapes,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestModelSlimMXFP4MoE(CustomTestCase):
    def test_mlaprolog_source_is_preserved_only_for_required_linears(self):
        qkv_scheme = ModelSlimMXFP8Scheme(
            {}, "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa"
        )
        q_b_scheme = ModelSlimMXFP8Scheme({}, "model.layers.0.self_attn.q_b_proj")
        o_proj_scheme = ModelSlimMXFP8Scheme({}, "model.layers.0.self_attn.o_proj")

        self.assertTrue(qkv_scheme.kernel.preserve_mlaprolog_source)
        self.assertTrue(q_b_scheme.kernel.preserve_mlaprolog_source)
        self.assertFalse(o_proj_scheme.kernel.preserve_mlaprolog_source)

    def test_fused_checkpoint_prefix_does_not_hide_mlaprolog_source(self):
        runtime_prefix = "model.layers.0.self_attn.fused_qkv_a_proj_with_mqa"
        checkpoint_prefix = "model.layers.0.self_attn.q_a_proj"
        config = ModelSlimConfig(
            {
                f"{checkpoint_prefix}.weight": "W8A8_MXFP8",
                "packed_modules_mapping": {
                    "model": {
                        "fused_qkv_a_proj_with_mqa": [
                            "q_a_proj",
                            "kv_a_proj_with_mqa",
                        ]
                    }
                },
            }
        )
        layer = SimpleNamespace()

        # Exercise the same post-selection hook used by get_quant_method
        # without constructing a device-specific LinearBase in a CPU test.
        scheme = config.get_linear_scheme(layer, checkpoint_prefix)
        scheme.configure_runtime_prefix(runtime_prefix)

        self.assertTrue(scheme.kernel.preserve_mlaprolog_source)

    def test_mlaprolog_source_views_share_checkpoint_storage(self):
        layer = torch.nn.Module()
        weight = torch.nn.Parameter(
            torch.empty((8, 16), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        scale = torch.nn.Parameter(
            torch.zeros((8, 4), dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("weight", weight)
        layer.register_parameter("weight_scale", scale)

        method = NPUMXFP8LinearMethod(preserve_mlaprolog_source=True)
        method.process_weights_after_loading(layer)

        self.assertEqual(layer.mlaprolog_weight_source.data_ptr(), weight.data_ptr())
        self.assertEqual(
            layer.mlaprolog_weight_scale_source.data_ptr(), scale.data_ptr()
        )
        self.assertEqual(layer.weight.data_ptr(), weight.data_ptr())
        self.assertEqual(layer.weight_scale_inv.data_ptr(), scale.data_ptr())

    def test_mxfp8_missing_scale_fails_before_mlaprolog_preservation(self):
        layer = torch.nn.Module()
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty((8, 16), dtype=torch.float8_e4m3fn),
                requires_grad=False,
            ),
        )
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(
                torch.full(
                    (8, 4),
                    MXFP8_E8M0_NOT_LOADED,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            ),
        )

        method = NPUMXFP8LinearMethod(preserve_mlaprolog_source=True)
        with self.assertRaisesRegex(RuntimeError, "not fully loaded"):
            method.process_weights_after_loading(layer)
        self.assertNotIn("mlaprolog_weight_scale_source", vars(layer))

    def test_checkpoint_shapes_are_derived_per_projection(self):
        self.assertEqual(
            _mxfp4_moe_weight_shapes("w13", 4, 128, 256),
            ((4, 512, 64), (4, 512, 4)),
        )
        self.assertEqual(
            _mxfp4_moe_weight_shapes("w2", 4, 128, 256),
            ((4, 128, 128), (4, 128, 8)),
        )

        with self.assertRaisesRegex(ValueError, "odd block count"):
            _mxfp4_moe_weight_shapes("w13", 4, 96, 256)

    def test_scheme_registers_packed_weights_and_required_scales(self):
        layer = torch.nn.Module()
        w13_scheme = ModelSlimMXFP4MoEScheme({}, "w13")
        w2_scheme = ModelSlimMXFP4MoEScheme({}, "w2")

        for scheme in (w13_scheme, w2_scheme):
            scheme.create_weights(
                layer,
                num_experts=4,
                hidden_size=128,
                intermediate_size_per_partition=256,
            )

        self.assertEqual(layer.w13_weight.shape, (4, 512, 64))
        self.assertEqual(layer.w13_weight_scale.shape, (4, 512, 4))
        self.assertEqual(layer.w2_weight.shape, (4, 128, 128))
        self.assertEqual(layer.w2_weight_scale.shape, (4, 128, 8))
        self.assertEqual(layer.w13_weight.dtype, torch.uint8)
        self.assertEqual(layer.w13_weight_scale.dtype, torch.uint8)
        self.assertTrue(torch.all(layer.w13_weight_scale == MXFP_E8M0_NOT_LOADED))
        self.assertIsNone(layer.w13_weight_offset)
        self.assertIsNone(layer.w2_weight_offset)

        with self.assertRaisesRegex(RuntimeError, "not fully loaded"):
            w13_scheme.process_weights_after_loading(layer)

    def test_selector_resolves_language_model_and_moe_aliases(self):
        checkpoint_prefix = "language_model.model.layers.0.block_sparse_moe.experts"
        runtime_prefix = "model.layers.0.mlp.experts"
        description = {
            f"{checkpoint_prefix}.0.gate_proj.weight": "W4A4_MXFP4",
            f"{checkpoint_prefix}.0.up_proj.weight": "W4A4_MXFP4",
            f"{checkpoint_prefix}.0.down_proj.weight": "W4A8_MXFP",
        }

        w13_scheme, w2_scheme = ModelSlimConfig(description).get_moe_scheme(
            torch.nn.Module(), runtime_prefix
        )

        self.assertIsInstance(w13_scheme, ModelSlimMXFP4MoEScheme)
        self.assertIsInstance(w2_scheme, ModelSlimMXFP4W4A8MoEScheme)
        self.assertEqual(w13_scheme.weight_prefix, "w13")
        self.assertEqual(w2_scheme.weight_prefix, "w2")

    def test_selector_requires_both_gate_and_up_descriptions(self):
        prefix = "model.layers.0.mlp.experts"
        description = {
            f"{prefix}.0.gate_proj.weight": "W4A4_MXFP4",
            f"{prefix}.0.down_proj.weight": "W4A4_MXFP4",
        }

        with self.assertRaisesRegex(ValueError, "1/2 found"):
            ModelSlimConfig(description).get_moe_scheme(torch.nn.Module(), prefix)

    def test_selector_rejects_mismatched_gate_and_up_schemes(self):
        prefix = "model.layers.0.mlp.experts"
        description = {
            f"{prefix}.0.gate_proj.weight": "W4A4_MXFP4",
            f"{prefix}.0.up_proj.weight": "W4A8_MXFP",
            f"{prefix}.0.down_proj.weight": "W4A4_MXFP4",
        }

        with self.assertRaisesRegex(ValueError, "Mismatched ModelSlim"):
            ModelSlimConfig(description).get_moe_scheme(torch.nn.Module(), prefix)

    def test_generic_expert_mapping_covers_weight_scales(self):
        mappings = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=1,
        )
        checkpoint_names = {
            "experts.0.gate_proj.weight_scale": "experts.w13_weight_scale",
            "experts.0.up_proj.weight_scale": "experts.w13_weight_scale",
            "experts.0.down_proj.weight_scale": "experts.w2_weight_scale",
        }

        for checkpoint_name, expected_name in checkpoint_names.items():
            matches = [
                checkpoint_name.replace(weight_name, param_name)
                for param_name, weight_name, _, _ in mappings
                if weight_name in checkpoint_name
            ]
            self.assertEqual(matches, [expected_name])

    def test_scale_pair_layout_and_input_scale_shape(self):
        checkpoint_scale = torch.arange(12, dtype=torch.uint8).reshape(1, 3, 4)
        packed = _pack_mxfp_weight_scale(checkpoint_scale)
        expected = torch.tensor(
            [[[[0, 1], [4, 5], [8, 9]], [[2, 3], [6, 7], [10, 11]]]],
            dtype=torch.uint8,
        )
        torch.testing.assert_close(packed, expected)

        flat_input_scale = torch.arange(8, dtype=torch.uint8).reshape(2, 4)
        normalized = _normalize_mxfp_input_scale(flat_input_scale)
        self.assertEqual(normalized.shape, (2, 2, 2))
        torch.testing.assert_close(normalized.reshape(2, 4), flat_input_scale)

    def test_missing_moe_scale_never_falls_back_to_one(self):
        kernel = NPUW4A4MXFP4MoEMethod("w13")
        quant_info = SimpleNamespace(w13_weight_scale=None)

        with self.assertRaisesRegex(RuntimeError, "unit-scale fallback"):
            kernel._weight_scale(quant_info, "w13")

    def test_prequantized_payload_dtype_must_match_mxfp_kernel(self):
        input_scale = torch.zeros((2, 2), dtype=torch.uint8)
        with patch.object(torch, "float4_e2m1fn_x2", torch.uint8, create=True):
            fp4_payload = torch.zeros((2, 2), dtype=torch.uint8)
            actual_payload, actual_scale = NPUW4A4MXFP4MoEMethod("w13")._quantize_input(
                fp4_payload, input_scale
            )
            self.assertIs(actual_payload, fp4_payload)
            self.assertEqual(actual_scale.shape, (2, 1, 2))

            with self.assertRaisesRegex(RuntimeError, "payload dtype"):
                NPUW4A4MXFP4MoEMethod("w13")._quantize_input(
                    torch.zeros((2, 4), dtype=torch.float8_e4m3fn),
                    input_scale,
                )

        with self.assertRaisesRegex(RuntimeError, "payload dtype"):
            NPUW4A8MXFPMoEMethod("w13")._quantize_input(
                torch.zeros((2, 4), dtype=torch.uint8),
                input_scale,
            )

        fp8_payload = torch.zeros((2, 4), dtype=torch.float8_e4m3fn)
        actual_payload, actual_scale = NPUW4A8MXFPMoEMethod("w13")._quantize_input(
            fp8_payload, input_scale
        )
        self.assertIs(actual_payload, fp8_payload)
        self.assertEqual(actual_scale.shape, (2, 1, 2))


class TestModelSlimKVScales(CustomTestCase):
    def test_public_config_registers_and_derives_k_scale(self):
        checkpoint_prefix = "language_model.model.layers.0.self_attn.attn"
        runtime_prefix = "model.layers.0.self_attn.attn"
        config = ModelSlimConfig(
            {f"{checkpoint_prefix}.quant_type": "Q_FP8_DYNAMIC_KV_FP8"}
        )
        layer = torch.nn.Module()
        layer.tp_q_head_num = 4
        layer.tp_k_head_num = 2

        method = config.get_quant_method(layer, runtime_prefix)
        self.assertIsInstance(method, ModelSlimQFP8DynamicKVFP8Method)
        method.create_weights(layer)

        self.assertNotIn("fa_q", layer._modules)
        self.assertEqual(layer.fa_k.scale.shape, (2, 1))
        self.assertTrue(torch.isnan(layer.fa_k.scale).all())
        with self.assertRaisesRegex(RuntimeError, "unit-scale fallback"):
            method.process_weights_after_loading(layer)

        loaded_scales = {
            "fa_k": torch.tensor([[0.5], [0.25]]),
            "fa_v": torch.tensor([[0.5], [0.25]]),
        }
        for name, value in loaded_scales.items():
            scale = layer._modules[name]._parameters["scale"]
            scale.weight_loader(scale, value)

        method.process_weights_after_loading(layer)
        self.assertTrue(layer._modelslim_fp8_kv_scale_ready)
        torch.testing.assert_close(layer.fak_descale_float, torch.tensor([[0.5, 0.25]]))
        torch.testing.assert_close(
            layer.fak_descale_reciprocal, torch.tensor([[2.0, 4.0]])
        )

    def test_kv_scale_contract_rejects_distinct_k_v_or_nonzero_offset(self):
        layer = torch.nn.Module()
        method = ModelSlimQFP8DynamicKVFP8Method(
            kv_scheme.ModelSlimQFP8DynamicKVFP8Scheme({}, "self_attn")
        )
        method.create_weights(layer, num_heads=4, num_kv_heads=1)

        layer.fa_k.scale.data.fill_(0.5)
        layer.fa_v.scale.data.fill_(0.25)
        with self.assertRaisesRegex(ValueError, "must be identical"):
            method.process_weights_after_loading(layer)
        self.assertFalse(layer._modelslim_fp8_kv_scale_ready)

        layer.fa_v.scale.data.copy_(layer.fa_k.scale)
        layer.fa_v.offset.data.fill_(1.0)
        with self.assertRaisesRegex(ValueError, "must be zero"):
            method.process_weights_after_loading(layer)
        self.assertFalse(layer._modelslim_fp8_kv_scale_ready)

    def test_kv_loader_shards_full_head_tensor_on_dim_zero(self):
        param = torch.nn.Parameter(torch.empty((2, 1)), requires_grad=False)
        loaded = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

        with (
            patch.object(kv_scheme, "get_tensor_model_parallel_rank", return_value=1),
            patch.object(
                kv_scheme, "get_tensor_model_parallel_world_size", return_value=2
            ),
        ):
            kv_scheme._modelslim_kv_weight_loader(param, loaded)

        torch.testing.assert_close(param, torch.tensor([[3.0], [4.0]]))


class TestAscendMXDeepEPBridge(CustomTestCase):
    def test_pre_a5_rejects_mxfp_dispatch(self):
        validator = deepep._DeepEPDispatcherImplBase._validate_and_adjust_dtype

        for dtype in (DispatcherOutputDtype.MXFP8, DispatcherOutputDtype.MXFP4):
            dispatcher = SimpleNamespace(deepep_output_dtype=dtype)
            with (
                patch.object(deepep, "_is_npu", True),
                patch.object(deepep, "is_npu_before_atlas_a5", return_value=True),
                self.assertRaisesRegex(RuntimeError, "before Atlas A5"),
            ):
                validator(dispatcher)

    def test_a5_accepts_mxfp_dispatch(self):
        validator = deepep._DeepEPDispatcherImplBase._validate_and_adjust_dtype
        dispatcher = SimpleNamespace(deepep_output_dtype=DispatcherOutputDtype.MXFP4)

        with (
            patch.object(deepep, "_is_npu", True),
            patch.object(deepep, "is_npu_before_atlas_a5", return_value=False),
        ):
            validator(dispatcher)

        self.assertEqual(dispatcher.deepep_output_dtype, DispatcherOutputDtype.MXFP4)

    def test_deepep_dtype_flags(self):
        self.assertEqual(
            deepep._deepep_dtype_flags(DispatcherOutputDtype.MXFP8),
            (True, False, True, False),
        )
        self.assertEqual(
            deepep._deepep_dtype_flags(DispatcherOutputDtype.MXFP4),
            (True, False, True, True),
        )

    def test_normal_dispatch_uses_empty_mx_dtype_marker(self):
        hidden_states = torch.ones((2, 64), dtype=torch.bfloat16)
        payload = deepep._npu_normal_dispatch_payload(
            hidden_states, DispatcherOutputDtype.MXFP8
        )

        self.assertIs(payload[0], hidden_states)
        self.assertEqual(payload[1].shape, (0,))
        self.assertEqual(payload[1].dtype, torch.float8_e4m3fn)

    def test_decode_unpermute_uses_absolute_row_indices(self):
        captured = {}

        def token_unpermute(*, permuted_tokens, sorted_indices, probs):
            captured["sorted_indices"] = sorted_indices
            return permuted_tokens

        fake_ops = SimpleNamespace(
            npu=SimpleNamespace(npu_moe_token_unpermute=token_unpermute)
        )
        hidden_states = torch.ones((2, 4))
        expanded_row_idx = torch.tensor([-3, 1], dtype=torch.int32)

        with patch.object(finalize_routing.torch, "ops", fake_ops):
            output = finalize_routing.NPUMoETokenUnpermute()._finalize_routing(
                hidden_states,
                torch.ones((1, 2)),
                expanded_row_idx,
                torch.zeros((1, 2), dtype=torch.int32),
            )

        self.assertIs(output, hidden_states)
        torch.testing.assert_close(
            captured["sorted_indices"], torch.tensor([3, 1], dtype=torch.int32)
        )


if __name__ == "__main__":
    unittest.main()
