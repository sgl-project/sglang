import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

# Load the quantization package first so `base_config`, `moe_methods`, and
# `linear_method_npu` initialize in dependency order. Importing `moe_methods`
# (or `linear_method_npu`) directly from a cold process triggers a circular
# import: linear_method_npu -> base_config -> quantization/__init__ ->
# gguf/unquant/gptq_moe -> moe_methods -> linear_method_npu (partially
# initialized, `_get_float8_e8m0fnu_dtype` not yet defined). Initializing the
# package first mirrors how the engine loads quantization at model-config time.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.moe.activation import NPUSwigluLimit
from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUW4A8MXFP4FusedMoEMethod,
    NPUW4A8MXFP4MoEMethod,
    prepare_w4a8_mxfp_weight,
    reshape_mxfp_activation_scale_for_npu,
    reshape_w4a8_mxfp_weight_scale_for_npu,
)
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod


class TestW4A8MxfpComputeOwnership(unittest.TestCase):
    def test_no_standalone_w4a8_mxfp_gmm_exists(self):
        from sglang.srt.hardware_backend.npu.quantization import moe_methods

        self.assertFalse(hasattr(moe_methods, "w4a8_mxfp_gmm"))


class TestFP4MethodGate(unittest.TestCase):
    def test_pre_arch35_keeps_fp8_moe_method(self):
        config = Fp8Config(is_fp4_experts=True)
        layer = FusedMoE.__new__(FusedMoE)

        with (
            patch("sglang.srt.layers.quantization.fp8.is_npu", return_value=True),
            patch(
                "sglang.srt.layers.quantization.fp8.is_npu_arch35",
                return_value=False,
            ),
        ):
            method = config.get_quant_method(layer, "model.layers.0.experts")

        self.assertIsInstance(method, Fp8MoEMethod)

    def test_arch35_uses_ascend_runner_method(self):
        config = Fp8Config(is_fp4_experts=True)
        layer = FusedMoE.__new__(FusedMoE)

        with (
            patch("sglang.srt.layers.quantization.fp8.is_npu", return_value=True),
            patch(
                "sglang.srt.layers.quantization.fp8.is_npu_arch35",
                return_value=True,
            ),
        ):
            method = config.get_quant_method(layer, "model.layers.0.experts")

        self.assertIsInstance(method, NPUW4A8MXFP4FusedMoEMethod)


class TestNPUSwigluLimit(unittest.TestCase):
    def test_clamps_before_swiglu(self):
        # DeepSeek-V4 clamps gate (first half) to <= limit but only the upper
        # bound, while up (second half) is clamped symmetrically to [-limit, limit].
        # A regression that swapped these would silently change expert activations.
        gate_up = torch.tensor([[8.0, -9.0, 9.0, -9.0]])
        with patch.object(
            torch.ops.npu, "npu_swiglu", return_value=torch.empty(1, 2), create=True
        ) as swiglu:
            NPUSwigluLimit(7.0)._apply_activation(gate_up)
        self.assertTrue(torch.equal(gate_up, torch.tensor([[7.0, -9.0, 7.0, -7.0]])))
        self.assertIs(swiglu.call_args.args[0], gate_up)


class TestReshapeMxfp4ScaleForNpu(unittest.TestCase):
    def test_packs_scale_to_gmm_layout(self):
        # [E, N, K/32] -> [E, K/64, N, 2] is the packed-pair layout the GMM reads;
        # getting the transpose axis wrong silently dequantizes with the wrong scale.
        scale = torch.arange(8, dtype=torch.uint8).view(1, 2, 4)
        out = reshape_w4a8_mxfp_weight_scale_for_npu(scale)
        self.assertEqual(tuple(out.shape), (1, 2, 2, 2))
        self.assertTrue(torch.equal(out, scale.view(1, 2, 2, 2).transpose(1, 2)))

    def test_rejects_odd_k_dim(self):
        with self.assertRaises(ValueError):
            reshape_w4a8_mxfp_weight_scale_for_npu(
                torch.zeros(1, 2, 3, dtype=torch.uint8)
            )


class TestPrepareW4A8MxfpWeight(unittest.TestCase):
    def test_uses_shared_weight_and_scale_layout(self):
        # A wrong transpose or scale packing makes both ModelSlim W4A8 and
        # DeepSeek-V4 W4A8 read different blocks from the same checkpoint.
        weight = torch.arange(16, dtype=torch.uint8).view(1, 2, 8)
        scale = torch.arange(8, dtype=torch.uint8).view(1, 2, 4)
        formatted_weight = torch.arange(16, dtype=torch.uint8).view(1, 2, 8)

        with patch(
            "sglang.srt.hardware_backend.npu.quantization.moe_methods.npu_format_cast",
            return_value=formatted_weight,
        ):
            prepared_weight, prepared_scale = prepare_w4a8_mxfp_weight(weight, scale)

        self.assertTrue(torch.equal(prepared_weight, formatted_weight.transpose(1, 2)))
        self.assertTrue(
            torch.equal(prepared_scale, scale.view(1, 2, 2, 2).transpose(1, 2))
        )


class TestMxfp4ScaleWeightLoader(unittest.TestCase):
    def test_reinterprets_e8m0_scale_as_raw_uint8(self):
        loaded = []

        def weight_loader(param, loaded_weight, *args, **kwargs):
            loaded.append(loaded_weight.clone())

        layer = torch.nn.Module()
        method = NPUW4A8MXFP4FusedMoEMethod(prefix="test")
        method.create_weights(
            layer,
            num_experts=1,
            hidden_size=64,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
            weight_loader=weight_loader,
        )
        checkpoint_scale = torch.tensor([0.5, 0.25, 0.125], dtype=torch.float8_e8m0fnu)

        layer.w13_weight_scale_inv.weight_loader(
            layer.w13_weight_scale_inv,
            checkpoint_scale,
            "model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
            "w1",
            0,
        )

        self.assertEqual(loaded[0].dtype, torch.uint8)
        self.assertTrue(torch.equal(loaded[0], checkpoint_scale.view(torch.uint8)))


class TestReshapeMxfpActivationScaleForNpu(unittest.TestCase):
    def test_packs_as_view(self):
        # The GMM expects a pair-packed *view* of the per-token scale, not a copy;
        # materializing a copy here would break the kernel's aliasing contract.
        flat = torch.arange(8).view(2, 4)
        packed = reshape_mxfp_activation_scale_for_npu(flat)
        self.assertEqual(tuple(packed.shape), (2, 2, 2))
        self.assertEqual(packed.data_ptr(), flat.data_ptr())

    def test_rejects_odd_scale_dim(self):
        with self.assertRaises(ValueError):
            reshape_mxfp_activation_scale_for_npu(torch.zeros(2, 3))

    def test_keeps_already_packed_scale(self):
        packed = torch.ones(2, 1, 2)
        self.assertIs(reshape_mxfp_activation_scale_for_npu(packed), packed)


class TestNPUW4A8MXFP4MoEMethod(unittest.TestCase):
    def setUp(self):
        self.input = torch.randn(2, 64)
        self.input_scale = torch.ones(2, 1, 2)
        self.weight = torch.empty(2, 64, 32, dtype=torch.uint8)
        self.weight_scale = torch.ones(2, 1, 32, 2, dtype=torch.uint8)
        self.group_list = torch.tensor([1, 1], dtype=torch.int32)
        self.quant_info = SimpleNamespace(
            w2_weight=self.weight,
            w2_weight_scale=self.weight_scale,
        )
        self.method = NPUW4A8MXFP4MoEMethod()

    def _apply(self, input_scale):
        return self.method.apply(
            self.quant_info,
            self.input,
            self.group_list,
            input_scale,
            torch.bfloat16,
            "w2",
            group_list_type=1,
        )

    def test_supplied_scale_skips_dynamic_quant(self):
        expected = torch.randn(2, 32)
        with (
            patch.object(
                torch.ops.npu, "npu_dynamic_mx_quant", create=True
            ) as dynamic_quant,
            patch.object(
                torch.ops.npu,
                "npu_grouped_matmul",
                return_value=[expected],
                create=True,
            ) as grouped_matmul,
        ):
            output = self._apply(self.input_scale)

        dynamic_quant.assert_not_called()
        self.assertIs(output, expected)
        call_kwargs = grouped_matmul.call_args.kwargs
        self.assertIs(call_kwargs["per_token_scale"][0], self.input_scale)
        self.assertEqual(call_kwargs["group_list"].dtype, torch.int64)
        self.assertTrue(torch.equal(call_kwargs["group_list"], self.group_list))

    def test_missing_scale_uses_shared_quantizer(self):
        quantized = torch.empty(2, 64, dtype=torch.float8_e4m3fn)
        quantized_scale = torch.ones(2, 1, 2)
        expected = torch.randn(2, 32)
        with (
            patch.object(
                torch.ops.npu,
                "npu_dynamic_mx_quant",
                return_value=(quantized, quantized_scale),
                create=True,
            ) as dynamic_quant,
            patch.object(
                torch.ops.npu,
                "npu_grouped_matmul",
                return_value=[expected],
                create=True,
            ) as grouped_matmul,
        ):
            self.method.hidden_states_quantizer = MagicMock(
                return_value=(quantized, quantized_scale)
            )
            output = self._apply(None)

        dynamic_quant.assert_not_called()
        self.method.hidden_states_quantizer.assert_called_once_with(self.input)
        self.assertIs(output, expected)
        self.assertIs(
            grouped_matmul.call_args.kwargs["per_token_scale"][0], quantized_scale
        )


class TestRunnerDelegation(unittest.TestCase):
    def test_apply_delegates_with_dsv4_scale_names(self):
        method = NPUW4A8MXFP4FusedMoEMethod(prefix="test")
        expected = object()
        method.runner = MagicMock()
        method.runner.run.return_value = expected
        layer = SimpleNamespace(
            w13_weight=MagicMock(),
            w13_weight_scale_inv=MagicMock(),
            w2_weight=MagicMock(),
            w2_weight_scale_inv=MagicMock(),
        )
        dispatch_output = object()

        output = method.apply(layer, dispatch_output)

        self.assertIs(output, expected)
        method.runner.run.assert_called_once()
        self.assertIs(method.runner.run.call_args.args[0], dispatch_output)
        quant_info = method.runner.run.call_args.args[1]
        self.assertIs(quant_info.w13_weight_scale, layer.w13_weight_scale_inv)
        self.assertIs(quant_info.w2_weight_scale, layer.w2_weight_scale_inv)

    def test_create_runner_installs_internal_kernels_before_runner(self):
        method = NPUW4A8MXFP4FusedMoEMethod(prefix="test")
        layer = SimpleNamespace()
        config = SimpleNamespace(layer=None)
        backend = MagicMock()
        backend.is_auto.return_value = True

        with (
            patch("sglang.srt.layers.moe.moe_runner.runner.MoeRunner") as moe_runner,
            patch(
                "sglang.srt.layers.moe.utils.get_moe_runner_backend",
                return_value=backend,
            ),
        ):
            method.create_moe_runner(layer, config)

        self.assertIs(layer.w13_kernel, method.w13_kernel)
        self.assertIs(layer.w2_kernel, method.w2_kernel)
        self.assertIs(config.layer, layer)
        moe_runner.assert_called_once()


class TestProcessWeightsAfterLoadingZeroScale(unittest.TestCase):
    @staticmethod
    def _method():
        return NPUW4A8MXFP4FusedMoEMethod(prefix="test")

    def test_raises_when_w13_scales_never_loaded(self):
        # An all-zero scale is the signature of a checkpoint whose scale names
        # never matched; without this guard every routed expert computes silently
        # as zero instead of failing loudly.
        layer = SimpleNamespace(
            w13_weight_scale_inv=torch.nn.Parameter(
                torch.zeros(2, 2, 4, dtype=torch.uint8), requires_grad=False
            ),
            w2_weight_scale_inv=torch.nn.Parameter(
                torch.zeros(2, 2, 4, dtype=torch.uint8), requires_grad=False
            ),
        )
        with self.assertRaises(RuntimeError):
            self._method().process_weights_after_loading(layer)

    def test_raises_when_w2_scales_never_loaded(self):
        layer = SimpleNamespace(
            w13_weight_scale_inv=torch.nn.Parameter(
                torch.ones(2, 2, 4, dtype=torch.uint8), requires_grad=False
            ),
            w2_weight_scale_inv=torch.nn.Parameter(
                torch.zeros(2, 2, 4, dtype=torch.uint8), requires_grad=False
            ),
        )
        with self.assertRaises(RuntimeError):
            self._method().process_weights_after_loading(layer)


if __name__ == "__main__":
    unittest.main()
