"""CPU contracts for GLM-5.3-Flash KDA PTPC projections."""

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.quantization import fp8_utils, unquant
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.glm5_next import (
    GLM53_KDA_PTPC_BF16_MAX_M,
    Glm5NextLinearAttention,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _Linear(torch.nn.Module):
    def __init__(self, quant_method=None):
        super().__init__()
        self.quant_method = quant_method or UnquantizedLinearMethod()


class TestGLM53KDAPTPC(CustomTestCase):
    def test_gate_truth_table(self):
        layer = SimpleNamespace()
        self.assertEqual(envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.default, ())
        for selected in (False, True):
            for use_aiter in (False, True):
                for marked in (False, True):
                    for gfx950 in (False, True):
                        selector = "qkv_proj" if selected else ""
                        with (
                            self.subTest(
                                selected=selected,
                                use_aiter=use_aiter,
                                marked=marked,
                                gfx950=gfx950,
                            ),
                            envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override(selector),
                            patch.object(unquant, "_use_aiter", use_aiter),
                            patch.object(
                                unquant,
                                "is_gfx95_supported",
                                return_value=gfx950,
                            ) as gfx95_supported,
                        ):
                            if marked:
                                layer._glm53_kda_ptpc_module = "qkv_proj"
                            elif hasattr(layer, "_glm53_kda_ptpc_module"):
                                del layer._glm53_kda_ptpc_module
                            self.assertEqual(
                                unquant._glm53_kda_ptpc_enabled(layer),
                                selected and use_aiter and marked and gfx950,
                            )
                            self.assertEqual(
                                gfx95_supported.call_count,
                                int(selected and use_aiter and marked),
                            )

    def test_default_off_preserves_bf16_linear(self):
        method = UnquantizedLinearMethod()
        layer = _Linear(method)
        weight = torch.randn(4, 8, dtype=torch.bfloat16)
        layer.register_parameter(
            "weight", torch.nn.Parameter(weight.clone(), requires_grad=False)
        )
        x = torch.randn(3, 8, dtype=torch.bfloat16)
        with (
            envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override(""),
            patch.object(unquant, "_is_cpu_amx_available", False),
            patch.object(method, "_repack_bf16_to_fp8_ptpc") as repack,
        ):
            method.process_weights_after_loading(layer)
        repack.assert_not_called()
        with (
            patch.object(unquant, "use_intel_amx_backend", return_value=False),
            patch.object(unquant, "_use_aiter", False),
        ):
            actual = method.apply(layer, x)
        torch.testing.assert_close(actual, F.linear(x, weight), rtol=0, atol=0)
        self.assertFalse(hasattr(layer, "_fp8_ptpc_ready"))

    def test_non_gfx950_does_not_repack(self):
        method = UnquantizedLinearMethod()
        layer = _Linear(method)
        layer._glm53_kda_ptpc_module = "qkv_proj"
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.zeros(4, 8, dtype=torch.bfloat16), requires_grad=False
            ),
        )
        with (
            envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override("qkv_proj"),
            patch.object(unquant, "_use_aiter", True),
            patch.object(unquant, "is_gfx95_supported", return_value=False),
            patch.object(method, "_repack_bf16_to_fp8_ptpc") as repack,
            patch.object(unquant, "_is_cpu_amx_available", False),
        ):
            method.process_weights_after_loading(layer)
        repack.assert_not_called()

    def test_repack_preserves_bf16_and_registers_nonpersistent_buffers(self):
        method = UnquantizedLinearMethod()
        layer = _Linear(method)
        weight = torch.arange(32, dtype=torch.bfloat16).view(4, 8)
        parameter = torch.nn.Parameter(weight.clone(), requires_grad=False)
        layer.register_parameter("weight", parameter)
        fp8_weight = torch.arange(32, dtype=torch.float32).view(4, 8)
        weight_scale = torch.arange(4, dtype=torch.float32).view(4, 1)
        shuffled = fp8_weight + 1
        fake_aiter = types.ModuleType("aiter")
        fake_aiter.dtypes = SimpleNamespace(fp8=object())
        fake_aiter.pertoken_quant = MagicMock(return_value=(fp8_weight, weight_scale))
        fake_ops = types.ModuleType("aiter.ops")
        fake_shuffle = types.ModuleType("aiter.ops.shuffle")
        fake_shuffle.shuffle_weight = MagicMock(return_value=shuffled)
        with patch.dict(
            sys.modules,
            {
                "aiter": fake_aiter,
                "aiter.ops": fake_ops,
                "aiter.ops.shuffle": fake_shuffle,
            },
        ):
            method._repack_bf16_to_fp8_ptpc(layer)
            method._repack_bf16_to_fp8_ptpc(layer)
        self.assertIs(layer.weight, parameter)
        torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(layer._fp8_ptpc_weight, shuffled)
        torch.testing.assert_close(layer._fp8_ptpc_weight_scale, weight_scale)
        self.assertEqual(
            set(dict(layer.named_buffers())),
            {
                "_fp8_ptpc_weight",
                "_fp8_ptpc_weight_scale",
            },
        )
        self.assertNotIn("_fp8_ptpc_weight", layer.state_dict())
        self.assertNotIn("_fp8_ptpc_weight_scale", layer.state_dict())
        self.assertTrue(layer._fp8_ptpc_ready)
        fake_aiter.pertoken_quant.assert_called_once()
        fake_shuffle.shuffle_weight.assert_called_once_with(fp8_weight, (16, 16))

    def test_repack_rejects_incompatible_selected_weight(self):
        for weight in (
            torch.zeros(4, 8),
            torch.zeros(2, 4, 8, dtype=torch.bfloat16),
        ):
            layer = _Linear()
            layer._glm53_kda_ptpc_module = "qkv_proj"
            layer.register_parameter(
                "weight", torch.nn.Parameter(weight, requires_grad=False)
            )
            with self.subTest(dtype=weight.dtype, shape=tuple(weight.shape)):
                with self.assertRaisesRegex(ValueError, "2-D BF16"):
                    UnquantizedLinearMethod._repack_bf16_to_fp8_ptpc(layer)

    def test_per_module_bf16_ptpc_boundaries_and_rollback(self):
        method = UnquantizedLinearMethod()
        for module_name, max_m in GLM53_KDA_PTPC_BF16_MAX_M.items():
            layer = _Linear(method)
            layer._fp8_ptpc_ready = True
            layer._fp8_ptpc_bf16_max_m = max_m
            layer.register_parameter(
                "weight",
                torch.nn.Parameter(
                    torch.empty(4, 8, dtype=torch.bfloat16, device="meta"),
                    requires_grad=False,
                ),
            )
            layer.register_buffer("_fp8_ptpc_weight", torch.empty(4, 8, device="meta"))
            layer.register_buffer(
                "_fp8_ptpc_weight_scale", torch.empty(4, 1, device="meta")
            )
            bf16_output = torch.empty(max_m, 4, device="meta")
            ptpc_output = torch.empty(max_m + 1, 4, device="meta")
            with (
                self.subTest(module=module_name),
                patch.object(unquant, "_use_aiter", False),
                patch.object(
                    unquant.F, "linear", return_value=bf16_output
                ) as bf16_linear,
                patch.object(
                    fp8_utils,
                    "apply_fp8_ptpc_linear",
                    return_value=ptpc_output,
                ) as apply_ptpc,
            ):
                x_bf16 = torch.empty(max_m, 8, device="meta")
                self.assertIs(method.apply(layer, x_bf16), bf16_output)
                bf16_linear.assert_called_once_with(x_bf16, layer.weight, None)
                apply_ptpc.assert_not_called()
                x_ptpc = torch.empty(max_m + 1, 8, device="meta")
                self.assertIs(method.apply(layer, x_ptpc), ptpc_output)
                apply_ptpc.assert_called_once_with(
                    x_ptpc,
                    layer._fp8_ptpc_weight,
                    layer._fp8_ptpc_weight_scale,
                    bias=None,
                )
                layer._fp8_ptpc_ready = False
                bf16_linear.reset_mock()
                fallback = torch.empty_like(ptpc_output)
                bf16_linear.return_value = fallback
                self.assertIs(method.apply(layer, x_ptpc), fallback)
                bf16_linear.assert_called_once_with(x_ptpc, layer.weight, None)

    def test_zero_tokens_remain_bf16(self):
        layer = SimpleNamespace(
            _fp8_ptpc_ready=True,
            _fp8_ptpc_bf16_max_m=0,
        )
        self.assertFalse(unquant.fp8_ptpc_linear_active(layer, 0))

    def test_tuple_dispatch_preserves_quantized_input_and_bias(self):
        method = UnquantizedLinearMethod()
        layer = SimpleNamespace(
            _fp8_ptpc_ready=True,
            _fp8_ptpc_bf16_max_m=512,
            _fp8_ptpc_weight=torch.empty(4, 8),
            _fp8_ptpc_weight_scale=torch.ones(4, 1),
        )
        qinput = torch.empty(3, 8)
        xscale = torch.ones(3, 1)
        bias = torch.zeros(4)
        expected = torch.empty(3, 4)
        with patch.object(
            fp8_utils, "apply_fp8_ptpc_linear", return_value=expected
        ) as apply_ptpc:
            actual = method.apply(layer, (qinput, xscale), bias)
        self.assertIs(actual, expected)
        apply_ptpc.assert_called_once_with(
            (qinput, xscale),
            layer._fp8_ptpc_weight,
            layer._fp8_ptpc_weight_scale,
            bias=bias,
        )

    def test_model_selector_marks_only_requested_unquantized_modules(self):
        attention = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
        torch.nn.Module.__init__(attention)
        attention.do_fuse_qkvbfg = False
        for name in ("qkv_proj", "o_proj", "b_proj", "f_a_proj", "g_a_proj"):
            setattr(attention, name, _Linear())
        with envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override("qkv_proj"):
            attention._configure_ptpc_modules()
        module = attention.qkv_proj
        self.assertEqual(module._glm53_kda_ptpc_module, "qkv_proj")
        self.assertEqual(
            module._fp8_ptpc_bf16_max_m,
            GLM53_KDA_PTPC_BF16_MAX_M["qkv_proj"],
        )
        for name in ("o_proj", "b_proj", "f_a_proj", "g_a_proj"):
            self.assertFalse(
                hasattr(getattr(attention, name), "_glm53_kda_ptpc_module")
            )

    def test_model_selector_rejects_unknown_fused_and_quantized_targets(self):
        attention = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
        torch.nn.Module.__init__(attention)
        attention.do_fuse_qkvbfg = True
        attention.o_proj = _Linear()
        for selector in ("unknown", "o_proj"):
            with (
                self.subTest(selector=selector),
                envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override(selector),
            ):
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    attention._configure_ptpc_modules()
        with envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override("qkv_proj"):
            with self.assertRaisesRegex(ValueError, "unavailable"):
                attention._configure_ptpc_modules()
        attention.do_fuse_qkvbfg = False
        attention.qkv_proj = _Linear(quant_method=object())
        with envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override("qkv_proj"):
            with self.assertRaisesRegex(ValueError, "UnquantizedLinearMethod"):
                attention._configure_ptpc_modules()

    def test_model_quantization_boundary_and_zero_tokens(self):
        threshold = GLM53_KDA_PTPC_BF16_MAX_M["qkv_proj"]
        layer = SimpleNamespace(
            _fp8_ptpc_ready=True,
            _fp8_ptpc_bf16_max_m=threshold,
        )
        x_small = torch.empty(threshold, 8)
        x_empty = torch.empty(0, 8)
        self.assertIs(
            Glm5NextLinearAttention._maybe_quantize_ptpc_input(layer, x_small),
            x_small,
        )
        self.assertIs(
            Glm5NextLinearAttention._maybe_quantize_ptpc_input(layer, x_empty),
            x_empty,
        )
        x_large = torch.empty(threshold + 1, 8)
        qinput = torch.empty(threshold + 1, 8)
        scale = torch.ones(threshold + 1, 1)
        fake_aiter = types.ModuleType("aiter")
        fake_aiter.dtypes = SimpleNamespace(fp8=object())
        fake_aiter.per_token_quant_hip = MagicMock(return_value=(qinput, scale))
        with patch.dict(sys.modules, {"aiter": fake_aiter}):
            actual = Glm5NextLinearAttention._maybe_quantize_ptpc_input(layer, x_large)
        self.assertIs(actual[0], qinput)
        self.assertIs(actual[1], scale)
        fake_aiter.per_token_quant_hip.assert_called_once()
        call = fake_aiter.per_token_quant_hip.call_args
        self.assertEqual(call.args[0].data_ptr(), x_large.data_ptr())
        self.assertEqual(call.args[0].shape, x_large.shape)
        self.assertIs(call.kwargs["quant_dtype"], fake_aiter.dtypes.fp8)


if __name__ == "__main__":
    unittest.main(verbosity=3)
