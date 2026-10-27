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
from sglang.srt.layers.quantization.unquant import (
    Glm53KdaPackedPtpcLinearMethod,
    Glm53KdaPtpcLinearMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.models.glm5_next import (
    GLM53_KDA_FUSED_O_NORM_MIN_M,
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


class _RecordingLinear(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x)
        return self.output, None


class _RecordingFusedLinear(torch.nn.Module):
    def __init__(self, output_size, quant_method=None):
        super().__init__()
        self.output_size = output_size
        self.quant_method = quant_method or UnquantizedLinearMethod()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x)
        tensor = x if isinstance(x, torch.Tensor) else x[0]
        return tensor.new_empty(tensor.shape[0], self.output_size)


class _RecordingBatchedLinear(torch.nn.Module):
    def forward(self, x):
        return x.new_empty(2, x.shape[1], 6).unbind(0)


class TestGLM53KDAPTPC(CustomTestCase):
    def test_method_activation_boundary(self):
        self.assertEqual(envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.default, ())
        method = Glm53KdaPtpcLinearMethod("qkv_proj", bf16_max_m=512)
        self.assertFalse(method.is_active(513))
        method._fp8_ptpc_ready = True
        self.assertFalse(method.is_active(0))
        self.assertFalse(method.is_active(512))
        self.assertTrue(method.is_active(513))

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
        ):
            method.process_weights_after_loading(layer)
        with (
            patch.object(unquant, "use_intel_amx_backend", return_value=False),
            patch.object(unquant, "_use_aiter", False),
        ):
            actual = method.apply(layer, x)
        torch.testing.assert_close(actual, F.linear(x, weight), rtol=0, atol=0)

    def test_repack_preserves_bf16_and_registers_nonpersistent_buffers(self):
        method = Glm53KdaPtpcLinearMethod("qkv_proj", bf16_max_m=0)
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
            method.process_weights_after_loading(layer)
            method.process_weights_after_loading(layer)
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
        self.assertTrue(method._fp8_ptpc_ready)
        fake_aiter.pertoken_quant.assert_called_once()
        fake_shuffle.shuffle_weight.assert_called_once_with(fp8_weight, (16, 16))

    def test_repack_rejects_incompatible_selected_weight(self):
        for weight in (
            torch.zeros(4, 8),
            torch.zeros(2, 4, 8, dtype=torch.bfloat16),
        ):
            layer = _Linear()
            layer.register_parameter(
                "weight", torch.nn.Parameter(weight, requires_grad=False)
            )
            with self.subTest(dtype=weight.dtype, shape=tuple(weight.shape)):
                with self.assertRaisesRegex(ValueError, "2-D BF16"):
                    Glm53KdaPtpcLinearMethod(
                        "qkv_proj", bf16_max_m=0
                    ).process_weights_after_loading(layer)

    def test_per_module_bf16_ptpc_boundaries_and_rollback(self):
        for module_name, max_m in GLM53_KDA_PTPC_BF16_MAX_M.items():
            method = Glm53KdaPtpcLinearMethod(module_name, bf16_max_m=max_m)
            method._fp8_ptpc_ready = True
            method.output_size = 4
            layer = _Linear(method)
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
                method._fp8_ptpc_ready = False
                bf16_linear.reset_mock()
                fallback = torch.empty_like(ptpc_output)
                bf16_linear.return_value = fallback
                self.assertIs(method.apply(layer, x_ptpc), fallback)
                bf16_linear.assert_called_once_with(x_ptpc, layer.weight, None)

    def test_zero_tokens_remain_bf16(self):
        method = Glm53KdaPtpcLinearMethod("qkv_proj", bf16_max_m=0)
        method._fp8_ptpc_ready = True
        self.assertFalse(method.is_active(0))

    def test_tuple_dispatch_preserves_quantized_input_and_bias(self):
        method = Glm53KdaPtpcLinearMethod("qkv_proj", bf16_max_m=512)
        method._fp8_ptpc_ready = True
        method.output_size = 4
        layer = SimpleNamespace(
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

    def test_model_selector_requires_aiter_gfx950(self):
        attention = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
        torch.nn.Module.__init__(attention)
        with (
            envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override("o_proj"),
            patch.object(
                sys.modules[Glm5NextLinearAttention.__module__],
                "_use_aiter_gfx95",
                False,
            ),
            self.assertRaisesRegex(ValueError, "requires AITER on gfx950"),
        ):
            attention._configure_ptpc_modules()

    def test_model_selector_marks_shared_input_modules_together(self):
        model_module = sys.modules[Glm5NextLinearAttention.__module__]
        attention = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
        torch.nn.Module.__init__(attention)
        attention.do_fuse_qkvbfg = False
        attention.fuse_bfg = False
        for name in ("qkv_proj", "f_a_proj", "g_a_proj", "o_proj"):
            setattr(attention, name, _Linear())
        attention.o_proj.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty(4096, 2048, device="meta"),
                requires_grad=False,
            ),
        )
        selector = "qkv_proj,f_a_proj,g_a_proj,o_proj"
        with (
            envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override(selector),
            patch.object(model_module, "_use_aiter_gfx95", True),
        ):
            attention._configure_ptpc_modules()
        for name in ("qkv_proj", "f_a_proj", "g_a_proj", "o_proj"):
            module = getattr(attention, name)
            self.assertIsInstance(
                module.quant_method,
                Glm53KdaPtpcLinearMethod,
            )
            self.assertEqual(
                module.quant_method.bf16_max_m,
                GLM53_KDA_PTPC_BF16_MAX_M[name],
            )

    def test_model_selector_preserves_fused_first_stage(self):
        model_module = sys.modules[Glm5NextLinearAttention.__module__]
        attention = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
        torch.nn.Module.__init__(attention)
        attention.do_fuse_qkvbfg = True
        attention.fuse_bfg = False
        attention.split_sizes = [3072, 8, 256]
        attention.fused_qkvbfg_a_proj = _Linear()
        attention.o_proj = _Linear()
        selector = "qkv_proj,f_a_proj,g_a_proj"
        with (
            envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override(selector),
            patch.object(model_module, "_use_aiter_gfx95", True),
        ):
            attention._configure_ptpc_modules()
        self.assertIsInstance(
            attention.fused_qkvbfg_a_proj.quant_method,
            Glm53KdaPackedPtpcLinearMethod,
        )
        self.assertTrue(attention.do_fuse_qkvbfg)

    def test_model_selector_rejects_unsupported_fused_qkv_size(self):
        model_module = sys.modules[Glm5NextLinearAttention.__module__]
        attention = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
        torch.nn.Module.__init__(attention)
        attention.do_fuse_qkvbfg = True
        attention.fuse_bfg = False
        attention.split_sizes = [12288, 32, 256]
        attention.fused_qkvbfg_a_proj = _Linear()
        attention.o_proj = _Linear()
        with (
            envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override(
                "qkv_proj,f_a_proj,g_a_proj"
            ),
            patch.object(model_module, "_use_aiter_gfx95", True),
            self.assertRaisesRegex(ValueError, "supports fused qkv sizes"),
        ):
            attention._configure_ptpc_modules()

    def test_fused_first_stage_keeps_decode_bf16_and_uses_ptpc_for_prefill(self):
        attention = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
        torch.nn.Module.__init__(attention)
        attention.head_dim = 4
        attention.split_sizes = [12, 2, 8]
        attention.fused_qkvbfg_a_proj = _RecordingFusedLinear(22)
        attention.fused_fg_b_proj = _RecordingBatchedLinear()
        method = Glm53KdaPackedPtpcLinearMethod(
            bf16_max_m=3,
            fp8_max_m=8,
            qkv_size=12,
            beta_size=2,
            fg_size=8,
        )
        method._fp8_ptpc_ready = True
        attention.fused_qkvbfg_a_proj.quant_method = method

        decode_input = torch.empty(2, 8)
        attention.forward_qkvbfg_fused(decode_input, forward_batch=None)
        self.assertIs(attention.fused_qkvbfg_a_proj.inputs[-1], decode_input)

        prefill_input = torch.empty(4, 8)
        packed_outputs = (
            torch.empty(4, 12),
            torch.empty(4, 2),
            torch.empty(4, 8),
        )
        with patch.object(
            method,
            "apply_ptpc_prefill",
            return_value=packed_outputs,
        ) as apply_ptpc:
            attention.forward_qkvbfg_fused(prefill_input, forward_batch=None)
        self.assertEqual(len(attention.fused_qkvbfg_a_proj.inputs), 1)
        apply_ptpc.assert_called_once_with(
            attention.fused_qkvbfg_a_proj,
            prefill_input,
            q_input=None,
        )

    def test_fused_first_stage_consumes_mhc_prequant(self):
        attention = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
        torch.nn.Module.__init__(attention)
        attention.head_dim = 4
        attention.split_sizes = [12, 2, 8]
        attention.fused_qkvbfg_a_proj = _RecordingFusedLinear(22)
        attention.fused_fg_b_proj = _RecordingBatchedLinear()
        method = Glm53KdaPackedPtpcLinearMethod(
            bf16_max_m=3,
            fp8_max_m=8,
            qkv_size=12,
            beta_size=2,
            fg_size=8,
        )
        method._fp8_ptpc_ready = True
        attention.fused_qkvbfg_a_proj.quant_method = method
        bf16 = torch.empty(4, 8)
        prequant = (torch.empty(4, 8), torch.ones(4, 1))
        packed_outputs = (
            torch.empty(4, 12),
            torch.empty(4, 2),
            torch.empty(4, 8),
        )
        with patch.object(
            method, "apply_ptpc_prefill", return_value=packed_outputs
        ) as apply_ptpc:
            attention.forward_qkvbfg_fused(
                (bf16, prequant[0], prequant[1]), forward_batch=None
            )
        apply_ptpc.assert_called_once_with(
            attention.fused_qkvbfg_a_proj,
            bf16,
            q_input=prequant,
        )

    def test_fused_o_norm_dispatch_uses_local_k_and_token_boundaries(self):
        model_module = sys.modules[Glm5NextLinearAttention.__module__]
        attention = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
        torch.nn.Module.__init__(attention)
        method = Glm53KdaPtpcLinearMethod(
            "o_proj",
            bf16_max_m=GLM53_KDA_PTPC_BF16_MAX_M["o_proj"],
        )
        method._fp8_ptpc_ready = True
        attention.o_proj = _Linear(method)
        attention.o_norm = _Linear()
        attention.o_norm.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty(128, device="meta"),
                requires_grad=False,
            ),
        )

        with (
            patch.object(
                model_module,
                "can_use_rms_norm_gated_per_token_fp8",
                return_value=True,
            ),
        ):
            for local_k, num_tokens, expected in (
                (1024, 0, False),
                (1024, 1, True),
                (2048, 255, False),
                (2048, 256, True),
                (2048, 4096, True),
                (1536, 4096, False),
            ):
                heads = local_k // 128
                attention.o_proj.register_parameter(
                    "weight",
                    torch.nn.Parameter(
                        torch.empty(4096, local_k, device="meta"),
                        requires_grad=False,
                    ),
                )
                core = torch.empty(
                    1,
                    num_tokens,
                    heads,
                    128,
                    device="meta",
                )
                gate = torch.empty(
                    num_tokens,
                    heads,
                    128,
                    device="meta",
                )
                with self.subTest(local_k=local_k, num_tokens=num_tokens):
                    self.assertEqual(
                        attention._use_fused_o_norm_ptpc(core, gate),
                        expected,
                    )

        self.assertEqual(
            GLM53_KDA_FUSED_O_NORM_MIN_M,
            {1024: 1, 2048: 256},
        )

    def test_model_selector_rejects_unknown_fused_and_quantized_targets(self):
        model_module = sys.modules[Glm5NextLinearAttention.__module__]
        attention = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
        torch.nn.Module.__init__(attention)
        attention.do_fuse_qkvbfg = True
        attention.o_proj = _Linear()
        attention.o_proj.register_parameter(
            "weight",
            torch.nn.Parameter(
                torch.empty(4096, 2048, device="meta"),
                requires_grad=False,
            ),
        )
        with envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override("unknown"):
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                attention._configure_ptpc_modules()
        with (
            envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override("o_proj"),
            patch.object(model_module, "_use_aiter_gfx95", True),
        ):
            attention._configure_ptpc_modules()
        self.assertIsInstance(
            attention.o_proj.quant_method,
            Glm53KdaPtpcLinearMethod,
        )
        with envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override("qkv_proj"):
            with self.assertRaisesRegex(ValueError, "selected together"):
                attention._configure_ptpc_modules()
        attention.do_fuse_qkvbfg = False
        attention.fuse_bfg = False
        attention.qkv_proj = _Linear(quant_method=object())
        attention.f_a_proj = _Linear()
        attention.g_a_proj = _Linear()
        with (
            envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override(
                "qkv_proj,f_a_proj,g_a_proj"
            ),
            patch.object(model_module, "_use_aiter_gfx95", True),
        ):
            with self.assertRaisesRegex(ValueError, "UnquantizedLinearMethod"):
                attention._configure_ptpc_modules()

    def test_model_selector_rejects_partial_shared_input_group(self):
        model_module = sys.modules[Glm5NextLinearAttention.__module__]
        attention = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
        torch.nn.Module.__init__(attention)
        attention.do_fuse_qkvbfg = False
        for name in ("qkv_proj", "f_a_proj", "g_a_proj"):
            setattr(attention, name, _Linear())
        for selector in (
            "qkv_proj",
            "f_a_proj",
            "g_a_proj",
            "qkv_proj,f_a_proj",
            "qkv_proj,g_a_proj",
        ):
            with (
                self.subTest(selector=selector),
                envs.SGLANG_OPT_GLM53_KDA_PTPC_MODULES.override(selector),
                patch.object(model_module, "_use_aiter_gfx95", True),
                self.assertRaisesRegex(ValueError, "selected together"),
            ):
                attention._configure_ptpc_modules()

    def test_forward_reuses_one_quantized_input_for_qkv_f_and_g(self):
        attention = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
        torch.nn.Module.__init__(attention)
        attention.fuse_bfg = False
        hidden_states = torch.randn(3, 8)
        quantized = (torch.empty(3, 8), torch.ones(3, 1))
        attention.qkv_proj = _RecordingLinear(torch.empty(3, 12))
        attention.b_proj = _RecordingLinear(torch.empty(3, 2))
        attention.f_a_proj = _RecordingLinear(torch.empty(3, 4))
        attention.f_b_proj = _RecordingLinear(torch.empty(3, 6))
        attention.g_a_proj = _RecordingLinear(torch.empty(3, 4))
        attention.g_b_proj = _RecordingLinear(torch.empty(3, 6))
        with (
            patch.object(
                Glm5NextLinearAttention,
                "_maybe_quantize_ptpc_input",
                return_value=quantized,
            ) as quantize,
            patch.object(
                Glm5NextLinearAttention,
                "_ptpc_linear_active",
                side_effect=lambda layer, _: (
                    layer in (attention.f_a_proj, attention.g_a_proj)
                ),
            ),
        ):
            attention.forward_qkvbfg(hidden_states, forward_batch=None)
        quantize.assert_called_once_with(attention.qkv_proj, hidden_states)
        self.assertIs(attention.qkv_proj.inputs[0], quantized)
        self.assertIs(attention.f_a_proj.inputs[0], quantized)
        self.assertIs(attention.g_a_proj.inputs[0], quantized)
        self.assertIs(attention.b_proj.inputs[0], hidden_states)

    def test_forward_reuses_mhc_prequant_without_requantizing(self):
        attention = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
        torch.nn.Module.__init__(attention)
        attention.fuse_bfg = False
        hidden_states = torch.randn(3, 8)
        quantized = (torch.empty(3, 8), torch.ones(3, 1))
        attention.qkv_proj = _RecordingLinear(torch.empty(3, 12))
        attention.b_proj = _RecordingLinear(torch.empty(3, 2))
        attention.f_a_proj = _RecordingLinear(torch.empty(3, 4))
        attention.f_b_proj = _RecordingLinear(torch.empty(3, 6))
        attention.g_a_proj = _RecordingLinear(torch.empty(3, 4))
        attention.g_b_proj = _RecordingLinear(torch.empty(3, 6))
        with (
            patch.object(
                Glm5NextLinearAttention,
                "_maybe_quantize_ptpc_input",
            ) as quantize,
            patch.object(
                Glm5NextLinearAttention,
                "_ptpc_linear_active",
                return_value=True,
            ),
        ):
            attention.forward_qkvbfg(
                (hidden_states, quantized[0], quantized[1]), forward_batch=None
            )
        quantize.assert_not_called()
        for layer in (attention.qkv_proj, attention.f_a_proj, attention.g_a_proj):
            self.assertIs(layer.inputs[0][0], quantized[0])
            self.assertIs(layer.inputs[0][1], quantized[1])
        self.assertIs(attention.b_proj.inputs[0], hidden_states)

    def test_mhc_prequant_capability_requires_every_shared_input_consumer(self):
        attention = Glm5NextLinearAttention.__new__(Glm5NextLinearAttention)
        torch.nn.Module.__init__(attention)
        attention.do_fuse_qkvbfg = False
        attention.fuse_bfg = False
        attention.qkv_proj = object()
        attention.f_a_proj = object()
        attention.g_a_proj = object()
        with patch.object(
            Glm5NextLinearAttention,
            "_ptpc_linear_active",
            side_effect=lambda layer, _: layer is not attention.g_a_proj,
        ):
            self.assertFalse(attention.can_consume_mhc_prequant(4096))
        with patch.object(
            Glm5NextLinearAttention,
            "_ptpc_linear_active",
            return_value=True,
        ):
            self.assertTrue(attention.can_consume_mhc_prequant(4096))

    def test_model_quantization_boundary_and_zero_tokens(self):
        threshold = GLM53_KDA_PTPC_BF16_MAX_M["qkv_proj"]
        method = Glm53KdaPtpcLinearMethod("qkv_proj", bf16_max_m=threshold)
        method._fp8_ptpc_ready = True
        layer = SimpleNamespace(quant_method=method)
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
