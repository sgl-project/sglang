import inspect
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

# Load the quantization package first so `base_config`, `moe_methods`, and
# `linear_method_npu` initialize in dependency order. Importing `fp4_moe_methods`
# (or `linear_method_npu`) directly from a cold process triggers a circular
# import: linear_method_npu -> base_config -> quantization/__init__ ->
# gguf/unquant/gptq_moe -> moe_methods -> linear_method_npu (partially
# initialized, `_get_float8_e8m0fnu_dtype` not yet defined). Initializing the
# package first mirrors how the engine loads quantization at model-config time.
import sglang.srt.layers.quantization  # noqa: F401
from sglang.srt.hardware_backend.npu.quantization import fp4_moe_methods
from sglang.srt.hardware_backend.npu.quantization.fp4_moe_methods import (
    NPUW4A4Fp4MoEMethod,
    _apply_swiglu_limit_npu,
    _configure_dsv4_deepep_dispatcher,
    _pair_pack_mxfp_act_scale,
    _reshape_mxfp4_scale_for_npu,
    npu_apply_without_routing_weights_w4a4_mxfp,
    w4a8_mxfp_gmm,
)
from sglang.srt.environ import envs
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.token_dispatcher import deepep
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod


_NOT_PASSED = object()


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


class TestApplySwiGLULimitNpu(unittest.TestCase):
    def test_clamps_gate_and_up_asymmetrically(self):
        # DeepSeek-V4 clamps gate (first half) to <= limit but only the upper
        # bound, while up (second half) is clamped symmetrically to [-limit, limit].
        # A regression that swapped these would silently change expert activations.
        gate_up = torch.tensor([[8.0, -9.0, 9.0, -9.0]])
        _apply_swiglu_limit_npu(gate_up, 7.0)
        self.assertTrue(torch.equal(gate_up, torch.tensor([[7.0, -9.0, 7.0, -7.0]])))

    def test_noop_when_limit_none(self):
        gate_up = torch.tensor([[8.0, -9.0]])
        _apply_swiglu_limit_npu(gate_up, None)
        self.assertTrue(torch.equal(gate_up, torch.tensor([[8.0, -9.0]])))

    def test_noop_when_limit_nonpositive(self):
        gate_up = torch.tensor([[8.0, -9.0]])
        _apply_swiglu_limit_npu(gate_up, 0.0)
        self.assertTrue(torch.equal(gate_up, torch.tensor([[8.0, -9.0]])))


class TestReshapeMxfp4ScaleForNpu(unittest.TestCase):
    def test_packs_scale_to_gmm_layout(self):
        # [E, N, K/32] -> [E, K/64, N, 2] is the packed-pair layout the GMM reads;
        # getting the transpose axis wrong silently dequantizes with the wrong scale.
        scale = torch.arange(8, dtype=torch.uint8).view(1, 2, 4)
        out = _reshape_mxfp4_scale_for_npu(scale)
        self.assertEqual(tuple(out.shape), (1, 2, 2, 2))
        self.assertTrue(torch.equal(out, scale.view(1, 2, 2, 2).transpose(1, 2)))

    def test_rejects_odd_k_dim(self):
        with self.assertRaises(ValueError):
            _reshape_mxfp4_scale_for_npu(torch.zeros(1, 2, 3, dtype=torch.uint8))


class TestMxfp4ScaleWeightLoader(unittest.TestCase):
    def test_reinterprets_e8m0_scale_as_raw_uint8(self):
        loaded = []

        def weight_loader(param, loaded_weight, *args, **kwargs):
            loaded.append(loaded_weight.clone())

        layer = torch.nn.Module()
        method = NPUW4A4Fp4MoEMethod(fp8_method=MagicMock(), prefix="test")
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


class TestPairPackMxfpActScale(unittest.TestCase):
    def test_packs_as_view(self):
        # The GMM expects a pair-packed *view* of the per-token scale, not a copy;
        # materializing a copy here would break the kernel's aliasing contract.
        flat = torch.arange(8).view(2, 4)
        packed = _pair_pack_mxfp_act_scale(flat)
        self.assertEqual(tuple(packed.shape), (2, 2, 2))
        self.assertEqual(packed.data_ptr(), flat.data_ptr())

    def test_rejects_odd_scale_dim(self):
        with self.assertRaises(ValueError):
            _pair_pack_mxfp_act_scale(torch.zeros(2, 3))

    def test_unflattens_low_latency_deepep_scale_as_view(self):
        # DeepEP returns one flat E8M0 scale per 32-element block.  Passing
        # that flat buffer to GMM would use the wrong scale layout and either
        # fail or dequantize activations incorrectly.
        flat = torch.arange(4, dtype=torch.uint8)
        packed = _pair_pack_mxfp_act_scale(flat, input_shape=(2, 64))

        self.assertEqual(tuple(packed.shape), (2, 1, 2))
        self.assertEqual(packed.data_ptr(), flat.data_ptr())
        self.assertTrue(torch.equal(packed, torch.tensor([[[0, 1]], [[2, 3]]])))

    def test_rejects_low_latency_deepep_scale_with_wrong_length(self):
        with self.assertRaises(ValueError):
            _pair_pack_mxfp_act_scale(
                torch.zeros(3, dtype=torch.uint8), input_shape=(2, 64)
            )


class TestDsv4DeepEPMxfp8DispatcherConfig(unittest.TestCase):
    @staticmethod
    def _deepep_backend():
        return SimpleNamespace(is_deepep=lambda: True)

    def test_a5_deepep_defaults_low_latency_dispatch_to_mxfp8(self):
        dispatcher = MagicMock()
        layer = SimpleNamespace(dispatcher=dispatcher)

        with (
            patch.object(fp4_moe_methods, "is_npu_arch35", return_value=True),
            patch(
                "sglang.srt.layers.moe.get_moe_a2a_backend",
                return_value=self._deepep_backend(),
            ),
            patch.dict(os.environ, {}, clear=True),
        ):
            _configure_dsv4_deepep_dispatcher(layer)

        dispatcher.set_quant_config.assert_called_once_with(
            {
                "normal_dispatcher_output_dtype": "bf16",
                "low_latency_dispatcher_output_dtype": "mxfp8",
            }
        )

    def test_non_deepep_ignores_the_low_latency_quant_environment(self):
        dispatcher = MagicMock()
        layer = SimpleNamespace(dispatcher=dispatcher)

        with (
            patch.object(fp4_moe_methods, "is_npu_arch35", return_value=True),
            patch(
                "sglang.srt.layers.moe.get_moe_a2a_backend",
                return_value=SimpleNamespace(is_deepep=lambda: False),
            ),
            envs.SGLANG_NPU_DSV4_DEEPEP_LL_DISPATCH_QUANT_MODE.override("invalid"),
        ):
            _configure_dsv4_deepep_dispatcher(layer)

        dispatcher.set_quant_config.assert_called_once_with(
            {"dispatcher_output_dtype": "bf16"}
        )

    def test_a5_deepep_allows_bf16_low_latency_fallback(self):
        dispatcher = MagicMock()
        layer = SimpleNamespace(dispatcher=dispatcher)

        with (
            patch.object(fp4_moe_methods, "is_npu_arch35", return_value=True),
            patch(
                "sglang.srt.layers.moe.get_moe_a2a_backend",
                return_value=self._deepep_backend(),
            ),
            envs.SGLANG_NPU_DSV4_DEEPEP_LL_DISPATCH_QUANT_MODE.override("bf16"),
        ):
            _configure_dsv4_deepep_dispatcher(layer)

        dispatcher.set_quant_config.assert_called_once_with(
            {
                "normal_dispatcher_output_dtype": "bf16",
                "low_latency_dispatcher_output_dtype": "bf16",
            }
        )

    def test_a5_deepep_rejects_an_invalid_low_latency_quant_mode(self):
        layer = SimpleNamespace(dispatcher=MagicMock())

        with (
            patch.object(fp4_moe_methods, "is_npu_arch35", return_value=True),
            patch(
                "sglang.srt.layers.moe.get_moe_a2a_backend",
                return_value=self._deepep_backend(),
            ),
            envs.SGLANG_NPU_DSV4_DEEPEP_LL_DISPATCH_QUANT_MODE.override("invalid"),
            self.assertRaisesRegex(ValueError, "SGLANG_NPU_DSV4"),
        ):
            _configure_dsv4_deepep_dispatcher(layer)

    def test_non_a5_ignores_the_low_latency_quant_environment(self):
        dispatcher = MagicMock()
        layer = SimpleNamespace(dispatcher=dispatcher)

        with (
            patch.object(fp4_moe_methods, "is_npu_arch35", return_value=False),
            patch(
                "sglang.srt.layers.moe.get_moe_a2a_backend",
                return_value=self._deepep_backend(),
            ),
            envs.SGLANG_NPU_DSV4_DEEPEP_LL_DISPATCH_QUANT_MODE.override("invalid"),
        ):
            _configure_dsv4_deepep_dispatcher(layer)

        dispatcher.set_quant_config.assert_called_once_with(
            {"dispatcher_output_dtype": "bf16"}
        )


class _LowLatencyBuffer:
    def __init__(self):
        self.kwargs = None

    def low_latency_dispatch(
        self,
        hidden_states,
        topk_ids,
        num_max_dispatch_tokens_per_rank,
        num_experts,
        *,
        use_fp8,
        quant_mode=_NOT_PASSED,
        **kwargs,
    ):
        self.kwargs = {"use_fp8": use_fp8, "quant_mode": quant_mode, **kwargs}
        return torch.empty(0), torch.empty(0), object(), object(), object()


class _LegacyLowLatencyBuffer:
    def low_latency_dispatch(
        self,
        hidden_states,
        topk_ids,
        num_max_dispatch_tokens_per_rank,
        num_experts,
        *,
        use_fp8,
        **kwargs,
    ):
        return torch.empty(0), torch.empty(0), object(), object(), object()


class TestDeepEPLowLatencyMxfp8Dispatch(unittest.TestCase):
    @staticmethod
    def _dispatcher(quant_mode, buffer):
        dispatcher = object.__new__(deepep._DeepEPDispatcherImplLowLatency)
        dispatcher.quant_config = {}
        dispatcher.use_fp8 = False
        dispatcher.use_nvfp4 = False
        dispatcher.low_latency_quant_mode = quant_mode
        dispatcher._low_latency_quant_mode_runtime_checked = False
        dispatcher.num_max_dispatch_tokens_per_rank = 2
        dispatcher.num_experts = 2
        dispatcher.return_recv_hook = False
        dispatcher._get_buffer = lambda: buffer
        return dispatcher

    def test_mxfp8_passes_the_kernel_quant_mode(self):
        buffer = _LowLatencyBuffer()
        dispatcher = self._dispatcher("mx_fp8_e4m3", buffer)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(deepep, "_deepep_precompile_tp_barrier"),
        ):
            dispatcher._dispatch_core(
                torch.zeros(1, 64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1, 1),
            )

        self.assertEqual(buffer.kwargs["quant_mode"], "mx_fp8_e4m3")

    def test_mxfp8_ops_strategy_uses_legacy_mxfp8_flags(self):
        buffer = _LowLatencyBuffer()
        dispatcher = self._dispatcher("mx_fp8_e4m3", buffer)

        with (
            patch.dict(os.environ, {"DEEP_USE_MODE": "ops"}, clear=True),
            patch.object(deepep, "_deepep_precompile_tp_barrier"),
        ):
            dispatcher._dispatch_core(
                torch.zeros(1, 64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1, 1),
            )

        self.assertTrue(buffer.kwargs["use_fp8"])
        self.assertTrue(buffer.kwargs["use_ue8m0"])
        self.assertEqual(buffer.kwargs["quant_mode"], "mx_fp8_e4m3")

    def test_mxfp8_rejects_an_unsupported_low_latency_strategy(self):
        dispatcher = self._dispatcher("mx_fp8_e4m3", _LowLatencyBuffer())

        with (
            patch.dict(os.environ, {"DEEP_USE_MODE": "alltoall"}, clear=True),
            self.assertRaisesRegex(RuntimeError, "DEEP_USE_MODE"),
        ):
            dispatcher._dispatch_core(
                torch.zeros(1, 64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1, 1),
            )

    def test_mxfp8_checks_runtime_interface_once_per_dispatcher(self):
        buffer = _LowLatencyBuffer()
        dispatcher = self._dispatcher("mx_fp8_e4m3", buffer)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(deepep, "_deepep_precompile_tp_barrier"),
            patch.object(
                deepep.inspect, "signature", wraps=inspect.signature
            ) as signature,
        ):
            dispatcher._dispatch_core(
                torch.zeros(1, 64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1, 1),
            )
            dispatcher._dispatch_core(
                torch.zeros(1, 64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1, 1),
            )

        self.assertEqual(signature.call_count, 1)

    def test_bf16_does_not_pass_a_quant_mode(self):
        buffer = _LowLatencyBuffer()
        dispatcher = self._dispatcher(None, buffer)

        with patch.object(deepep, "_deepep_precompile_tp_barrier"):
            dispatcher._dispatch_core(
                torch.zeros(1, 64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1, 1),
            )

        self.assertIs(buffer.kwargs["quant_mode"], _NOT_PASSED)

    def test_mxfp8_rejects_legacy_runtime_without_quant_mode(self):
        dispatcher = self._dispatcher("mx_fp8_e4m3", _LegacyLowLatencyBuffer())

        with self.assertRaisesRegex(RuntimeError, "quant_mode"):
            dispatcher._dispatch_core(
                torch.zeros(1, 64),
                torch.zeros(1, 1, dtype=torch.int64),
                torch.ones(1, 1),
            )


class TestW4A8MxfpGmmInputScale(unittest.TestCase):
    def setUp(self):
        self.input = torch.randn(2, 64)
        self.input_scale = torch.ones(2, 1, 2)
        self.weight = torch.empty(2, 64, 32, dtype=torch.uint8)
        self.weight_scale = torch.ones(2, 1, 32, 2, dtype=torch.uint8)
        self.group_list = torch.tensor([1, 1], dtype=torch.int32)

    def _call_gmm(self, input_scale):
        return w4a8_mxfp_gmm(
            input=self.input,
            input_scale=input_scale,
            weight=self.weight,
            weight_scale=self.weight_scale,
            group_list_type=1,
            group_list=self.group_list,
            output_dtype=torch.bfloat16,
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
            output = self._call_gmm(self.input_scale)

        dynamic_quant.assert_not_called()
        self.assertIs(output, expected)
        call_kwargs = grouped_matmul.call_args.kwargs
        self.assertIs(call_kwargs["per_token_scale"][0], self.input_scale)
        self.assertEqual(call_kwargs["group_list"].dtype, torch.int64)
        self.assertTrue(torch.equal(call_kwargs["group_list"], self.group_list))

    def test_flat_deepep_scale_skips_dynamic_quant_after_layout_adaptation(self):
        flat_scale = torch.arange(4, dtype=torch.uint8)
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
            output = self._call_gmm(flat_scale)

        dynamic_quant.assert_not_called()
        self.assertIs(output, expected)
        packed_scale = grouped_matmul.call_args.kwargs["per_token_scale"][0]
        self.assertEqual(tuple(packed_scale.shape), (2, 1, 2))
        self.assertEqual(packed_scale.data_ptr(), flat_scale.data_ptr())

    def test_missing_scale_uses_dynamic_quant(self):
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
            output = self._call_gmm(None)

        dynamic_quant.assert_called_once()
        self.assertIs(output, expected)
        self.assertIs(
            grouped_matmul.call_args.kwargs["per_token_scale"][0], quantized_scale
        )


class TestW4A8MxfpGmmChain(unittest.TestCase):
    def test_applies_swiglu_limit_before_swiglu(self):
        # The clamp must be wired in *before* npu_swiglu using the layer's
        # configured swiglu_limit; dropping the clamp (or applying it after)
        # silently changes routed-expert output on near-limit activations.
        gate_up = torch.tensor([[8.0, -9.0, 9.0, -9.0]])
        activated = torch.randn(1, 2)
        expected = torch.randn(1, 2)
        layer = SimpleNamespace(
            w13_weight=MagicMock(),
            w13_weight_scale_inv=MagicMock(),
            w2_weight=MagicMock(),
            w2_weight_scale_inv=MagicMock(),
            moe_runner_config=SimpleNamespace(swiglu_limit=7.0),
        )

        with (
            patch.object(
                fp4_moe_methods, "w4a8_mxfp_gmm", side_effect=[gate_up, expected]
            ) as gmm,
            patch.object(
                torch.ops.npu, "npu_swiglu", return_value=activated, create=True
            ) as swiglu,
        ):
            output = npu_apply_without_routing_weights_w4a4_mxfp(
                layer,
                torch.randn(1, 4),
                torch.ones(1, 1, 2),
                group_list_type=1,
                group_list=torch.tensor([1], dtype=torch.int64),
                output_dtype=torch.bfloat16,
            )

        self.assertIs(output, expected)
        self.assertTrue(
            torch.equal(
                swiglu.call_args.args[0], torch.tensor([[7.0, -9.0, 7.0, -7.0]])
            )
        )
        self.assertIs(gmm.call_args_list[1].kwargs["input"], activated)


class TestProcessWeightsAfterLoadingZeroScale(unittest.TestCase):
    @staticmethod
    def _method():
        return NPUW4A4Fp4MoEMethod(fp8_method=MagicMock(), prefix="test")

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
