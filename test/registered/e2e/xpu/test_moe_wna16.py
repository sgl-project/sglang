"""
Unit tests for XPU MoE WnA16 quantization methods and execution paths.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from compressed_tensors.quantization import QuantizationStrategy

from sglang.kernels.ops.quantization.fp8_utils import ue8m0_to_fp32
from sglang.srt.arg_groups.overrides import _moe_runner_backend_quant_constraints
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import (
    MoeA2ABackend,
    MoeRunnerBackend,
    resolve_moe_runner_backend,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
    CompressedTensorsW8A8Fp8MoE,
)
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod
from sglang.srt.layers.quantization.fp8_utils import (
    can_use_xpu_fp8_moe,
    mxfp8_group_quantize,
)
from sglang.srt.runtime_context import get_context, get_flags, get_parallel
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=15, suite="stage-b-test-1-gpu-xpu")


@triton.jit
def _decode_ue8m0_kernel(scales, output, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(output + offsets, ue8m0_to_fp32(tl.load(scales + offsets)))


class TestXpuMoeWna16(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.xpu.is_available():
            raise unittest.SkipTest("XPU device not available")
        from sgl_kernel import is_xe2_arch

        if not is_xe2_arch():
            raise unittest.SkipTest("FP8 W8A16 MoE tests require an Xe2 XPU")
        cls.device = "xpu"
        cls.num_tokens = 4
        cls.num_experts = 8
        cls.topk = 2
        cls.hidden = 256
        cls.intermediate = 256

    def setUp(self):
        torch.manual_seed(42)
        self.enterContext(
            get_context().override_server_args(model_path="dummy", device="xpu")
        )
        self.enterContext(
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.AUTO, a2a_backend=MoeA2ABackend.NONE
            )
        )
        self.enterContext(get_parallel().override(tp_group=None))

    def _reference_moe(
        self,
        dispatch_output,
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        block_shape=None,
        mxfp8=False,
        b1=None,
        b2=None,
        routed_scaling_factor=None,
        runner_config=None,
    ):
        config = runner_config or MoeRunnerConfig()
        activation_fn = {
            "silu": F.silu,
            "gelu": F.gelu,
            "gelu_tanh": lambda x: F.gelu(x, approximate="tanh"),
            "relu2": lambda x: torch.square(F.relu(x)),
            "situ": lambda x: 4.0 * torch.tanh(x / 4.0) * torch.sigmoid(x),
        }[config.activation]
        x = dispatch_output.hidden_states.float().cpu()
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        topk_weights = topk_weights.float().cpu()
        topk_ids = topk_ids.cpu()
        w13_weight = w13_weight.float().cpu()
        w2_weight = w2_weight.float().cpu()
        w13_scale = w13_scale.cpu() if w13_scale is not None else None
        w2_scale = w2_scale.cpu() if w2_scale is not None else None
        b1_cpu = b1.float().cpu() if b1 is not None else None
        b2_cpu = b2.float().cpu() if b2 is not None else None
        rsf = 1.0 if routed_scaling_factor is None else routed_scaling_factor
        reference = torch.zeros(
            (self.num_tokens, topk_ids.shape[1], self.hidden)
            if config.no_combine
            else (self.num_tokens, self.hidden),
            dtype=torch.float32,
        )

        def reference_gemm(activation, weight, scale, expert_id, bias):
            weight = weight[expert_id]
            scale = scale[expert_id] if scale is not None else None
            expert_bias = bias[expert_id] if bias is not None else 0.0
            if scale is None:
                return activation @ weight.transpose(0, 1) + expert_bias
            if scale.numel() == 1:
                output = activation @ weight.transpose(0, 1)
                return output * scale.item() + expert_bias
            if mxfp8:
                scale = torch.pow(2.0, scale.float() - 127.0)
                output = torch.zeros(weight.shape[0], dtype=torch.float32)
                for group_start in range(0, weight.shape[1], 32):
                    group_end = group_start + 32
                    partial = activation[group_start:group_end] @ weight[
                        :, group_start:group_end
                    ].transpose(0, 1)
                    output += partial * scale[:, group_start // 32]
                return output + expert_bias
            block_n, block_k = block_shape
            expanded_scale = scale.repeat_interleave(block_n, dim=0).repeat_interleave(
                block_k, dim=1
            )[: weight.shape[0], : weight.shape[1]]
            scaled_weight = weight * expanded_scale
            return activation @ scaled_weight.transpose(0, 1) + expert_bias

        for token_id in range(self.num_tokens):
            for topk_id in range(topk_ids.shape[1]):
                expert_id = int(topk_ids[token_id, topk_id])
                expert_input = x[token_id]
                if config.apply_router_weight_on_input:
                    expert_input = expert_input * topk_weights[token_id, topk_id]
                gate_up = reference_gemm(
                    expert_input, w13_weight, w13_scale, expert_id, b1_cpu
                )
                if config.is_gated:
                    gate = gate_up[: self.intermediate]
                    up = gate_up[self.intermediate :]
                    if config.swiglu_limit is not None:
                        gate = activation_fn(gate).clamp(max=config.swiglu_limit)
                        up = up.clamp(min=-config.swiglu_limit, max=config.swiglu_limit)
                        intermediate = gate * up
                    else:
                        intermediate = activation_fn(gate) * up
                else:
                    intermediate = activation_fn(gate_up)
                expert_output = reference_gemm(
                    intermediate, w2_weight, w2_scale, expert_id, b2_cpu
                )
                if config.no_combine:
                    reference[token_id, topk_id] = expert_output
                else:
                    weight = (
                        1.0
                        if config.apply_router_weight_on_input
                        else topk_weights[token_id, topk_id]
                    )
                    reference[token_id] += weight * expert_output * rsf

        return reference

    def _assert_output_matches_reference(self, output, reference, rtol=0.03, atol=0.03):
        self.assertIsInstance(output, StandardCombineInput)
        self.assertEqual(output.hidden_states.shape, reference.shape)
        self.assertEqual(output.hidden_states.dtype, torch.bfloat16)
        torch.testing.assert_close(
            output.hidden_states.float().cpu(), reference, rtol=rtol, atol=atol
        )

    def _create_dispatch_output(self):
        x = (
            torch.randn(
                self.num_tokens, self.hidden, dtype=torch.bfloat16, device=self.device
            )
            * 0.5
        )
        topk_weights = torch.full(
            (self.num_tokens, self.topk), 0.5, dtype=torch.float32, device=self.device
        )
        topk_ids = torch.tensor(
            [[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int32, device=self.device
        )
        return StandardDispatchOutput(
            x, None, StandardTopKOutput(topk_weights, topk_ids, None)
        )

    def _build_moe_layer(
        self,
        method,
        params_dtype=torch.float8_e4m3fn,
        with_bias=False,
        routed_scaling_factor=None,
        runner_config=None,
    ):
        if runner_config is None:
            runner_config = MoeRunnerConfig(routed_scaling_factor=routed_scaling_factor)
        layer = nn.Module()
        layer.moe_tp_size = 1
        layer.moe_runner_config = runner_config
        layer.num_local_experts = self.num_experts
        layer.intermediate_size_per_partition = self.intermediate

        create_kwargs = dict(
            layer=layer,
            num_experts=self.num_experts,
            hidden_size=self.hidden,
            intermediate_size_per_partition=self.intermediate,
            params_dtype=params_dtype,
        )
        if with_bias:
            create_kwargs["with_bias"] = True
        method.create_weights(**create_kwargs)
        method.create_moe_runner(layer, runner_config)

        w13_intermediate = (
            2 * self.intermediate if runner_config.is_gated else self.intermediate
        )
        w13_bf16 = (
            torch.randn(
                self.num_experts,
                w13_intermediate,
                self.hidden,
                dtype=torch.bfloat16,
            )
            * 0.1
        )
        w2_bf16 = (
            torch.randn(
                self.num_experts,
                self.hidden,
                self.intermediate,
                dtype=torch.bfloat16,
            )
            * 0.1
        )
        layer.w13_weight.data.copy_(w13_bf16.to(layer.w13_weight.dtype))
        layer.w2_weight.data.copy_(w2_bf16.to(layer.w2_weight.dtype))
        return layer, w13_bf16, w2_bf16

    def test_moe_runner_backend_is_intel_xpu(self):
        backend = resolve_moe_runner_backend("intel_xpu")
        self.assertTrue(
            backend.is_intel_xpu(),
            "MoeRunnerBackend.INTEL_XPU.is_intel_xpu() must return True",
        )
        view = MagicMock(quantization="mxfp8", moe_runner_backend="auto")
        self.assertEqual(_moe_runner_backend_quant_constraints(view), {})

    def test_fp8_moe_method_scalar_scale(self):
        dispatch_output = self._create_dispatch_output()
        method = Fp8MoEMethod(Fp8Config(is_checkpoint_fp8_serialized=True))
        layer, _, _ = self._build_moe_layer(method)
        layer.w13_weight_scale.data.copy_(
            torch.linspace(
                0.75, 1.25, layer.w13_weight_scale.numel(), dtype=torch.float32
            ).view_as(layer.w13_weight_scale)
        )
        layer.w2_weight_scale.data.copy_(
            torch.linspace(
                1.25, 0.75, layer.w2_weight_scale.numel(), dtype=torch.float32
            ).view_as(layer.w2_weight_scale)
        )

        layer.to(self.device)
        method.process_weights_after_loading(layer)

        reference = self._reference_moe(
            dispatch_output,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )
        output = method.apply(layer, dispatch_output)
        self._assert_output_matches_reference(output, reference)

    def test_fp8_moe_method_block_scale_and_modifiers(self):
        dispatch_output = self._create_dispatch_output()
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[128, 128],
        )
        method = Fp8MoEMethod(quant_config)
        layer, _, _ = self._build_moe_layer(
            method, with_bias=True, routed_scaling_factor=1.5
        )
        layer.w13_weight_scale_inv.data.copy_(
            torch.linspace(
                0.75, 1.25, layer.w13_weight_scale_inv.numel(), dtype=torch.float32
            ).view_as(layer.w13_weight_scale_inv)
        )
        layer.w2_weight_scale_inv.data.copy_(
            torch.linspace(
                1.25, 0.75, layer.w2_weight_scale_inv.numel(), dtype=torch.float32
            ).view_as(layer.w2_weight_scale_inv)
        )
        layer.w13_weight_bias.data.copy_(torch.randn_like(layer.w13_weight_bias) * 0.01)
        layer.w2_weight_bias.data.copy_(torch.randn_like(layer.w2_weight_bias) * 0.01)

        layer.to(self.device)
        method.process_weights_after_loading(layer)

        reference = self._reference_moe(
            dispatch_output,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            block_shape=[128, 128],
            b1=layer.w13_weight_bias,
            b2=layer.w2_weight_bias,
            routed_scaling_factor=1.5,
        )
        output = method.apply(layer, dispatch_output)
        self._assert_output_matches_reference(output, reference)

    def test_fp8_moe_swiglu_limit_native(self):
        from sgl_kernel import fused_experts

        dispatch_output = self._create_dispatch_output()
        config = MoeRunnerConfig(swiglu_limit=10.0)
        method = Fp8MoEMethod(Fp8Config(is_checkpoint_fp8_serialized=True))
        layer, _, _ = self._build_moe_layer(method, runner_config=config)
        layer.to(self.device)
        method.process_weights_after_loading(layer)

        reference = self._reference_moe(
            dispatch_output,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            runner_config=config,
        )
        with patch("sgl_kernel.fused_experts", wraps=fused_experts) as native:
            output = method.apply(layer, dispatch_output)
            native.assert_called_once()
            self.assertEqual(native.call_args.kwargs["swiglu_limit"], 10.0)
        self._assert_output_matches_reference(output, reference)

    def test_mxfp8_moe_method_serialized(self):
        from sgl_kernel import fused_experts

        self.enterContext(
            get_flags().moe.override(runner_backend=MoeRunnerBackend.INTEL_XPU)
        )
        dispatch_output = self._create_dispatch_output()
        quant_config = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            use_mxfp8=True,
            weight_block_size=[1, 32],
        )
        method = Fp8MoEMethod(quant_config)
        layer, _, _ = self._build_moe_layer(method)
        scale_pattern = torch.tensor(
            [126, 127, 128, 127, 126, 128, 127, 126], dtype=torch.uint8
        )
        layer.w13_weight_scale_inv.data.copy_(
            scale_pattern.view(1, 1, -1).expand_as(layer.w13_weight_scale_inv)
        )
        layer.w2_weight_scale_inv.data.copy_(
            scale_pattern.view(1, 1, -1).expand_as(layer.w2_weight_scale_inv)
        )

        layer.to(self.device)
        method.process_weights_after_loading(layer)

        reference = self._reference_moe(
            dispatch_output,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            block_shape=[1, 32],
            mxfp8=True,
        )
        with patch("sgl_kernel.fused_experts", wraps=fused_experts) as native:
            output = method.apply(layer, dispatch_output)
            native.assert_called_once()
            self.assertIsNone(native.call_args.kwargs["block_shape"])
        self._assert_output_matches_reference(output, reference)

    def test_mxfp8_moe_method_online_quant(self):
        empty_q, empty_s = mxfp8_group_quantize(
            torch.empty((0, 64), dtype=torch.bfloat16, device=self.device)
        )
        self.assertEqual(empty_q.shape, (0, 64))
        self.assertEqual(empty_s.shape, (0, 2))

        dispatch_output = self._create_dispatch_output()
        quant_config = Fp8Config(is_checkpoint_fp8_serialized=False, use_mxfp8=True)
        method = Fp8MoEMethod(quant_config)
        layer, w13_bf16, w2_bf16 = self._build_moe_layer(
            method, params_dtype=torch.bfloat16
        )

        layer.to(self.device)
        method.process_weights_after_loading(layer)

        self.assertEqual(layer.w13_weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(layer.w13_weight_scale_inv.dtype, torch.uint8)
        self.assertEqual(
            layer.w13_weight_scale_inv.shape,
            (self.num_experts, 2 * self.intermediate, self.hidden // 32),
        )

        output = method.apply(layer, dispatch_output)
        dequant_ref = self._reference_moe(
            dispatch_output,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            block_shape=[1, 32],
            mxfp8=True,
        )
        bf16_ref = self._reference_moe(dispatch_output, w13_bf16, w2_bf16, None, None)
        self._assert_output_matches_reference(output, dequant_ref)
        self._assert_output_matches_reference(output, bf16_ref, rtol=0.1, atol=0.1)

    def test_compressed_tensors_w8a8_fp8_moe_tensor(self):
        dispatch_output = self._create_dispatch_output()
        weight_quant = MagicMock(strategy=QuantizationStrategy.TENSOR)
        input_quant = MagicMock(strategy=QuantizationStrategy.TENSOR, dynamic=True)
        scheme = CompressedTensorsW8A8Fp8MoE(weight_quant, input_quant)
        layer, _, _ = self._build_moe_layer(scheme)
        layer.w13_weight_scale.data.copy_(
            torch.linspace(
                0.75, 1.25, layer.w13_weight_scale.numel(), dtype=torch.float32
            ).view_as(layer.w13_weight_scale)
        )
        layer.w2_weight_scale.data.copy_(
            torch.linspace(
                1.25, 0.75, layer.w2_weight_scale.numel(), dtype=torch.float32
            ).view_as(layer.w2_weight_scale)
        )

        layer.to(self.device)
        scheme.process_weights_after_loading(layer)

        reference = self._reference_moe(
            dispatch_output,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )
        output = scheme.apply_weights(layer, dispatch_output)
        self._assert_output_matches_reference(output, reference)

    def test_compressed_tensors_w8a8_fp8_moe_block(self):
        dispatch_output = self._create_dispatch_output()
        weight_quant = MagicMock(
            strategy=QuantizationStrategy.BLOCK,
            block_structure=[128, 128],
            dynamic=False,
        )
        input_quant = MagicMock(strategy=QuantizationStrategy.TENSOR, dynamic=True)
        scheme = CompressedTensorsW8A8Fp8MoE(weight_quant, input_quant)
        layer, _, _ = self._build_moe_layer(scheme)
        layer.w13_weight_scale.data.copy_(
            torch.linspace(
                0.75, 1.25, layer.w13_weight_scale.numel(), dtype=torch.float32
            ).view_as(layer.w13_weight_scale)
        )
        layer.w2_weight_scale.data.copy_(
            torch.linspace(
                1.25, 0.75, layer.w2_weight_scale.numel(), dtype=torch.float32
            ).view_as(layer.w2_weight_scale)
        )

        layer.to(self.device)
        scheme.process_weights_after_loading(layer)

        reference = self._reference_moe(
            dispatch_output,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            block_shape=[128, 128],
        )
        output = scheme.apply_weights(layer, dispatch_output)
        self._assert_output_matches_reference(output, reference)

    def test_compressed_tensors_w8a8_fp8_moe_channel_fallback(self):
        dispatch_output = self._create_dispatch_output()
        weight_quant = MagicMock(strategy=QuantizationStrategy.CHANNEL)
        input_quant = MagicMock(strategy=QuantizationStrategy.TOKEN, dynamic=True)
        scheme = CompressedTensorsW8A8Fp8MoE(weight_quant, input_quant)

        with patch(
            "sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe.get_moe_runner_backend",
            return_value=MoeRunnerBackend.INTEL_XPU,
        ):
            layer, _, _ = self._build_moe_layer(scheme)
            self.assertEqual(scheme.runner.runner_backend, MoeRunnerBackend.TRITON)
            with patch.object(
                scheme.runner,
                "run",
                return_value=StandardCombineInput(dispatch_output.hidden_states),
            ) as mock_run:
                scheme.apply_weights(layer, dispatch_output)
                mock_run.assert_called_once()
                quant_info = mock_run.call_args[0][1]
                self.assertTrue(quant_info.per_channel_quant)

        method = Fp8MoEMethod(Fp8Config(is_checkpoint_fp8_serialized=True))
        with patch(
            "sglang.srt.layers.quantization.fp8.get_moe_runner_backend",
            return_value=MoeRunnerBackend.TRITON,
        ):
            fp8_layer, _, _ = self._build_moe_layer(method)
            with patch.object(
                method.runner,
                "run",
                return_value=StandardCombineInput(dispatch_output.hidden_states),
            ) as mock_fp8_run:
                method.apply(fp8_layer, dispatch_output)
                mock_fp8_run.assert_called_once()

    def test_xpu_fp8_moe_activation_support(self):
        cases = (
            ("silu", True, True),
            ("gelu_tanh", True, True),
            ("relu2", False, True),
            ("gelu", True, False),
            ("situ", True, False),
            ("unsupported", True, False),
            ("silu", False, False),
            ("gelu_tanh", False, False),
            ("relu2", True, False),
        )
        for activation, is_gated, expected in cases:
            with self.subTest(activation=activation, is_gated=is_gated):
                config = MoeRunnerConfig(activation=activation, is_gated=is_gated)
                self.assertEqual(can_use_xpu_fp8_moe(config, None), expected)

        default_config = MoeRunnerConfig(activation="silu", is_gated=True)
        # Block size checks
        self.assertTrue(can_use_xpu_fp8_moe(default_config, [1, 32], use_mxfp8=True))
        self.assertFalse(can_use_xpu_fp8_moe(default_config, [32, 32], use_mxfp8=True))
        self.assertTrue(
            can_use_xpu_fp8_moe(default_config, [128, 128], use_mxfp8=False)
        )
        self.assertFalse(can_use_xpu_fp8_moe(default_config, [32, 32], use_mxfp8=False))

        # EP checks
        with get_parallel().override(tp_size=2, moe_ep_size=2, attn_tp_size=2):
            self.assertFalse(can_use_xpu_fp8_moe(default_config, None))
        non_local_config = MoeRunnerConfig(
            activation="silu", is_gated=True, num_experts=8, num_local_experts=4
        )
        self.assertFalse(can_use_xpu_fp8_moe(non_local_config, None))

        # Architecture checks
        with patch("sgl_kernel.is_xe2_arch", return_value=False):
            self.assertFalse(can_use_xpu_fp8_moe(default_config, None))

    def test_compressed_tensors_runner_config_fallback(self):
        from sgl_kernel import fused_experts

        self.enterContext(
            get_flags().moe.override(runner_backend=MoeRunnerBackend.INTEL_XPU)
        )
        configs = (
            MoeRunnerConfig(inplace=False, apply_router_weight_on_input=True),
            MoeRunnerConfig(inplace=False, no_combine=True),
            MoeRunnerConfig(inplace=False, activation="gelu"),
            MoeRunnerConfig(inplace=False, activation="situ"),
        )
        for config in configs:
            with self.subTest(
                activation=config.activation,
                no_combine=config.no_combine,
                apply_router_weight_on_input=config.apply_router_weight_on_input,
            ):
                scheme = CompressedTensorsW8A8Fp8MoE(
                    MagicMock(strategy=QuantizationStrategy.TENSOR),
                    MagicMock(strategy=QuantizationStrategy.TENSOR, dynamic=True),
                )
                layer, _, _ = self._build_moe_layer(scheme, runner_config=config)
                layer.to(self.device)
                scheme.process_weights_after_loading(layer)
                dispatch = self._create_dispatch_output()
                if config.apply_router_weight_on_input:
                    dispatch = StandardDispatchOutput(
                        dispatch.hidden_states * 2,
                        None,
                        StandardTopKOutput(
                            dispatch.topk_output.topk_weights[:, :1].contiguous(),
                            dispatch.topk_output.topk_ids[:, :1].contiguous(),
                            None,
                        ),
                    )
                reference = self._reference_moe(
                    dispatch,
                    layer.w13_weight,
                    layer.w2_weight,
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                    runner_config=config,
                )
                with patch("sgl_kernel.fused_experts", wraps=fused_experts) as native:
                    output = scheme.apply_weights(layer, dispatch)
                    native.assert_not_called()
                self._assert_output_matches_reference(
                    output, reference, rtol=0.1, atol=0.1
                )

    def test_compressed_tensors_block32_fallback(self):
        from sgl_kernel import fused_experts

        scheme = CompressedTensorsW8A8Fp8MoE(
            MagicMock(
                strategy=QuantizationStrategy.BLOCK,
                block_structure=[32, 32],
                dynamic=False,
            ),
            MagicMock(strategy=QuantizationStrategy.TENSOR, dynamic=True),
        )
        layer, _, _ = self._build_moe_layer(scheme)
        layer.to(self.device)
        scheme.process_weights_after_loading(layer)
        dispatch = self._create_dispatch_output()
        reference = self._reference_moe(
            dispatch,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            block_shape=[32, 32],
        )
        with patch("sgl_kernel.fused_experts", wraps=fused_experts) as native:
            output = scheme.apply_weights(layer, dispatch)
            native.assert_not_called()
        self._assert_output_matches_reference(output, reference, rtol=0.1, atol=0.1)

    def test_mxfp8_default_config_device_scope(self):
        from sglang.srt.layers.moe.moe_runner.triton_utils import (
            fused_moe_triton_config as configs,
        )

        for is_xpu, is_hip, expected_n in (
            (True, False, 16),
            (False, False, 1),
            (False, True, 1),
        ):
            with self.subTest(is_xpu=is_xpu, is_hip=is_hip):
                with patch.multiple(configs, _is_xpu=is_xpu, _is_hip=is_hip):
                    config = configs.get_default_config(
                        self.num_tokens,
                        self.num_experts,
                        self.intermediate,
                        self.hidden,
                        self.topk,
                        "fp8_w8a8",
                        False,
                        block_shape=[1, 32],
                    )
                    self.assertEqual(config["BLOCK_SIZE_N"], expected_n)

    def test_mxfp8_scale_decode(self):
        scales = torch.arange(256, dtype=torch.int16).to(torch.uint8)
        reference = scales.view(torch.float8_e8m0fnu).float()
        output = torch.empty(256, dtype=torch.float32, device=self.device)
        _decode_ue8m0_kernel[(1,)](scales.to(self.device), output, BLOCK_SIZE=256)
        torch.testing.assert_close(
            output.cpu(), reference, rtol=0, atol=0, equal_nan=True
        )

    def test_mxfp8_fallback(self):
        from sgl_kernel import fused_experts

        # Exercise 32-element groups without 128-element block alignment.
        self.hidden = 288
        self.intermediate = 160
        cases = (
            (MoeRunnerBackend.TRITON, {}),
            (MoeRunnerBackend.TRITON, dict(activation="gelu")),
            (MoeRunnerBackend.AUTO, dict(no_combine=True)),
            (MoeRunnerBackend.AUTO, dict(activation="gelu")),
            (MoeRunnerBackend.INTEL_XPU, dict(activation="gelu")),
            (MoeRunnerBackend.INTEL_XPU, dict(apply_router_weight_on_input=True)),
        )
        for backend, settings in cases:
            with self.subTest(backend=backend, **settings):
                with get_flags().moe.override(runner_backend=backend):
                    serialized = backend.is_triton()
                    method = Fp8MoEMethod(
                        Fp8Config(
                            is_checkpoint_fp8_serialized=serialized, use_mxfp8=True
                        )
                    )
                    config = MoeRunnerConfig(inplace=False, **settings)
                    layer, w13_bf16, w2_bf16 = self._build_moe_layer(
                        method,
                        params_dtype=(
                            torch.float8_e4m3fn if serialized else torch.bfloat16
                        ),
                        runner_config=config,
                    )
                    layer.to(self.device)
                    if serialized:
                        for weight, scale, original in (
                            (
                                layer.w13_weight,
                                layer.w13_weight_scale_inv,
                                w13_bf16,
                            ),
                            (
                                layer.w2_weight,
                                layer.w2_weight_scale_inv,
                                w2_bf16,
                            ),
                        ):
                            qweight, qscale = mxfp8_group_quantize(
                                original.to(self.device).flatten(0, 1)
                            )
                            weight.data.copy_(qweight.view_as(weight))
                            scale.data.copy_(qscale.view_as(scale))
                    method.process_weights_after_loading(layer)
                    self.assertTrue(method.runner.runner_backend.is_triton())
                    dispatch = self._create_dispatch_output()
                    if config.apply_router_weight_on_input:
                        topk = dispatch.topk_output
                        dispatch = StandardDispatchOutput(
                            dispatch.hidden_states * 2,
                            None,
                            StandardTopKOutput(
                                topk.topk_weights[:, :1].contiguous(),
                                topk.topk_ids[:, :1].contiguous(),
                                None,
                            ),
                        )
                    reference = self._reference_moe(
                        dispatch,
                        layer.w13_weight,
                        layer.w2_weight,
                        layer.w13_weight_scale_inv,
                        layer.w2_weight_scale_inv,
                        block_shape=[1, 32],
                        mxfp8=True,
                        runner_config=config,
                    )
                    with patch(
                        "sgl_kernel.fused_experts", wraps=fused_experts
                    ) as native:
                        output = method.apply(layer, dispatch)
                        native.assert_not_called()
                    self._assert_output_matches_reference(
                        output, reference, rtol=0.1, atol=0.1
                    )

    def test_xpu_native_backends_and_gelu_tanh(self):
        from sgl_kernel import fused_experts

        for backend in (MoeRunnerBackend.AUTO, MoeRunnerBackend.INTEL_XPU):
            with get_flags().moe.override(runner_backend=backend):
                methods = (
                    Fp8MoEMethod(Fp8Config(is_checkpoint_fp8_serialized=True)),
                    CompressedTensorsW8A8Fp8MoE(
                        MagicMock(strategy=QuantizationStrategy.TENSOR),
                        MagicMock(strategy=QuantizationStrategy.TENSOR, dynamic=True),
                    ),
                )
                for method in methods:
                    with self.subTest(backend=backend, method=type(method).__name__):
                        config = MoeRunnerConfig(activation="gelu_tanh")
                        layer, _, _ = self._build_moe_layer(
                            method,
                            runner_config=config,
                        )
                        layer.to(self.device)
                        method.process_weights_after_loading(layer)
                        dispatch = self._create_dispatch_output()
                        reference = self._reference_moe(
                            dispatch,
                            layer.w13_weight,
                            layer.w2_weight,
                            layer.w13_weight_scale,
                            layer.w2_weight_scale,
                            runner_config=config,
                        )
                        with patch(
                            "sgl_kernel.fused_experts", wraps=fused_experts
                        ) as native:
                            output = (
                                method.apply_weights(layer, dispatch)
                                if isinstance(method, CompressedTensorsW8A8Fp8MoE)
                                else method.apply(layer, dispatch)
                            )
                            native.assert_called_once()
                            self.assertEqual(
                                native.call_args.kwargs["activation"], "gelu"
                            )
                        self._assert_output_matches_reference(output, reference)

    def test_fp8_moe_relu2_native(self):
        from sgl_kernel import fused_experts

        dispatch_output = self._create_dispatch_output()
        config = MoeRunnerConfig(activation="relu2", is_gated=False)
        method = Fp8MoEMethod(Fp8Config(is_checkpoint_fp8_serialized=True))
        layer, _, _ = self._build_moe_layer(method, runner_config=config)
        layer.to(self.device)
        method.process_weights_after_loading(layer)

        reference = self._reference_moe(
            dispatch_output,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
            runner_config=config,
        )
        with patch("sgl_kernel.fused_experts", wraps=fused_experts) as native:
            output = method.apply(layer, dispatch_output)
            native.assert_called_once()
            self.assertEqual(native.call_args.kwargs["activation"], "relu2")
        self._assert_output_matches_reference(output, reference)


if __name__ == "__main__":
    unittest.main()
