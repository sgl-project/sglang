"""
Unit tests for XPU MoE WnA16 quantization methods and execution paths.
"""

import unittest
from unittest.mock import MagicMock

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.token_dispatcher import (
    StandardCombineInput,
    StandardDispatchOutput,
)
from sglang.srt.layers.moe.topk import StandardTopKOutput
from sglang.srt.layers.moe.utils import resolve_moe_runner_backend
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import (
    CompressedTensorsW8A8Fp8MoE,
    QuantizationStrategy,
)
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod
from sglang.srt.layers.quantization.w8a8_fp8 import W8A8FP8MoEMethod
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=15, suite="stage-b-test-1-gpu-xpu")


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

    def _reference_moe(
        self,
        dispatch_output,
        w13_weight,
        w2_weight,
        w13_scale,
        w2_scale,
        block_shape=None,
        mxfp8=False,
    ):
        x = dispatch_output.hidden_states.float().cpu()
        topk_weights, topk_ids, _ = dispatch_output.topk_output
        topk_weights = topk_weights.float().cpu()
        topk_ids = topk_ids.cpu()
        w13_weight = w13_weight.float().cpu()
        w2_weight = w2_weight.float().cpu()
        w13_scale = w13_scale.cpu() if w13_scale is not None else None
        w2_scale = w2_scale.cpu() if w2_scale is not None else None
        reference = torch.zeros(self.num_tokens, self.hidden, dtype=torch.float32)

        def reference_gemm(activation, weight, scale, expert_id):
            weight = weight[expert_id]
            scale = scale[expert_id] if scale is not None else None
            if scale is None:
                return activation @ weight.transpose(0, 1)
            if scale.numel() == 1:
                output = activation @ weight.transpose(0, 1)
                return output * scale.item()
            if mxfp8:
                scale = torch.pow(2.0, scale.float() - 127.0)
                output = torch.zeros(weight.shape[0], dtype=torch.float32)
                for group_start in range(0, weight.shape[1], 32):
                    group_end = group_start + 32
                    partial = activation[group_start:group_end] @ weight[
                        :, group_start:group_end
                    ].transpose(0, 1)
                    output += partial * scale[:, group_start // 32]
                return output
            block_n, block_k = block_shape
            expanded_scale = scale.repeat_interleave(block_n, dim=0).repeat_interleave(
                block_k, dim=1
            )[: weight.shape[0], : weight.shape[1]]
            scaled_weight = weight * expanded_scale
            return activation @ scaled_weight.transpose(0, 1)

        for token_id in range(self.num_tokens):
            for topk_id in range(self.topk):
                expert_id = int(topk_ids[token_id, topk_id])
                gate_up = reference_gemm(x[token_id], w13_weight, w13_scale, expert_id)
                intermediate = (
                    F.silu(gate_up[: self.intermediate]) * gate_up[self.intermediate :]
                )
                expert_output = reference_gemm(
                    intermediate, w2_weight, w2_scale, expert_id
                )
                reference[token_id] += topk_weights[token_id, topk_id] * expert_output

        return reference

    def _assert_output_matches_reference(self, output, reference):
        self.assertIsInstance(output, StandardCombineInput)
        self.assertEqual(output.hidden_states.shape, (self.num_tokens, self.hidden))
        self.assertEqual(output.hidden_states.dtype, torch.bfloat16)
        torch.testing.assert_close(
            output.hidden_states.float().cpu(), reference, rtol=0.01, atol=0.01
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

    def test_moe_runner_backend_is_intel_xpu(self):
        backend = resolve_moe_runner_backend("intel_xpu")
        self.assertTrue(
            backend.is_intel_xpu(),
            "MoeRunnerBackend.INTEL_XPU.is_intel_xpu() must return True",
        )

    def test_fp8_moe_method_scalar_scale(self):
        dispatch_output = self._create_dispatch_output()
        quant_config = MagicMock(
            is_checkpoint_fp8_serialized=False,
            is_fp4_experts=False,
            use_mxfp8=False,
            weight_block_size=None,
        )
        method = Fp8MoEMethod(quant_config)
        layer = nn.Module()
        method.create_moe_runner(layer, MoeRunnerConfig())

        layer.w13_weight = (
            torch.randn(
                self.num_experts,
                2 * self.intermediate,
                self.hidden,
                dtype=torch.bfloat16,
                device=self.device,
            )
            .mul_(0.1)
            .to(torch.float8_e4m3fn)
        )
        layer.w2_weight = (
            torch.randn(
                self.num_experts,
                self.hidden,
                self.intermediate,
                dtype=torch.bfloat16,
                device=self.device,
            )
            .mul_(0.1)
            .to(torch.float8_e4m3fn)
        )
        layer.w13_weight_scale = torch.ones(
            self.num_experts, 1, dtype=torch.float32, device=self.device
        )
        layer.w13_weight_scale.copy_(
            torch.linspace(0.75, 1.25, self.num_experts, device=self.device).view(-1, 1)
        )
        layer.w2_weight_scale = torch.linspace(
            1.25, 0.75, self.num_experts, dtype=torch.float32, device=self.device
        ).view(-1, 1)

        reference = self._reference_moe(
            dispatch_output,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )
        output = method.apply(layer, dispatch_output)
        self._assert_output_matches_reference(output, reference)

    def test_fp8_moe_method_block_scale(self):
        dispatch_output = self._create_dispatch_output()
        block_quant_config = MagicMock(
            is_checkpoint_fp8_serialized=False,
            is_fp4_experts=False,
            use_mxfp8=False,
            weight_block_size=[128, 128],
        )
        method = Fp8MoEMethod(block_quant_config)
        layer = nn.Module()
        method.create_moe_runner(layer, MoeRunnerConfig())

        layer.w13_weight = (
            torch.randn(
                self.num_experts,
                2 * self.intermediate,
                self.hidden,
                dtype=torch.bfloat16,
                device=self.device,
            )
            .mul_(0.1)
            .to(torch.float8_e4m3fn)
        )
        layer.w2_weight = (
            torch.randn(
                self.num_experts,
                self.hidden,
                self.intermediate,
                dtype=torch.bfloat16,
                device=self.device,
            )
            .mul_(0.1)
            .to(torch.float8_e4m3fn)
        )
        layer.w13_weight_scale_inv = torch.linspace(
            0.75,
            1.25,
            self.num_experts * 4 * 2,
            dtype=torch.float32,
            device=self.device,
        ).view(self.num_experts, 4, 2)
        layer.w2_weight_scale_inv = torch.linspace(
            1.25,
            0.75,
            self.num_experts * 2 * 2,
            dtype=torch.float32,
            device=self.device,
        ).view(self.num_experts, 2, 2)

        reference = self._reference_moe(
            dispatch_output,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            block_shape=[128, 128],
        )
        output = method.apply(layer, dispatch_output)
        self._assert_output_matches_reference(output, reference)

    def test_mxfp8_moe_method(self):
        dispatch_output = self._create_dispatch_output()
        mxfp8_quant_config = MagicMock(
            is_checkpoint_fp8_serialized=True,
            is_fp4_experts=False,
            use_mxfp8=True,
            weight_block_size=[1, 32],
        )
        method = Fp8MoEMethod(mxfp8_quant_config)
        layer = nn.Module()
        method.create_moe_runner(layer, MoeRunnerConfig())

        layer.w13_weight = (
            torch.randn(
                self.num_experts,
                2 * self.intermediate,
                self.hidden,
                dtype=torch.bfloat16,
                device=self.device,
            )
            .mul_(0.1)
            .to(torch.float8_e4m3fn)
        )
        layer.w2_weight = (
            torch.randn(
                self.num_experts,
                self.hidden,
                self.intermediate,
                dtype=torch.bfloat16,
                device=self.device,
            )
            .mul_(0.1)
            .to(torch.float8_e4m3fn)
        )
        scale_pattern = torch.tensor(
            [126, 127, 128, 127, 126, 128, 127, 126],
            dtype=torch.uint8,
            device=self.device,
        )
        layer.w13_weight_scale_inv = (
            scale_pattern.view(1, 1, -1)
            .expand(self.num_experts, 2 * self.intermediate, self.hidden // 32)
            .contiguous()
        )
        layer.w2_weight_scale_inv = (
            scale_pattern.view(1, 1, -1)
            .expand(self.num_experts, self.hidden, self.intermediate // 32)
            .contiguous()
        )

        reference = self._reference_moe(
            dispatch_output,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale_inv,
            layer.w2_weight_scale_inv,
            block_shape=[1, 32],
            mxfp8=True,
        )
        output = method.apply(layer, dispatch_output)
        self._assert_output_matches_reference(output, reference)

    def test_w8a8_fp8_moe_method(self):
        dispatch_output = self._create_dispatch_output()
        quant_config = MagicMock(
            is_checkpoint_fp8_serialized=False,
            is_fp4_experts=False,
            use_mxfp8=False,
            weight_block_size=None,
        )
        method = W8A8FP8MoEMethod(quant_config)
        layer = nn.Module()
        method.create_moe_runner(layer, MoeRunnerConfig())

        layer.w13_weight = (
            torch.randn(
                self.num_experts,
                2 * self.intermediate,
                self.hidden,
                dtype=torch.bfloat16,
                device=self.device,
            )
            .mul_(0.1)
            .to(torch.float8_e4m3fn)
        )
        layer.w2_weight = (
            torch.randn(
                self.num_experts,
                self.hidden,
                self.intermediate,
                dtype=torch.bfloat16,
                device=self.device,
            )
            .mul_(0.1)
            .to(torch.float8_e4m3fn)
        )
        layer.w13_weight_scale = torch.ones(
            self.num_experts, 1, dtype=torch.float32, device=self.device
        )
        layer.w13_weight_scale.copy_(
            torch.linspace(0.75, 1.25, self.num_experts, device=self.device).view(-1, 1)
        )
        layer.w2_weight_scale = torch.linspace(
            1.25, 0.75, self.num_experts, dtype=torch.float32, device=self.device
        ).view(-1, 1)

        reference = self._reference_moe(
            dispatch_output,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )
        output = method.apply(layer, dispatch_output)
        self._assert_output_matches_reference(output, reference)

    def test_compressed_tensors_w8a8_fp8_moe(self):
        dispatch_output = self._create_dispatch_output()
        weight_quant = MagicMock(strategy=QuantizationStrategy.TENSOR)
        input_quant = MagicMock(strategy=QuantizationStrategy.TENSOR, dynamic=True)
        scheme = CompressedTensorsW8A8Fp8MoE(weight_quant, input_quant)
        scheme.moe_runner_config = MoeRunnerConfig()
        layer = nn.Module()

        layer.w13_weight = (
            torch.randn(
                self.num_experts,
                2 * self.intermediate,
                self.hidden,
                dtype=torch.bfloat16,
                device=self.device,
            )
            .mul_(0.1)
            .to(torch.float8_e4m3fn)
        )
        layer.w2_weight = (
            torch.randn(
                self.num_experts,
                self.hidden,
                self.intermediate,
                dtype=torch.bfloat16,
                device=self.device,
            )
            .mul_(0.1)
            .to(torch.float8_e4m3fn)
        )
        layer.w13_weight_scale = torch.ones(
            self.num_experts, 1, dtype=torch.float32, device=self.device
        )
        layer.w2_weight_scale = torch.ones(
            self.num_experts, 1, dtype=torch.float32, device=self.device
        )
        layer.w13_weight_scale.copy_(
            torch.linspace(0.75, 1.25, self.num_experts, device=self.device).view(-1, 1)
        )
        layer.w2_weight_scale.copy_(
            torch.linspace(1.25, 0.75, self.num_experts, device=self.device).view(-1, 1)
        )

        reference = self._reference_moe(
            dispatch_output,
            layer.w13_weight,
            layer.w2_weight,
            layer.w13_weight_scale,
            layer.w2_weight_scale,
        )
        output = scheme.apply_weights(layer, dispatch_output)
        self._assert_output_matches_reference(output, reference)


if __name__ == "__main__":
    unittest.main()
