"""
Unit tests for XPU MoE WnA16 quantization methods and execution paths.
"""

from unittest.mock import MagicMock

import torch
import torch.nn as nn

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
        cls.device = "xpu"
        cls.num_tokens = 4
        cls.num_experts = 4
        cls.topk = 2
        cls.hidden = 256
        cls.intermediate = 256

    def _create_dispatch_output(self):
        x = torch.randn(
            self.num_tokens, self.hidden, dtype=torch.bfloat16, device=self.device
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

        layer.w13_weight = torch.randn(
            self.num_experts,
            2 * self.intermediate,
            self.hidden,
            dtype=torch.bfloat16,
            device=self.device,
        ).to(torch.float8_e4m3fn)
        layer.w2_weight = torch.randn(
            self.num_experts,
            self.hidden,
            self.intermediate,
            dtype=torch.bfloat16,
            device=self.device,
        ).to(torch.float8_e4m3fn)
        layer.w13_weight_scale = torch.ones(
            self.num_experts, 1, dtype=torch.float32, device=self.device
        )
        layer.w2_weight_scale = torch.ones(
            self.num_experts, 1, dtype=torch.float32, device=self.device
        )

        output = method.apply(layer, dispatch_output)
        self.assertIsInstance(output, StandardCombineInput)
        self.assertEqual(output.hidden_states.shape, (self.num_tokens, self.hidden))
        self.assertEqual(output.hidden_states.dtype, torch.bfloat16)

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

        layer.w13_weight = torch.randn(
            self.num_experts,
            2 * self.intermediate,
            self.hidden,
            dtype=torch.bfloat16,
            device=self.device,
        ).to(torch.float8_e4m3fn)
        layer.w2_weight = torch.randn(
            self.num_experts,
            self.hidden,
            self.intermediate,
            dtype=torch.bfloat16,
            device=self.device,
        ).to(torch.float8_e4m3fn)
        layer.w13_weight_scale_inv = torch.ones(
            self.num_experts,
            (2 * self.intermediate + 127) // 128,
            (self.hidden + 127) // 128,
            dtype=torch.float32,
            device=self.device,
        )
        layer.w2_weight_scale_inv = torch.ones(
            self.num_experts,
            (self.hidden + 127) // 128,
            (self.intermediate + 127) // 128,
            dtype=torch.float32,
            device=self.device,
        )

        output = method.apply(layer, dispatch_output)
        self.assertIsInstance(output, StandardCombineInput)
        self.assertEqual(output.hidden_states.shape, (self.num_tokens, self.hidden))
        self.assertEqual(output.hidden_states.dtype, torch.bfloat16)

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

        layer.w13_weight = torch.randn(
            self.num_experts,
            2 * self.intermediate,
            self.hidden,
            dtype=torch.bfloat16,
            device=self.device,
        ).to(torch.float8_e4m3fn)
        layer.w2_weight = torch.randn(
            self.num_experts,
            self.hidden,
            self.intermediate,
            dtype=torch.bfloat16,
            device=self.device,
        ).to(torch.float8_e4m3fn)
        layer.w13_weight_scale = torch.ones(
            self.num_experts, 1, dtype=torch.float32, device=self.device
        )
        layer.w2_weight_scale = torch.ones(
            self.num_experts, 1, dtype=torch.float32, device=self.device
        )

        output = method.apply(layer, dispatch_output)
        self.assertIsInstance(output, StandardCombineInput)
        self.assertEqual(output.hidden_states.shape, (self.num_tokens, self.hidden))
        self.assertEqual(output.hidden_states.dtype, torch.bfloat16)

    def test_compressed_tensors_w8a8_fp8_moe(self):
        dispatch_output = self._create_dispatch_output()
        weight_quant = MagicMock(strategy=QuantizationStrategy.TENSOR)
        input_quant = MagicMock(strategy=QuantizationStrategy.TENSOR, dynamic=True)
        scheme = CompressedTensorsW8A8Fp8MoE(weight_quant, input_quant)
        scheme.moe_runner_config = MoeRunnerConfig()
        layer = nn.Module()

        layer.w13_weight = torch.randn(
            self.num_experts,
            2 * self.intermediate,
            self.hidden,
            dtype=torch.bfloat16,
            device=self.device,
        ).to(torch.float8_e4m3fn)
        layer.w2_weight = torch.randn(
            self.num_experts,
            self.hidden,
            self.intermediate,
            dtype=torch.bfloat16,
            device=self.device,
        ).to(torch.float8_e4m3fn)
        layer.w13_weight_scale = torch.ones(
            self.num_experts, 1, dtype=torch.float32, device=self.device
        )
        layer.w2_weight_scale = torch.ones(
            self.num_experts, 1, dtype=torch.float32, device=self.device
        )

        output = scheme.apply_weights(layer, dispatch_output)
        self.assertIsInstance(output, StandardCombineInput)
        self.assertEqual(output.hidden_states.shape, (self.num_tokens, self.hidden))
        self.assertEqual(output.hidden_states.dtype, torch.bfloat16)


if __name__ == "__main__":
    import unittest

    unittest.main()
