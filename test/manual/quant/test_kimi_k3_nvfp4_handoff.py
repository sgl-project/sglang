"""CPU state/ABI tests; these do not establish GPU numerical correctness."""

import unittest
from unittest.mock import patch

import torch

import sglang.srt.layers.quantization  # noqa: F401
from sglang.kernels.ops.moe import moe_route_nvfp4_fused as kernel
from sglang.srt.layers.moe import route_quant_handoff as h
from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
    quantize_hidden_states_fp4,
)


class TestNvfp4Handoff(unittest.TestCase):
    def setUp(self):
        h.clear()
        self.x = torch.randn(8, 6016)[:, 2432:]
        self.scale = torch.tensor([1.0])
        self.scores = torch.randn(8, 896)
        self.bias = torch.zeros(896)
        self.outputs = (
            torch.ones(8, 16),
            torch.zeros(8, 16, dtype=torch.int32),
            torch.zeros(8, 16, dtype=torch.int32),
            torch.ones(8, 1792, dtype=torch.uint8),
            torch.zeros(8, 224, dtype=torch.uint8),
        )

    def tearDown(self):
        h.clear()

    def produce(self):
        h.stage_nvfp4(self.x, self.scale)
        with (
            patch.object(kernel, "covered", return_value=True),
            patch.object(kernel, "available", return_value=True),
            patch.object(kernel, "route_nvfp4_fused", return_value=self.outputs),
        ):
            return h.try_route_quant_fused(
                self.scores,
                self.bias,
                16,
                num_fused_shared_experts=0,
                renormalize=True,
                routed_scaling_factor=1.0,
                apply_routed_scaling_factor_on_output=False,
            )

    def test_consume_once_preserves_fp32_routing(self):
        weights, ids = self.produce()
        self.assertIs(weights, self.outputs[0])
        self.assertEqual(weights.dtype, torch.float32)
        self.assertIsNone(h.take(self.x))
        q, s = quantize_hidden_states_fp4(self.x, self.scale)
        self.assertIs(q, self.outputs[3])
        self.assertEqual(s.dtype, torch.float8_e4m3fn)
        self.assertFalse(h.has_nvfp4(self.x, self.scale))
        self.assertIsNone(h.take_nvfp4(self.x, self.scale))

    def test_scale_and_layout_identity(self):
        self.produce()
        self.assertIsNone(h.take_nvfp4(self.x, self.scale.clone()))
        self.assertFalse(h.has_nvfp4(self.x.contiguous(), self.scale))
        self.assertTrue(h.has_nvfp4(self.x, self.scale))

    def test_clear_and_restage_do_not_leak(self):
        self.produce()
        h.clear()
        self.assertFalse(h.has_nvfp4(self.x, self.scale))
        self.produce()
        h.stage(self.x)
        self.assertFalse(h.has_nvfp4(self.x, self.scale))
        self.assertIsNone(h.take(self.x))

    def test_uncovered_falls_back_without_publishing(self):
        h.stage_nvfp4(self.x, self.scale)
        with patch.object(kernel, "covered", return_value=False):
            result = h.try_route_quant_fused(
                self.scores,
                self.bias,
                16,
                num_fused_shared_experts=0,
                renormalize=True,
                routed_scaling_factor=1.0,
                apply_routed_scaling_factor_on_output=False,
            )
        self.assertIsNone(result)
        self.assertFalse(h.has_nvfp4(self.x, self.scale))


if __name__ == "__main__":
    unittest.main()
