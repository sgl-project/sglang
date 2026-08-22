"""Regression tests for GPTQ Marlin MoE scale allocation and TP loading."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinConfig
from sglang.srt.layers.quantization.gptq.schemes.gptq_moe import GPTQMarlinMoEScheme
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestGPTQMarlinMoE(CustomTestCase):
    def test_marlin_moe_scales_follow_params_dtype(self):
        quant_config = GPTQMarlinConfig(weight_bits=4, group_size=128, desc_act=False, is_sym=True, lm_head_quantized=False, dynamic={}, full_config={})
        for params_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(params_dtype=params_dtype):
                layer = torch.nn.Module()
                layer.moe_tp_size = 1
                GPTQMarlinMoEScheme(quant_config).create_weights(layer=layer, num_experts=2, hidden_size=256, intermediate_size_per_partition=128, params_dtype=params_dtype)
                self.assertEqual(layer.w13_scales.dtype, params_dtype)
                self.assertEqual(layer.w2_scales.dtype, params_dtype)


if __name__ == "__main__":
    unittest.main()
