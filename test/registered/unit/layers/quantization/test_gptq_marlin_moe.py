"""Regression tests for GPTQ MoE scale allocation and TP loading."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig, GPTQMarlinConfig
from sglang.srt.layers.quantization.gptq.schemes.gptq_cpu import (
    GPTQIntelAMXMoEScheme,
)
from sglang.srt.layers.quantization.gptq.schemes.gptq_moe import GPTQMarlinMoEScheme
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestGPTQMarlinMoE(CustomTestCase):
    def test_marlin_moe_scales_follow_params_dtype(self):
        quant_config = GPTQMarlinConfig(
            weight_bits=4,
            group_size=128,
            desc_act=False,
            is_sym=True,
            lm_head_quantized=False,
            dynamic={},
            full_config={},
        )
        for params_dtype in (torch.float16, torch.bfloat16):
            with self.subTest(params_dtype=params_dtype):
                layer = torch.nn.Module()
                layer.moe_tp_size = 1
                GPTQMarlinMoEScheme(quant_config).create_weights(
                    layer=layer,
                    num_experts=2,
                    hidden_size=256,
                    intermediate_size_per_partition=128,
                    params_dtype=params_dtype,
                )
                self.assertEqual(layer.w13_scales.dtype, params_dtype)
                self.assertEqual(layer.w2_scales.dtype, params_dtype)

    def test_marlin_moe_w2_scale_shape_for_tp_and_act_order(self):
        for desc_act, expected_groups in ((False, 1), (True, 2)):
            with self.subTest(desc_act=desc_act):
                quant_config = GPTQMarlinConfig(
                    weight_bits=4,
                    group_size=128,
                    desc_act=desc_act,
                    is_sym=True,
                    lm_head_quantized=False,
                    dynamic={},
                    full_config={},
                )
                layer = torch.nn.Module()
                layer.moe_tp_size = 2
                GPTQMarlinMoEScheme(quant_config).create_weights(
                    layer=layer,
                    num_experts=2,
                    hidden_size=256,
                    intermediate_size_per_partition=128,
                    params_dtype=torch.bfloat16,
                )
                self.assertEqual(layer.w2_scales.shape, (2, expected_groups, 256))
                self.assertEqual(layer.w2_scales.load_full_w2, desc_act)

    def test_intel_amx_static_act_order_keeps_w2_tp_sharded(self):
        quant_config = GPTQConfig(
            weight_bits=4,
            group_size=128,
            desc_act=True,
            lm_head_quantized=False,
            dynamic={},
            true_sequential=True,
            static_groups=True,
        )
        layer = torch.nn.Module()
        layer.moe_tp_size = 2
        GPTQIntelAMXMoEScheme(quant_config).create_weights(
            layer=layer,
            num_experts=2,
            hidden_size=256,
            intermediate_size_per_partition=128,
            params_dtype=torch.float16,
        )

        self.assertEqual(layer.w2_scales.shape, (2, 1, 256))
        self.assertEqual(layer.w2_qzeros.shape, (2, 1, 32))
        self.assertFalse(hasattr(layer.w2_scales, "load_full_w2"))
        self.assertFalse(hasattr(layer.w2_qzeros, "load_full_w2"))

    def test_fused_moe_w2_loader_honors_load_full(self):
        loader = SimpleNamespace(
            quant_config=None,
            use_padded_loading=False,
            use_presharded_weights=False,
            use_triton_kernels=False,
            moe_tp_size=2,
            moe_tp_rank=1,
        )
        loaded_weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        sharded = torch.empty(2, 4)
        FusedMoE._load_w2(
            loader,
            expert_data=sharded,
            shard_dim=0,
            shard_id="w2",
            loaded_weight=loaded_weight,
        )
        self.assertTrue(torch.equal(sharded, loaded_weight[2:]))
        for use_padded_loading in (False, True):
            with self.subTest(use_padded_loading=use_padded_loading):
                loader.use_padded_loading = use_padded_loading
                full = torch.empty_like(loaded_weight)
                FusedMoE._load_w2(
                    loader,
                    expert_data=full,
                    shard_dim=0,
                    shard_id="w2",
                    loaded_weight=loaded_weight,
                    load_full=True,
                )
                self.assertTrue(torch.equal(full, loaded_weight))


if __name__ == "__main__":
    unittest.main()
