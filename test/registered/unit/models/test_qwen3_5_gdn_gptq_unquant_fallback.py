import unittest

import torch

from sglang.srt.layers.linear import ColumnParallelLinear, MergedColumnParallelLinear
from sglang.srt.layers.quantization.gptq.gptq import GPTQMarlinConfig
from sglang.srt.layers.quantization.marlin_utils import check_marlin_supports_layer
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import (
    _FALLBACK_FUSED_SHARDS,
    get_dynamic_override,
)
from sglang.srt.models.qwen3_5 import Qwen3_5GatedDeltaNet
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestQwen35GdnGptqUnquantFallback(CustomTestCase):
    def setUp(self):
        self.quant_config = GPTQMarlinConfig(
            weight_bits=4,
            group_size=128,
            desc_act=False,
            is_sym=True,
            lm_head_quantized=False,
            dynamic={},
            full_config={},
        )

    def test_fallback_fused_shards_contains_in_proj(self):
        """Verify _FALLBACK_FUSED_SHARDS includes in_proj_ba and in_proj_qkvz."""
        self.assertIn("in_proj_ba", _FALLBACK_FUSED_SHARDS)
        self.assertEqual(
            _FALLBACK_FUSED_SHARDS["in_proj_ba"], ["in_proj_b", "in_proj_a"]
        )
        self.assertIn("in_proj_qkvz", _FALLBACK_FUSED_SHARDS)
        self.assertEqual(
            _FALLBACK_FUSED_SHARDS["in_proj_qkvz"], ["in_proj_qkv", "in_proj_z"]
        )

    def test_get_dynamic_override_fused_shards(self):
        """Verify negative dynamic rules matching shards match the fused projection."""
        config = GPTQMarlinConfig(
            weight_bits=4,
            group_size=128,
            desc_act=False,
            is_sym=True,
            lm_head_quantized=False,
            dynamic={"-:.*in_proj_b.*": {}},
            full_config={},
        )
        self.assertFalse(
            get_dynamic_override(config, "model.layers.0.linear_attn.in_proj_ba")
        )

    def test_check_marlin_supports_layer_tile_indivisible(self):
        """Verify check_marlin_supports_layer rejects size_n not divisible by 64."""
        layer_96 = MergedColumnParallelLinear(
            input_size=5120,
            output_sizes=[48, 48],
            bias=False,
            quant_config=None,
            tp_rank=0,
            tp_size=1,
        )
        self.assertFalse(check_marlin_supports_layer(layer_96, group_size=128))

        layer_128 = ColumnParallelLinear(
            input_size=5120,
            output_size=128,
            bias=False,
            quant_config=None,
            tp_rank=0,
            tp_size=1,
        )
        self.assertTrue(check_marlin_supports_layer(layer_128, group_size=128))

    def test_gptq_marlin_get_quant_method_fallback(self):
        """Verify GPTQMarlinConfig.get_quant_method returns UnquantizedLinearMethod for indivisible shapes."""
        layer_96 = MergedColumnParallelLinear(
            input_size=5120,
            output_sizes=[48, 48],
            bias=False,
            quant_config=None,
            tp_rank=0,
            tp_size=1,
        )
        quant_method = self.quant_config.get_quant_method(
            layer_96, prefix="model.layers.0.linear_attn.in_proj_ba"
        )
        self.assertIsInstance(quant_method, UnquantizedLinearMethod)

    def test_create_ba_proj_instantiates_unquantized_with_weight_param(self):
        """Verify create_ba_proj builds in_proj_ba with unquantized weight when quant_config is GPTQ/Marlin."""
        ba_proj = Qwen3_5GatedDeltaNet.create_ba_proj(
            None,
            hidden_size=5120,
            num_v_heads=48,
            quant_config=self.quant_config,
            prefix="model.layers.0.linear_attn.in_proj_ba",
            tp_rank=0,
            tp_size=1,
        )
        self.assertIsInstance(ba_proj.quant_method, UnquantizedLinearMethod)
        self.assertTrue(hasattr(ba_proj, "weight"))
        self.assertEqual(ba_proj.weight.shape, (96, 5120))

        # Simulate loading in_proj_b (shard 0) and in_proj_a (shard 1)
        target_dtype = ba_proj.weight.dtype
        b_weight = torch.ones(48, 5120, dtype=target_dtype)
        a_weight = torch.ones(48, 5120, dtype=target_dtype) * 2.0
        ba_proj.weight.weight_loader(ba_proj.weight, b_weight, 0)
        ba_proj.weight.weight_loader(ba_proj.weight, a_weight, 1)

        self.assertTrue(torch.allclose(ba_proj.weight[:48], b_weight))
        self.assertTrue(torch.allclose(ba_proj.weight[48:], a_weight))

        # Test forward pass
        x = torch.randn(2, 5120, dtype=target_dtype)
        out, bias = ba_proj(x)
        self.assertEqual(out.shape, (2, 96))
        self.assertIsNone(bias)


if __name__ == "__main__":
    unittest.main()
