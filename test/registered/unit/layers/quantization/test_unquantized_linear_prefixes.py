"""Fusion eligibility must follow source prefixes, including fused QKV shards."""

import unittest
from unittest.mock import Mock

from sglang.srt.layers.linear import LinearBase, QKVParallelLinear
from sglang.srt.layers.quantization.fp8 import Fp8Config
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.quantization.utils import are_linear_prefixes_unquantized
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUnquantizedLinearPrefixes(unittest.TestCase):
    def test_without_quantization(self):
        self.assertTrue(are_linear_prefixes_unquantized(None, ["attn.qkv_proj"]))

    def test_resolves_every_source_prefix_without_allocating_weights(self):
        prefixes = ["attn.qkv_proj", "attn.f_a_proj", "attn.f_b_proj"]
        config = Mock()

        def resolve(layer, prefix):
            self.assertIsInstance(layer, LinearBase)
            self.assertEqual(list(layer.parameters()), [])
            return UnquantizedLinearMethod()

        config.get_quant_method.side_effect = resolve
        self.assertTrue(are_linear_prefixes_unquantized(config, iter(prefixes)))
        self.assertEqual(
            [call.kwargs["prefix"] for call in config.get_quant_method.call_args_list],
            prefixes,
        )

    def test_one_quantized_projection_prevents_fusion(self):
        config = Mock()
        config.get_quant_method.side_effect = [UnquantizedLinearMethod(), object()]
        self.assertFalse(
            are_linear_prefixes_unquantized(config, ["attn.qkv_proj", "attn.f_b_proj"])
        )

    def test_none_method_does_not_prove_unquantized(self):
        config = Mock()
        config.get_quant_method.return_value = None
        self.assertFalse(are_linear_prefixes_unquantized(config, ["attn.qkv_proj"]))

    def test_unmatched_concrete_linear_class_falls_back(self):
        config = Mock()
        config.get_quant_method.side_effect = lambda layer, prefix: (
            object() if isinstance(layer, QKVParallelLinear) else None
        )
        self.assertFalse(are_linear_prefixes_unquantized(config, ["attn.qkv_proj"]))

    def test_unknown_quantization_config_falls_back(self):
        config = Mock()
        config.get_quant_method.side_effect = ValueError("unmatched target class")
        self.assertFalse(are_linear_prefixes_unquantized(config, ["attn.qkv_proj"]))

    def test_fp8_checkpoint_with_bf16_qkv_shards(self):
        config = Fp8Config(
            ignored_layers=[
                "attn.q_proj",
                "attn.k_proj",
                "attn.v_proj",
                "attn.f_a_proj",
            ]
        )
        self.assertTrue(
            are_linear_prefixes_unquantized(config, ["attn.qkv_proj", "attn.f_a_proj"])
        )

    def test_mixed_qkv_shards_cannot_enable_fusion(self):
        config = Fp8Config(ignored_layers=["attn.q_proj", "attn.k_proj"])
        self.assertFalse(are_linear_prefixes_unquantized(config, ["attn.qkv_proj"]))


if __name__ == "__main__":
    unittest.main()
