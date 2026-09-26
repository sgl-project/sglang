"""MiMo checkpoint policies must survive runtime projection fusion."""

import unittest

from sglang.srt.layers.quantization.modelopt_quant import ModelOptMixedPrecisionConfig
from sglang.srt.models.mimo_v2 import MiMoV2ForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMiMoModelOptFusion(CustomTestCase):
    def policy(self, entries):
        # Resolution is metadata-only; model construction and CUDA state are not
        # needed to exercise the same resolver used by get_quant_method.
        config = object.__new__(ModelOptMixedPrecisionConfig)
        config.quantized_layers = entries
        config.packed_modules_mapping = MiMoV2ForCausalLM.packed_modules_mapping
        return config

    def test_dense_fp8_gate_up_keeps_its_block_scales(self):
        config = self.policy(
            {
                "model.layers.0.mlp.gate_proj": {"quant_algo": "FP8_BLOCK_SCALES"},
                "model.layers.0.mlp.up_proj": {"quant_algo": "FP8_BLOCK_SCALES"},
                "model.layers.1.mlp.experts.0.gate_proj": {"quant_algo": "NVFP4"},
            }
        )
        self.assertEqual(
            config.resolve_quant_algo("model.layers.0.mlp.gate_up_proj"),
            "FP8_BLOCK_SCALES",
        )
        self.assertEqual(
            config.resolve_quant_algo("model.layers.1.mlp.experts"), "NVFP4"
        )

    def test_split_qkv_checkpoint_policy_resolves_after_fusion(self):
        config = self.policy(
            {
                f"model.layers.2.self_attn.{projection}_proj": {"quant_algo": "FP8"}
                for projection in ("q", "k", "v")
            }
        )
        self.assertEqual(
            config.resolve_quant_algo("model.layers.2.self_attn.qkv_proj"), "FP8"
        )

    def test_already_fused_qkv_policy_is_preserved(self):
        config = self.policy(
            {"model.layers.0.self_attn.qkv_proj": {"quant_algo": "FP8_BLOCK_SCALES"}}
        )
        self.assertEqual(
            config.resolve_quant_algo("model.layers.0.self_attn.qkv_proj"),
            "FP8_BLOCK_SCALES",
        )

    def test_conflicting_gate_up_policies_are_rejected(self):
        config = self.policy(
            {
                "model.layers.0.mlp.gate_proj": {"quant_algo": "NVFP4"},
                "model.layers.0.mlp.up_proj": {"quant_algo": "FP8"},
            }
        )
        with self.assertRaisesRegex(ValueError, "Mixed quant_algo"):
            config.resolve_quant_algo("model.layers.0.mlp.gate_up_proj")

    def test_unquantized_output_projection_stays_unquantized(self):
        config = self.policy(
            {"model.layers.0.self_attn.qkv_proj": {"quant_algo": "FP8_BLOCK_SCALES"}}
        )
        self.assertIsNone(config.resolve_quant_algo("model.layers.0.self_attn.o_proj"))


if __name__ == "__main__":
    unittest.main()
