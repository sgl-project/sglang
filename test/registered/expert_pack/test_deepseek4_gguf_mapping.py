# SPDX-License-Identifier: Apache-2.0

import unittest

from sglang.srt.model_loader.deepseek4_gguf import (
    _split_suffix,
    _v4_checkpoint_name,
    routed_expert_tensor,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDeepseek4GGUFMapping(unittest.TestCase):
    def test_routed_expert_detection_is_exact(self):
        self.assertEqual(
            routed_expert_tensor("blk.17.ffn_gate_exps.weight"), (17, "gate")
        )
        self.assertEqual(
            routed_expert_tensor("blk.42.ffn_down_exps.weight"), (42, "down")
        )
        self.assertIsNone(routed_expert_tensor("blk.1.ffn_gate_exps.bias"))
        self.assertIsNone(routed_expert_tensor("blk.1.ffn_gate_shexp.weight"))

    def test_suffix_is_not_duplicated(self):
        self.assertEqual(
            _split_suffix("blk.0.attn_norm.weight"), ("blk.0.attn_norm", "weight")
        )
        self.assertEqual(_split_suffix("unusual.name"), ("unusual.name", ""))

    def test_v4_only_names(self):
        cases = {
            "output_hc_base.weight": "hc_head_base",
            "output_hc_fn.weight": "hc_head_fn",
            "output_hc_scale.weight": "hc_head_scale",
            "blk.3.attn_kv.weight": "layers.3.attn.wkv.weight",
            "blk.3.attn_compressor_gate.weight": (
                "layers.3.attn.compressor.wgate.weight"
            ),
            "blk.3.attn_compressor_kv.weight": ("layers.3.attn.compressor.wkv.weight"),
            "blk.4.indexer_compressor_ape.weight": (
                "layers.4.attn.indexer.compressor.ape"
            ),
            "blk.2.attn_sinks.weight": "layers.2.attn.attn_sink",
            "blk.2.hc_attn_fn.weight": "layers.2.hc_attn_fn",
            "blk.2.ffn_gate_tid2eid.weight": "layers.2.ffn.gate.tid2eid",
            "blk.9.ffn_gate_inp.weight": "layers.9.ffn.gate.weight",
            "blk.9.ffn_gate_exps.weight": "layers.9.ffn.experts.w1.weight",
            "blk.9.ffn_down_shexp.weight": ("layers.9.ffn.shared_experts.w2.weight"),
            "blk.10.indexer.proj.weight": (
                "layers.10.attn.indexer.weights_proj.weight"
            ),
            "token_embd.weight": "embed.weight",
            "output.weight": "head.weight",
            "output_norm.weight": "norm.weight",
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertEqual(_v4_checkpoint_name(source), expected)


if __name__ == "__main__":
    unittest.main()
