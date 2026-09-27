"""Unit tests for the Qwen3-Next weight-name matching fix (issue #13214).

A plain substring match in ``load_weights`` treated a checkpoint that already
ships a fused ``gate_up_proj`` as containing ``up_proj`` and double-fused it
into ``gate_gate_up_proj``. ``_match_weight_name`` anchors matches to
``.``-separated path segments; these tests pin that behavior.
"""

import unittest

from sglang.srt.models.qwen3_next import _match_weight_name


class TestMatchWeightName(unittest.TestCase):
    def test_fused_gate_up_proj_not_double_matched(self):
        name = "model.layers.14.mlp.shared_expert.gate_up_proj.weight"
        self.assertFalse(_match_weight_name(name, "up_proj"))
        self.assertFalse(_match_weight_name(name, "gate_proj"))

    def test_fused_shared_expert_with_fusion_enabled(self):
        name = "model.layers.14.mlp.experts.128.gate_up_proj.weight"
        self.assertFalse(_match_weight_name(name, "up_proj"))
        self.assertFalse(_match_weight_name(name, "experts.128.up_proj."))

    def test_shard_names_still_match(self):
        self.assertTrue(
            _match_weight_name("model.layers.0.mlp.up_proj.weight", "up_proj")
        )
        self.assertTrue(
            _match_weight_name("model.layers.0.mlp.gate_proj.weight", "gate_proj")
        )
        self.assertTrue(
            _match_weight_name("model.layers.0.mlp.q_proj.weight", "q_proj")
        )

    def test_multi_segment_expert_names_match(self):
        name = "model.layers.0.mlp.experts.3.gate_proj.weight"
        self.assertTrue(_match_weight_name(name, "experts.3.gate_proj."))

    def test_trailing_dot_names_match(self):
        name = "model.layers.0.linear_attn.in_proj_qkv.weight"
        self.assertTrue(_match_weight_name(name, "in_proj_qkv."))
        # An already-fused GDN weight must not match the unfused shard name.
        self.assertFalse(
            _match_weight_name(
                "model.layers.0.linear_attn.in_proj_qkvz.weight", "in_proj_qkv."
            )
        )


if __name__ == "__main__":
    unittest.main()
