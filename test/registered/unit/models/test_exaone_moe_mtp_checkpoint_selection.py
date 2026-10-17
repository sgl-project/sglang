"""EXAONE-MoE checkpoint selection for target and MTP draft passes."""

import unittest

from sglang.srt.models.exaone_moe import ExaoneMoEForCausalLM
from sglang.srt.models.exaone_moe_mtp import ExaoneMoEForCausalLMMTP
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestExaoneMoeCheckpointSelection(unittest.TestCase):
    def test_target_keeps_model_weights_and_skips_mtp(self):
        model = object.__new__(ExaoneMoEForCausalLM)
        for name, unused in (
            ("model.layers.0.self_attn.q_proj.weight", False),
            ("mtp.fc.weight", True),
            ("mtp.layers.0.self_attn.q_proj.weight", True),
            ("model.layers.0.self_attn.rotary_emb.inv_freq", True),
            ("model.layers.0.self_attn.rotary_emb.cos_cached", True),
        ):
            with self.subTest(name=name):
                self.assertEqual(model.is_unused_checkpoint_weight(name), unused)

    def test_draft_keeps_only_mtp_weights_it_loads(self):
        model = object.__new__(ExaoneMoEForCausalLMMTP)
        for name, unused in (
            ("model.layers.0.self_attn.q_proj.weight", True),
            ("mtp.fc.weight", False),
            ("mtp.pre_fc_norm_embedding.weight", False),
            ("mtp.layers.0.self_attn.q_proj.weight", False),
            ("mtp.layers.0.self_attn.rotary_emb.inv_freq", True),
            ("mtp.projector.weight", True),
        ):
            with self.subTest(name=name):
                self.assertEqual(model.is_unused_checkpoint_weight(name), unused)


if __name__ == "__main__":
    unittest.main()
