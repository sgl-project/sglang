"""Checkpoint selection follows the Ernie 4.5 main and MTP load rules."""

import unittest
from types import SimpleNamespace

from sglang.srt.models.ernie4 import Ernie4_5_MoeForCausalLM
from sglang.srt.models.ernie4_eagle import Ernie4_5_MoeForCausalLMMTP
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestErnie45CheckpointSelection(unittest.TestCase):
    def test_main_excludes_mtp_and_tied_lm_head(self):
        model = object.__new__(Ernie4_5_MoeForCausalLM)
        model.config = SimpleNamespace(tie_word_embeddings=True)
        for name, unused in (
            ("model.layers.0.self_attn.q_proj.weight", False),
            ("model.mtp_block.0.self_attn.q_proj.weight", True),
            ("lm_head.weight", True),
            ("model.embed_tokens.weight", False),
        ):
            with self.subTest(name=name):
                self.assertEqual(model.is_unused_checkpoint_weight(name), unused)

    def test_draft_keeps_only_its_mtp_layer(self):
        model = object.__new__(Ernie4_5_MoeForCausalLMMTP)
        model.mtp_layer_id = 0
        for name, unused in (
            ("model.layers.0.self_attn.q_proj.weight", True),
            ("model.mtp_block.0.self_attn.q_proj.weight", False),
            ("model.mtp_emb_norm.0.weight", False),
            ("model.mtp_hidden_norm.0.weight", False),
            ("model.mtp_linear_proj.0.weight", False),
            ("model.mtp_block.1.self_attn.q_proj.weight", True),
        ):
            with self.subTest(name=name):
                self.assertEqual(model.is_unused_checkpoint_weight(name), unused)


if __name__ == "__main__":
    unittest.main()
