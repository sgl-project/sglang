"""LongCat target and NextN draft checkpoint selection."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models.longcat_flash import LongcatFlashForCausalLM
from sglang.srt.models.longcat_flash_nextn import LongcatFlashForCausalLMNextN
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLongcatCheckpointSelection(unittest.TestCase):
    def test_target_excludes_mtp_and_rotary_weights(self):
        model = object.__new__(LongcatFlashForCausalLM)
        for name, unused in (
            ("model.layers.0.self_attn.q_b_proj.weight", False),
            ("model.mtp.layers.0.self_attn.q_b_proj.weight", True),
            ("model.layers.0.self_attn.rotary_emb.inv_freq", True),
        ):
            with self.subTest(name=name):
                self.assertEqual(model.is_unused_checkpoint_weight(name), unused)

    def test_draft_excludes_main_and_target_owned_embedding(self):
        model = object.__new__(LongcatFlashForCausalLMNextN)
        for name, unused in (
            ("model.layers.0.self_attn.q_b_proj.weight", True),
            ("model.mtp.layers.0.self_attn.q_b_proj.weight", False),
            ("model.mtp.norm.weight", False),
            ("model.mtp.embed_tokens.weight", True),
            ("model.mtp.layers.0.self_attn.rotary_emb.inv_freq", True),
        ):
            with self.subTest(name=name):
                self.assertEqual(model.is_unused_checkpoint_weight(name), unused)

    def test_target_retains_rotary_name_for_post_load_bookkeeping(self):
        model = object.__new__(LongcatFlashForCausalLM)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(n_routed_experts=1, num_hidden_layers=1)
        model.use_ngram_embedding = False
        observed = []
        model.post_load_weights = lambda weight_names: observed.extend(weight_names)

        name = "model.layers.0.self_attn.rotary_emb.inv_freq"
        model.load_weights([(name, torch.empty(0))])

        self.assertEqual(observed, [name])


if __name__ == "__main__":
    unittest.main()
