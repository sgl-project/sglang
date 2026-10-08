import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.hunyuan_v4 import HYV4ForCausalLM
from sglang.srt.models.hunyuan_v4_nextn import HYV4ForCausalLMNextN
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestHunyuanV4NextNWeightLoading(unittest.TestCase):
    def test_indexer_checkpoint_layout_is_permuted(self):
        model = object.__new__(HYV4ForCausalLMNextN)
        model.config = SimpleNamespace(
            num_hidden_layers=80,
            index_n_heads=2,
            index_head_dim=4,
            qk_rope_head_dim=2,
        )
        captured = []
        object.__setattr__(
            model,
            "do_load_weights",
            lambda weights, **kwargs: captured.extend(weights),
        )
        loaded_weight = torch.arange(8).reshape(8, 1)

        model.load_weights(
            [("model.mtp_layers.0.self_attn.indexer.wq_b.weight", loaded_weight)]
        )

        self.assertEqual(
            captured[0][0], "model.layers.80.self_attn.indexer.wq_b.weight"
        )
        torch.testing.assert_close(
            captured[0][1].flatten(), torch.tensor([2, 3, 0, 1, 6, 7, 4, 5])
        )

    def test_draft_checkpoint_selection_keeps_only_first_mtp_layer(self):
        model = object.__new__(HYV4ForCausalLMNextN)
        for name, unused in (
            ("model.layers.0.self_attn.q_proj.weight", True),
            ("model.mtp_layers.0.self_attn.indexer.wq_b.weight", False),
            ("model.mtp_layers.0.final_layernorm.weight", False),
            ("model.mtp_layers.1.self_attn.q_proj.weight", True),
        ):
            with self.subTest(name=name):
                self.assertEqual(model.is_unused_checkpoint_weight(name), unused)


class TestHunyuanV4MainCheckpointSelection(unittest.TestCase):
    def test_main_excludes_mtp_layers(self):
        model = object.__new__(HYV4ForCausalLM)
        for name, unused in (
            ("model.layers.0.self_attn.q_proj.weight", False),
            ("model.mtp_layers.0.self_attn.q_proj.weight", True),
            ("model.embed_tokens.weight", False),
        ):
            with self.subTest(name=name):
                self.assertEqual(model.is_unused_checkpoint_weight(name), unused)


if __name__ == "__main__":
    unittest.main()
