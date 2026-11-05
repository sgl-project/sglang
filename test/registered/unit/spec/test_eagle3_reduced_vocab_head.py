"""A reduced-vocabulary EAGLE3 checkpoint cannot tie its head to target embeddings."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models import llama_eagle3
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestEagle3ReducedVocabHead(unittest.TestCase):
    def build(self, draft_vocab):
        config = SimpleNamespace(
            vocab_size=8,
            draft_vocab_size=draft_vocab,
            hidden_size=2,
            tie_word_embeddings=True,
        )
        inner = nn.Module()
        inner.embed_tokens = nn.Embedding(8, 2)
        with (
            patch.object(llama_eagle3, "LlamaModel", return_value=inner),
            patch.object(
                llama_eagle3,
                "get_parallel",
                return_value=SimpleNamespace(pp_group=None),
            ),
            patch.object(
                llama_eagle3,
                "get_spec",
                return_value=SimpleNamespace(speculative_draft_window_size=None),
            ),
            patch.object(
                llama_eagle3,
                "ParallelLMHead",
                side_effect=lambda vocab, hidden, **kw: nn.Linear(
                    hidden, vocab, bias=False
                ),
            ),
            patch.object(llama_eagle3, "LogitsProcessor"),
        ):
            return llama_eagle3.LlamaForCausalLMEagle3(config)

    def test_reduced_head_is_loaded_and_survives_target_embedding_share(self):
        model = self.build(3)
        trained = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        model.load_weights([("lm_head.weight", trained)])
        self.assertIsNot(model.lm_head, model.model.embed_tokens)
        torch.testing.assert_close(model.lm_head.weight, trained)
        target_embedding = nn.Parameter(torch.full((8, 2), 99.0))
        with patch("sglang.srt.models.llama.current_platform"):
            model.set_embed(target_embedding)
        self.assertIs(model.model.embed_tokens.weight, target_embedding)
        torch.testing.assert_close(model.lm_head.weight, trained)
        self.assertEqual(tuple(model.lm_head(torch.ones(1, 2)).shape), (1, 3))

    def test_equal_vocab_retains_existing_weight_tying(self):
        model = self.build(8)
        self.assertIs(model.lm_head, model.model.embed_tokens)


if __name__ == "__main__":
    unittest.main()
