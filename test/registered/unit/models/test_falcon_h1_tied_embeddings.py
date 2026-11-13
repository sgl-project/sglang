import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.models import falcon_h1
from sglang.srt.runtime_context import get_parallel

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _init_logits_processor(self, config, logit_scale=None):
    # Avoid distributed runtime setup, but use the real LM-head computation.
    nn.Module.__init__(self)
    self.logit_scale = logit_scale
    self.use_fp32_lm_head = False
    self.rl_on_policy_target = None


class TestFalconH1TiedEmbeddings(CustomTestCase):
    def _build_model(self, dtype, tied):
        config = SimpleNamespace(
            vocab_size=16,
            hidden_size=8,
            tie_word_embeddings=tied,
            lm_head_multiplier=0.25,
        )
        backbone = nn.Module()
        backbone.embed_tokens = nn.Embedding(16, 8, dtype=dtype)
        head = nn.Linear(8, 16, bias=False, dtype=dtype)
        with (
            get_parallel().override(
                pp_group=SimpleNamespace(is_first_rank=True, is_last_rank=True),
                enable_dp_lm_head=False,
            ),
            patch.object(falcon_h1, "FalconH1Model", return_value=backbone),
            patch.object(falcon_h1, "ParallelLMHead", return_value=head),
            patch.object(falcon_h1.LogitsProcessor, "__init__", _init_logits_processor),
        ):
            return falcon_h1.FalconH1ForCausalLM(config)

    def test_tied_embedding_preserves_dtype_and_fp32_logits(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                model = self._build_model(dtype, tied=True)
                embedding = model.model.embed_tokens
                self.assertIs(model.lm_head, embedding)
                self.assertEqual(embedding.weight.dtype, dtype)
                shared_weight = embedding.weight

                # A checkpoint without a separate lm_head must update both uses.
                checkpoint_weight = torch.linspace(-1, 1, 128).reshape(16, 8)
                loaded = model.load_weights(
                    [("model.embed_tokens.weight", checkpoint_weight)]
                )
                self.assertEqual(loaded, {"model.embed_tokens.weight"})
                self.assertIs(model.lm_head.weight, shared_weight)
                torch.testing.assert_close(
                    embedding.weight, checkpoint_weight.to(dtype)
                )

                hidden_states = embedding(torch.tensor([1, 3, 5]))
                self.assertEqual(hidden_states.dtype, dtype)
                logits = model.logits_processor._compute_lm_head(
                    hidden_states, model.lm_head
                )
                expected = hidden_states.float() @ shared_weight.float().T
                self.assertEqual(logits.dtype, torch.float32)
                torch.testing.assert_close(logits, expected)
                self.assertEqual(model.logits_processor.logit_scale, 0.25)
                self.assertEqual(shared_weight.dtype, dtype)
                self.assertIs(model.lm_head.weight, embedding.weight)

    def test_untied_head_keeps_existing_fp32_behavior(self):
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            with self.subTest(dtype=dtype):
                model = self._build_model(dtype, tied=False)
                embedding = model.model.embed_tokens
                self.assertIsNot(model.lm_head.weight, embedding.weight)
                self.assertEqual(embedding.weight.dtype, dtype)
                self.assertEqual(model.lm_head.weight.dtype, torch.float32)
                hidden_states = embedding(torch.tensor([1, 3, 5]))
                logits = model.logits_processor._compute_lm_head(
                    hidden_states, model.lm_head
                )
                torch.testing.assert_close(
                    logits, hidden_states.float() @ model.lm_head.weight.T
                )


if __name__ == "__main__":
    unittest.main()
