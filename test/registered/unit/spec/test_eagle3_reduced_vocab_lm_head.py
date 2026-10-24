# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""EAGLE3 draft lm_head resolution for reduced/untied/tied vocabularies.

A draft checkpoint that declares *both* ``tie_word_embeddings: true`` and a
reduced ``draft_vocab_size`` ships its own ``lm_head.weight``. Resolving the
tied embedding first used to bind the full-size target embedding as the draft
head while the logits processor still used the reduced vocab; the processor
silently truncated the logits and acceptance collapsed to ~0. These tests pin
the resolution order so ``llama_eagle3`` and ``kimi_k25_eagle3`` cannot
re-diverge.
"""

import unittest

from transformers import LlamaConfig

from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import published_topology

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

VOCAB_SIZE = 151936
DRAFT_VOCAB_SIZE = 32000


def _config(**overrides):
    base = dict(
        vocab_size=VOCAB_SIZE,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        max_position_embeddings=128,
    )
    base.update(overrides)
    return LlamaConfig(**base)


class TestLlamaEagle3HeadResolution(unittest.TestCase):
    """The four-row resolution matrix, end to end on the real draft model."""

    def _build(self, **overrides):
        config = _config(**overrides)
        topology = published_topology(tp_size=1, pp_size=1)
        topology.__enter__()
        try:
            with get_parallel().override(pp_group=None):
                return LlamaForCausalLMEagle3(config)
        finally:
            topology.__exit__(None, None, None)

    def test_reduced_vocab_wins_over_tied_embedding(self):
        # The bug: tie=True previously bound the full-size embed_tokens as the
        # head while the logits processor used the reduced vocab.
        model = self._build(tie_word_embeddings=True, draft_vocab_size=DRAFT_VOCAB_SIZE)
        self.assertIsNot(model.lm_head, model.model.embed_tokens)
        self.assertEqual(model.lm_head.weight.shape[0], DRAFT_VOCAB_SIZE)
        self.assertEqual(model.logits_processor.vocab_size, DRAFT_VOCAB_SIZE)
        self.assertFalse(model.load_lm_head_from_target)

    def test_reduced_vocab_with_untied_embedding(self):
        model = self._build(
            tie_word_embeddings=False, draft_vocab_size=DRAFT_VOCAB_SIZE
        )
        self.assertIsNot(model.lm_head, model.model.embed_tokens)
        self.assertEqual(model.lm_head.weight.shape[0], DRAFT_VOCAB_SIZE)
        self.assertEqual(model.logits_processor.vocab_size, DRAFT_VOCAB_SIZE)
        self.assertFalse(model.load_lm_head_from_target)

    def test_tied_embedding_without_draft_vocab_is_unchanged(self):
        model = self._build(tie_word_embeddings=True)
        self.assertIs(model.lm_head, model.model.embed_tokens)
        self.assertEqual(model.logits_processor.vocab_size, VOCAB_SIZE)
        self.assertFalse(model.load_lm_head_from_target)

    def test_untied_embedding_without_draft_vocab_loads_from_target(self):
        model = self._build(tie_word_embeddings=False)
        self.assertIsNot(model.lm_head, model.model.embed_tokens)
        self.assertEqual(model.lm_head.weight.shape[0], VOCAB_SIZE)
        self.assertEqual(model.logits_processor.vocab_size, VOCAB_SIZE)
        self.assertTrue(model.load_lm_head_from_target)


if __name__ == "__main__":
    unittest.main()
