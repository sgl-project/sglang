"""Prefix-cache reuse vs. embedding pooling.

A radix prefix hit recomputes only the suffix of a request (at most the last
token, since the match is capped one short of the input).  Pooling that reads
hidden states at positions inside the cached prefix -- CLS, MEAN, pooled
classification scores, or request-level custom pooling positions -- would
silently read rows that were never recomputed, returning the embedding of the
wrong tokens.  These tests pin the contract: such requests must recompute the
full sequence, while last-token pooling keeps reusing prefixes.
"""

import unittest
from array import array

from sglang.srt.configs.embedding_model_spec import resolve_embedding_model_spec
from sglang.srt.managers.schedule_batch import Req
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _spec(architectures, *, is_embedding_requested=False, is_embedding_gemma=False):
    return resolve_embedding_model_spec(
        architectures,
        is_embedding_requested=is_embedding_requested,
        is_embedding_gemma=is_embedding_gemma,
    )


class TestPoolingMayReadPrefix(unittest.TestCase):
    def test_cls_encoder_reads_prefix(self):
        self.assertTrue(_spec(["BertModel"]).pooling_may_read_prefix)
        self.assertTrue(_spec(["XLMRobertaModel"]).pooling_may_read_prefix)

    def test_mean_embedding_gemma_reads_prefix(self):
        self.assertTrue(
            _spec(["Gemma3TextModel"], is_embedding_gemma=True).pooling_may_read_prefix
        )
        self.assertTrue(_spec(["EmbeddingGemma2Model"]).pooling_may_read_prefix)

    def test_classification_reads_prefix(self):
        self.assertTrue(
            _spec(["BertForSequenceClassification"]).pooling_may_read_prefix
        )

    def test_last_token_decoder_pooling_keeps_reuse(self):
        self.assertFalse(_spec(["LlamaEmbeddingModel"]).pooling_may_read_prefix)
        self.assertFalse(
            _spec(
                ["Qwen3ForCausalLM"], is_embedding_requested=True
            ).pooling_may_read_prefix
        )

    def test_generation_model_keeps_reuse(self):
        self.assertFalse(_spec(["Qwen3ForCausalLM"]).pooling_may_read_prefix)


class TestComputeMaxPrefixLen(unittest.TestCase):
    def _req(self, **kwargs):
        return Req("r0", None, array("q", [10, 11, 12, 13]), None, **kwargs)

    def test_plain_request_matches_up_to_last_minus_one(self):
        self.assertEqual(self._req()._compute_max_prefix_len(4), 3)

    def test_prefix_reading_pooling_forces_full_recompute(self):
        self.assertEqual(
            self._req(pooling_may_read_prefix=True)._compute_max_prefix_len(4), 0
        )

    def test_custom_pooling_positions_force_full_recompute(self):
        self.assertEqual(
            self._req(token_indices_to_pool=[0])._compute_max_prefix_len(4), 0
        )
        self.assertEqual(
            self._req(multi_item_delimiter_indices=[1])._compute_max_prefix_len(4), 0
        )


if __name__ == "__main__":
    unittest.main()
