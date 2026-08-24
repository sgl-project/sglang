"""Unit tests for the multi-tokenizer BatchEmbeddingOutput repack."""

import unittest

import torch

from sglang.srt.managers.io_struct import (
    BatchEmbeddingOutput,
    unwrap_from_pickle,
    wrap_as_pickle,
)
from sglang.srt.managers.multi_tokenizer_mixin import _handle_output_by_index
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _batch_output():
    return BatchEmbeddingOutput(
        rids=["req-0", "req-1"],
        finished_reasons=[{"type": "length"}, {"type": "length"}],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        prompt_tokens=[3, 5],
        cached_tokens=[0, 2],
        placeholder_tokens_idx=None,
        placeholder_tokens_val=None,
        retraction_counts=[0, 7],
        cached_tokens_details=[{"radix": 0}, {"radix": 2}],
        time_stats=wrap_as_pickle(["stats-0", "stats-1"]),
        pooled_hidden_states=[[1.0], [2.0]],
    )


class TestMultiTokenizerEmbeddingRepack(CustomTestCase):
    def test_repack_preserves_per_request_fields(self):
        out = _handle_output_by_index(_batch_output(), 1)
        self.assertEqual(out.rids, ["req-1"])
        self.assertEqual(out.embeddings, [[0.3, 0.4]])
        self.assertEqual(out.prompt_tokens, [5])
        self.assertEqual(out.cached_tokens, [2])
        # The regression: these were dropped, and the tokenizer manager
        # indexes retraction_counts unconditionally on every response.
        self.assertEqual(out.retraction_counts, [7])
        self.assertEqual(out.cached_tokens_details, [{"radix": 2}])
        self.assertEqual(unwrap_from_pickle(out.time_stats), ["stats-1"])
        self.assertEqual(out.pooled_hidden_states, [[2.0]])

    def test_repack_unstacks_pooled_hidden_states(self):
        # The scheduler's output streamer may pack pooled_hidden_states as a
        # single stacked tensor ([tensor(N, ...)], len 1) when all requests
        # share a shape; the repack must index into the tensor, not the list.
        output = _batch_output()
        stacked = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
        output.pooled_hidden_states = [stacked]
        out = _handle_output_by_index(output, 1)
        self.assertEqual(len(out.pooled_hidden_states), 1)
        self.assertTrue(
            torch.equal(out.pooled_hidden_states[0], torch.tensor([2.0, 20.0]))
        )

    def test_repack_handles_absent_optional_fields(self):
        output = _batch_output()
        output.retraction_counts = None
        output.time_stats = None
        output.pooled_hidden_states = None
        out = _handle_output_by_index(output, 0)
        self.assertEqual(out.rids, ["req-0"])
        self.assertIsNone(out.retraction_counts)
        self.assertIsNone(out.time_stats)
        self.assertIsNone(out.pooled_hidden_states)


if __name__ == "__main__":
    unittest.main()
