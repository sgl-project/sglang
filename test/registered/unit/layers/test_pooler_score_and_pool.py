"""Unit tests for score_and_pool in sglang.srt.layers.pooler.

All tests run on CPU — no GPU required.  MIS delimiter positions are passed
via forward_batch.multi_item_delimiter_indices (pre-computed by the caller).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.layers import pooler as pooler_module
from sglang.srt.layers.pooler import (
    EmbeddingPoolerOutput,
    Pooler,
    PoolingType,
    score_and_pool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _make_forward_batch(
    extend_seq_lens,
    multi_item_delimiter_indices=None,
    token_indices_to_pool=None,
    return_pooled_hidden_states=False,
    is_prefill_only=True,
):
    """Build a minimal ForwardBatch stub for pooler unit tests."""
    return SimpleNamespace(
        extend_seq_lens=torch.tensor(extend_seq_lens, dtype=torch.long),
        extend_seq_lens_cpu=extend_seq_lens,
        multi_item_delimiter_indices=multi_item_delimiter_indices,
        token_indices_to_pool=token_indices_to_pool,
        dimensions=None,
        return_pooled_hidden_states=return_pooled_hidden_states,
        is_prefill_only=is_prefill_only,
    )


class TestScoreAndPool(CustomTestCase):
    """Unit tests for the score_and_pool helper function."""

    def setUp(self):
        torch.manual_seed(42)
        self.hidden_dim = 8
        self.num_labels = 2
        self.score_head = nn.Linear(self.hidden_dim, self.num_labels, bias=False)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)

    def test_single_item_returns_scores(self):
        """No delimiter indices -> single-item path returns [batch, num_labels]."""
        hidden = torch.randn(8, self.hidden_dim)
        fb = _make_forward_batch(extend_seq_lens=[5, 3])
        input_ids = torch.arange(8)

        out = score_and_pool(self.score_head, self.pooler, hidden, fb, input_ids)

        self.assertIsInstance(out, EmbeddingPoolerOutput)
        self.assertEqual(out.embeddings.shape, (2, self.num_labels))

    def test_mis_returns_per_request_list(self):
        """Delimiter indices provided -> returns a list with one tensor per request."""
        # Sequence: [0, 1, 2, D, 3, 4, 5, D, 6, 7, 8, D]
        # Delimiters at positions 3, 7, 11 -> extract at 2, 6, 10
        input_ids = torch.arange(12)
        hidden = torch.randn(len(input_ids), self.hidden_dim)
        fb = _make_forward_batch(
            extend_seq_lens=[len(input_ids)],
            multi_item_delimiter_indices=[torch.tensor([3, 7, 11])],
        )

        out = score_and_pool(self.score_head, self.pooler, hidden, fb, input_ids)

        self.assertIsInstance(out.embeddings, list)
        self.assertEqual(len(out.embeddings), 1)
        self.assertEqual(out.embeddings[0].shape, (3, self.num_labels))

    def test_mis_batched_splits_per_request(self):
        """Two batched MIS requests -> returns a list of length 2."""
        # Request 1: [10, 11, D, 12, 13, D]  -> delimiters at 2, 5
        # Request 2: [20, 21, 22, D]          -> delimiter at 3
        req1 = [10, 11, 99, 12, 13, 99]
        req2 = [20, 21, 22, 99]
        input_ids = torch.tensor(req1 + req2)
        hidden = torch.randn(len(input_ids), self.hidden_dim)
        fb = _make_forward_batch(
            extend_seq_lens=[len(req1), len(req2)],
            multi_item_delimiter_indices=[
                torch.tensor([2, 5]),
                torch.tensor([3]),
            ],
        )

        out = score_and_pool(self.score_head, self.pooler, hidden, fb, input_ids)

        self.assertIsInstance(out.embeddings, list)
        self.assertEqual(len(out.embeddings), 2)
        self.assertEqual(out.embeddings[0].shape, (2, self.num_labels))
        self.assertEqual(out.embeddings[1].shape, (1, self.num_labels))

    def test_mis_extracts_positions_before_delimiter(self):
        """Verify MIS picks hidden states at index (delimiter_position - 1)."""
        # Delimiters at indices 2 and 5 -> extract hidden at indices 1 and 4
        input_ids = torch.tensor([10, 11, 99, 20, 21, 99])
        hidden = (
            torch.arange(len(input_ids))
            .unsqueeze(1)
            .float()
            .expand(-1, self.hidden_dim)
            .clone()
        )
        fb = _make_forward_batch(
            extend_seq_lens=[len(input_ids)],
            multi_item_delimiter_indices=[torch.tensor([2, 5])],
        )

        identity_head = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        nn.init.eye_(identity_head.weight)

        out = score_and_pool(identity_head, self.pooler, hidden, fb, input_ids)

        scores = out.embeddings[0]
        torch.testing.assert_close(scores[0], hidden[1])
        torch.testing.assert_close(scores[1], hidden[4])

    def test_mis_delimiter_at_position_one(self):
        """Delimiters at positions 1 and 3 extract at indices 0 and 2."""
        input_ids = torch.tensor([10, 99, 11, 99])
        hidden = (
            torch.arange(len(input_ids))
            .unsqueeze(1)
            .float()
            .expand(-1, self.hidden_dim)
            .clone()
        )
        fb = _make_forward_batch(
            extend_seq_lens=[len(input_ids)],
            multi_item_delimiter_indices=[torch.tensor([1, 3])],
        )

        identity_head = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        nn.init.eye_(identity_head.weight)

        out = score_and_pool(identity_head, self.pooler, hidden, fb, input_ids)

        self.assertEqual(len(out.embeddings), 1)
        self.assertEqual(out.embeddings[0].shape[0], 2)
        torch.testing.assert_close(out.embeddings[0][0], hidden[0])
        torch.testing.assert_close(out.embeddings[0][1], hidden[2])

    def test_single_item_scores_match_manual_computation(self):
        """Single-item scores equal score_head applied to pooled hidden states."""
        hidden = torch.randn(8, self.hidden_dim)
        fb = _make_forward_batch(extend_seq_lens=[5, 3])
        input_ids = torch.arange(8)

        out = score_and_pool(self.score_head, self.pooler, hidden, fb, input_ids)

        pooled = self.pooler(hidden, fb).embeddings
        expected = self.score_head(pooled)
        torch.testing.assert_close(out.embeddings, expected)

    def test_mean_pooling_respects_packed_sequence_boundaries(self):
        hidden = torch.tensor([[1.0], [3.0], [7.0], [9.0], [11.0]])
        fb = _make_forward_batch(extend_seq_lens=[2, 3])
        pooler = Pooler(pooling_type=PoolingType.MEAN, normalize=False)

        pooled = pooler(hidden, fb).embeddings

        torch.testing.assert_close(pooled, torch.tensor([[2.0], [9.0]]))

    def test_mean_pooling_short_request_after_long_one(self):
        """On NVIDIA GPUs (segmented reduction), a short request packed after a
        long, large-valued one keeps its mean; a running-sum difference would
        cancel it away."""
        torch.manual_seed(0)
        lens = [4096, 1, 3]
        long = 1e4 + torch.randn(lens[0], 8)
        short = torch.randn(sum(lens[1:]), 8) * 1e-3
        hidden = torch.cat([long, short])
        fb = _make_forward_batch(extend_seq_lens=lens)
        pooler = Pooler(pooling_type=PoolingType.MEAN, normalize=False)

        with patch.object(pooler_module, "_is_cuda", True):
            pooled = pooler(hidden, fb).embeddings

        expected = torch.stack(
            [chunk.double().mean(0) for chunk in hidden.split(lens)]
        ).float()
        torch.testing.assert_close(pooled, expected, rtol=1e-5, atol=1e-6)

    def test_mean_pooling_keeps_a_nan_in_its_request(self):
        """On NVIDIA GPUs a NaN row (in a request or in padding past the last
        request) leaves the other requests' means finite."""
        lens = [2, 3, 1]
        hidden = torch.arange(8, dtype=torch.bfloat16).unsqueeze(-1).repeat(1, 4)
        hidden[3] = float("nan")  # inside request 1
        hidden[7] = float("nan")  # padding row
        fb = _make_forward_batch(extend_seq_lens=lens)
        pooler = Pooler(pooling_type=PoolingType.MEAN, normalize=False)

        with patch.object(pooler_module, "_is_cuda", True):
            pooled = pooler(hidden, fb).embeddings

        self.assertEqual(pooled.dtype, torch.bfloat16)
        self.assertEqual(pooled.isnan().any(-1).tolist(), [False, True, False])
        self.assertEqual(pooled[0].tolist(), [0.5] * 4)
        self.assertEqual(pooled[2].tolist(), [5.0] * 4)

    def test_mean_pooling_paths_agree_on_ordinary_inputs(self):
        """Other devices keep the cumsum; both paths give the same means."""
        torch.manual_seed(1)
        lens = [5, 1, 7]
        hidden = torch.randn(sum(lens), 8)
        fb = _make_forward_batch(extend_seq_lens=lens)
        pooler = Pooler(pooling_type=PoolingType.MEAN, normalize=False)
        means = {}
        for on_cuda in (True, False):
            with patch.object(pooler_module, "_is_cuda", on_cuda):
                means[on_cuda] = pooler(hidden, fb).embeddings
        torch.testing.assert_close(means[True], means[False], rtol=1e-5, atol=1e-6)

    def test_empty_delimiter_indices(self):
        """Empty delimiter tensor per request -> returns list with empty tensor."""
        input_ids = torch.arange(6)
        hidden = torch.randn(6, self.hidden_dim)
        fb = _make_forward_batch(
            extend_seq_lens=[6],
            multi_item_delimiter_indices=[torch.tensor([], dtype=torch.long)],
        )

        out = score_and_pool(self.score_head, self.pooler, hidden, fb, input_ids)

        self.assertIsInstance(out.embeddings, list)
        self.assertEqual(len(out.embeddings), 1)
        self.assertEqual(out.embeddings[0].shape, (0, self.num_labels))

    def test_setwise_returns_per_request_list(self):
        """Anchor positions provided -> one [num_anchors, num_labels] tensor per request."""
        # Sequence: query + items + suffix; anchors at positions 8, 10, 12.
        input_ids = torch.arange(14)
        hidden = torch.randn(len(input_ids), self.hidden_dim)
        fb = _make_forward_batch(
            extend_seq_lens=[len(input_ids)],
            token_indices_to_pool=[torch.tensor([8, 10, 12])],
        )

        out = score_and_pool(self.score_head, self.pooler, hidden, fb, input_ids)

        self.assertIsInstance(out.embeddings, list)
        self.assertEqual(len(out.embeddings), 1)
        # 3 anchors -> 3 score rows (no query-boundary row, unlike MIS).
        self.assertEqual(out.embeddings[0].shape, (3, self.num_labels))

    def test_setwise_extracts_at_anchor_positions(self):
        """Setwise pools AT the anchor index (no delimiter-1 shift)."""
        input_ids = torch.tensor([0, 1, 2, 3, 4, 5])
        hidden = (
            torch.arange(len(input_ids))
            .unsqueeze(1)
            .float()
            .expand(-1, self.hidden_dim)
            .clone()
        )
        fb = _make_forward_batch(
            extend_seq_lens=[len(input_ids)],
            token_indices_to_pool=[torch.tensor([2, 5])],
        )

        identity_head = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        nn.init.eye_(identity_head.weight)

        out = score_and_pool(identity_head, self.pooler, hidden, fb, input_ids)

        scores = out.embeddings[0]
        # Pooled exactly at indices 2 and 5 (not 1 and 4).
        torch.testing.assert_close(scores[0], hidden[2])
        torch.testing.assert_close(scores[1], hidden[5])

    def test_setwise_batched_splits_per_request(self):
        """Two batched setwise requests -> a list of length 2."""
        req1 = list(range(10, 17))  # 7 tokens, anchors at 5, 6
        req2 = list(range(20, 24))  # 4 tokens, anchor at 3
        input_ids = torch.tensor(req1 + req2)
        hidden = torch.randn(len(input_ids), self.hidden_dim)
        fb = _make_forward_batch(
            extend_seq_lens=[len(req1), len(req2)],
            token_indices_to_pool=[
                torch.tensor([5, 6]),
                torch.tensor([3]),
            ],
        )

        out = score_and_pool(self.score_head, self.pooler, hidden, fb, input_ids)

        self.assertIsInstance(out.embeddings, list)
        self.assertEqual(len(out.embeddings), 2)
        self.assertEqual(out.embeddings[0].shape, (2, self.num_labels))
        self.assertEqual(out.embeddings[1].shape, (1, self.num_labels))

    def test_setwise_returns_pooled_hidden_states(self):
        """return_pooled_hidden_states -> per-anchor pre-head hidden states."""
        input_ids = torch.arange(6)
        hidden = torch.randn(len(input_ids), self.hidden_dim)
        fb = _make_forward_batch(
            extend_seq_lens=[len(input_ids)],
            token_indices_to_pool=[torch.tensor([3, 5])],
            return_pooled_hidden_states=True,
        )

        out = score_and_pool(self.score_head, self.pooler, hidden, fb, input_ids)

        self.assertIsNotNone(out.pooled_hidden_states)
        self.assertEqual(len(out.pooled_hidden_states), 1)
        self.assertEqual(out.pooled_hidden_states[0].shape, (2, self.hidden_dim))


if __name__ == "__main__":
    unittest.main()
