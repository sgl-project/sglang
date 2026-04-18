"""Unit tests for score_and_pool in sglang.srt.layers.pooler.

All tests run on CPU — no GPU required.  The global server_args singleton
is mocked so the tests are hermetic.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from sglang.srt.layers.pooler import (
    EmbeddingPoolerOutput,
    Pooler,
    PoolingType,
    score_and_pool,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


def _make_forward_batch(
    extend_seq_lens, is_prefill_only=False, return_pooled_hidden_states=False
):
    """Build a minimal ForwardBatch stub for pooler unit tests."""
    return SimpleNamespace(
        extend_seq_lens=torch.tensor(extend_seq_lens, dtype=torch.long),
        extend_seq_lens_cpu=extend_seq_lens,
        is_prefill_only=is_prefill_only,
        dimensions=None,
        return_pooled_hidden_states=return_pooled_hidden_states,
    )


def _mock_server_args(delimiter=None):
    return SimpleNamespace(multi_item_scoring_delimiter=delimiter)


class TestScoreAndPool(CustomTestCase):
    """Unit tests for the score_and_pool helper function."""

    def setUp(self):
        torch.manual_seed(42)
        self.hidden_dim = 8
        self.num_labels = 2
        self.score_head = nn.Linear(self.hidden_dim, self.num_labels, bias=False)
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)

    @patch("sglang.srt.layers.pooler.get_global_server_args")
    def test_single_item_returns_scores(self, mock_get_args):
        """No delimiter -> single-item path returns [batch, num_labels]."""
        mock_get_args.return_value = _mock_server_args(delimiter=None)

        hidden = torch.randn(8, self.hidden_dim)
        fb = _make_forward_batch(extend_seq_lens=[5, 3])
        input_ids = torch.arange(8)

        out = score_and_pool(self.score_head, self.pooler, hidden, fb, input_ids)

        self.assertIsInstance(out, EmbeddingPoolerOutput)
        self.assertEqual(out.embeddings.shape, (2, self.num_labels))

    @patch("sglang.srt.layers.pooler.get_global_server_args")
    def test_mis_returns_per_request_list(self, mock_get_args):
        """Delimiter found -> returns a list with one tensor per request."""
        delimiter_token = 99
        mock_get_args.return_value = _mock_server_args(delimiter=delimiter_token)

        input_ids = torch.tensor(
            [
                0,
                1,
                2,
                delimiter_token,
                3,
                4,
                5,
                delimiter_token,
                6,
                7,
                8,
                delimiter_token,
            ]
        )
        hidden = torch.randn(len(input_ids), self.hidden_dim)
        fb = _make_forward_batch(extend_seq_lens=[len(input_ids)], is_prefill_only=True)

        out = score_and_pool(self.score_head, self.pooler, hidden, fb, input_ids)

        self.assertIsInstance(out.embeddings, list)
        self.assertEqual(len(out.embeddings), 1)
        self.assertEqual(out.embeddings[0].shape, (3, self.num_labels))

    @patch("sglang.srt.layers.pooler.get_global_server_args")
    def test_mis_batched_splits_per_request(self, mock_get_args):
        """Two batched MIS requests -> returns a list of length 2."""
        delimiter_token = 99
        mock_get_args.return_value = _mock_server_args(delimiter=delimiter_token)

        # Request 1: [10, 11, delim, 12, 13, delim]  -> 2 delimiters
        # Request 2: [20, 21, 22, delim]              -> 1 delimiter
        req1 = [10, 11, delimiter_token, 12, 13, delimiter_token]
        req2 = [20, 21, 22, delimiter_token]
        input_ids = torch.tensor(req1 + req2)
        hidden = torch.randn(len(input_ids), self.hidden_dim)
        fb = _make_forward_batch(
            extend_seq_lens=[len(req1), len(req2)], is_prefill_only=True
        )

        out = score_and_pool(self.score_head, self.pooler, hidden, fb, input_ids)

        self.assertIsInstance(out.embeddings, list)
        self.assertEqual(len(out.embeddings), 2)
        self.assertEqual(out.embeddings[0].shape, (2, self.num_labels))
        self.assertEqual(out.embeddings[1].shape, (1, self.num_labels))

    @patch("sglang.srt.layers.pooler.get_global_server_args")
    def test_mis_falls_back_when_no_delimiters_in_input(self, mock_get_args):
        """Delimiter configured but absent from input_ids -> single-item fallback."""
        mock_get_args.return_value = _mock_server_args(delimiter=99)

        input_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        hidden = torch.randn(8, self.hidden_dim)
        fb = _make_forward_batch(extend_seq_lens=[5, 3], is_prefill_only=True)

        out = score_and_pool(self.score_head, self.pooler, hidden, fb, input_ids)

        self.assertIsInstance(out.embeddings, torch.Tensor)
        self.assertEqual(out.embeddings.shape, (2, self.num_labels))

    @patch("sglang.srt.layers.pooler.get_global_server_args")
    def test_mis_falls_back_when_not_prefill_only(self, mock_get_args):
        """Delimiter configured, is_prefill_only=False -> single-item fallback."""
        mock_get_args.return_value = _mock_server_args(delimiter=99)

        input_ids = torch.tensor([0, 1, 2, 99, 3, 4, 5, 99])
        hidden = torch.randn(8, self.hidden_dim)
        fb = _make_forward_batch(extend_seq_lens=[5, 3], is_prefill_only=False)

        out = score_and_pool(self.score_head, self.pooler, hidden, fb, input_ids)

        self.assertIsInstance(out.embeddings, torch.Tensor)
        self.assertEqual(out.embeddings.shape, (2, self.num_labels))

    @patch("sglang.srt.layers.pooler.get_global_server_args")
    def test_mis_extracts_positions_before_delimiter(self, mock_get_args):
        """Verify MIS picks hidden states at index (delimiter_position - 1)."""
        delimiter_token = 99
        mock_get_args.return_value = _mock_server_args(delimiter=delimiter_token)

        # Delimiters at indices 2 and 5 -> extract hidden at indices 1 and 4
        input_ids = torch.tensor([10, 11, delimiter_token, 20, 21, delimiter_token])
        hidden = (
            torch.arange(len(input_ids))
            .unsqueeze(1)
            .float()
            .expand(-1, self.hidden_dim)
            .clone()
        )
        fb = _make_forward_batch(extend_seq_lens=[len(input_ids)], is_prefill_only=True)

        identity_head = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        nn.init.eye_(identity_head.weight)

        out = score_and_pool(identity_head, self.pooler, hidden, fb, input_ids)

        scores = out.embeddings[0]
        torch.testing.assert_close(scores[0], hidden[1])
        torch.testing.assert_close(scores[1], hidden[4])

    @patch("sglang.srt.layers.pooler.get_global_server_args")
    def test_mis_ignores_delimiter_at_position_zero(self, mock_get_args):
        """A delimiter at flat index 0 has no preceding token and must be skipped."""
        delimiter_token = 99
        mock_get_args.return_value = _mock_server_args(delimiter=delimiter_token)

        # Delimiter at index 0 should be ignored; only the one at index 3 counts
        input_ids = torch.tensor([delimiter_token, 10, 11, delimiter_token])
        hidden = (
            torch.arange(len(input_ids))
            .unsqueeze(1)
            .float()
            .expand(-1, self.hidden_dim)
            .clone()
        )
        fb = _make_forward_batch(extend_seq_lens=[len(input_ids)], is_prefill_only=True)

        identity_head = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        nn.init.eye_(identity_head.weight)

        out = score_and_pool(identity_head, self.pooler, hidden, fb, input_ids)

        self.assertEqual(len(out.embeddings), 1)
        self.assertEqual(out.embeddings[0].shape[0], 1)
        torch.testing.assert_close(out.embeddings[0][0], hidden[2])

    @patch("sglang.srt.layers.pooler.get_global_server_args")
    def test_single_item_scores_match_manual_computation(self, mock_get_args):
        """Single-item scores equal score_head applied to all tokens then pooled."""
        mock_get_args.return_value = _mock_server_args(delimiter=None)

        hidden = torch.randn(8, self.hidden_dim)
        fb = _make_forward_batch(extend_seq_lens=[5, 3])
        input_ids = torch.arange(8)

        out = score_and_pool(self.score_head, self.pooler, hidden, fb, input_ids)

        # score-first-then-pool: matches the original Qwen3/Qwen2 classification forward
        logits = self.score_head(hidden)
        expected = self.pooler(logits, fb).embeddings
        torch.testing.assert_close(out.embeddings, expected)


if __name__ == "__main__":
    unittest.main()
