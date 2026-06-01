"""Unit tests for the Double Sparsity selection-recall oracle (AC-1 diagnostic math).

CPU-only. Pins the score-only diagnostic contract:
- ranking honors the selector's (score DESC, position ASC) tie-break;
- the multi-token needle rule is worst-rank / all-needle-tokens-in-top-K;
- score-only recall@K works for K beyond the locked budget (no decode);
- the invariant recall@budget == selected_contains_needle holds against the
  real ``select_topk_sequence_order`` output;
- empty / out-of-range needle spans fail fast (no guessing).
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.attention.double_sparsity.selection_recall_oracle import (
    DEFAULT_RECALL_K_VALUES,
    needle_all_tokens_in_topk,
    needle_best_rank,
    needle_ranks,
    needle_worst_rank,
    score_only_recall_at_k,
    selected_contains_needle,
)
from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
    select_topk_sequence_order,
)


class TestNeedleRankTieBreak(unittest.TestCase):
    def test_distinct_scores_rank_is_count_strictly_greater(self):
        # order by score desc: pos2(9), pos4(7), pos0(5), pos1(3), pos3(1)
        scores = torch.tensor([[5.0, 3.0, 9.0, 1.0, 7.0]])
        ranks = needle_ranks(scores, torch.tensor([0, 1, 2, 3, 4]))
        self.assertEqual(ranks.tolist(), [[2, 3, 0, 4, 1]])

    def test_equal_scores_break_toward_lower_position(self):
        scores = torch.tensor([[5.0, 5.0, 5.0]])
        ranks = needle_ranks(scores, torch.tensor([0, 1, 2]))
        # lower position ranks higher (smaller rank) on a tie
        self.assertEqual(ranks.tolist(), [[0, 1, 2]])

    def test_matches_select_topk_ordering_on_ties(self):
        # A tie that the selector resolves toward lower positions; the oracle
        # rank must agree with which tokens select_topk_sequence_order keeps.
        scores = torch.tensor([[1.0, 5.0, 5.0, 5.0, 2.0]])
        sel, _ = select_topk_sequence_order(scores, max_top_k=2)
        # top-2 by (score desc, pos asc) = positions {1, 2}
        self.assertEqual(sorted(p for p in sel[0].tolist() if p >= 0), [1, 2])
        self.assertTrue(bool(needle_all_tokens_in_topk(scores, torch.tensor([1]), 2)))
        self.assertFalse(bool(needle_all_tokens_in_topk(scores, torch.tensor([3]), 2)))

    def test_neg_inf_position_ranks_below_finite(self):
        scores = torch.tensor([[5.0, float("-inf"), 7.0]])
        ranks = needle_ranks(scores, torch.tensor([1]))
        # -inf is outranked by both finite tokens
        self.assertEqual(ranks.tolist(), [[2]])


class TestMultiTokenWorstRank(unittest.TestCase):
    def test_worst_and_best_rank_over_span(self):
        scores = torch.tensor([[5.0, 3.0, 9.0, 1.0, 7.0]])
        span = torch.tensor([0, 4])  # ranks 2 and 1
        self.assertEqual(needle_worst_rank(scores, span).tolist(), [2])
        self.assertEqual(needle_best_rank(scores, span).tolist(), [1])

    def test_all_in_topk_uses_worst_not_best(self):
        scores = torch.tensor([[5.0, 3.0, 9.0, 1.0, 7.0]])
        span = torch.tensor([0, 4])  # worst rank 2
        self.assertFalse(bool(needle_all_tokens_in_topk(scores, span, 2)))  # 2 < 2 false
        self.assertTrue(bool(needle_all_tokens_in_topk(scores, span, 3)))  # 2 < 3 true


class TestScoreOnlyRecallCurve(unittest.TestCase):
    def test_default_k_grid_and_beyond_budget(self):
        # 6000 tokens with a known needle rank; recall@K beyond 2048 is score-only.
        torch.manual_seed(0)
        scores = torch.randn(1, 6000)
        # Force a single needle whose rank we control by making it the 3000th-best.
        order = torch.argsort(scores[0], descending=True)
        needle_pos = int(order[3000].item())  # rank exactly 3000
        curve = score_only_recall_at_k(scores, torch.tensor([needle_pos]))
        self.assertEqual(set(curve.keys()), set(int(k) for k in DEFAULT_RECALL_K_VALUES))
        self.assertFalse(bool(curve[2048]))  # rank 3000 not in top-2048
        self.assertTrue(bool(curve[4096]))  # but is in top-4096 (score-only)
        self.assertTrue(bool(curve[8192]))

    def test_per_row_independent(self):
        scores = torch.tensor([[9.0, 1.0, 2.0], [1.0, 9.0, 2.0]])
        span = torch.tensor([0])  # row0 rank 0; row1 rank 2
        worst = needle_worst_rank(scores, span)
        self.assertEqual(worst.tolist(), [0, 2])


class TestRecall2048Invariant(unittest.TestCase):
    def test_recall_at_budget_equals_selected_contains_needle(self):
        # T > budget so selection genuinely excludes tokens.
        scores = torch.tensor([[5.0, 3.0, 9.0, 1.0, 7.0, 8.0, 2.0, 6.0, 4.0, 0.0]])
        budget = 4
        selected, _ = select_topk_sequence_order(scores, max_top_k=budget)

        for needle in ([4], [0], [2, 5], [0, 3]):
            span = torch.tensor(needle)
            oracle = bool(needle_all_tokens_in_topk(scores, span, budget))
            contains = bool(selected_contains_needle(selected, span))
            self.assertEqual(
                oracle,
                contains,
                msg=f"invariant violated for needle {needle}: "
                f"recall@{budget}={oracle} vs selected_contains_needle={contains}",
            )

    def test_selected_contains_needle_ignores_pad(self):
        selected = torch.tensor([[2, 4, 5, 7, -1, -1]])
        self.assertTrue(bool(selected_contains_needle(selected, torch.tensor([4, 7]))))
        self.assertFalse(bool(selected_contains_needle(selected, torch.tensor([3]))))
        # a needle equal to the pad sentinel value can never be "selected"
        self.assertFalse(bool(selected_contains_needle(selected, torch.tensor([4, 6]))))


class TestFailFast(unittest.TestCase):
    def test_empty_needle_span_raises(self):
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError):
            needle_ranks(scores, torch.tensor([], dtype=torch.int64))

    def test_out_of_range_needle_raises(self):
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError):
            needle_ranks(scores, torch.tensor([3]))  # max valid index is 2
        with self.assertRaises(ValueError):
            needle_ranks(scores, torch.tensor([-1]))

    def test_non_2d_scores_raises(self):
        with self.assertRaises(ValueError):
            needle_ranks(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([0]))

    def test_zero_k_raises(self):
        scores = torch.tensor([[1.0, 2.0, 3.0]])
        with self.assertRaises(ValueError):
            needle_all_tokens_in_topk(scores, torch.tensor([0]), 0)


if __name__ == "__main__":
    unittest.main()
