"""Unit tests for the Loop-7 AC-3 non-learned selector variants: length-
conditional hybrid scorer, head-aggregation, anchor-budget, and the graph-safe
default guard. CPU-only.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
    _compute_logical_token_scores,
    _force_include_recency_anchor,
    compute_token_scores,
    ds_scorer_is_default,
    select_topk_sequence_order,
)


def _logical_inputs(seq_lens):
    """Identity-mapped logical-mode inputs. token0=needle (aligned, modest mag),
    token1=background (high mag, off-axis), rest small. H=1, D=4, max_tokens=16."""
    bs = len(seq_lens)
    queries = torch.zeros(bs, 1, 4)
    queries[:, 0, 0] = 1.0  # [1,0,0,0] per row
    sigs = torch.zeros(1, 16, 1, 4)
    sigs[0, 0, 0] = torch.tensor([3.0, 0.0, 0.0, 0.0])  # needle (raw dot 3, cos 1.0)
    sigs[0, 1, 0] = torch.tensor([4.0, 4.0, 0.0, 0.0])  # background (raw 4, cos .707)
    for p in range(2, 16):
        sigs[0, p, 0] = torch.tensor([0.1, 0.0, 0.0, 0.0])
    written = torch.ones(1, 16, dtype=torch.bool)
    chan_sel = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int32)
    chan_w = torch.ones(1, 1, 4, dtype=torch.float32)
    req_pool = torch.arange(bs, dtype=torch.int32)
    req_to_token = torch.arange(16, dtype=torch.int32).unsqueeze(0).expand(bs, -1).contiguous()
    sl = torch.tensor(seq_lens, dtype=torch.int32)
    return dict(
        queries=queries, token_signatures=sigs, written=written,
        channel_selection=chan_sel, channel_weights=chan_w, layer_id=0,
        req_pool_indices=req_pool, req_to_token=req_to_token, seq_lens=sl,
        max_seq_len=16,
    )


class TestHybridScorer(unittest.TestCase):
    def test_hybrid_picks_raw_below_threshold_cosine_above(self):
        kw = _logical_inputs([4, 16])  # row0 short, row1 long; threshold=8
        raw = _compute_logical_token_scores(**kw, scorer_norm="off")
        cos = _compute_logical_token_scores(**kw, scorer_norm="cosine")
        hyb = _compute_logical_token_scores(**kw, scorer_norm="hybrid", hybrid_threshold=8)
        # row0 (seq_len 4 <= 8) -> raw; row1 (16 > 8) -> cosine
        self.assertTrue(torch.allclose(hyb[0, :4], raw[0, :4]))
        self.assertTrue(torch.allclose(hyb[1], cos[1]))
        # and raw vs cosine genuinely differ on the needle-vs-background ranking
        self.assertGreater(raw[0, 1].item(), raw[0, 0].item())  # raw: bg > needle
        self.assertGreater(cos[1, 0].item(), cos[1, 1].item())  # cosine: needle > bg

    def test_hybrid_off_equals_raw(self):
        kw = _logical_inputs([4, 16])
        self.assertTrue(
            torch.equal(
                _compute_logical_token_scores(**kw, scorer_norm="off"),
                _compute_logical_token_scores(**kw),  # default
            )
        )


class TestHeadAggregation(unittest.TestCase):
    def test_mean_differs_from_max_multi_head(self):
        # 2 heads; query aligns with head0 only -> max != mean.
        q = torch.zeros(1, 2, 4)
        q[0, 0, 0] = 1.0
        sig = torch.zeros(1, 1, 2, 4)
        sig[0, 0, 0] = torch.tensor([5.0, 0, 0, 0])  # head0 dot 5
        sig[0, 0, 1] = torch.tensor([1.0, 0, 0, 0])  # head1 dot ~0 (q head1 = 0)
        written = torch.ones(1, 1, dtype=torch.bool)
        cs = torch.tensor([[[0, 1, 2, 3], [0, 1, 2, 3]]], dtype=torch.int32)
        cw = torch.ones(1, 2, 4)
        smax = compute_token_scores(q, sig, written, cs, cw, 0, head_agg="max")
        smean = compute_token_scores(q, sig, written, cs, cw, 0, head_agg="mean")
        self.assertAlmostEqual(smax[0, 0].item(), 5.0, places=4)   # max over heads
        self.assertAlmostEqual(smean[0, 0].item(), 2.5, places=4)  # (5+0)/2
        self.assertNotAlmostEqual(smax[0, 0].item(), smean[0, 0].item())


class TestAnchorBudget(unittest.TestCase):
    def test_forces_recent_positions_into_selection(self):
        # scores favor early tokens; anchor must force in the most-recent ones.
        scores = torch.tensor([[10.0, 9.0, 8.0, 1.0, 2.0, 3.0, 0.0, 0.0]])
        sel, _ = select_topk_sequence_order(scores, max_top_k=3)  # picks {0,1,2}
        self.assertEqual(sorted(p for p in sel[0].tolist() if p >= 0), [0, 1, 2])
        seq_lens = torch.tensor([8], dtype=torch.int32)
        forced, vl = _force_include_recency_anchor(sel, scores, seq_lens, anchor_budget=2)
        real = sorted(p for p in forced[0].tolist() if p >= 0)
        # positions 6 and 7 (the 2 most recent of seq_len 8) now present, count kept
        self.assertIn(6, real)
        self.assertIn(7, real)
        self.assertEqual(len(real), 3)
        self.assertEqual(int(vl[0]), 3)


class TestScorerDefaultGuard(unittest.TestCase):
    def test_default_vs_variants(self):
        from types import SimpleNamespace as NS
        self.assertTrue(ds_scorer_is_default(None))
        self.assertTrue(ds_scorer_is_default(NS(scorer_norm="off", head_agg="max", anchor_budget=0)))
        self.assertFalse(ds_scorer_is_default(NS(scorer_norm="cosine", head_agg="max", anchor_budget=0)))
        self.assertFalse(ds_scorer_is_default(NS(scorer_norm="off", head_agg="mean", anchor_budget=0)))
        self.assertFalse(ds_scorer_is_default(NS(scorer_norm="off", head_agg="max", anchor_budget=64)))


if __name__ == "__main__":
    unittest.main()
