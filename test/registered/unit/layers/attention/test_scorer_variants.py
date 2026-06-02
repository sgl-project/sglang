"""Unit tests for the Loop-7 AC-3 non-learned selector variants: length-
conditional hybrid scorer, head-aggregation, anchor-budget, and the graph-safe
default guard. CPU-only.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
    _anchor_positions,
    _compute_logical_token_scores,
    _force_include_anchor,
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


class TestAnchorModes(unittest.TestCase):
    def test_anchor_position_generators(self):
        self.assertEqual(_anchor_positions(16, 3, "recency"), [13, 14, 15])
        self.assertEqual(_anchor_positions(16, 3, "global"), [0, 1, 2])
        self.assertEqual(_anchor_positions(16, 4, "strided"), [0, 5, 10, 15])
        self.assertEqual(_anchor_positions(16, 3, "off"), [])
        # budget > seq_len clamps to seq_len; short sequence
        self.assertEqual(_anchor_positions(2, 5, "recency"), [0, 1])
        self.assertEqual(_anchor_positions(1, 4, "strided"), [0])

    def _forced_real(self, mode, budget):
        # scores favor early tokens; anchors must be forced in regardless.
        scores = torch.tensor([[10.0, 9.0, 8.0, 1.0, 2.0, 3.0, 0.0, 0.0]])
        sel, _ = select_topk_sequence_order(scores, max_top_k=3)  # picks {0,1,2}
        seq_lens = torch.tensor([8], dtype=torch.int32)
        forced, vl = _force_include_anchor(sel, scores, seq_lens, budget, mode)
        real = sorted(p for p in forced[0].tolist() if p >= 0)
        return real, int(vl[0])

    def test_recency_forces_recent_positions(self):
        real, vl = self._forced_real("recency", 2)
        self.assertIn(6, real)
        self.assertIn(7, real)
        self.assertEqual(len(real), 3)  # count preserved
        self.assertEqual(real, sorted(real))  # ascending
        self.assertEqual(vl, 3)

    def test_strided_forces_spread_positions(self):
        real, _ = self._forced_real("strided", 3)  # [0, ~3-4, 7] over [0,8)
        self.assertIn(7, real)  # last strided position
        self.assertEqual(len(real), 3)

    def test_off_is_noop(self):
        real, _ = self._forced_real("off", 4)
        self.assertEqual(real, [0, 1, 2])  # unchanged

    def test_recency_budget_over_topk_forces_most_recent(self):
        # top_k=3, seq_len=8, budget 5 > top_k: must force the most-recent 3.
        real, vl = self._forced_real("recency", 5)
        self.assertEqual(real, [5, 6, 7])
        self.assertEqual(vl, 3)

    def test_strided_budget_over_topk_clamps_to_topk(self):
        # budget 5 > top_k=3: 3 evenly-spaced anchors over [0, 8), not the first 3.
        real, _ = self._forced_real("strided", 5)
        self.assertEqual(len(real), 3)
        self.assertEqual(real, [0, 4, 7])

    def test_recency_budget_over_seq_len(self):
        # budget >= seq_len: still bounded by the selected count (3).
        real, _ = self._forced_real("recency", 10)
        self.assertEqual(real, [5, 6, 7])

    def test_no_duplicates_when_anchor_already_selected(self):
        scores = torch.tensor([[10.0, 9.0, 8.0, 1.0, 2.0, 3.0, 0.0, 0.0]])
        sel, _ = select_topk_sequence_order(scores, max_top_k=3)  # {0,1,2}
        seq_lens = torch.tensor([8], dtype=torch.int32)
        forced, _ = _force_include_anchor(sel, scores, seq_lens, 2, "global")  # {0,1}
        real = [p for p in forced[0].tolist() if p >= 0]
        self.assertEqual(len(real), len(set(real)))  # no dupes
        self.assertEqual(sorted(real), [0, 1, 2])  # already present -> unchanged


class TestPhysicalHybridRejected(unittest.TestCase):
    def test_physical_hybrid_raises(self):
        q = torch.zeros(1, 1, 4)
        q[0, 0, 0] = 1.0
        sig = torch.zeros(1, 1, 1, 4)
        sig[0, 0, 0, 0] = 3.0
        w = torch.ones(1, 1, dtype=torch.bool)
        cs = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.int32)
        cw = torch.ones(1, 1, 4)
        with self.assertRaises(ValueError):
            compute_token_scores(q, sig, w, cs, cw, 0, scorer_norm="hybrid")


class TestScorerDefaultGuard(unittest.TestCase):
    def test_default_vs_variants(self):
        from types import SimpleNamespace as NS
        self.assertTrue(ds_scorer_is_default(None))
        self.assertTrue(ds_scorer_is_default(NS(scorer_norm="off", head_agg="max", anchor_mode="off")))
        self.assertFalse(ds_scorer_is_default(NS(scorer_norm="cosine", head_agg="max", anchor_mode="off")))
        self.assertFalse(ds_scorer_is_default(NS(scorer_norm="off", head_agg="mean", anchor_mode="off")))
        self.assertFalse(ds_scorer_is_default(NS(scorer_norm="off", head_agg="max", anchor_mode="recency")))


class TestNonDefaultScorerGraphGuard(unittest.TestCase):
    """Server init rejects a non-default scorer when CUDA graph is enabled (the
    production-path safety guard), and allows it with --disable-cuda-graph."""

    def _server_args(self, scorer_norm, disable_cuda_graph):
        from types import SimpleNamespace
        cfg = f'{{"channel_mask_path": "/tmp/cm.safetensors", "scorer_norm": "{scorer_norm}"}}'
        return SimpleNamespace(
            enable_double_sparsity=True,
            enable_hisparse=False,
            disaggregation_mode=None,
            double_sparsity_config=cfg,
            disable_cuda_graph=disable_cuda_graph,
            page_size=64,
        )

    def test_non_default_scorer_with_cuda_graph_rejected(self):
        from sglang.srt.layers.attention.double_sparsity.validator import (
            validate_double_sparsity,
        )
        with self.assertRaises(ValueError) as cm:
            validate_double_sparsity(self._server_args("cosine", disable_cuda_graph=False))
        self.assertIn("CUDA graph", str(cm.exception))

    def test_non_default_scorer_eager_passes_guard(self):
        # With --disable-cuda-graph the scorer guard must NOT fire (a later
        # validation check may raise, but not the cuda-graph guard).
        from sglang.srt.layers.attention.double_sparsity.validator import (
            validate_double_sparsity,
        )
        try:
            validate_double_sparsity(self._server_args("cosine", disable_cuda_graph=True))
        except Exception as e:
            # A later validation check may raise (e.g. missing channel-mask file),
            # but the scorer/cuda-graph guard must NOT be what fired.
            self.assertNotIn("CUDA graph", str(e))


class TestRecallOracleGraphGuard(unittest.TestCase):
    """Server init rejects the recall_oracle diagnostic under CUDA graph (the
    hook host-syncs and would record nothing under replay), and allows it with
    --disable-cuda-graph."""

    def _server_args(self, recall_oracle, disable_cuda_graph):
        from types import SimpleNamespace
        cfg = (
            '{"channel_mask_path": "/tmp/cm.safetensors", "recall_oracle": '
            + ("true" if recall_oracle else "false")
            + "}"
        )
        return SimpleNamespace(
            enable_double_sparsity=True,
            enable_hisparse=False,
            disaggregation_mode=None,
            double_sparsity_config=cfg,
            disable_cuda_graph=disable_cuda_graph,
            page_size=64,
        )

    def test_recall_oracle_with_cuda_graph_rejected(self):
        from sglang.srt.layers.attention.double_sparsity.validator import (
            validate_double_sparsity,
        )
        with self.assertRaises(ValueError) as cm:
            validate_double_sparsity(self._server_args(True, disable_cuda_graph=False))
        self.assertIn("recall_oracle", str(cm.exception))

    def test_recall_oracle_eager_passes_guard(self):
        from sglang.srt.layers.attention.double_sparsity.validator import (
            validate_double_sparsity,
        )
        try:
            validate_double_sparsity(self._server_args(True, disable_cuda_graph=True))
        except Exception as e:
            # A later check (missing channel-mask file) may raise, but the
            # recall_oracle/cuda-graph guard must NOT be what fired.
            self.assertNotIn("recall_oracle", str(e))
            self.assertNotIn("CUDA graph", str(e))


if __name__ == "__main__":
    unittest.main()
