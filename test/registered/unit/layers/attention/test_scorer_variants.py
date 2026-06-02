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


class TestScorerGraphSafeGuard(unittest.TestCase):
    """As of R6, scorer_norm (cosine/hybrid) + head_agg (mean) are graph-safe and
    pass the startup guard under CUDA graph; only a non-default anchor_mode is
    rejected (still eager-only)."""

    def _server_args(self, extra_cfg, disable_cuda_graph):
        from types import SimpleNamespace
        cfg = '{"channel_mask_path": "/tmp/cm.safetensors"' + extra_cfg + "}"
        return SimpleNamespace(
            enable_double_sparsity=True,
            enable_hisparse=False,
            disaggregation_mode=None,
            double_sparsity_config=cfg,
            disable_cuda_graph=disable_cuda_graph,
            page_size=64,
        )

    def test_graph_safe_scorer_passes_under_cuda_graph(self):
        # cosine/hybrid/mean are graph-safe -> the cuda-graph guard must NOT fire
        # under CUDA graph (a later check, e.g. missing mask file, may raise).
        from sglang.srt.layers.attention.double_sparsity.validator import (
            validate_double_sparsity,
        )
        for extra in (
            ', "scorer_norm": "cosine"',
            ', "scorer_norm": "hybrid"',
            ', "head_agg": "mean"',
        ):
            try:
                validate_double_sparsity(self._server_args(extra, disable_cuda_graph=False))
            except Exception as e:
                self.assertNotIn("CUDA graph", str(e), f"cfg {extra} wrongly graph-rejected")

    def test_non_default_anchor_passes_under_cuda_graph(self):
        # As of R9 anchor_mode is graph-safe (tensorized force-include) — the
        # cuda-graph guard must NOT fire under CUDA graph (a later check, e.g.
        # missing mask file, may raise).
        from sglang.srt.layers.attention.double_sparsity.validator import (
            validate_double_sparsity,
        )
        try:
            validate_double_sparsity(
                self._server_args(', "anchor_mode": "recency", "anchor_budget": 4', disable_cuda_graph=False)
            )
        except Exception as e:
            self.assertNotIn("CUDA graph", str(e), "anchor wrongly graph-rejected (R9 graph-safe)")
            self.assertNotIn("graph-safe", str(e))


class TestGraphSafeScorerEqualsEager(unittest.TestCase):
    """R6: the graph-safe Triton scorer produces selection IDENTICAL to the eager
    scorer for raw/cosine/hybrid x max/mean, on fp16 and int8 signatures (GPU)."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_graph_safe_matches_eager_all_variants(self):
        import itertools
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            retrieve_topk_via_labels, retrieve_topk_graph_safe,
        )
        from sglang.srt.layers.attention.double_sparsity.cuda_graph import (
            allocate_graph_state,
        )
        dev = torch.device("cuda")
        torch.manual_seed(0)
        BS, H, LD, MAXT, MSL, K, HYB = 3, 8, 8, 512, 320, 64, 128
        for int8 in (False, True):
            q = torch.randn(BS, H, LD, device=dev)
            if int8:
                sig = torch.randint(-127, 127, (1, MAXT, H, LD), device=dev, dtype=torch.int8)
                scales = torch.rand(1, MAXT, H, device=dev, dtype=torch.float16) * 0.1 + 0.01
            else:
                sig = torch.randn(1, MAXT, H, LD, device=dev, dtype=torch.float16)
                scales = None
            written = torch.ones(1, MAXT, dtype=torch.bool, device=dev)
            chsel = torch.arange(LD, device=dev).view(1, 1, -1).expand(1, H, -1).to(torch.int32).contiguous()
            chw = torch.ones(1, H, LD, device=dev)
            rpi = torch.arange(BS, device=dev, dtype=torch.int32)
            rtt = torch.arange(MSL, device=dev, dtype=torch.int32).unsqueeze(0).expand(BS, -1).contiguous()
            sl = torch.tensor([60, 200, MSL], device=dev, dtype=torch.int32)  # below/above HYB
            st = allocate_graph_state(max_bs=BS, max_top_k=K, max_seq_len=MSL, device=dev)
            AB = 16  # anchor_budget (<= K)
            for sn, ha, am in itertools.product(
                ("off", "cosine", "hybrid"), ("max", "mean"),
                ("off", "recency", "global", "strided"),
            ):
                idx_e, vl_e = retrieve_topk_via_labels(
                    queries=q, token_signatures=sig, written=written, channel_selection=chsel,
                    channel_weights=chw, layer_id=0, max_top_k=K, req_pool_indices=rpi,
                    req_to_token=rtt, seq_lens=sl, max_seq_len=MSL, token_scales=scales,
                    scorer_norm=sn, head_agg=ha, hybrid_threshold=HYB,
                    anchor_mode=am, anchor_budget=AB,
                )
                st.selected_indices.fill_(-1); st.valid_lengths.fill_(0)
                retrieve_topk_graph_safe(
                    queries=q, token_signatures=sig, written=written, channel_selection=chsel,
                    channel_weights=chw, layer_id=0, req_pool_indices=rpi, req_to_token=rtt,
                    seq_lens=sl, max_seq_len=MSL, max_top_k=K,
                    out_indices=st.selected_indices, out_lengths=st.valid_lengths,
                    scratch_scores=st.scratch_scores, scratch_topk_values=st.scratch_topk_values,
                    scratch_topk_indices=st.scratch_topk_indices, scratch_invalid_mask=st.scratch_invalid_mask,
                    scratch_sorted_vals=st.scratch_sorted_vals, scratch_boundary=st.scratch_boundary,
                    scratch_valid_i64=st.scratch_valid_i64, scratch_throwaway_idx=st.scratch_throwaway_idx,
                    token_scales=scales, scorer_norm=sn, head_agg=ha, hybrid_threshold=HYB,
                    anchor_mode=am, anchor_budget=AB,
                )
                tag = "int8" if int8 else "fp16"
                self.assertTrue(
                    torch.equal(st.selected_indices[:BS, :K], idx_e.to(torch.int32)),
                    f"[{tag}] sn={sn} ha={ha} am={am}: graph-safe indices != eager",
                )
                self.assertTrue(
                    torch.equal(st.valid_lengths[:BS], vl_e.to(torch.int32)),
                    f"[{tag}] sn={sn} ha={ha} am={am}: graph-safe lengths != eager",
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_anchor_over_budget_graph_matches_eager(self):
        """R10: over-budget anchor (anchor_budget > top_k) is bit-identical
        eager-vs-graph (substantiates the over-budget coverage claim + exercises
        the A=min(anchor_budget,K,max_seq) temp-shape clamp)."""
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            retrieve_topk_via_labels, retrieve_topk_graph_safe,
        )
        from sglang.srt.layers.attention.double_sparsity.cuda_graph import (
            allocate_graph_state,
        )
        dev = torch.device("cuda")
        torch.manual_seed(3)
        BS, H, LD, MAXT, MSL, K = 2, 4, 8, 256, 96, 16
        AB = K + 40  # anchor_budget WELL over top_k
        q = torch.randn(BS, H, LD, device=dev)
        sig = torch.randn(1, MAXT, H, LD, device=dev, dtype=torch.float16)
        written = torch.ones(1, MAXT, dtype=torch.bool, device=dev)
        chsel = torch.arange(LD, device=dev).view(1, 1, -1).expand(1, H, -1).to(torch.int32).contiguous()
        chw = torch.ones(1, H, LD, device=dev)
        rpi = torch.arange(BS, device=dev, dtype=torch.int32)
        rtt = torch.arange(MSL, device=dev, dtype=torch.int32).unsqueeze(0).expand(BS, -1).contiguous()
        sl = torch.tensor([12, MSL], device=dev, dtype=torch.int32)  # one with seq_len<K
        st = allocate_graph_state(max_bs=BS, max_top_k=K, max_seq_len=MSL, device=dev)
        for am in ("recency", "global", "strided"):
            idx_e, vl_e = retrieve_topk_via_labels(
                queries=q, token_signatures=sig, written=written, channel_selection=chsel,
                channel_weights=chw, layer_id=0, max_top_k=K, req_pool_indices=rpi,
                req_to_token=rtt, seq_lens=sl, max_seq_len=MSL,
                anchor_mode=am, anchor_budget=AB,
            )
            st.selected_indices.fill_(-1); st.valid_lengths.fill_(0)
            retrieve_topk_graph_safe(
                queries=q, token_signatures=sig, written=written, channel_selection=chsel,
                channel_weights=chw, layer_id=0, req_pool_indices=rpi, req_to_token=rtt,
                seq_lens=sl, max_seq_len=MSL, max_top_k=K,
                out_indices=st.selected_indices, out_lengths=st.valid_lengths,
                scratch_scores=st.scratch_scores, scratch_topk_values=st.scratch_topk_values,
                scratch_topk_indices=st.scratch_topk_indices, scratch_invalid_mask=st.scratch_invalid_mask,
                scratch_sorted_vals=st.scratch_sorted_vals, scratch_boundary=st.scratch_boundary,
                scratch_valid_i64=st.scratch_valid_i64, scratch_throwaway_idx=st.scratch_throwaway_idx,
                anchor_mode=am, anchor_budget=AB,
            )
            self.assertTrue(torch.equal(st.selected_indices[:BS, :K], idx_e.to(torch.int32)), f"over-budget am={am} idx")
            self.assertTrue(torch.equal(st.valid_lengths[:BS], vl_e.to(torch.int32)), f"over-budget am={am} len")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_anchor_graph_safe_replay_zero_alloc(self):
        """R9: a hybrid+anchor graph-safe selection captured in a real CUDA graph
        replays byte-identically + with zero new allocations."""
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            retrieve_topk_via_labels, retrieve_topk_graph_safe,
        )
        from sglang.srt.layers.attention.double_sparsity.cuda_graph import (
            allocate_graph_state, assert_no_alloc_in_region,
        )
        dev = torch.device("cuda")
        torch.manual_seed(1)
        BS, H, LD, MAXT, MSL, K, AB = 2, 4, 8, 256, 160, 32, 8
        q = torch.randn(BS, H, LD, device=dev)
        sig = torch.randn(1, MAXT, H, LD, device=dev, dtype=torch.float16)
        written = torch.ones(1, MAXT, dtype=torch.bool, device=dev)
        chsel = torch.arange(LD, device=dev).view(1, 1, -1).expand(1, H, -1).to(torch.int32).contiguous()
        chw = torch.ones(1, H, LD, device=dev)
        rpi = torch.arange(BS, device=dev, dtype=torch.int32)
        rtt = torch.arange(MSL, device=dev, dtype=torch.int32).unsqueeze(0).expand(BS, -1).contiguous()
        sl = torch.tensor([90, MSL], device=dev, dtype=torch.int32)
        kw = dict(scorer_norm="hybrid", head_agg="mean", hybrid_threshold=64,
                  anchor_mode="recency", anchor_budget=AB)
        idx_e, vl_e = retrieve_topk_via_labels(
            queries=q, token_signatures=sig, written=written, channel_selection=chsel,
            channel_weights=chw, layer_id=0, max_top_k=K, req_pool_indices=rpi,
            req_to_token=rtt, seq_lens=sl, max_seq_len=MSL, **kw,
        )
        st = allocate_graph_state(max_bs=BS, max_top_k=K, max_seq_len=MSL, device=dev)

        def _call():
            retrieve_topk_graph_safe(
                queries=q, token_signatures=sig, written=written, channel_selection=chsel,
                channel_weights=chw, layer_id=0, req_pool_indices=rpi, req_to_token=rtt,
                seq_lens=sl, max_seq_len=MSL, max_top_k=K,
                out_indices=st.selected_indices, out_lengths=st.valid_lengths,
                scratch_scores=st.scratch_scores, scratch_topk_values=st.scratch_topk_values,
                scratch_topk_indices=st.scratch_topk_indices, scratch_invalid_mask=st.scratch_invalid_mask,
                scratch_sorted_vals=st.scratch_sorted_vals, scratch_boundary=st.scratch_boundary,
                scratch_valid_i64=st.scratch_valid_i64, scratch_throwaway_idx=st.scratch_throwaway_idx,
                **kw,
            )

        # Warmup on a side stream (required before CUDA graph capture).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _call()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _call()
        st.selected_indices.fill_(-1); st.valid_lengths.fill_(0)
        with assert_no_alloc_in_region("anchor-graph-replay"):
            g.replay()
            torch.cuda.synchronize()
        self.assertTrue(torch.equal(st.selected_indices[:BS, :K], idx_e.to(torch.int32)))
        self.assertTrue(torch.equal(st.valid_lengths[:BS], vl_e.to(torch.int32)))


class TestScorerGraphSafePredicate(unittest.TestCase):
    def test_graph_safe_vs_default(self):
        from types import SimpleNamespace as NS
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            ds_scorer_is_graph_safe,
        )
        # As of R9 ALL non-learned variants are graph-safe (scorer_norm + head_agg
        # [R6], anchor_mode [R9]).
        self.assertTrue(ds_scorer_is_graph_safe(None))
        self.assertTrue(ds_scorer_is_graph_safe(NS(scorer_norm="cosine", head_agg="mean", anchor_mode="off")))
        self.assertTrue(ds_scorer_is_graph_safe(NS(scorer_norm="hybrid", head_agg="max", anchor_mode="off")))
        self.assertTrue(ds_scorer_is_graph_safe(NS(scorer_norm="off", head_agg="max", anchor_mode="recency")))
        self.assertTrue(ds_scorer_is_graph_safe(NS(scorer_norm="hybrid", head_agg="mean", anchor_mode="strided")))


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


class TestLiftedBudgetABI(unittest.TestCase):
    """AC-4 task13: the opt-in lifted-budget ABI (config fields + validation)."""

    def _parse(self, extra):
        from sglang.srt.layers.attention.double_sparsity.config import (
            parse_double_sparsity_config,
        )
        return parse_double_sparsity_config(
            '{"channel_mask_path": "/tmp/cm.safetensors"' + extra + "}"
        )

    def test_default_off(self):
        c = self._parse("")
        self.assertFalse(c.enable_lifted_budget_decode)
        self.assertEqual(c.lifted_budget_top_k, 0)

    def test_valid_lifted_budget(self):
        c = self._parse(', "enable_lifted_budget_decode": true, "lifted_budget_top_k": 4096')
        self.assertTrue(c.enable_lifted_budget_decode)
        self.assertEqual(c.lifted_budget_top_k, 4096)

    def test_rejects_lifted_budget_le_top_k(self):
        with self.assertRaises(ValueError) as cm:
            self._parse(', "enable_lifted_budget_decode": true, "lifted_budget_top_k": 2048')
        self.assertIn("lifted_budget_top_k", str(cm.exception))

    def test_rejects_lifted_budget_without_flag(self):
        # Set without the enable flag would silently no-op -> fail closed.
        with self.assertRaises(ValueError) as cm:
            self._parse(', "lifted_budget_top_k": 4096')
        self.assertIn("enable_lifted_budget_decode", str(cm.exception))

    def test_rejects_flag_without_budget(self):
        with self.assertRaises(ValueError):
            self._parse(', "enable_lifted_budget_decode": true')

    def test_abi_not_max_top_k_or_twilight(self):
        # The reserved Twilight fields must still be rejected as unknown keys;
        # the lifted budget is the ONLY sanctioned wider-than-index_topk mechanism.
        for bad in ('"max_top_k": 4096', '"selection_mode": "top_p"', '"top_p": 0.9'):
            with self.assertRaises(ValueError):
                self._parse(", " + bad)

    def test_rejects_lifted_budget_not_multiple_of_128(self):
        # flash_mla_sparse_fwd tiles topk by 128 (topk % (2*B_TOPK) == 0).
        with self.assertRaises(ValueError) as cm:
            self._parse(
                ', "enable_lifted_budget_decode": true, "lifted_budget_top_k": 4097'
            )
        self.assertIn("128", str(cm.exception))

    def test_accepts_lifted_budget_multiple_of_128(self):
        c = self._parse(
            ', "enable_lifted_budget_decode": true, "lifted_budget_top_k": 8192'
        )
        self.assertEqual(c.lifted_budget_top_k, 8192)

    def _server_args(self, cfg_extra, top_k, disable_cuda_graph=False):
        from types import SimpleNamespace
        cfg = (
            '{"channel_mask_path": "/tmp/cm.safetensors", "top_k": %d' % top_k
            + cfg_extra + "}"
        )
        hf = object()
        return SimpleNamespace(
            enable_double_sparsity=True, enable_hisparse=False, disaggregation_mode=None,
            double_sparsity_config=cfg, disable_cuda_graph=disable_cuda_graph,
            page_size=64, kv_cache_dtype="auto",
            get_model_config=lambda: SimpleNamespace(hf_config=hf),
        )

    def test_validator_topk_gt_index_topk_requires_flag(self):
        # top_k > index_topk WITHOUT the lifted flag is steered to the ABI
        # (not SGLANG_DS_ALLOW_TOPK_MISMATCH) by the model-topk gate.
        import sglang.srt.configs.model_config as mc
        from sglang.srt.layers.attention.double_sparsity.validator import (
            validate_double_sparsity,
        )
        o1, o2 = mc.is_deepseek_dsa, mc.get_dsa_index_topk
        mc.is_deepseek_dsa = lambda hf: True
        mc.get_dsa_index_topk = lambda hf: 2048
        try:
            with self.assertRaises(ValueError) as cm:
                validate_double_sparsity(self._server_args("", top_k=4096))
            msg = str(cm.exception)
            self.assertIn("lifted-budget", msg)
            self.assertIn("enable_lifted_budget_decode", msg)
        finally:
            mc.is_deepseek_dsa, mc.get_dsa_index_topk = o1, o2

    def test_validator_lifted_allows_cuda_graph(self):
        # The lifted path is now CUDA-graph-safe (fixed-shape builder + alloc-free
        # out= dequant + preallocated scratch), so the validator must NOT reject it
        # for running under CUDA graph. (A later channel-mask load on the fake path
        # may raise, but not a `--disable-cuda-graph` rejection.)
        import sglang.srt.configs.model_config as mc
        from sglang.srt.layers.attention.double_sparsity.validator import (
            validate_double_sparsity,
        )
        o1, o2 = mc.is_deepseek_dsa, mc.get_dsa_index_topk
        mc.is_deepseek_dsa = lambda hf: True
        mc.get_dsa_index_topk = lambda hf: 2048
        try:
            try:
                validate_double_sparsity(self._server_args(
                    ', "enable_lifted_budget_decode": true, "lifted_budget_top_k": 4096',
                    top_k=2048, disable_cuda_graph=False,
                ))
            except Exception as e:  # noqa: BLE001
                self.assertNotIn("disable-cuda-graph", str(e))
        finally:
            mc.is_deepseek_dsa, mc.get_dsa_index_topk = o1, o2

    def test_validator_lifted_requires_top_k_eq_index_topk(self):
        # The base budget must stay == index_topk; lifted_budget_top_k is the
        # SEPARATE wider width. top_k=4096 (!= index_topk 2048) is rejected.
        import sglang.srt.configs.model_config as mc
        from sglang.srt.layers.attention.double_sparsity.validator import (
            validate_double_sparsity,
        )
        o1, o2 = mc.is_deepseek_dsa, mc.get_dsa_index_topk
        mc.is_deepseek_dsa = lambda hf: True
        mc.get_dsa_index_topk = lambda hf: 2048
        try:
            with self.assertRaises(ValueError) as cm:
                validate_double_sparsity(self._server_args(
                    ', "enable_lifted_budget_decode": true, "lifted_budget_top_k": 8192',
                    top_k=4096, disable_cuda_graph=True,
                ))
            self.assertIn("index_topk", str(cm.exception))
        finally:
            mc.is_deepseek_dsa, mc.get_dsa_index_topk = o1, o2

    def test_validator_lifted_valid_config_passes_lifted_gates(self):
        # top_k==index_topk, lifted>index_topk, %128, eager: the lifted gates must
        # NOT fire. (A later channel-mask load on the fake path may raise, but it
        # must not be one of the lifted-gate rejections.)
        import sglang.srt.configs.model_config as mc
        from sglang.srt.layers.attention.double_sparsity.validator import (
            validate_double_sparsity,
        )
        o1, o2 = mc.is_deepseek_dsa, mc.get_dsa_index_topk
        mc.is_deepseek_dsa = lambda hf: True
        mc.get_dsa_index_topk = lambda hf: 2048
        try:
            try:
                validate_double_sparsity(self._server_args(
                    ', "enable_lifted_budget_decode": true, "lifted_budget_top_k": 4096',
                    top_k=2048, disable_cuda_graph=True,
                ))
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                self.assertNotIn("disable-cuda-graph", msg)
                self.assertNotIn("must be > DSA index_topk", msg)
                self.assertNotIn("base top_k to equal", msg)
                self.assertNotIn("not implemented", msg)
        finally:
            mc.is_deepseek_dsa, mc.get_dsa_index_topk = o1, o2


if __name__ == "__main__":
    unittest.main()
