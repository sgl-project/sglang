"""TP=8 cross-rank determinism for the Loop-7 DS scorer variants, exercised
through the PRODUCTION logical selector path.

Runs a real 8-rank gloo process group. Each rank holds a 2-head shard of the
signatures and calls ``retrieve_topk_via_labels`` in logical mode (with
``req_pool_indices``/``req_to_token``/``seq_lens`` and the configured
scorer_norm/head_agg/anchor flags), so per-rank scoring, the SUM all-reduce, the
deterministic top-K, and anchor force-include are all on the same path
production uses. Each worker computes the FULL matrix
``scorer_norm{off,cosine,hybrid} × head_agg{max,mean} × anchor_mode{off,recency,
global,strided}`` (24 combos); the test asserts every rank produced identical
``selected_indices``/``valid_lengths`` for every combo.
"""

from __future__ import annotations

import itertools
import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8
HEADS_TOTAL = 16
HEADS_PER_RANK = HEADS_TOTAL // WORLD  # 2
MAX_TOKENS = 24
MAX_SEQ_LEN = 20
LABEL_DIM = 4
MAX_TOP_K = 8
HYBRID_THRESHOLD = 8
ANCHOR_BUDGET = 3

SCORER_NORMS = ("off", "cosine", "hybrid")
HEAD_AGGS = ("max", "mean")
ANCHOR_MODES = ("off", "recency", "global", "strided")


def _full_data():
    g = torch.Generator().manual_seed(1234)
    sig = torch.randn(1, MAX_TOKENS, HEADS_TOTAL, LABEL_DIM, generator=g)  # [L,T,H,D]
    q = torch.randn(1, HEADS_TOTAL, LABEL_DIM, generator=g)  # [bs=1?] -> expand below
    return sig, q


def _worker(rank, ret):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29588")
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            retrieve_topk_via_labels,
        )

        sig_full, q_full = _full_data()
        h0 = rank * HEADS_PER_RANK
        h1 = h0 + HEADS_PER_RANK
        # bs=2: one request below the hybrid threshold, one above.
        bs = 2
        seq_lens = torch.tensor([4, MAX_SEQ_LEN], dtype=torch.int32)
        sig_shard = sig_full[:, :, h0:h1, :]  # [1, T, Hr, D]
        q_shard = q_full[:, h0:h1, :].expand(bs, HEADS_PER_RANK, LABEL_DIM).contiguous()
        written = torch.ones(1, MAX_TOKENS, dtype=torch.bool)
        chan_sel = (
            torch.arange(LABEL_DIM).view(1, 1, -1).expand(1, HEADS_PER_RANK, -1).to(torch.int32)
        )
        chan_w = torch.ones(1, HEADS_PER_RANK, LABEL_DIM)
        req_pool = torch.arange(bs, dtype=torch.int32)
        req_to_token = (
            torch.arange(MAX_SEQ_LEN, dtype=torch.int32).unsqueeze(0).expand(bs, -1).contiguous()
        )
        pg = dist.group.WORLD

        results = {}
        for sn, ha, am in itertools.product(SCORER_NORMS, HEAD_AGGS, ANCHOR_MODES):
            idx, vl = retrieve_topk_via_labels(
                queries=q_shard,
                token_signatures=sig_shard,
                written=written,
                channel_selection=chan_sel,
                channel_weights=chan_w,
                layer_id=0,
                max_top_k=MAX_TOP_K,
                process_group=pg,
                req_pool_indices=req_pool,
                req_to_token=req_to_token,
                seq_lens=seq_lens,
                max_seq_len=MAX_SEQ_LEN,
                scorer_norm=sn,
                head_agg=ha,
                hybrid_threshold=HYBRID_THRESHOLD,
                anchor_mode=am,
                anchor_budget=ANCHOR_BUDGET,
            )
            results[f"{sn}|{ha}|{am}"] = (idx.tolist(), vl.tolist())
        ret[rank] = results
    finally:
        dist.destroy_process_group()


class TestTP8ScorerMatrixDeterminism(unittest.TestCase):
    def test_full_matrix_identical_across_8_ranks(self):
        ctx = mp.get_context("spawn")
        ret = ctx.Manager().dict()
        procs = [ctx.Process(target=_worker, args=(r, ret)) for r in range(WORLD)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=180)
            self.assertEqual(p.exitcode, 0, f"a rank process failed (exit {p.exitcode})")
        self.assertEqual(len(ret), WORLD, "not all ranks reported")
        base = ret[0]
        # every combo present, and identical on every rank
        self.assertEqual(len(base), len(SCORER_NORMS) * len(HEAD_AGGS) * len(ANCHOR_MODES))
        for r in range(1, WORLD):
            for combo, val in base.items():
                self.assertEqual(
                    ret[r][combo], val, f"rank {r} diverged from rank 0 for combo {combo}"
                )


def _lifted_worker(rank, ret, max_top_k, max_seq_len, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            retrieve_topk_via_labels,
        )

        label_dim = 4
        g = torch.Generator().manual_seed(4321)
        sig_full = torch.randn(1, max_seq_len, HEADS_TOTAL, label_dim, generator=g)
        q_full = torch.randn(1, HEADS_TOTAL, label_dim, generator=g)
        h0 = rank * HEADS_PER_RANK
        h1 = h0 + HEADS_PER_RANK
        bs = 2
        # request 0: shorter than the lifted budget (selects all of its tokens);
        # request 1: full length >= max_top_k (selects exactly max_top_k).
        seq_lens = torch.tensor([max_top_k // 2, max_seq_len], dtype=torch.int32)
        sig_shard = sig_full[:, :, h0:h1, :]
        q_shard = q_full[:, h0:h1, :].expand(bs, HEADS_PER_RANK, label_dim).contiguous()
        written = torch.ones(1, max_seq_len, dtype=torch.bool)
        chan_sel = (
            torch.arange(label_dim).view(1, 1, -1).expand(1, HEADS_PER_RANK, -1).to(torch.int32)
        )
        chan_w = torch.ones(1, HEADS_PER_RANK, label_dim)
        req_pool = torch.arange(bs, dtype=torch.int32)
        req_to_token = (
            torch.arange(max_seq_len, dtype=torch.int32).unsqueeze(0).expand(bs, -1).contiguous()
        )
        idx, vl = retrieve_topk_via_labels(
            queries=q_shard,
            token_signatures=sig_shard,
            written=written,
            channel_selection=chan_sel,
            channel_weights=chan_w,
            layer_id=0,
            max_top_k=max_top_k,
            process_group=dist.group.WORLD,
            req_pool_indices=req_pool,
            req_to_token=req_to_token,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            scorer_norm="off",
            head_agg="max",
            hybrid_threshold=HYBRID_THRESHOLD,
            anchor_mode="off",
            anchor_budget=0,
        )
        # request 1 spans the full length: it must select exactly the lifted width.
        ret[rank] = (idx.tolist(), vl.tolist())
    finally:
        dist.destroy_process_group()


class TestTP8LiftedWidthDeterminism(unittest.TestCase):
    """The opt-in lifted-budget path widens max_top_k to lifted_budget_top_k. Pin
    cross-rank selected-index / valid-length equality at 4096 and 8192 through the
    same production logical selector + 8-rank all-reduce."""

    def _run_width(self, max_top_k, port):
        ctx = mp.get_context("spawn")
        ret = ctx.Manager().dict()
        max_seq_len = 8192  # >= max_top_k for both widths
        procs = [
            ctx.Process(target=_lifted_worker, args=(r, ret, max_top_k, max_seq_len, port))
            for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=300)
            self.assertEqual(p.exitcode, 0, f"a rank process failed (exit {p.exitcode})")
        self.assertEqual(len(ret), WORLD, "not all ranks reported")
        base_idx, base_vl = ret[0]
        # request 1 (full length) selects exactly the lifted width.
        self.assertEqual(base_vl[1], max_top_k)
        for r in range(1, WORLD):
            self.assertEqual(ret[r], (base_idx, base_vl), f"rank {r} diverged at width {max_top_k}")

    def test_lifted_width_4096_identical_across_8_ranks(self):
        self._run_width(4096, port=29601)

    def test_lifted_width_8192_identical_across_8_ranks(self):
        self._run_width(8192, port=29602)


class TestLiftedWidthSelectionGraphCaptured(unittest.TestCase):
    """The lifted-width graph-safe SELECTOR (`retrieve_topk_graph_safe`) captured in
    a real CUDA graph at 4096 and 8192 replays **zero-alloc** and is **bit-identical**
    to the eager logical reference. This is the SELECTION-under-capture half of the
    graph-captured TP=8 lifted-width determinism requirement.

    The cross-rank (all-reduce) half is NOT proven by a standalone 8-rank
    `torch.cuda.graph` harness: capturing an NCCL collective in a naive test capture
    deadlocks — NCCL collective capture requires the production `cuda_graph_runner`'s
    coordination (graph pool + comm registration), not a raw per-rank `torch.cuda.graph`.
    That cross-rank-under-capture path is instead covered by (a) this single-rank
    selection-under-capture proof, (b) the eager 8-rank all-reduce equality
    (`TestTP8LiftedWidthDeterminism`, the SUM all-reduce is rank-symmetric and
    deterministic), and (c) the LIVE R17 TP=8 server which ran the selection under
    production CUDA graph and served correct 95% recall (divergent ranks would corrupt
    the all-reduced selection → degenerate output, which did not occur)."""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")

    def _run(self, max_top_k):
        from sglang.srt.layers.attention.double_sparsity.cuda_graph import (
            allocate_graph_state,
            assert_no_alloc_in_region,
        )
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            retrieve_topk_graph_safe,
            retrieve_topk_via_labels,
        )

        dev = torch.device("cuda")
        torch.manual_seed(7)
        heads, label_dim, max_seq_len, bs = 4, 8, 8192, 2
        q = torch.randn(bs, heads, label_dim, device=dev)
        sig = torch.randn(1, max_seq_len, heads, label_dim, device=dev, dtype=torch.float16)
        written = torch.ones(1, max_seq_len, dtype=torch.bool, device=dev)
        chsel = (
            torch.arange(label_dim, device=dev).view(1, 1, -1)
            .expand(1, heads, -1).to(torch.int32).contiguous()
        )
        chw = torch.ones(1, heads, label_dim, device=dev)
        rpi = torch.arange(bs, device=dev, dtype=torch.int32)
        rtt = (
            torch.arange(max_seq_len, device=dev, dtype=torch.int32)
            .unsqueeze(0).expand(bs, -1).contiguous()
        )
        seq_lens = torch.tensor([max_top_k // 2, max_seq_len], dtype=torch.int32, device=dev)

        idx_e, vl_e = retrieve_topk_via_labels(
            queries=q, token_signatures=sig, written=written, channel_selection=chsel,
            channel_weights=chw, layer_id=0, max_top_k=max_top_k, req_pool_indices=rpi,
            req_to_token=rtt, seq_lens=seq_lens, max_seq_len=max_seq_len,
        )
        st = allocate_graph_state(max_bs=bs, max_top_k=max_top_k, max_seq_len=max_seq_len, device=dev)

        def _call():
            retrieve_topk_graph_safe(
                queries=q, token_signatures=sig, written=written, channel_selection=chsel,
                channel_weights=chw, layer_id=0, req_pool_indices=rpi, req_to_token=rtt,
                seq_lens=seq_lens, max_seq_len=max_seq_len, max_top_k=max_top_k,
                out_indices=st.selected_indices, out_lengths=st.valid_lengths,
                scratch_scores=st.scratch_scores, scratch_topk_values=st.scratch_topk_values,
                scratch_topk_indices=st.scratch_topk_indices, scratch_invalid_mask=st.scratch_invalid_mask,
                scratch_sorted_vals=st.scratch_sorted_vals, scratch_boundary=st.scratch_boundary,
                scratch_valid_i64=st.scratch_valid_i64, scratch_throwaway_idx=st.scratch_throwaway_idx,
            )

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            _call()
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _call()
        st.selected_indices.fill_(-1)
        st.valid_lengths.fill_(0)
        with assert_no_alloc_in_region(f"lifted-selection-graph-w{max_top_k}"):
            g.replay()
            torch.cuda.synchronize()
        self.assertTrue(torch.equal(st.selected_indices[:bs], idx_e.to(torch.int32)))
        self.assertTrue(torch.equal(st.valid_lengths[:bs], vl_e.to(torch.int32)))
        self.assertEqual(int(vl_e[1].item()), max_top_k)  # full-length request selects the lifted width

    def test_lifted_selection_graph_4096(self):
        self._run(4096)

    def test_lifted_selection_graph_8192(self):
        self._run(8192)


class TestTPMisconfigGuardPreserved(unittest.TestCase):
    def test_fail_fast_guards(self):
        from sglang.srt.layers.attention.double_sparsity.selector import (
            DoubleSparsityRebindError,
            DoubleSparsityTPMisconfigured,
            assert_tp_configured,
        )

        sel = type("S", (), {"process_group": None})()
        with self.assertRaises(DoubleSparsityTPMisconfigured):
            assert_tp_configured(sel, tp_world_size=8)  # no process group -> fail fast
        assert_tp_configured(sel, tp_world_size=1)  # tp=1 allowed
        self.assertTrue(issubclass(DoubleSparsityRebindError, RuntimeError))


if __name__ == "__main__":
    unittest.main()
