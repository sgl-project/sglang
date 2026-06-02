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
