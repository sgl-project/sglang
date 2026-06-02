"""TP cross-rank determinism for the Loop-7 DS scorer variants.

Runs a real 2-rank gloo process group: each rank holds a head-shard of the
signatures, computes its per-rank scalar scores (with the configured
scorer_norm/head_agg), all-reduces (SUM, the production semantics), then runs
the shared deterministic top-K + anchor. Asserts every rank produces identical
``selected_indices``/``valid_lengths`` for each scorer/head/anchor combination.
Also keeps the TP-misconfig fail-fast guard in the matrix.
"""

from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 2
HEADS_TOTAL = 4
HEADS_PER_RANK = HEADS_TOTAL // WORLD
T = 48
LABEL_DIM = 4
MAX_TOP_K = 8


def _build_full_signatures(seed=0):
    g = torch.Generator().manual_seed(seed)
    # [T, HEADS_TOTAL, LABEL_DIM]; one query aligned with channel 0.
    sig = torch.randn(T, HEADS_TOTAL, LABEL_DIM, generator=g)
    q = torch.randn(HEADS_TOTAL, LABEL_DIM, generator=g)
    return sig, q


def _worker(rank, scorer_norm, head_agg, anchor_mode, ret):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    dist.init_process_group("gloo", rank=rank, world_size=WORLD)
    try:
        from sglang.srt.layers.attention.double_sparsity.selection_kernel import (
            _force_include_anchor,
            all_reduce_token_scores,
            compute_token_scores,
            select_topk_sequence_order,
        )

        sig, q = _build_full_signatures(seed=0)  # identical full data on all ranks
        h0 = rank * HEADS_PER_RANK
        h1 = h0 + HEADS_PER_RANK
        sig_shard = sig[:, h0:h1, :].unsqueeze(0)  # [1, T, Hr, D] -> [L=1,...]
        q_shard = q[h0:h1, :].unsqueeze(0)  # [1, Hr, D]
        written = torch.ones(1, T, dtype=torch.bool)
        chan_sel = (
            torch.arange(LABEL_DIM).view(1, 1, -1).expand(1, HEADS_PER_RANK, -1).to(torch.int32)
        )
        chan_w = torch.ones(1, HEADS_PER_RANK, LABEL_DIM)

        # Per-rank scalar score over this rank's heads (cosine if requested).
        local = compute_token_scores(
            q_shard, sig_shard, written, chan_sel, chan_w, 0,
            scorer_norm=scorer_norm, head_agg=head_agg,
        )  # [1, T]
        pg = dist.group.WORLD
        summed = all_reduce_token_scores(local.clone(), process_group=pg)
        idx, vl = select_topk_sequence_order(summed, MAX_TOP_K)
        if anchor_mode != "off":
            seq_lens = torch.full((1,), T, dtype=torch.int32)
            idx, vl = _force_include_anchor(idx, summed, seq_lens, 4, anchor_mode)
        ret[rank] = (idx.tolist(), vl.tolist())
    finally:
        dist.destroy_process_group()


class TestTPScorerDeterminism(unittest.TestCase):
    def _run_matrix(self, scorer_norm, head_agg, anchor_mode):
        ctx = mp.get_context("spawn")
        ret = ctx.Manager().dict()
        procs = [
            ctx.Process(target=_worker, args=(r, scorer_norm, head_agg, anchor_mode, ret))
            for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=120)
            self.assertEqual(p.exitcode, 0, f"rank proc failed (exit {p.exitcode})")
        self.assertEqual(len(ret), WORLD)
        r0, r1 = ret[0], ret[1]
        self.assertEqual(r0, r1, f"cross-rank mismatch for ({scorer_norm},{head_agg},{anchor_mode})")

    def test_off_max_off(self):
        self._run_matrix("off", "max", "off")

    def test_cosine_max_off(self):
        self._run_matrix("cosine", "max", "off")

    def test_off_mean_off(self):
        self._run_matrix("off", "mean", "off")

    def test_off_max_recency(self):
        self._run_matrix("off", "max", "recency")

    def test_cosine_mean_strided(self):
        self._run_matrix("cosine", "mean", "strided")


class TestTPMisconfigGuardPreserved(unittest.TestCase):
    def test_fail_fast_guards_importable(self):
        from sglang.srt.layers.attention.double_sparsity.selector import (
            DoubleSparsityRebindError,
            DoubleSparsityTPMisconfigured,
            assert_tp_configured,
        )

        sel = type("S", (), {"process_group": None})()
        with self.assertRaises(DoubleSparsityTPMisconfigured):
            assert_tp_configured(sel, tp_world_size=8)  # no process group -> fail fast
        # tp_world_size 1 is allowed
        assert_tp_configured(sel, tp_world_size=1)
        self.assertTrue(issubclass(DoubleSparsityRebindError, RuntimeError))


if __name__ == "__main__":
    unittest.main()
