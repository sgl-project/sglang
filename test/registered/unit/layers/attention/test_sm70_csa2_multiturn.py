"""Multi-turn CSA2 testbench: sink / window / sparse mass and needle retrieval.

The serving model is not full n×n attention. Each query sees a sliding window
plus a top-k of compressed keys, with a per-head sink term in the softmax
denominator. Long-context cohesion failures split into named modes this file
reproduces on the CPU oracle (no server, no weights dump):

1. **Sink starved** — local scores ≫ sink logit, sink mass collapses.
2. **Sink takeover** — sink ≫ every key, output vanishes (already in the
   reference tests; re-checked here via mass).
3. **Indexer miss** — the distinctive early key is not in top-k.
4. **Candidate-block drop** — two-level indexer never offers the needle to top-k.
5. **SWA lock** — needle *is* retrieved, but the window is full of a repeated
   local key the query matches, so needle mass is ~0. This is the
   ``The file. The file.`` pattern: the last 128 tokens *are* the loop.
6. **Turn ≠ one-shot** — chat turns are chunked prefill + decode on the same
   cache; ratio-2 pending and top-k must match a single prefill.

fp16 vs bf16 is the SM70 gap against Hopper. The test prints a per-turn table
(``pytest -s``) so a miss is a row of numbers, not a vibe.

Default tests include a **30 000-token / 10-turn** pass at the real Flash
window (128), top-k (512), and candidate budget (2048×8), both constructed
and a 5-layer ``CSA2Reference`` session (~1 min)::

    PYTHONPATH=python pytest -s \\
        test/registered/unit/layers/attention/test_sm70_csa2_multiturn.py
"""

from __future__ import annotations

import math
import time
import unittest
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

import torch

from sglang.srt.layers.attention.dsv4 import sm70_csa2_reference as R
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=120, suite="base-a-test-cpu")

torch.set_num_threads(min(4, torch.get_num_threads()))

FP16 = torch.float16
BF16 = torch.bfloat16

# DeepSeek-V4.1-Flash config.json attention slice (context, not hidden size).
FLASH_WINDOW = 128
FLASH_TOPK = 512
FLASH_CAND_BLOCKS = 2048
FLASH_CAND_BLOCK = 8
LONG_TOKENS = 30000
LONG_TURNS = 10
LONG_TURN = LONG_TOKENS // LONG_TURNS  # 3000
NEEDLE_START, NEEDLE_END = 120, 200
PREFILL_CHUNK = 2048  # serving --chunked-prefill-size


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def _mean(mass: R.AttnMassSplit) -> Dict[str, float]:
    m = mass.mean_over_heads()
    return {k: float(v[0]) for k, v in m.items()}


def _format_mass(label: str, mass: R.AttnMassSplit, **extra) -> str:
    m = _mean(mass)
    bits = " ".join(f"{k}={v:.4f}" for k, v in extra.items() if v is not None)
    return (
        f"{label:44s} sink={m['sink']:.4f} window={m['window']:.4f} "
        f"sparse={m['sparse']:.4f} needle={m['needle']:.4f} "
        f"max_score={m['max_score']:.3f} sink_logit={float(mass.sink_logit.mean()):.3f}"
        + (f" {bits}" if bits else "")
    )


def _assert_partition(tc: CustomTestCase, mass: R.AttnMassSplit, places: int = 4):
    m = _mean(mass)
    for k, v in m.items():
        tc.assertTrue(math.isfinite(v), k)
    if int(mass.n_window[0] + mass.n_sparse[0]) == 0:
        tc.assertAlmostEqual(m["sink"] + m["window"] + m["sparse"], 0.0, places=places)
        return
    tc.assertAlmostEqual(
        m["sink"] + m["window"] + m["sparse"],
        1.0,
        places=places,
        msg=_format_mass("mass does not partition", mass),
    )


@dataclass
class IndexerReport:
    needle_ids: List[int]
    in_topk: bool
    in_candidates: Optional[bool]
    best_needle_rank: int  # 0 = highest score among reachable; -1 if all -inf
    best_needle_score: float
    kth_score: float
    n_topk: int
    n_candidates: Optional[int]


def indexer_report(
    logits: torch.Tensor,
    compress_len: int,
    topk: int,
    needle_ids: Iterable[int],
    candidate_mask: Optional[torch.Tensor] = None,
) -> IndexerReport:
    """One query row. ``logits`` is [n] or [1, n] fp32, -inf already OK."""
    row = logits.reshape(-1).float().clone()
    n = row.numel()
    reach = torch.arange(n) < int(compress_len)
    row = row.masked_fill(~reach, -torch.inf)
    in_cand: Optional[bool] = None
    n_cand: Optional[int] = None
    if candidate_mask is not None:
        mask = candidate_mask.reshape(-1)[:n]
        n_cand = int(mask[: int(compress_len)].sum())
        in_cand = any(mask[i].item() for i in needle_ids if 0 <= i < n)
        row = row.masked_fill(~mask, -torch.inf)
    needles = [i for i in needle_ids if 0 <= i < n]
    finite = row > -torch.inf
    n_reach = int(finite.sum())
    kk = min(int(topk), n_reach)
    if kk == 0:
        return IndexerReport(needles, False, in_cand, -1, float("-inf"), float("-inf"), 0, n_cand)
    top = row.topk(kk)
    sel = set(int(i) for i in top.indices.tolist())
    kth = float(top.values.min())
    in_topk = any(i in sel for i in needles)
    best_rank, best_score = -1, float("-inf")
    order = torch.argsort(row, descending=True)
    for rank, j in enumerate(order.tolist()):
        if row[j] <= -torch.inf:
            break
        if j in needles:
            best_rank, best_score = rank, float(row[j])
            break
    return IndexerReport(
        needles, in_topk, in_cand, best_rank, best_score, kth, kk, n_cand
    )


def _dir(dim: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    return v / v.norm().clamp_min(1e-8) * scale


def _unit_q(heads: int, dim: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    v = _dir(dim, seed, scale)
    return v.view(1, 1, dim).expand(1, heads, dim).contiguous()


# --------------------------------------------------------------------------- #
# Tiny session topology (4 layers, ratio-1 source at 2)
# --------------------------------------------------------------------------- #


def _session_cfg(**kw) -> R.CSA2Config:
    base = dict(
        num_layers=4,
        compress_ratios=(0, 0, 1, 1),
        kv_source_layer_ids=(2,),
        index_source_layer_ids=(2,),
        candidate_source_layer_id=-1,
        sliding_window=32,
        index_topk=16,
        max_seq_len=2048,
    )
    base.update(kw)
    return R.tiny_config(**base)


def _long_cfg(**kw) -> R.CSA2Config:
    """Same tiny hidden size, real Flash context hyperparameters, 30k RoPE table."""
    base = dict(
        num_layers=5,
        compress_ratios=(0, 0, 1, 1, 1),
        kv_source_layer_ids=(2,),
        index_source_layer_ids=(2, 4),
        candidate_source_layer_id=2,
        sliding_window=FLASH_WINDOW,
        index_topk=FLASH_TOPK,
        candidate_topk_blocks=FLASH_CAND_BLOCKS,
        candidate_block_size=FLASH_CAND_BLOCK,
        max_seq_len=LONG_TOKENS + 256,
    )
    base.update(kw)
    return R.tiny_config(**base)


def diagnose_csa2_row(
    ref: R.CSA2Reference,
    lid: int,
    x_row: torch.Tensor,
    pos: int,
    topk_row: Optional[torch.Tensor],
    needle_comp: Set[int],
) -> R.AttnMassSplit:
    """Rebuild the last query's key union from live caches and split mass."""
    cfg, lw = ref.cfg, ref.weights[lid]
    x = x_row.view(1, -1).to(ref.act_dtype)
    positions = torch.tensor([pos], dtype=torch.int64)
    q_lora = R.rmsnorm(R.linear(x, lw.wq_a), lw.q_norm, cfg.rms_norm_eps)
    q = R.linear(q_lora, lw.wq_b).view(1, cfg.n_heads, cfg.head_dim)
    q = R.rope_tail(q, ref.freqs[lid][positions], cfg.rope_head_dim)
    w = cfg.sliding_window
    all_k = torch.stack(ref.swa_k[lid])
    offs = torch.arange(w)
    idx = positions[:, None] - (w - 1) + offs
    valid_w = idx >= 0
    keys = all_k[idx.clamp_min(0)]
    valid = valid_w
    needle_mask = torch.zeros(1, w, dtype=torch.bool)
    if topk_row is not None and topk_row.numel():
        src = cfg.kv_source_for(lid)
        lat = torch.stack(ref.kv_state[src].latents)
        ci = topk_row.view(1, -1).to(torch.int64)
        keys = torch.cat([keys, lat[ci.clamp_min(0)]], dim=1)
        valid = torch.cat([valid, ci >= 0], dim=1)
        extra = torch.zeros(1, ci.shape[1], dtype=torch.bool)
        for j, c in enumerate(ci[0].tolist()):
            extra[0, j] = c >= 0 and c in needle_comp
        needle_mask = torch.cat([needle_mask, extra], dim=1)
    return R.sparse_attention_mass_split(
        q,
        keys,
        valid,
        lw.attn_sink,
        cfg.softmax_scale,
        window_width=w,
        needle_mask=needle_mask,
    )


def _jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    sa = {int(i) for i in a.tolist() if i >= 0}
    sb = {int(i) for i in b.tolist() if i >= 0}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


# --------------------------------------------------------------------------- #
# Sink / window / sparse mass (constructed keys)
# --------------------------------------------------------------------------- #


class TestAttnMassSplit(CustomTestCase):
    def test_mass_partitions_and_matches_sink_column(self):
        g = torch.Generator().manual_seed(0)
        t, h, d, w, s = 3, 4, 64, 8, 6
        scale = d**-0.5
        q = (torch.randn(t, h, d, generator=g) * 0.4).to(FP16)
        keys = (torch.randn(t, w + s, d, generator=g) * 0.4).to(FP16)
        valid = torch.ones(t, w + s, dtype=torch.bool)
        valid[:, 3] = False
        sink = torch.randn(h, generator=g)
        mass = R.sparse_attention_mass_split(
            q, keys, valid, sink, scale, window_width=w
        )
        for r in range(t):
            row = R.AttnMassSplit(
                sink=mass.sink[r : r + 1],
                window=mass.window[r : r + 1],
                sparse=mass.sparse[r : r + 1],
                needle=mass.needle[r : r + 1],
                max_score=mass.max_score[r : r + 1],
                sink_logit=mass.sink_logit,
                n_window=mass.n_window[r : r + 1],
                n_sparse=mass.n_sparse[r : r + 1],
                n_needle=mass.n_needle[r : r + 1],
            )
            _assert_partition(self, row)
        sm = R.sparse_softmax_terms(q, keys, valid, sink, scale)
        expect_sink = (sm.sink_p / sm.l).squeeze(-1)
        self.assertTrue(torch.allclose(mass.sink, expect_sink, atol=1e-5, rtol=1e-5))
        # output still matches the previous formula (refactor identity)
        got = R.sparse_attention_rows(q, keys, valid, sink, scale)
        p_dtype = q.dtype
        o = torch.einsum("thk,tkd->thd", sm.p.to(p_dtype).float(), keys.float())
        expect = (o / sm.l).to(q.dtype)
        self.assertTrue(torch.equal(got, expect))

    def test_empty_row_is_zero_not_nan(self):
        h, d = 4, 64
        q = torch.ones(1, h, d, dtype=FP16)
        keys = torch.ones(1, 8, d, dtype=FP16)
        valid = torch.zeros(1, 8, dtype=torch.bool)
        sink = torch.zeros(h)
        mass = R.sparse_attention_mass_split(
            q, keys, valid, sink, d**-0.5, window_width=4
        )
        self.assertFalse(torch.isnan(mass.sink).any())
        self.assertEqual(float(mass.sink.sum() + mass.window.sum() + mass.sparse.sum()), 0.0)

    def test_sink_takeover_is_visible_in_mass(self):
        h, d, k = 4, 64, 16
        q = torch.ones(1, h, d, dtype=FP16)
        keys = torch.ones(1, k, d, dtype=FP16) * 0.05
        valid = torch.ones(1, k, dtype=torch.bool)
        huge = torch.full((h,), 30.0)
        mass = R.sparse_attention_mass_split(
            q, keys, valid, huge, d**-0.5, window_width=k
        )
        print("\n" + _format_mass("sink takeover (sink=30)", mass))
        _assert_partition(self, mass)
        self.assertGreater(_mean(mass)["sink"], 0.99)
        tiny = R.sparse_attention_rows(q, keys, valid, huge, d**-0.5)
        self.assertLess(tiny.float().abs().max().item(), 1e-3)

    def test_sink_starved_when_local_scores_dominate(self):
        """Named failure: window keys score ~30, sink logit ~0.5 → sink mass ~0.

        This is the formula working, not a missing sink term. The testbench has
        to flag it, because a long noisy window does the same thing as dropping
        the sink from the kernel.
        """
        h, d, w = 4, 64, 32
        scale = d**-0.5
        loop = _dir(d, 3, scale=8.0).to(FP16)
        q = loop.view(1, 1, d).expand(1, h, d).contiguous()
        keys = loop.view(1, 1, d).expand(1, w, d).contiguous()
        valid = torch.ones(1, w, dtype=torch.bool)
        sink = torch.full((h,), 0.5)
        mass = R.sparse_attention_mass_split(
            q, keys, valid, sink, scale, window_width=w
        )
        m = _mean(mass)
        print("\n" + _format_mass("sink starved (aligned window)", mass))
        _assert_partition(self, mass)
        self.assertLess(m["sink"], 1e-4)
        self.assertGreater(m["window"], 0.99)
        self.assertGreater(m["max_score"] - float(sink.mean()), 5.0)


# --------------------------------------------------------------------------- #
# Indexer: needle stays / candidate drops
# --------------------------------------------------------------------------- #


class TestIndexerNeedle(CustomTestCase):
    def _scored(self, n: int, needle: int, filler_scale: float = 0.25):
        h, d = 8, 128
        q = R.fake_quant_fp4_ue8m0(_unit_q(h, d, 1, 4.0).to(FP16))
        keys = []
        for j in range(n):
            keys.append(_dir(d, 200 + j, filler_scale))
        keys[needle] = _dir(d, 1, 4.0)
        k = R.fake_quant_fp4_ue8m0(torch.stack(keys).to(FP16))
        w = torch.ones(1, h, dtype=FP16)
        logits = R.index_scores(q, k, w)
        return logits[0], q, k

    def test_distinct_needle_stays_in_topk_after_filler(self):
        n, needle, topk = 256, 7, 16
        logits, _, _ = self._scored(n, needle)
        rep = indexer_report(logits, n, topk, [needle])
        print(
            f"\nneedle in topk={rep.in_topk} rank={rep.best_needle_rank} "
            f"score={rep.best_needle_score:.4f} kth={rep.kth_score:.4f} n={n} topk={topk}"
        )
        self.assertTrue(rep.in_topk, "distinct needle lost to filler at the indexer")
        self.assertLess(rep.best_needle_rank, topk)

    def test_candidate_blocks_drop_early_needle_when_budget_tight(self):
        """Two-level indexer: last block is pinned; a tight block budget then
        keeps later high-scoring filler and never offers the early needle.

        On the real Flash config this kicks in past 2048*8 compressed positions.
        The test uses a toy budget so the drop is deterministic.
        """
        n, block, needle = 64, 8, 1
        logits = torch.full((1, n), 0.5)
        logits[0, needle] = 3.0
        # One later position sets the block-max. The rest of that block stays
        # below the needle so a *wide* candidate mask still lets top-k keep it.
        logits[0, 48] = 6.0
        lens = torch.tensor([[n]])
        tight = R.select_candidate_blocks(logits, lens, topk_blocks=2, block_size=block)
        wide = R.select_candidate_blocks(logits, lens, topk_blocks=8, block_size=block)
        tight_rep = indexer_report(logits[0], n, topk=8, needle_ids=[needle], candidate_mask=tight[0])
        wide_rep = indexer_report(logits[0], n, topk=8, needle_ids=[needle], candidate_mask=wide[0])
        print(
            f"\ncandidate drop: tight in_cand={tight_rep.in_candidates} "
            f"in_topk={tight_rep.in_topk} n_cand={tight_rep.n_candidates}; "
            f"wide in_cand={wide_rep.in_candidates} in_topk={wide_rep.in_topk}"
        )
        self.assertFalse(tight_rep.in_candidates)
        self.assertFalse(tight_rep.in_topk)
        self.assertTrue(wide_rep.in_candidates)
        self.assertTrue(wide_rep.in_topk)
        # last block is always pinned
        self.assertTrue(bool(tight[0, n - 1]))

    def test_fp16_bf16_near_tie_is_reported(self):
        """SM70 stores fp16; Hopper bf16. A needle that barely wins in fp32 can
        leave the top-k after the FP4 grid. This is the documented Jaccard gap,
        pinned as a retrieval miss rather than a sink miss.
        """
        h, d, n, needle, topk = 8, 128, 64, 3, 4
        g = torch.Generator().manual_seed(9)
        q_f = torch.randn(1, h, d, generator=g)
        k_f = torch.randn(n, d, generator=g)
        # needle almost matches q's first head; filler is close
        k_f[needle] = q_f[0, 0] + 0.01 * torch.randn(d, generator=g)
        w = torch.ones(1, h)
        q16 = R.fake_quant_fp4_ue8m0(q_f.to(FP16))
        k16 = R.fake_quant_fp4_ue8m0(k_f.to(FP16))
        qbf = R.fake_quant_fp4_ue8m0(q_f.to(BF16))
        kbf = R.fake_quant_fp4_ue8m0(k_f.to(BF16))
        r16 = indexer_report(R.index_scores(q16, k16, w.to(FP16))[0], n, topk, [needle])
        rbf = indexer_report(R.index_scores(qbf, kbf, w.to(BF16))[0], n, topk, [needle])
        print(
            f"\nnear-tie needle fp16 in_topk={r16.in_topk} rank={r16.best_needle_rank} "
            f"score={r16.best_needle_score:.4f}; bf16 in_topk={rbf.in_topk} "
            f"rank={rbf.best_needle_rank} score={rbf.best_needle_score:.4f}"
        )
        # Not an assert on equality: the point is the report exists. We do
        # require both paths to produce finite ranks (the needle is reachable).
        self.assertNotEqual(r16.best_needle_rank, -1)
        self.assertNotEqual(rbf.best_needle_rank, -1)


# --------------------------------------------------------------------------- #
# SWA lock: retrieved needle, local loop still owns the softmax
# --------------------------------------------------------------------------- #


class TestSwaLock(CustomTestCase):
    def _pack(self, q, window, sparse, needle_index, sink):
        w = window.shape[1]
        keys = torch.cat([window, sparse], dim=1)
        valid = torch.ones(1, keys.shape[1], dtype=torch.bool)
        needle_mask = torch.zeros(1, keys.shape[1], dtype=torch.bool)
        needle_mask[0, w + needle_index] = True
        mass = R.sparse_attention_mass_split(
            q,
            keys,
            valid,
            sink,
            q.shape[-1] ** -0.5,
            window_width=w,
            needle_mask=needle_mask,
        )
        return mass

    def test_loop_aligned_query_starves_retrieved_needle(self):
        h, d, w, s = 4, 64, 32, 16
        loop = _dir(d, 11, 6.0).to(FP16)
        needle = _dir(d, 23, 6.0).to(FP16)
        q = loop.view(1, 1, d).expand(1, h, d).contiguous()
        window = loop.view(1, 1, d).expand(1, w, d).contiguous()
        sparse = (torch.randn(1, s, d) * 0.05).to(FP16)
        sparse[0, 0] = needle
        sink = torch.zeros(h)
        mass = self._pack(q, window, sparse, 0, sink)
        m = _mean(mass)
        print("\n" + _format_mass("SWA lock (query=loop, needle retrieved)", mass))
        _assert_partition(self, mass)
        self.assertGreater(m["window"], 0.9)
        self.assertLess(m["needle"], 0.05)
        # Indexer did its job: the needle is in the key set (n_needle=1).
        self.assertEqual(int(mass.n_needle[0]), 1)

    def test_needle_aligned_query_beats_loop_window(self):
        h, d, w, s = 4, 64, 32, 16
        loop = _dir(d, 11, 6.0).to(FP16)
        needle = _dir(d, 23, 6.0).to(FP16)
        q = needle.view(1, 1, d).expand(1, h, d).contiguous()
        window = loop.view(1, 1, d).expand(1, w, d).contiguous()
        sparse = (torch.randn(1, s, d) * 0.05).to(FP16)
        sparse[0, 0] = needle
        sink = torch.zeros(h)
        mass = self._pack(q, window, sparse, 0, sink)
        m = _mean(mass)
        print("\n" + _format_mass("needle query vs loop window", mass))
        _assert_partition(self, mass)
        self.assertGreater(m["needle"], 0.5)
        self.assertGreater(m["needle"], m["window"])


# --------------------------------------------------------------------------- #
# Multi-turn CSA2Reference (chat turns = chunked forward on one cache)
# --------------------------------------------------------------------------- #


class TestMultiTurnCSA2(CustomTestCase):
    def _turns(self, hidden: int, g: torch.Generator):
        facts = torch.randn(48, hidden, generator=g)
        filler = torch.randn(64, hidden, generator=g)
        decode = torch.randn(1, hidden, generator=g).expand(24, -1).contiguous()
        ask = torch.randn(8, hidden, generator=g)
        return [("facts", facts), ("filler", filler), ("decode-loop", decode), ("ask", ask)]

    def test_turns_match_one_shot_prefill(self):
        cfg = _session_cfg()
        g = torch.Generator().manual_seed(70)
        turns = self._turns(cfg.hidden_size, g)
        x = torch.cat([t[1] for t in turns])
        one = R.build_reference(cfg, FP16, seed=3)
        one_out = one.forward_chunk(x)
        multi = R.build_reference(cfg, FP16, seed=3)
        parts = [multi.forward_chunk(chunk) for _, chunk in turns]
        # last row of each path
        self.assertEqual(multi.pos, one.pos)
        last = x.shape[0] - 1
        self.assertTrue(torch.equal(one_out.topk_idx[2][last], parts[-1].topk_idx[2][-1]))
        d = (one_out.attn_out[2][last].float() - parts[-1].attn_out[2][-1].float()).abs().max()
        self.assertLess(float(d), 1e-2, f"ask-row attn drifted {float(d)}")

    def test_ratio2_odd_turn_boundary_matches_one_shot(self):
        cfg = _session_cfg(compress_ratios=(0, 0, 2, 2))
        g = torch.Generator().manual_seed(71)
        hidden = cfg.hidden_size
        # 5 tokens: last is mid-pair pending. Then 11 more.
        a = torch.randn(5, hidden, generator=g)
        b = torch.randn(11, hidden, generator=g)
        x = torch.cat([a, b])
        one = R.build_reference(cfg, FP16, seed=4).forward_chunk(x)
        multi = R.build_reference(cfg, FP16, seed=4)
        multi.forward_chunk(a)
        self.assertIsNotNone(multi.kv_state[2].pending_kv)
        two = multi.forward_chunk(b)
        self.assertIsNone(multi.kv_state[2].pending_kv)
        self.assertTrue(torch.equal(one.topk_idx[2][-1], two.topk_idx[2][-1]))
        d = (one.attn_out[2][-1].float() - two.attn_out[2][-1].float()).abs().max()
        self.assertLess(float(d), 1e-2)

    def test_per_turn_mass_table_and_early_needle_recall(self):
        """Facts live at compressed 8..16. After filler+decode they are outside
        the 32-token window, so recall is indexer-only. Random weights do not
        guarantee a hit; the table is the artifact. We do require: masses
        partition, no NaNs, and SWA is full after the first window.
        """
        cfg = _session_cfg()
        g = torch.Generator().manual_seed(72)
        turns = self._turns(cfg.hidden_size, g)
        needle = set(range(8, 16))
        ref16 = R.build_reference(cfg, FP16, seed=5)
        refbf = R.build_reference(cfg, BF16, seed=5)
        pos = 0
        print("\n--- multi-turn CSA2 (fp16 vs bf16) ---")
        hdr = (
            f"{'turn':16s} {'pos':>4} {'sink':>7} {'window':>7} {'sparse':>7} "
            f"{'needle':>7} {'in_topk':>8} {'jacc':>6}"
        )
        print(hdr)
        last_in = None
        for name, chunk in turns:
            out16 = ref16.forward_chunk(chunk)
            outbf = refbf.forward_chunk(chunk)
            row = chunk.shape[0] - 1
            p = pos + row
            mass = diagnose_csa2_row(
                ref16, 2, chunk[row], p, out16.topk_idx[2][row], needle
            )
            _assert_partition(self, mass, places=3)
            m = _mean(mass)
            sel16 = out16.topk_idx[2][row]
            selbf = outbf.topk_idx[2][row]
            in_topk = any(int(i) in needle for i in sel16.tolist() if i >= 0)
            last_in = in_topk
            jac = _jaccard(sel16, selbf)
            print(
                f"{name:16s} {p:4d} {m['sink']:7.4f} {m['window']:7.4f} "
                f"{m['sparse']:7.4f} {m['needle']:7.4f} {str(in_topk):>8} {jac:6.3f}"
            )
            self.assertEqual(int(out16.swa_valid_count[row].item()), min(p + 1, cfg.sliding_window))
            self.assertGreater(jac, 0.5, f"fp16/bf16 top-k Jaccard collapsed at {name}")
            pos += chunk.shape[0]
        self.assertEqual(int(ref16.swa_k[2].__len__()), pos)
        # After the loop the window cannot contain the needle tokens.
        self.assertGreater(pos, cfg.sliding_window + 16)
        print(
            f"ask-row needle in fp16 top-k: {last_in} "
            f"(False is an indexer miss, not a sink miss; window cannot see pos 8..16)"
        )

    def test_planted_needle_survives_loop_turns_at_indexer(self):
        """After the session has filled a looped SWA ring, overwrite compressed
        index-K at the fact positions with a unique direction and query it.
        If this fails, the bug is top-k / candidate / visibility, not softmax.
        """
        cfg = _session_cfg(index_topk=8)
        g = torch.Generator().manual_seed(73)
        turns = self._turns(cfg.hidden_size, g)
        ref = R.build_reference(cfg, FP16, seed=6)
        for _, chunk in turns:
            ref.forward_chunk(chunk)
        needle = list(range(8, 16))
        d = cfg.index_head_dim
        planted = R.fake_quant_fp4_ue8m0(_dir(d, 1, 4.0).to(FP16))
        for j in needle:
            ref.kv_state[2].index_k[j] = planted
        q = R.fake_quant_fp4_ue8m0(_unit_q(cfg.index_n_heads, d, 1, 4.0).to(FP16))
        k = torch.stack(ref.kv_state[2].index_k)
        w = torch.ones(1, cfg.index_n_heads, dtype=FP16)
        logits = R.index_scores(q, k, w)[0]
        n = k.shape[0]
        rep = indexer_report(logits, n, cfg.index_topk, needle)
        print(
            f"\nplanted needle after loop turns: in_topk={rep.in_topk} "
            f"rank={rep.best_needle_rank} n_compressed={n} window={cfg.sliding_window}"
        )
        self.assertTrue(rep.in_topk)
        self.assertLess(rep.best_needle_rank, cfg.index_topk)

        # Same query through attention: SWA is whatever the session stored,
        # sparse keys include the planted latents for the selected positions.
        lid = 2
        topk = R.topk_positions(logits[None], torch.tensor([[n]]), cfg.index_topk)[0]
        pos = sum(c.shape[0] for _, c in turns) - 1
        # Align the *attention* query with the planted latent direction so a
        # retrieval hit is visible in needle mass, not drowned by a random q.
        lat = torch.stack(ref.kv_state[2].latents)
        planted_lat = _dir(cfg.head_dim, 1, 4.0).to(FP16)
        for j in needle:
            ref.kv_state[2].latents[j] = planted_lat
        q_attn = planted_lat.view(1, 1, -1).expand(1, cfg.n_heads, -1).contiguous()
        w_swa = cfg.sliding_window
        all_k = torch.stack(ref.swa_k[lid])
        offs = torch.arange(w_swa)
        idx = torch.tensor([pos])[:, None] - (w_swa - 1) + offs
        valid_w = idx >= 0
        keys = all_k[idx.clamp_min(0)]
        ci = topk.view(1, -1).to(torch.int64)
        lat = torch.stack(ref.kv_state[2].latents)
        keys = torch.cat([keys, lat[ci.clamp_min(0)]], dim=1)
        valid = torch.cat([valid_w, ci >= 0], dim=1)
        nmask = torch.zeros(1, keys.shape[1], dtype=torch.bool)
        for j, c in enumerate(ci[0].tolist()):
            nmask[0, w_swa + j] = c in needle
        mass = R.sparse_attention_mass_split(
            q_attn,
            keys,
            valid,
            ref.weights[lid].attn_sink,
            cfg.softmax_scale,
            window_width=w_swa,
            needle_mask=nmask,
        )
        print(_format_mass("planted needle + real SWA ring", mass, in_topk=float(rep.in_topk)))
        _assert_partition(self, mass, places=3)
        self.assertGreater(_mean(mass)["needle"], 0.2)


def _turn_name(t: int) -> str:
    if t == 0:
        return "facts"
    if t == LONG_TURNS - 2:
        return "decode-loop"
    if t == LONG_TURNS - 1:
        return "ask"
    return f"filler-{t}"


class TestFlashScale30k(CustomTestCase):
    """30k tokens / 10 turns at the real Flash window, top-k, and candidate budget.

    Constructed tensors (no 40-layer net). Cheap enough for default CPU CI.
    """

    def test_distinct_needle_survives_30k_and_flash_topk(self):
        n, d, h = LONG_TOKENS, 128, 32
        needle = list(range(NEEDLE_START, NEEDLE_END))
        g = torch.Generator().manual_seed(80)
        k = torch.randn(n, d, generator=g) * 0.25
        k[needle] = _dir(d, 1, 4.0)
        q = R.fake_quant_fp4_ue8m0(_unit_q(h, d, 1, 4.0).to(FP16))
        kq = R.fake_quant_fp4_ue8m0(k.to(FP16))
        logits = R.index_scores(q, kq, torch.ones(1, h, dtype=FP16))
        lens = torch.tensor([[n]])
        cand = R.select_candidate_blocks(
            logits, lens, FLASH_CAND_BLOCKS, FLASH_CAND_BLOCK
        )
        pub = indexer_report(logits[0], n, FLASH_TOPK, needle)
        cons = indexer_report(logits[0], n, FLASH_TOPK, needle, cand[0])
        print(
            f"\n30k distinct needle: publisher in_topk={pub.in_topk} rank={pub.best_needle_rank}; "
            f"consumer in_cand={cons.in_candidates} in_topk={cons.in_topk} "
            f"n_cand={cons.n_candidates}/{n} topk={FLASH_TOPK} blocks={FLASH_CAND_BLOCKS}"
        )
        self.assertTrue(pub.in_topk)
        self.assertEqual(pub.best_needle_rank, 0)
        self.assertTrue(cons.in_candidates)
        self.assertTrue(cons.in_topk)
        self.assertEqual(cons.n_candidates, FLASH_CAND_BLOCKS * FLASH_CAND_BLOCK)

    def test_real_candidate_budget_drops_early_moderate_needle_at_30k(self):
        """At 30k, ratio-1 has 3750 blocks and only 2048 are kept. A needle that
        is merely *good* (not unique-max) in an early block is dropped before top-k.
        At 144 tokens this cannot happen (18 blocks < 2048).
        """
        n, needle = LONG_TOKENS, NEEDLE_START
        logits = torch.full((1, n), 0.5)
        logits[0, needle] = 3.0
        # Every block from 100 onward: high block-max. 3650 winners, budget 2048.
        logits[0, 100 * FLASH_CAND_BLOCK : n : FLASH_CAND_BLOCK] = 6.0
        lens = torch.tensor([[n]])
        cand = R.select_candidate_blocks(
            logits, lens, FLASH_CAND_BLOCKS, FLASH_CAND_BLOCK
        )
        pub = indexer_report(logits[0], n, FLASH_TOPK, [needle])
        cons = indexer_report(logits[0], n, FLASH_TOPK, [needle], cand[0])
        print(
            f"\n30k moderate needle: publisher in_topk={pub.in_topk} rank={pub.best_needle_rank}; "
            f"consumer in_cand={cons.in_candidates} in_topk={cons.in_topk} "
            f"n_cand={cons.n_candidates}"
        )
        self.assertFalse(cons.in_candidates)
        self.assertFalse(cons.in_topk)
        self.assertTrue(bool(cand[0, n - 1]))
        # Publisher still ranks it, but not inside 512 of 30k high later keys.
        self.assertFalse(pub.in_topk)
        self.assertGreater(pub.best_needle_rank, FLASH_TOPK)

    def test_swa_lock_at_flash_window_128_topk_512(self):
        """Once the loop has been going, both the 128-window *and* the 512
        retrieved keys are loop tokens. One fact in the shortlist is 1/640.
        """
        h, d, w, s = 4, 512, FLASH_WINDOW, FLASH_TOPK
        loop = _dir(d, 11, 6.0).to(FP16)
        needle = _dir(d, 23, 6.0).to(FP16)
        q = loop.view(1, 1, d).expand(1, h, d).contiguous()
        window = loop.view(1, 1, d).expand(1, w, d).contiguous()
        sparse = loop.view(1, 1, d).expand(1, s, d).contiguous().clone()
        sparse[0, 0] = needle
        nmask = torch.zeros(1, w + s, dtype=torch.bool)
        nmask[0, w] = True
        mass = R.sparse_attention_mass_split(
            q,
            torch.cat([window, sparse], dim=1),
            torch.ones(1, w + s, dtype=torch.bool),
            torch.zeros(h),
            d**-0.5,
            window_width=w,
            needle_mask=nmask,
        )
        m = _mean(mass)
        print("\n" + _format_mass("SWA lock @128+511 loop + 1 needle", mass))
        _assert_partition(self, mass)
        self.assertLess(m["needle"], 0.01)
        self.assertGreater(m["window"] + m["sparse"] - m["needle"], 0.98)

    def test_ten_turns_30k_mass_table(self):
        """10 chat turns, 3000 tokens each. Facts at 120..200. Turn 8 fills the
        128-token window with a loop; turn 9 asks for the fact.
        """
        d_ix, d_attn, h_ix, h_attn = 128, 512, 32, 4
        g = torch.Generator().manual_seed(81)
        needle_ids = list(range(NEEDLE_START, NEEDLE_END))
        needle_ix = _dir(d_ix, 1, 4.0)
        loop_ix = _dir(d_ix, 99, 4.0)
        needle_attn = _dir(d_attn, 1, 4.0)
        loop_attn = _dir(d_attn, 99, 4.0)
        k_ix = torch.randn(LONG_TOKENS, d_ix, generator=g) * 0.25
        k_ix[needle_ids] = needle_ix
        k_attn = torch.randn(LONG_TOKENS, d_attn, generator=g) * 0.25
        k_attn[needle_ids] = needle_attn
        w_ix = torch.ones(1, h_ix, dtype=FP16)
        sink = torch.zeros(h_attn)
        print(
            f"\n--- 10 turns × {LONG_TURN} = {LONG_TOKENS}, "
            f"window={FLASH_WINDOW}, topk={FLASH_TOPK}, "
            f"cand={FLASH_CAND_BLOCKS}×{FLASH_CAND_BLOCK} ---"
        )
        print(
            f"{'turn':12s} {'n':>6} {'in_top':>7} {'in_cand':>8} {'rank':>6} "
            f"{'sink':>7} {'window':>7} {'sparse':>7} {'needle':>7} {'jacc':>6}"
        )
        last = {}
        loop_mass = None
        loop_in_topk = None
        for t in range(LONG_TURNS):
            n = (t + 1) * LONG_TURN
            name = _turn_name(t)
            if t == LONG_TURNS - 2:
                k_ix[n - FLASH_WINDOW : n] = loop_ix
                k_attn[n - FLASH_WINDOW : n] = loop_attn
                q_ix, q_attn = loop_ix, loop_attn
            elif t == LONG_TURNS - 1:
                q_ix, q_attn = needle_ix, needle_attn
            elif t == 0:
                q_ix, q_attn = needle_ix, needle_attn
            else:
                q_ix, q_attn = k_ix[n - 1], k_attn[n - 1]
            kq = R.fake_quant_fp4_ue8m0(k_ix[:n].to(FP16))
            q16 = R.fake_quant_fp4_ue8m0(
                q_ix.view(1, 1, d_ix).expand(1, h_ix, d_ix).contiguous().to(FP16)
            )
            qbf = R.fake_quant_fp4_ue8m0(
                q_ix.view(1, 1, d_ix).expand(1, h_ix, d_ix).contiguous().to(BF16)
            )
            kbf = R.fake_quant_fp4_ue8m0(k_ix[:n].to(BF16))
            logits = R.index_scores(q16, kq, w_ix)
            logits_bf = R.index_scores(qbf, kbf, w_ix.to(BF16))
            lens = torch.tensor([[n]])
            cand = R.select_candidate_blocks(
                logits, lens, FLASH_CAND_BLOCKS, FLASH_CAND_BLOCK
            )
            pub = indexer_report(logits[0], n, FLASH_TOPK, needle_ids)
            cons = indexer_report(logits[0], n, FLASH_TOPK, needle_ids, cand[0])
            masked = logits.masked_fill(~cand[:, :n], -torch.inf)
            topk = R.topk_positions(masked, lens, FLASH_TOPK)[0]
            topk_bf = R.topk_positions(logits_bf, lens, FLASH_TOPK)[0]
            jac = _jaccard(topk, topk_bf)
            ci = topk.to(torch.int64)
            window = k_attn[n - FLASH_WINDOW : n].to(FP16).unsqueeze(0)
            sparse = k_attn[ci.clamp_min(0)].to(FP16).unsqueeze(0)
            keys = torch.cat([window, sparse], dim=1)
            valid = torch.ones(1, keys.shape[1], dtype=torch.bool)
            nmask = torch.zeros(1, keys.shape[1], dtype=torch.bool)
            for j, c in enumerate(ci.tolist()):
                nmask[0, FLASH_WINDOW + j] = c in needle_ids
            q_a = (
                q_attn.view(1, 1, d_attn)
                .expand(1, h_attn, d_attn)
                .contiguous()
                .to(FP16)
            )
            mass = R.sparse_attention_mass_split(
                q_a,
                keys,
                valid,
                sink,
                d_attn**-0.5,
                window_width=FLASH_WINDOW,
                needle_mask=nmask,
            )
            _assert_partition(self, mass, places=3)
            m = _mean(mass)
            print(
                f"{name:12s} {n:6d} {str(cons.in_topk):>7} {str(cons.in_candidates):>8} "
                f"{cons.best_needle_rank:6d} {m['sink']:7.4f} {m['window']:7.4f} "
                f"{m['sparse']:7.4f} {m['needle']:7.4f} {jac:6.3f}"
            )
            last = dict(name=name, n=n, cons=cons, pub=pub, mass=m, jac=jac)
            if t == LONG_TURNS - 2:
                loop_mass = m
                loop_in_topk = cons.in_topk
        self.assertIsNotNone(loop_mass)
        # Generating the loop: the early fact is not in the 512. Window+sparse
        # are local; needle mass is ~0. Asking for the fact (next turn) recovers.
        self.assertFalse(loop_in_topk)
        self.assertLess(loop_mass["needle"], 0.05)
        # Ask turn: unique fact must still be retrieved at 30k.
        self.assertEqual(last["name"], "ask")
        self.assertEqual(last["n"], LONG_TOKENS)
        self.assertTrue(last["cons"].in_topk)
        self.assertGreater(last["mass"]["needle"], 0.2)
        self.assertGreater(last["jac"], 0.5)

    def test_csa2_session_30k_ten_turns(self):
        """Full CSA2Reference: 10 turns × 3000, chunked 2048, real Flash context
        hyperparams. About a minute on this CPU.
        """
        cfg = _long_cfg()
        g = torch.Generator().manual_seed(82)
        needle = set(range(NEEDLE_START, NEEDLE_END))
        ref = R.build_reference(cfg, FP16, seed=7)
        print(
            f"\n--- CSA2Reference 30k session window={cfg.sliding_window} "
            f"topk={cfg.index_topk} cand={cfg.candidate_topk_blocks}×"
            f"{cfg.candidate_block_size} layers={cfg.num_layers} ---"
        )
        print(
            f"{'turn':12s} {'pos':>6} {'s':>6} {'sink':>7} {'window':>7} "
            f"{'sparse':>7} {'needle':>7} {'L2top':>7} {'L4top':>7} {'L4cand':>7} {'sec':>7}"
        )
        pos = 0
        last_out = None
        for t in range(LONG_TURNS):
            name = _turn_name(t)
            chunk = torch.randn(LONG_TURN, cfg.hidden_size, generator=g)
            if t == LONG_TURNS - 2:
                chunk = (
                    torch.randn(1, cfg.hidden_size, generator=g)
                    .expand(LONG_TURN, -1)
                    .contiguous()
                )
            t0 = time.perf_counter()
            out = None
            for s in range(0, LONG_TURN, PREFILL_CHUNK):
                out = ref.forward_chunk(chunk[s : s + PREFILL_CHUNK])
            elapsed = time.perf_counter() - t0
            row = out.topk_idx[2].shape[0] - 1
            p = pos + LONG_TURN - 1
            mass = diagnose_csa2_row(
                ref, 4, chunk[-1], p, out.topk_idx[4][row], needle
            )
            _assert_partition(self, mass, places=3)
            m = _mean(mass)
            sel2 = out.topk_idx[2][row]
            sel4 = out.topk_idx[4][row]
            in2 = any(int(i) in needle for i in sel2.tolist() if i >= 0)
            in4 = any(int(i) in needle for i in sel4.tolist() if i >= 0)
            in_cand = True
            if out.candidate_mask is not None:
                in_cand = any(
                    bool(out.candidate_mask[row, i]) for i in needle if i < out.candidate_mask.shape[1]
                )
            print(
                f"{name:12s} {p:6d} {elapsed:6.1f} {m['sink']:7.4f} {m['window']:7.4f} "
                f"{m['sparse']:7.4f} {m['needle']:7.4f} {str(in2):>7} {str(in4):>7} "
                f"{str(in_cand):>7} {elapsed:7.1f}"
            )
            last_out = out
            pos += LONG_TURN
        self.assertEqual(pos, LONG_TOKENS)
        self.assertEqual(int(ref.swa_k[2].__len__()), LONG_TOKENS)
        self.assertEqual(
            int(last_out.swa_valid_count[-1].item()), FLASH_WINDOW
        )
        print(
            f"done: n={pos} compressed={ref.kv_state[2].num_compressed} "
            f"(ratio-1 should be {LONG_TOKENS})"
        )
        self.assertEqual(ref.kv_state[2].num_compressed, LONG_TOKENS)


if __name__ == "__main__":
    unittest.main()
