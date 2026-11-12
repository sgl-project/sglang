# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SGLANG_OPT_DFLASH_DRAFT_SPLITKV_ATTN: the DFLASH MLA draft block's split-KV extend attention
(kernels/ops/attention/dflash_draft_splitkv.py) computes the stock serial kernel's causal sliding-window attention
over the fp8 rail plus the block's own bf16 keys.

References are FP64 softmaxes built here from the stated semantics, never from the module under test:
- exact mode (every dot in IEEE fp32) matches them to the bf16 output rounding on every shape the draft call can see;
- the shipped modes (knob 1: stock operand dtypes, knob 2: bf16 rail operands) are no less accurate than the stock
  serial kernel under check A2: pooled over 64 seeds x rails of 1056 / 1087 / 4095 keys, mean and max
  relative error <= stock x 1.01, no NaN;
- shipped outputs are deterministic (eager, PDL and CUDA-graph replay bitwise equal);
- a rail longer than the grid sized from max_kv_len writes NaN rows (fail loud), never a silently cut rail.
"""

import itertools
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-large")

H, D, DV, EXT, PAGE = 6, 576, 512, 8, 64
SM_SCALE = 1.0 / (192**0.5)
DRAFT_MAX_KV = 1024 + PAGE - 1  # the serving grid bound: draft window 1024 + page - 1
WIN_SERVED = 4095  # the draft layers' own sliding window
DEV = "cuda"
# Exact-mode tolerance per element: bf16 output rounding (<= 2^-9 |ref|) plus fp32 dot / exp / combine error.
REL_TOL, ABS_TOL_FRAC = 2.0**-8, 4e-5
A2_LINE, A2_SEEDS, A2_LENS = 1.01, 64, (1056, 1087, 4095)


def _op():
    from sglang.kernels.ops.attention.dflash_draft_splitkv import (
        dflash_draft_splitkv_attention_fwd,
    )

    return dflash_draft_splitkv_attention_fwd


def _case(seed, lens, qlens, slots=1 << 16):
    """bs sequences: rail j of sequence s = pool slot kv_indices[kv_indptr[s] + j] (distinct random slots)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = sum(qlens)
    q = (torch.randn(n, H, D, device=DEV, generator=g) * 0.5).to(torch.bfloat16)
    k_ext = (torch.randn(n, 1, D, device=DEV, generator=g) * 0.5).to(torch.bfloat16)
    kb = (torch.randn(slots, 1, D, device=DEV, generator=g) * 0.5).to(torch.float8_e4m3fn)
    perm = torch.randperm(slots, device=DEV, generator=g)
    return dict(
        q=q,
        k_ext=k_ext,
        v_ext=k_ext[..., :DV].contiguous(),
        kb=kb,
        vb=kb[..., :DV],
        kv_indices=perm[: sum(lens)].to(torch.int64).contiguous(),
        kv_indptr=torch.tensor([0, *itertools.accumulate(lens)], dtype=torch.int32, device=DEV),
        qo_indptr=torch.tensor([0, *itertools.accumulate(qlens)], dtype=torch.int32, device=DEV),
        n=n,
        maxq=max(qlens),
    )


def _ref64(c, win, k_scale=1.0, v_scale=1.0):
    """Row m of a sequence with rail length L keeps rail key j iff (L + m) <= (j + win) and block key n iff
    n <= m and m <= n + win (win <= 0: no window); k_scale scales rail scores, v_scale rail values."""
    out = torch.zeros(c["n"], H, DV, dtype=torch.float64, device=DEV)
    qo, kv = c["qo_indptr"].tolist(), c["kv_indptr"].tolist()
    for s in range(len(qo) - 1):
        q0, q1, k0, k1 = qo[s], qo[s + 1], kv[s], kv[s + 1]
        L = k1 - k0
        idx = c["kv_indices"][k0:k1]
        rail, railv = c["kb"][idx, 0].double(), c["vb"][idx, 0].double()
        blk, blkv = c["k_ext"][q0:q1, 0].double(), c["v_ext"][q0:q1, 0].double()
        j = torch.arange(L, device=DEV)
        for m in range(q1 - q0):
            nn = torch.arange(m + 1, device=DEV)
            keep = torch.ones_like(j, dtype=torch.bool) if win <= 0 else (L + m) <= (j + win)
            keepb = torch.ones_like(nn, dtype=torch.bool) if win <= 0 else m <= nn + win
            keys = torch.cat([rail[keep] * k_scale, blk[: m + 1][keepb]], 0)
            vals = torch.cat([railv[keep] * v_scale, blkv[: m + 1][keepb]], 0)
            p = torch.softmax((c["q"][q0 + m].double() @ keys.T) * SM_SCALE, dim=-1)
            out[q0 + m] = p @ vals
    return out


def _run(c, win, *, max_kv, k_scale=1.0, v_scale=1.0, **kw):
    o = torch.full((c["n"], H, DV), float("nan"), dtype=torch.bfloat16, device=DEV)
    _op()(
        c["q"], c["k_ext"], c["v_ext"], o, c["kb"], c["vb"], c["qo_indptr"], c["kv_indptr"], c["kv_indices"],
        c["maxq"], k_scale, v_scale, SM_SCALE, max_kv_len=max_kv, sliding_window_size=win, **kw,
    )
    torch.cuda.synchronize()
    return o


def _a2_inputs(seed, slots=1 << 16, bsmax=4):
    """Check A2's inputs (benchmark shapes): q / block keys of 8 rows, an fp8 pool, one request's page table."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = (torch.randn(EXT * bsmax, H, D, device=DEV, generator=g) * 0.5).to(torch.bfloat16)
    k_ext = (torch.randn(EXT * bsmax, 1, D, device=DEV, generator=g) * 0.5).to(torch.bfloat16)
    v_ext = k_ext[..., :DV].contiguous()
    kb = (torch.randn(slots, 1, D, device=DEV, generator=g) * 0.5).to(torch.float8_e4m3fn)
    pages = torch.randperm(slots // PAGE, device=DEV, generator=g)
    req = (pages[:, None] * PAGE + torch.arange(PAGE, device=DEV)).reshape(-1)
    return dict(q=q[:EXT], k_ext=k_ext[:EXT], v_ext=v_ext[:EXT], kb=kb, vb=kb[..., :DV], req=req, n=EXT, maxq=EXT)


def _a2_rail(inp, L):
    """The rail = the last L slots of the request's first P committed tokens."""
    P = 8704 + (L - 1024) if L < 4095 else 8704
    return dict(
        inp,
        kv_indices=inp["req"][P - L : P].to(torch.int64).contiguous(),
        kv_indptr=torch.tensor([0, L], dtype=torch.int32, device=DEV),
        qo_indptr=torch.tensor([0, EXT], dtype=torch.int32, device=DEV),
    )


def _a2_ref(c, L):
    q, keys, kext = c["q"].double(), c["kb"][c["kv_indices"], 0].double(), c["k_ext"][:, 0].double()
    out = torch.empty(EXT, H, DV, dtype=torch.float64, device=DEV)
    for m in range(EXT):
        kk = torch.cat([keys[max(0, L + m - WIN_SERVED) :], kext[: m + 1]], 0)
        out[m] = torch.softmax((q[m] @ kk.T) * SM_SCALE, dim=-1) @ kk[:, :DV]
    return out


_A2 = None


def _a2_pooled():
    """Per arm: mean over (seed, L) of rel_mean = mean|o - ref| / mean|ref| and max of rel_max = max|o - ref| / max|ref|."""
    global _A2
    if _A2 is not None:
        return _A2
    from sglang.kernels.ops.attention.extend_attention import extend_attention_fwd

    rows = {"stock": [], "knob1": [], "knob2": []}
    nan = dict.fromkeys(rows, 0)
    for seed in range(2000, 2000 + A2_SEEDS):
        inp = _a2_inputs(seed)
        for L in A2_LENS:
            c = _a2_rail(inp, L)
            ref = _a2_ref(c, L)
            outs = {}
            o = torch.full((EXT, H, DV), float("nan"), dtype=torch.bfloat16, device=DEV)
            extend_attention_fwd(
                c["q"], c["k_ext"], c["v_ext"], o, c["kb"], c["vb"], c["qo_indptr"], c["kv_indptr"], c["kv_indices"],
                None, True, None, EXT, 1.0, 1.0, sm_scale=SM_SCALE, sliding_window_size=WIN_SERVED, page_size=PAGE,
            )
            torch.cuda.synchronize()
            outs["stock"] = o
            max_kv = max(DRAFT_MAX_KV, L)
            outs["knob1"] = _run(c, WIN_SERVED, max_kv=max_kv)
            outs["knob2"] = _run(c, WIN_SERVED, max_kv=max_kv, hp=True)
            for arm, o in outs.items():
                err = (o.double() - ref).abs()
                nan[arm] += int(torch.isnan(o).sum())
                rows[arm].append((float(err.mean() / ref.abs().mean()), float(err.max() / ref.abs().max())))
    _A2 = {
        arm: dict(mean_rel_mean=sum(r[0] for r in v) / len(v), max_rel_max=max(r[1] for r in v), nan=nan[arm], n=len(v))
        for arm, v in rows.items()
    }
    print("A2 pooled", _A2, flush=True)
    return _A2


class TestDFlashDraftSplitKV(unittest.TestCase):
    def _exact(self, seed, lens, qlens, win, k_scale=1.0, v_scale=1.0, pdl=False):
        c = _case(seed, lens, qlens)
        o = _run(c, win, max_kv=max(DRAFT_MAX_KV, *lens), k_scale=k_scale, v_scale=v_scale, exact=True, pdl=pdl)
        ref = _ref64(c, win, k_scale, v_scale)
        self.assertEqual(int(torch.isnan(o).sum()), 0, "NaN in the output")
        err = (o.double() - ref).abs()
        tol = REL_TOL * ref.abs() + ABS_TOL_FRAC * ref.abs().max()
        worst = float((err / tol).max())
        self.assertLessEqual(worst, 1.0, f"max |o - ref| / tol = {worst:.3f}")

    def test_exact_rails(self):
        # empty, single-key, tile-edge, draft-window and long rails; the serving grid (1087) leaves spare tiles
        for i, L in enumerate((0, 1, 63, 64, 65, 1024, 1056, 1087, 4095)):
            with self.subTest(L=L):
                self._exact(3000 + i, [L], [EXT], WIN_SERVED)

    def test_exact_windows(self):
        for seed, L, win in ((3100, 1056, 100), (3101, 1056, 1000), (3102, 1087, -1), (3103, 4095, 1024)):
            with self.subTest(L=L, win=win):
                self._exact(seed, [L], [EXT], win)

    def test_exact_scales(self):
        for seed, L, ks, vs in ((3200, 1087, 0.5, 2.0), (3201, 4095, 1.5, 0.75)):
            with self.subTest(L=L, k_scale=ks, v_scale=vs):
                self._exact(seed, [L], [EXT], WIN_SERVED, ks, vs)

    def test_exact_batch(self):
        cases = (
            (3300, [1056, 1087], [EXT, EXT], WIN_SERVED),
            (3301, [1024, 65, 4095], [EXT, 3, 1], WIN_SERVED),  # ragged block lengths
            (3302, [0, 1, 64, 1087], [EXT, EXT, EXT, EXT], 100),
        )
        for seed, lens, qlens, win in cases:
            with self.subTest(lens=lens, qlens=qlens, win=win):
                self._exact(seed, lens, qlens, win)

    def test_exact_pdl(self):
        cases = ((3400, [1056], [EXT], WIN_SERVED), (3401, [1024, 65, 4095], [EXT, 3, 1], WIN_SERVED), (3402, [1087], [EXT], 100))
        for seed, lens, qlens, win in cases:
            with self.subTest(lens=lens, win=win):
                self._exact(seed, lens, qlens, win, pdl=True)

    def test_shipped_deterministic_and_graph_safe(self):
        for hp, (L, seed) in itertools.product((False, True), ((1000, 3500), (1087, 3501))):
            with self.subTest(hp=hp, L=L):
                c = _case(seed, [L], [EXT])
                outs = [_run(c, WIN_SERVED, max_kv=DRAFT_MAX_KV, hp=hp, pdl=pdl) for pdl in (False, False, False, True)]
                for pdl in (False, True):
                    o = torch.full((c["n"], H, DV), float("nan"), dtype=torch.bfloat16, device=DEV)
                    args = (c["q"], c["k_ext"], c["v_ext"], o, c["kb"], c["vb"], c["qo_indptr"], c["kv_indptr"],
                            c["kv_indices"], c["maxq"], 1.0, 1.0, SM_SCALE)
                    kw = dict(max_kv_len=DRAFT_MAX_KV, sliding_window_size=WIN_SERVED, hp=hp, pdl=pdl)
                    s = torch.cuda.Stream()
                    s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(s):
                        _op()(*args, **kw)
                    torch.cuda.current_stream().wait_stream(s)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        _op()(*args, **kw)
                    o.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    outs.append(o.clone())
                    del graph
                self.assertEqual(int(torch.isnan(outs[0]).sum()), 0)
                base = outs[0].view(torch.int16)
                diffs = [int((x.view(torch.int16) != base).sum()) for x in outs[1:]]
                self.assertEqual(diffs, [0] * len(diffs), "eager x2, PDL, graph, graph + PDL vs the first eager call")

    def test_rail_longer_than_grid_writes_nan(self):
        # grid sized for 1024 keys (16 rail tiles): sequence 0 (1000 keys) fits, sequence 1 (1087 keys, 17 tiles) does not
        c = _case(3700, [1000, 1087], [EXT, EXT])
        ref = _ref64(c, WIN_SERVED)
        for exact in (True, False):
            with self.subTest(exact=exact):
                o = _run(c, WIN_SERVED, max_kv=1024, exact=exact)
                self.assertTrue(bool(torch.isnan(o[EXT:]).all()), "the over-long rail's rows must be NaN")
                self.assertEqual(int(torch.isnan(o[:EXT]).sum()), 0)
                if exact:
                    err = (o[:EXT].double() - ref[:EXT]).abs()
                    tol = REL_TOL * ref[:EXT].abs() + ABS_TOL_FRAC * ref[:EXT].abs().max()
                    self.assertLessEqual(float((err / tol).max()), 1.0)

    def _a2(self, arm):
        a2 = _a2_pooled()
        stock, got = a2["stock"], a2[arm]
        self.assertEqual(got["nan"], 0)
        self.assertLessEqual(got["mean_rel_mean"], A2_LINE * stock["mean_rel_mean"], f"{arm} mean rel error vs stock {stock}")
        self.assertLessEqual(got["max_rel_max"], A2_LINE * stock["max_rel_max"], f"{arm} max rel error vs stock {stock}")

    def test_a2_knob1_stock_operands_vs_stock(self):
        self._a2("knob1")

    def test_a2_knob2_bf16_operands_vs_stock(self):
        self._a2("knob2")


if __name__ == "__main__":
    unittest.main()
