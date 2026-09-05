"""Unit tests for the tiered (int8-g64 ring over int4-g32) QSA KV path: the
``qsa/sparse_attn.py`` kernels and ``MHATokenToKVPoolTiered`` (~40 MB VRAM,
no server, no model).

A small ring R = 64 over a 4096-slot pool, so ring rows are overwritten by
later slots (s + R). Torch references: int4-g32 for cold rows, int8-g64
dequant (q8 * s8, fp32 -> bf16) of the ring row for hot rows; ring rows are
compared bit-exact with the int8 kernel (``quant_store_kv_int8``) itself.
  1. ``quant_store_kv_tiered``: N = 1000 tokens into random ascending slots
     (incl. slot 0) written in alias-free chunks (span < R, as chunked prefill
     with chunk <= R); int32 and int64 loc; int4 rows / scales of EVERY slot
     bit-exact vs the reference (never overwritten); owner[r] == the last slot
     of class r; ring rows of every hot slot bit-exact vs
     ``quant_store_kv_int8``; cold slots' ring rows hold the owning slot;
  2. hot / cold boundary: write s then s + R -> owner flips to s + R (s cold,
     s + R hot, int4 row of s intact); write s + R then s -> owner == s; N > R
     in one launch raises;
  3. same-launch ring-row collisions (slots congruent mod R in ONE write, as
     with several requests in a batch): exactly one owner per contested ring
     row, its ring row bit-exact, the losers cold, every int4 row exact,
     uncontested rows hot; 20 random collision patterns;
  4. compact gather: 3 requests (300 / 1200 / 4000 tokens, permuted
     req_to_token), random owner pattern (per ring row: -1 or a random member
     of its slot class, with the ring row re-quantized from that slot),
     poisoned ring rows for the unowned classes -> rows bit-exact vs the
     per-tier torch reference; rows beyond the packed range untouched; an
     invalid position inside the packed region untouched; the trtllm strided
     layout (cu_k = arange * 64, topk 40, -1 padding, pos >= seq_len); a stale
     owner (owner = slot +/- R) -> int4 path; a call without ring / owner takes
     the plain int4 path;
  5. prefix row gather: 3 lengths, 32-row gaps after each request, mixed tiers,
     bit-exact, gaps untouched, missing ring args rejected;
  6. fp16-max scale clamp on the ring: absmax 1e6 / bf16-max groups -> s8 =
     65504 (never inf), ring row bit-exact vs the clamped torch reference, int4
     half bit-exact, both gathers finite + bit-exact, after eviction the int4
     row is served;
  7. an all-cold gather (owner = -1) equals ``_compact_kv_int4`` and an all-hot
     gather (a 4096-row ring with owner = arange) equals ``_compact_kv_int8``;
  8. ``MHATokenToKVPoolTiered`` (SGLANG_KV_TIERS_W=64): eager and lazy-VMM
     paths; ring shapes / dtypes, owner -1, kv_tiered / kv_bits / ring_mask,
     descs identical to the int4 pool (the ring is not on the owner),
     ``get_kv_size_bytes`` includes the ring, ``set_kv_buffer`` bit-exact (both
     tiers + owner), N > R raises, ``lazy_release`` resets the owner, guards;
     bad SGLANG_KV_TIERS_W rejected.
"""

import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.qsa.sparse_attn import KV_INT4_GROUP as G4
from sglang.srt.layers.attention.qsa.sparse_attn import KV_INT8_GROUP as G8
from sglang.srt.layers.attention.qsa.sparse_attn import (
    quant_store_kv_int8,
    quant_store_kv_tiered,
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_prefix_gather_dequant_tiered,
)
from sglang.srt.mem_cache.int4_kv_pool import MHATokenToKVPoolInt4
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.mem_cache.tiered_kv_pool import (
    MHATokenToKVPoolTiered,
    ring_slots_from_env,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-small")

SLOTS, HEADS, DIM, BATCH, TOPK = 4096, 2, 256, 3, 64
R = 64
MASK = R - 1
NG4, DH, NG8 = DIM // G4, DIM // 2, DIM // G8


# ---------------------------------------------------------------------- torch references
def ref_pack(q):
    return ((q[..., 0::2] + 8) | ((q[..., 1::2] + 8) << 4)).to(torch.uint8)


def ref_unpack(p):
    b = p.to(torch.int32)
    return torch.stack([(b & 15) - 8, (b >> 4) - 8], dim=-1).reshape(*p.shape[:-1], -1)


def ref_quant4(x):
    xf = x.float().reshape(*x.shape[:-1], NG4, G4)
    a = xf.abs().amax(-1)
    s = torch.where(a > 0, a / 7.0, torch.ones_like(a)).clamp(max=65504.0).half()
    q = (
        torch.clamp(torch.round(xf / s.float()[..., None]), -7, 7)
        .to(torch.int32)
        .reshape(x.shape)
    )
    return ref_pack(q), s


def ref_dequant4(p, s):
    return (ref_unpack(p).float() * s.float().repeat_interleave(G4, dim=-1)).to(
        torch.bfloat16
    )


def ref_quant8(x):
    """int8-g64 with the fp16-max clamp (what the tiered writer stores in the ring)."""
    xf = x.float().reshape(*x.shape[:-1], NG8, G8)
    a = xf.abs().amax(-1)
    s = torch.where(a > 0, a / 127.0, torch.ones_like(a)).clamp(max=65504.0).half()
    q = (
        torch.clamp(torch.round(xf / s.float()[..., None]), -127, 127)
        .to(torch.int8)
        .reshape(x.shape)
    )
    return q, s


def ref_dequant8(q, s):
    return (q.float() * s.float().repeat_interleave(G8, dim=-1)).to(torch.bfloat16)


def chunks_alias_free(sl):
    """Split an ascending slot list into consecutive chunks whose span is < R."""
    out, start = [], 0
    for i in range(1, len(sl) + 1):
        if i == len(sl) or int(sl[i]) - int(sl[start]) >= R:
            out.append((start, i))
            start = i
    return out


def new_bufs(dev):
    k4 = torch.full((SLOTS, HEADS, DH), 0x77, dtype=torch.uint8, device=dev)
    v4 = torch.full((SLOTS, HEADS, DH), 0x99, dtype=torch.uint8, device=dev)
    ks4 = torch.full((SLOTS, HEADS, NG4), 3.0, dtype=torch.float16, device=dev)
    vs4 = torch.full((SLOTS, HEADS, NG4), 5.0, dtype=torch.float16, device=dev)
    rk = torch.full((R, HEADS, DIM), 0x33, dtype=torch.int8, device=dev)
    rv = torch.full((R, HEADS, DIM), 0x44, dtype=torch.int8, device=dev)
    rks = torch.full((R, HEADS, NG8), 7.0, dtype=torch.float16, device=dev)
    rvs = torch.full((R, HEADS, NG8), 9.0, dtype=torch.float16, device=dev)
    owner = torch.full((R,), -1, dtype=torch.int32, device=dev)
    return [k4, v4, ks4, vs4, rk, rv, rks, rvs, owner]


def tiered_write(xk, xv, loc, bufs, ones):
    k4, v4, ks4, vs4, rk, rv, rks, rvs, owner = bufs
    quant_store_kv_tiered(
        xk, xv, loc, k4, v4, ks4, vs4, rk, rv, rks, rvs, owner, ones, ones, MASK
    )


def one(i, dev):
    return torch.tensor([i], dtype=torch.int32, device=dev)


@unittest.skipUnless(torch.cuda.is_available(), "Triton KV kernels require CUDA")
class TestTieredKVKernels(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        dev = cls.dev = "cuda"
        cls.ones = torch.ones(HEADS, DIM, dtype=torch.float16, device=dev)
        # ---- inputs of the write tests
        cls.N = 1000
        cls.xk = torch.randn(cls.N, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        cls.xv = torch.randn(cls.N, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        cls.xk[5] = 0  # zero row -> s = 1 in both tiers
        sl = torch.randperm(SLOTS, device=dev)[: cls.N].sort().values
        sl[0] = 0  # slot 0 (the dummy-write slot) is part of the set
        cls.sl = sl
        cls.pk_ref, cls.sk_ref = ref_quant4(cls.xk)
        cls.pv_ref, cls.sv_ref = ref_quant4(cls.xv)
        # int8 reference straight from the int8 kernel (identical arithmetic)
        cls.k8_ref = torch.empty(SLOTS, HEADS, DIM, dtype=torch.int8, device=dev)
        cls.v8_ref = torch.empty_like(cls.k8_ref)
        cls.ks8_ref = torch.empty(SLOTS, HEADS, NG8, dtype=torch.float16, device=dev)
        cls.vs8_ref = torch.empty_like(cls.ks8_ref)
        quant_store_kv_int8(
            cls.xk,
            cls.xv,
            sl.to(torch.int32),
            cls.k8_ref,
            cls.v8_ref,
            cls.ks8_ref,
            cls.vs8_ref,
            cls.ones,
            cls.ones,
        )
        torch.cuda.synchronize()
        cls.q8k, cls.s8k = ref_quant8(cls.xk)
        cls.q8v, cls.s8v = ref_quant8(cls.xv)
        # ---- full pool with a random owner pattern for the gather tests
        cls.k16 = torch.randn(SLOTS, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        cls.v16 = torch.randn(SLOTS, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        bufs = cls.bufs = new_bufs(dev)
        for a in range(0, SLOTS, R):  # ascending chunks of exactly R slots
            tiered_write(
                cls.k16[a : a + R],
                cls.v16[a : a + R],
                torch.arange(a, a + R, device=dev, dtype=torch.int32),
                bufs,
                cls.ones,
            )
        torch.cuda.synchronize()
        k4, v4, ks4, vs4, rk, rv, rks, rvs, owner = bufs
        cls.pk, cls.sk = ref_quant4(cls.k16)
        cls.pv, cls.sv = ref_quant4(cls.v16)
        cls.owner_after_fill = owner.clone()
        # random owner pattern: per ring row r either -1 (nobody) or a random member
        # s' of the class {r, r + R, ...}; the ring row is then re-quantized from s'
        n_cls = cls.n_cls = SLOTS // R
        pick = torch.randint(0, n_cls, (R,), device=dev)
        pick = torch.where(
            torch.rand(R, device=dev) < 0.35, torch.full_like(pick, -1), pick
        )
        own = torch.where(
            pick < 0, torch.full_like(pick, -1), pick * R + torch.arange(R, device=dev)
        )
        own[0] = 0  # ring row 0 owned by slot 0 (dummy-write slot)
        own[1] = -1
        own[2] = 2 + R * (n_cls - 1)  # the last slot of its class
        owner.copy_(own.to(torch.int32))
        cls.hot_rows = torch.nonzero(own >= 0).flatten()
        quant_store_kv_int8(
            cls.k16[own[cls.hot_rows]],
            cls.v16[own[cls.hot_rows]],
            cls.hot_rows.to(torch.int32),
            rk,
            rv,
            rks,
            rvs,
            cls.ones,
            cls.ones,
        )
        # poison every unowned ring row: garbage payload and NaN scales must never
        # reach the output
        cls.dead = torch.nonzero(own < 0).flatten()
        rk[cls.dead] = -128
        rv[cls.dead] = 127
        rks[cls.dead] = float("nan")
        rvs[cls.dead] = float("inf")
        torch.cuda.synchronize()
        cls.k8_all = torch.empty(SLOTS, HEADS, DIM, dtype=torch.int8, device=dev)
        cls.v8_all = torch.empty_like(cls.k8_all)
        cls.ks8_all = torch.empty(SLOTS, HEADS, NG8, dtype=torch.float16, device=dev)
        cls.vs8_all = torch.empty_like(cls.ks8_all)
        quant_store_kv_int8(
            cls.k16,
            cls.v16,
            torch.arange(SLOTS, device=dev, dtype=torch.int32),
            cls.k8_all,
            cls.v8_all,
            cls.ks8_all,
            cls.vs8_all,
            cls.ones,
            cls.ones,
        )
        torch.cuda.synchronize()
        ar = torch.arange(SLOTS, device=dev)
        cls.is_hot = owner.long()[ar & MASK] == ar  # [SLOTS] bool
        cls.d4k, cls.d4v = ref_dequant4(cls.pk, cls.sk), ref_dequant4(cls.pv, cls.sv)
        cls.d8k = ref_dequant8(cls.k8_all, cls.ks8_all)
        cls.d8v = ref_dequant8(cls.v8_all, cls.vs8_all)
        cls.dk_all = torch.where(cls.is_hot[:, None, None], cls.d8k, cls.d4k)
        cls.dv_all = torch.where(cls.is_hot[:, None, None], cls.d8v, cls.d4v)
        # ---- requests
        cls.seq_lens = torch.tensor([300, 1200, 4000], dtype=torch.int32, device=dev)
        cls.req_to_token = torch.stack(
            [torch.randperm(SLOTS, device=dev) for _ in range(BATCH)]
        ).to(torch.int32)
        cls.req_indices = torch.arange(BATCH, dtype=torch.int32, device=dev)
        indices = torch.stack(
            [
                torch.randperm(int(n), device=dev)[:TOPK].sort().values
                for n in cls.seq_lens
            ]
        ).to(torch.int32)
        # the random owner pattern leaves only ~2/3 R hot slots in 4096: make sure
        # every request selects up to 8 of them
        for b in range(BATCH):
            hot_pos = torch.nonzero(
                cls.is_hot[cls.req_to_token[b, : int(cls.seq_lens[b])].long()]
            ).flatten()
            if hot_pos.numel():
                take = hot_pos[torch.randperm(hot_pos.numel(), device=dev)[:8]]
                taken = set(take.tolist())
                rest = torch.tensor(
                    [p for p in indices[b].tolist() if p not in taken],
                    device=dev,
                    dtype=torch.int64,
                )
                indices[b] = (
                    torch.cat([rest[: TOPK - take.numel()], take])
                    .sort()
                    .values.to(torch.int32)
                )
        cls.indices = indices

    # ------------------------------------------------------------ helpers
    def tier_kwargs(self, bufs=None, owner=None):
        k4, v4, ks4, vs4, rk, rv, rks, rvs, own = bufs or self.bufs
        return dict(
            k_scale=ks4,
            v_scale=vs4,
            sm_k=self.ones,
            sm_v=self.ones,
            kv_bits=4,
            ring_k=rk,
            ring_v=rv,
            ring_ks=rks,
            ring_vs=rvs,
            owner=own if owner is None else owner,
            ring_mask=MASK,
        )

    def int4_kwargs(self, bufs=None):
        k4, v4, ks4, vs4 = (bufs or self.bufs)[:4]
        return dict(k_scale=ks4, v_scale=vs4, sm_k=self.ones, sm_v=self.ones, kv_bits=4)

    def run_compact(self, idx, kw=None, bufs=None):
        dev = self.dev
        k4, v4 = (bufs or self.bufs)[:2]
        cu_k = torch.empty(BATCH + 1, dtype=torch.int32, device=dev)
        counts = torch.empty(BATCH, dtype=torch.int32, device=dev)
        qwen_sparse_fa2_cu_seqlens_triton(self.seq_lens, idx, counts, cu_k, BATCH, TOPK)
        n = int(cu_k[-1])
        out_k = torch.full(
            (BATCH * TOPK + 16, HEADS, DIM),
            float("nan"),
            device=dev,
            dtype=torch.bfloat16,
        )
        out_v = torch.full_like(out_k, float("nan"))
        qwen_sparse_kv_extraction_compact_triton(
            k4,
            v4,
            self.req_to_token,
            self.req_indices,
            idx,
            self.seq_lens,
            cu_k,
            out_k,
            out_v,
            BATCH,
            TOPK,
            **(kw or self.tier_kwargs(bufs)),
        )
        torch.cuda.synchronize()
        return cu_k, n, out_k, out_v

    # ------------------------------------------------------------ 1. dual write
    def test_dual_write_ascending_slots_ring_wrap(self):
        dev, sl = self.dev, self.sl
        self.assertEqual(int(sl.unique().numel()), self.N)
        self.assertTrue(torch.equal(self.k8_ref[sl], self.q8k))
        self.assertTrue(torch.equal(self.ks8_ref[sl], self.s8k))
        ch = chunks_alias_free(sl)
        self.assertGreater(len(ch), 20)
        self.assertLessEqual(max(e - a for a, e in ch), R)
        exp_owner = torch.full((R,), -1, dtype=torch.int64, device=dev)
        for s in sl.tolist():
            exp_owner[s & MASK] = s  # ascending -> the last writer of each class
        n_wrapped = int(((sl & MASK).bincount(minlength=R) > 1).sum())
        self.assertGreater(n_wrapped, 40, "the slot set must make most ring rows wrap")
        for loc_dtype in (torch.int32, torch.int64):
            with self.subTest(loc_dtype=loc_dtype):
                bufs = new_bufs(dev)
                k4, v4, ks4, vs4, rk, rv, rks, rvs, owner = bufs
                for a, e in ch:
                    tiered_write(
                        self.xk[a:e],
                        self.xv[a:e],
                        sl[a:e].to(loc_dtype),
                        bufs,
                        self.ones,
                    )
                torch.cuda.synchronize()
                ll = sl.long()
                self.assertTrue(torch.equal(k4[ll], self.pk_ref))
                self.assertTrue(torch.equal(ks4[ll], self.sk_ref))
                self.assertTrue(torch.equal(v4[ll], self.pv_ref))
                self.assertTrue(torch.equal(vs4[ll], self.sv_ref))
                untouched = torch.ones(SLOTS, dtype=torch.bool, device=dev)
                untouched[ll] = False
                self.assertTrue((k4[untouched] == 0x77).all())
                self.assertTrue((ks4[untouched] == 3.0).all())
                self.assertTrue(torch.equal(owner.long(), exp_owner))
                self.assertGreaterEqual(int(exp_owner[0]), 0)
                hot = ll[
                    owner.long()[ll & MASK] == ll
                ]  # slots that still own their row
                cold = ll[owner.long()[ll & MASK] != ll]
                self.assertEqual(hot.numel(), int((exp_owner >= 0).sum()))
                self.assertEqual(cold.numel(), self.N - hot.numel())
                self.assertGreater(cold.numel(), 900)
                self.assertTrue(torch.equal(rk[hot & MASK], self.k8_ref[hot]))
                self.assertTrue(torch.equal(rks[hot & MASK], self.ks8_ref[hot]))
                self.assertTrue(torch.equal(rv[hot & MASK], self.v8_ref[hot]))
                self.assertTrue(torch.equal(rvs[hot & MASK], self.vs8_ref[hot]))
                own_of_cold = owner.long()[cold & MASK]
                self.assertTrue(torch.equal(rk[cold & MASK], self.k8_ref[own_of_cold]))
                # zero row: s = 1, nibbles 8 (and s8 = 1, q8 = 0 while hot)
                self.assertTrue((ks4[ll[5]] == 1.0).all() and (k4[ll[5]] == 0x88).all())
                if int(owner[ll[5] & MASK]) == int(ll[5]):
                    self.assertTrue((rks[ll[5] & MASK] == 1.0).all())
                    self.assertTrue((rk[ll[5] & MASK] == 0).all())

    # ------------------------------------------------------------ 2. hot/cold boundary
    def test_hot_cold_boundary(self):
        dev, s0 = self.dev, 100
        xk, xv = self.xk, self.xv
        bufs = new_bufs(dev)
        k4, v4, ks4, vs4, rk, rv, rks, rvs, owner = bufs
        tiered_write(xk[:1], xv[:1], one(s0, dev), bufs, self.ones)
        torch.cuda.synchronize()
        self.assertEqual(int(owner[s0 & MASK]), s0)
        self.assertTrue(torch.equal(rk[s0 & MASK], self.q8k[0]))
        self.assertTrue(torch.equal(k4[s0], self.pk_ref[0]))
        tiered_write(xk[1:2], xv[1:2], one(s0 + R, dev), bufs, self.ones)  # evicts s0
        torch.cuda.synchronize()
        self.assertEqual(int(owner[s0 & MASK]), s0 + R, "owner must flip to s + R")
        self.assertTrue(torch.equal(rk[s0 & MASK], self.q8k[1]))
        self.assertTrue(torch.equal(rks[s0 & MASK], self.s8k[1]))
        self.assertTrue(torch.equal(k4[s0], self.pk_ref[0]), "int4 row of s intact")
        self.assertTrue(torch.equal(ks4[s0], self.sk_ref[0]))
        self.assertTrue(torch.equal(k4[s0 + R], self.pk_ref[1]))
        self.assertTrue(torch.equal(v4[s0 + R], self.pv_ref[1]))
        self.assertEqual(int((owner == -1).sum()), R - 1)
        bufs = new_bufs(dev)
        k4, v4, ks4, vs4, rk, rv, rks, rvs, owner = bufs
        tiered_write(xk[1:2], xv[1:2], one(s0 + R, dev), bufs, self.ones)
        tiered_write(xk[:1], xv[:1], one(s0, dev), bufs, self.ones)  # s after s + R
        torch.cuda.synchronize()
        self.assertEqual(int(owner[s0 & MASK]), s0)
        self.assertTrue(torch.equal(rk[s0 & MASK], self.q8k[0]))
        self.assertTrue(torch.equal(k4[s0 + R], self.pk_ref[1]))
        with self.assertRaises(AssertionError):  # N > R in one launch
            tiered_write(
                xk[: R + 1],
                xv[: R + 1],
                torch.arange(R + 1, device=dev, dtype=torch.int32),
                bufs,
                self.ones,
            )

    # ------------------------------------------------------------ 3. collisions
    def test_same_launch_ring_row_collisions(self):
        # slots congruent mod R in the same write: the stamp launch picks one owner
        # per ring row, only the owner's programs write the ring row, the losers
        # are cold; no row may mix two tokens' bytes.
        dev = self.dev
        n_contested_total = 0
        for _ in range(20):
            bufs = new_bufs(dev)
            k4, v4, ks4, vs4, rk, rv, rks, rvs, owner = bufs
            n_c = R  # N = R tokens, heavy aliasing
            classes = torch.randint(0, 8, (n_c,), device=dev)  # ring rows 0..7 only
            members = torch.randperm(SLOTS // R, device=dev)[:n_c]  # distinct slots
            loc_c = (members * R + classes).to(torch.int32)
            self.assertEqual(int(loc_c.unique().numel()), n_c)
            perm = torch.randperm(self.N, device=dev)[:n_c]
            tiered_write(self.xk[perm], self.xv[perm], loc_c, bufs, self.ones)
            torch.cuda.synchronize()
            ll = loc_c.long()
            self.assertTrue(torch.equal(k4[ll], self.pk_ref[perm]))
            self.assertTrue(torch.equal(ks4[ll], self.sk_ref[perm]))
            self.assertTrue(torch.equal(v4[ll], self.pv_ref[perm]))
            self.assertTrue(torch.equal(vs4[ll], self.sv_ref[perm]))
            for r_ in range(R):
                cand = ll[(ll & MASK) == r_]
                if cand.numel() == 0:
                    self.assertEqual(int(owner[r_]), -1)
                    self.assertTrue((rk[r_] == 0x33).all() and (rks[r_] == 7.0).all())
                    continue
                w = int(owner[r_])
                self.assertIn(w, set(cand.tolist()), f"owner of row {r_}")
                # the token (row of xk) written to slot w
                i_w = int(perm[int(torch.nonzero(loc_c == w).item())])
                self.assertTrue(torch.equal(rk[r_], self.q8k[i_w]), f"ring row {r_}")
                self.assertTrue(torch.equal(rks[r_], self.s8k[i_w]), f"ring row {r_}")
                self.assertTrue(torch.equal(rv[r_], self.q8v[i_w]), f"ring row {r_}")
                self.assertTrue(torch.equal(rvs[r_], self.s8v[i_w]), f"ring row {r_}")
                n_contested_total += int(cand.numel() > 1)
            hot_c = ll[owner.long()[ll & MASK] == ll]
            self.assertEqual(hot_c.numel(), int((owner >= 0).sum()))
            self.assertLessEqual(hot_c.numel(), 8)
        self.assertGreater(n_contested_total, 100)
        # an uncontested token in the same launch as a contested row stays hot
        bufs = new_bufs(dev)
        k4, v4, ks4, vs4, rk, rv, rks, rvs, owner = bufs
        loc_m = torch.tensor([5, 5 + R, 5 + 2 * R, 7], dtype=torch.int32, device=dev)
        tiered_write(self.xk[:4], self.xv[:4], loc_m, bufs, self.ones)
        torch.cuda.synchronize()
        w5 = int(owner[5])
        self.assertIn(w5, (5, 5 + R, 5 + 2 * R))
        self.assertEqual(int(owner[7]), 7)
        i5 = {5: 0, 5 + R: 1, 5 + 2 * R: 2}[w5]
        self.assertTrue(
            torch.equal(rk[5], self.q8k[i5]) and torch.equal(rks[5], self.s8k[i5])
        )
        self.assertTrue(
            torch.equal(rv[5], self.q8v[i5]) and torch.equal(rvs[5], self.s8v[i5])
        )
        self.assertTrue(
            torch.equal(rk[7], self.q8k[3]) and torch.equal(rv[7], self.q8v[3])
        )
        self.assertTrue(torch.equal(k4[loc_m.long()], self.pk_ref[:4]))
        self.assertEqual(int((owner >= 0).sum()), 2)

    # ------------------------------------------------------------ 4. compact gather
    def test_full_pool_and_owner_pattern_premises(self):
        k4, v4, ks4, vs4 = self.bufs[:4]
        self.assertTrue(torch.equal(k4, self.pk) and torch.equal(ks4, self.sk))
        self.assertTrue(torch.equal(v4, self.pv) and torch.equal(vs4, self.sv))
        self.assertTrue(
            torch.equal(
                self.owner_after_fill.long(),
                torch.arange(SLOTS - R, SLOTS, device=self.dev),
            )
        )
        self.assertTrue(10 < self.hot_rows.numel() < R - 5)
        self.assertEqual(int(self.is_hot.sum()), self.hot_rows.numel())
        self.assertTrue(bool(self.is_hot[0]) and not bool(self.is_hot[1]))
        self.assertTrue(bool(self.is_hot[2 + R * (self.n_cls - 1)]))
        # the tiers must differ for the gather tests to discriminate
        self.assertFalse(torch.equal(self.d8k[self.is_hot], self.d4k[self.is_hot]))

    def test_compact_gather_per_tier_bit_exact(self):
        cu_k, n, out_k, out_v = self.run_compact(self.indices)
        self.assertEqual(n, BATCH * TOPK)
        n_hot_sel = 0
        for b in range(BATCH):
            rows = self.req_to_token[b, self.indices[b].long()].long()
            a, e = int(cu_k[b]), int(cu_k[b + 1])
            self.assertTrue(torch.equal(out_k[a:e], self.dk_all[rows]), f"req {b} K")
            self.assertTrue(torch.equal(out_v[a:e], self.dv_all[rows]), f"req {b} V")
            n_hot_sel += int(self.is_hot[rows].sum())
        self.assertTrue(
            10 < n_hot_sel < n, f"needs a hot/cold mix, got {n_hot_sel}/{n}"
        )
        self.assertTrue(torch.isnan(out_k[n:]).all() and torch.isnan(out_v[n:]).all())
        self.assertTrue(torch.isfinite(out_k[:n]).all(), "a poisoned ring row leaked")
        # the same call with the int4 kwargs only (no owner) takes the plain int4 path
        _, _, ok4, ov4 = self.run_compact(self.indices, self.int4_kwargs())
        for b in range(BATCH):
            rows = self.req_to_token[b, self.indices[b].long()].long()
            a, e = int(cu_k[b]), int(cu_k[b + 1])
            self.assertTrue(torch.equal(ok4[a:e], self.d4k[rows]))
            self.assertTrue(torch.equal(ov4[a:e], self.d4v[rows]))

    def test_compact_gather_invalid_position_untouched(self):
        cu_k, n, out_k, _ = self.run_compact(self.indices)
        bad = self.indices.clone()
        bad[1, 0] = int(self.seq_lens[1]) + 5
        cu_b, n_b, ok_b, ov_b = self.run_compact(bad)
        self.assertEqual(n_b, BATCH * TOPK - 1)
        a = int(cu_b[1])
        self.assertTrue(torch.isnan(ok_b[a]).all() and torch.isnan(ov_b[a]).all())
        rows = self.req_to_token[1, bad[1, 1 : TOPK - 1].long()].long()
        self.assertTrue(torch.equal(ok_b[a + 1 : a + TOPK - 1], self.dk_all[rows]))
        self.assertTrue(torch.equal(ov_b[a + 1 : a + TOPK - 1], self.dv_all[rows]))
        self.assertTrue(torch.equal(ok_b[:a], out_k[:a]))
        self.assertTrue(torch.equal(ok_b[int(cu_b[2]) : n_b], out_k[int(cu_k[2]) : n]))
        self.assertTrue(torch.isnan(ok_b[n_b:]).all())

    def test_compact_gather_trtllm_strided_layout(self):
        # cu_k = arange * stride (stride 64 > topk 40), -1 padding, a position >= seq_len
        dev = self.dev
        page_s, topk_s = 64, 40
        stride_s = -(-topk_s // page_s) * page_s
        cu_s = torch.arange(BATCH + 1, dtype=torch.int32, device=dev) * stride_s
        nsel = [topk_s, 25, 33]
        idx_s = torch.full((BATCH, topk_s), -1, dtype=torch.int32, device=dev)
        for b in range(BATCH):
            idx_s[b, : nsel[b]] = (
                torch.randperm(int(self.seq_lens[b]), device=dev)[: nsel[b]]
                .sort()
                .values.to(torch.int32)
            )
        idx_s[2, 5] = int(self.seq_lens[2]) + 3
        ok_s = torch.full(
            (BATCH * stride_s, HEADS, DIM),
            float("nan"),
            device=dev,
            dtype=torch.bfloat16,
        )
        ov_s = torch.full_like(ok_s, float("nan"))
        k4, v4 = self.bufs[:2]
        qwen_sparse_kv_extraction_compact_triton(
            k4,
            v4,
            self.req_to_token,
            self.req_indices,
            idx_s,
            self.seq_lens,
            cu_s,
            ok_s,
            ov_s,
            BATCH,
            topk_s,
            **self.tier_kwargs(),
        )
        torch.cuda.synchronize()
        written = torch.zeros(BATCH * stride_s, dtype=torch.bool, device=dev)
        for b in range(BATCH):
            for c in range(topk_s):
                pos = int(idx_s[b, c])
                if 0 <= pos < int(self.seq_lens[b]):
                    r_ = b * stride_s + c
                    slot = int(self.req_to_token[b, pos])
                    self.assertTrue(torch.equal(ok_s[r_], self.dk_all[slot]), (b, c))
                    self.assertTrue(torch.equal(ov_s[r_], self.dv_all[slot]), (b, c))
                    written[r_] = True
        self.assertEqual(int(written.sum()), sum(nsel) - 1)
        self.assertTrue(torch.isnan(ok_s[~written]).all())
        self.assertTrue(torch.isnan(ov_s[~written]).all())

    def test_stale_owner_falls_back_to_int4(self):
        # a hot slot's ring row is re-owned by slot +/- R (as after an eviction)
        owner = self.bufs[8].clone()
        sel_slot = int(self.req_to_token[0, self.indices[0, 3].long()])
        owner[sel_slot & MASK] = sel_slot + R if sel_slot + R < SLOTS else sel_slot - R
        cu_t, _, ok_t, ov_t = self.run_compact(
            self.indices, self.tier_kwargs(owner=owner)
        )
        r0 = int(cu_t[0]) + 3
        self.assertTrue(torch.equal(ok_t[r0], self.d4k[sel_slot]))
        self.assertTrue(torch.equal(ov_t[r0], self.d4v[sel_slot]))

    # ------------------------------------------------------------ 5. prefix gather
    def _prefix_gather(self, bufs, lens, req_idx, cu, pk, pv, batch, max_len):
        k4, v4, ks4, vs4, rk, rv, rks, rvs, owner = bufs
        qwen_sparse_prefix_gather_dequant_tiered(
            k4,
            v4,
            ks4,
            vs4,
            self.ones,
            self.ones,
            self.req_to_token,
            req_idx,
            lens,
            cu,
            pk,
            pv,
            batch,
            max_len,
            ring_k=rk,
            ring_v=rv,
            ring_ks=rks,
            ring_vs=rvs,
            owner=owner,
            ring_mask=MASK,
        )
        torch.cuda.synchronize()

    def test_prefix_gather_mixed_tiers_gaps_untouched(self):
        dev = self.dev
        lens = torch.tensor([300, 1203, 4001], dtype=torch.int32, device=dev)
        req_idx = torch.tensor([2, 0, 1], dtype=torch.int64, device=dev)
        gap = 32
        cu = F.pad((lens + gap).cumsum(0), (1, 0)).to(torch.int32).contiguous()
        total = int(cu[-1])
        pkk = torch.full(
            (total + 100, HEADS, DIM), float("nan"), device=dev, dtype=torch.bfloat16
        )
        pvv = torch.full_like(pkk, float("nan"))
        self._prefix_gather(
            self.bufs, lens, req_idx, cu, pkk, pvv, BATCH, int(lens.max())
        )
        written = torch.zeros(pkk.shape[0], dtype=torch.bool, device=dev)
        n_hot_pre = 0
        for b in range(BATCH):
            rows = self.req_to_token[int(req_idx[b]), : int(lens[b])].long()
            a, e = int(cu[b]), int(cu[b]) + int(lens[b])
            self.assertTrue(torch.equal(pkk[a:e], self.dk_all[rows]), f"req {b} K")
            self.assertTrue(torch.equal(pvv[a:e], self.dv_all[rows]), f"req {b} V")
            written[a:e] = True
            n_hot_pre += int(self.is_hot[rows].sum())
            self.assertTrue(torch.isnan(pkk[e : e + gap]).all(), f"gap after req {b}")
            self.assertTrue(torch.isnan(pvv[e : e + gap]).all(), f"gap after req {b}")
        self.assertTrue(
            torch.isnan(pkk[~written]).all() and torch.isnan(pvv[~written]).all()
        )
        self.assertEqual(int(written.sum()), int(lens.sum()))
        self.assertTrue(0 < n_hot_pre < int(lens.sum()))
        k4, v4, ks4, vs4 = self.bufs[:4]
        with self.assertRaises(AssertionError):  # ring / owner missing
            qwen_sparse_prefix_gather_dequant_tiered(
                k4,
                v4,
                ks4,
                vs4,
                self.ones,
                self.ones,
                self.req_to_token,
                req_idx,
                lens,
                cu,
                pkk,
                pvv,
                BATCH,
                int(lens.max()),
            )

    # ------------------------------------------------------------ 6. fp16-max clamp
    def test_fp16_max_scale_clamp_on_ring(self):
        dev = self.dev
        bufs = [t.clone() for t in self.bufs]
        k4, v4, ks4, vs4, rk, rv, rks, rvs, owner = bufs
        big = torch.zeros(1, HEADS, DIM, device=dev, dtype=torch.bfloat16)
        big[0, 0, :32] = torch.tensor(
            [1e6, -1e6, 5e5, -5e5, 4.6e5, -4.6e5, 1e5, -1e5] * 4, device=dev
        ).to(torch.bfloat16)
        big[0, 1, 64:96] = torch.finfo(torch.bfloat16).max
        bigv = -big
        pos7 = 17
        slot7 = int(self.req_to_token[0, pos7])
        tiered_write(big, bigv, one(slot7, dev), bufs, self.ones)
        torch.cuda.synchronize()
        q8b, s8b = ref_quant8(big)
        q8bv, s8bv = ref_quant8(bigv)
        p4b, s4b = ref_quant4(big)
        p4bv, s4bv = ref_quant4(bigv)
        self.assertEqual(float(s8b[0, 1, 1]), 65504.0)
        self.assertEqual(float(s4b[0, 1, 2]), 65504.0)
        self.assertTrue(torch.isfinite(s8b).all())
        self.assertEqual(int(owner[slot7 & MASK]), slot7)
        self.assertTrue(torch.isfinite(rks[slot7 & MASK]).all(), "ring scale inf")
        self.assertTrue(torch.isfinite(rvs[slot7 & MASK]).all(), "ring scale inf")
        self.assertTrue(torch.equal(rk[slot7 & MASK], q8b[0]))
        self.assertTrue(torch.equal(rks[slot7 & MASK], s8b[0]))
        self.assertTrue(torch.equal(rv[slot7 & MASK], q8bv[0]))
        self.assertTrue(torch.equal(rvs[slot7 & MASK], s8bv[0]))
        self.assertTrue(
            torch.equal(k4[slot7], p4b[0]) and torch.equal(ks4[slot7], s4b[0])
        )
        self.assertTrue(
            torch.equal(v4[slot7], p4bv[0]) and torch.equal(vs4[slot7], s4bv[0])
        )
        d8b, d8bv = ref_dequant8(q8b, s8b)[0], ref_dequant8(q8bv, s8bv)[0]
        self.assertTrue(torch.isfinite(d8b).all())
        self.assertEqual(
            float(d8b[1, 64]), float(torch.tensor(127 * 65504.0).to(torch.bfloat16))
        )
        idx7 = self.indices.clone()
        idx7[0, 0] = pos7
        cu7, n7, ok7, ov7 = self.run_compact(idx7, bufs=bufs)
        r7 = int(cu7[0])
        self.assertTrue(
            torch.isfinite(ok7[:n7]).all() and torch.isfinite(ov7[:n7]).all()
        )
        self.assertTrue(torch.equal(ok7[r7], d8b) and torch.equal(ov7[r7], d8bv))
        pk7 = torch.full(
            (pos7 + 8, HEADS, DIM), float("nan"), device=dev, dtype=torch.bfloat16
        )
        pv7 = torch.full_like(pk7, float("nan"))
        self._prefix_gather(
            bufs,
            torch.tensor([pos7 + 8], dtype=torch.int32, device=dev),
            torch.tensor([0], dtype=torch.int64, device=dev),
            torch.tensor([0, pos7 + 8], dtype=torch.int32, device=dev),
            pk7,
            pv7,
            1,
            pos7 + 8,
        )
        self.assertTrue(torch.isfinite(pk7).all() and torch.isfinite(pv7).all())
        self.assertTrue(torch.equal(pk7[pos7], d8b) and torch.equal(pv7[pos7], d8bv))
        # evict it (write slot7 +/- R) and read again: the (clamped) int4 row is served
        owner[slot7 & MASK] = slot7 + R if slot7 + R < SLOTS else slot7 - R
        cu7c, n7c, ok7c, _ = self.run_compact(idx7, bufs=bufs)
        self.assertTrue(torch.equal(ok7c[int(cu7c[0])], ref_dequant4(p4b, s4b)[0]))
        self.assertTrue(torch.isfinite(ok7c[:n7c]).all())

    # ------------------------------------------------------------ 7. single-tier equivalence
    def test_all_cold_and_all_hot_match_single_tier_kernels(self):
        dev = self.dev
        tk_ = 2048
        seq_b = torch.tensor([4000], dtype=torch.int32, device=dev)
        idx_b = (
            torch.randperm(4000, device=dev)[:tk_].sort().values.to(torch.int32)[None]
        )
        cu_b = torch.tensor([0, tk_], dtype=torch.int32, device=dev)
        k4, v4, ks4, vs4 = self.bufs[:4]
        rows_b = self.req_to_token[0, idx_b[0].long()].long()

        def gather(k, v, kw):
            ob_k = torch.empty(tk_, HEADS, DIM, device=dev, dtype=torch.bfloat16)
            ob_v = torch.empty_like(ob_k)
            qwen_sparse_kv_extraction_compact_triton(
                k, v, self.req_to_token, self.req_indices, idx_b, seq_b, cu_b,
                ob_k, ob_v, 1, tk_, **kw,
            )  # fmt: skip
            torch.cuda.synchronize()
            return ob_k, ob_v

        # all-cold: owner = -1 everywhere -> equals the int4 kernel
        owner_all_cold = torch.full((R,), -1, dtype=torch.int32, device=dev)
        cold_k, cold_v = gather(k4, v4, self.tier_kwargs(owner=owner_all_cold))
        i4_k, i4_v = gather(k4, v4, self.int4_kwargs())
        self.assertTrue(torch.equal(cold_k, i4_k) and torch.equal(cold_v, i4_v))
        self.assertTrue(torch.equal(cold_k, self.d4k[rows_b]))
        # all-hot: a ring as large as the pool (slot & mask == slot) whose rows ARE
        # the int8 pool and whose owner table is arange -> every selected slot is hot
        owner_all_hot = torch.arange(SLOTS, device=dev, dtype=torch.int32)
        n_hot_b = int((owner_all_hot.long()[rows_b & (SLOTS - 1)] == rows_b).sum())
        self.assertEqual(n_hot_b, tk_)
        kw_hot = dict(
            k_scale=ks4,
            v_scale=vs4,
            sm_k=self.ones,
            sm_v=self.ones,
            kv_bits=4,
            ring_k=self.k8_all,
            ring_v=self.v8_all,
            ring_ks=self.ks8_all,
            ring_vs=self.vs8_all,
            owner=owner_all_hot,
            ring_mask=SLOTS - 1,
        )
        hot_k, hot_v = gather(k4, v4, kw_hot)
        kw_i8 = dict(
            k_scale=self.ks8_all,
            v_scale=self.vs8_all,
            sm_k=self.ones,
            sm_v=self.ones,
            kv_bits=8,
        )
        i8_k, i8_v = gather(self.k8_all, self.v8_all, kw_i8)
        self.assertTrue(torch.equal(hot_k, i8_k) and torch.equal(hot_v, i8_v))
        self.assertTrue(torch.equal(hot_k, self.d8k[rows_b]))
        self.assertTrue(torch.equal(hot_v, self.d8v[rows_b]))


@unittest.skipUnless(torch.cuda.is_available(), "KV pools require CUDA")
class TestTieredKVPool(CustomTestCase):
    size, page = 1024, 64

    def test_class_keys_and_hybrid_pool_hooks(self):
        self.assertEqual(MHATokenToKVPoolTiered.kv_bits, 4)
        self.assertIs(MHATokenToKVPoolTiered.kv_tiered, True)
        self.assertFalse(getattr(MHATokenToKVPoolInt4, "kv_tiered", False))
        self.assertTrue(
            callable(getattr(HybridLinearKVPool, "get_kv_ring_buffer", None))
        )
        self.assertTrue(
            callable(getattr(HybridLinearKVPool, "get_kv_ring_owner", None))
        )

    def test_ring_slots_from_env(self):
        for bad_w in ("0", "-8", "100", "abc"):
            with patch.dict(os.environ, {"SGLANG_KV_TIERS_W": bad_w}):
                with self.assertRaises(ValueError):
                    ring_slots_from_env()
        env = dict(os.environ)
        env.pop("SGLANG_KV_TIERS_W", None)
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(ring_slots_from_env(), 8192)
        with patch.dict(os.environ, {"SGLANG_KV_TIERS_W": str(R)}):
            self.assertEqual(ring_slots_from_env(), R)

    def _pool_env(self, lazy):
        env = dict(os.environ)
        env.pop("SGLANG_KV_LAZY", None)
        env["SGLANG_KV_TIERS_W"] = str(R)
        if lazy:
            env["SGLANG_KV_LAZY"] = "1"
            env["SGLANG_KV_LAZY_FLOOR"] = "512"
        return env

    def _make_pools(self, lazy):
        mk = dict(
            size=self.size,
            page_size=self.page,
            dtype=torch.uint8,
            head_num=HEADS,
            head_dim=DIM,
            layer_num=2,
            device="cuda",
            enable_memory_saver=False,
            start_layer=0,
        )
        # the reference int4 pool is always eager: its descs and byte counts do not
        # depend on the backing mode, and one VMM arena per process keeps the
        # teardown simple
        with patch.dict(os.environ, self._pool_env(lazy=False), clear=True):
            ref_pool = MHATokenToKVPoolInt4(**mk)
            ref_descs = [
                (d.name, d.shape, d.row_bytes, d.tokens_per_row)
                for d in ref_pool._kv_buffer_descs
            ]
            ref_kb, ref_vb = ref_pool.get_kv_size_bytes()
            ref_pool._clear_buffers()
        with patch.dict(os.environ, self._pool_env(lazy), clear=True):
            pool = MHATokenToKVPoolTiered(**mk)
        return pool, ref_descs, ref_kb, ref_vb

    def _check_pool(self, pool, ref_descs, ref_kb, ref_vb, lazy):
        dev = "cuda"
        ones = torch.ones(HEADS, DIM, dtype=torch.float16, device=dev)
        rows = self.size + self.page
        self.assertTrue(pool.kv_tiered)
        self.assertEqual(pool.kv_bits, 4)
        self.assertEqual(pool.ring_slots, R)
        self.assertEqual(pool.ring_mask, MASK)
        self.assertEqual(pool.dtype, torch.uint8)
        self.assertEqual(pool.store_dtype, torch.uint8)
        self.assertEqual(pool.head_dim, DIM)
        for ring in (pool.ring_k, pool.ring_v, pool.ring_ks, pool.ring_vs):
            self.assertEqual(len(ring), 2)
        self.assertEqual(pool.ring_k[0].shape, (R, HEADS, DIM))
        self.assertEqual(pool.ring_k[0].dtype, torch.int8)
        self.assertEqual(pool.ring_v[1].shape, (R, HEADS, DIM))
        self.assertEqual(pool.ring_v[1].dtype, torch.int8)
        self.assertEqual(pool.ring_ks[0].shape, (R, HEADS, NG8))
        self.assertEqual(pool.ring_ks[0].dtype, torch.float16)
        self.assertEqual(pool.ring_vs[1].shape, (R, HEADS, NG8))
        self.assertEqual(pool.ring_vs[1].dtype, torch.float16)
        self.assertEqual(pool.ring_owner.shape, (R,))
        self.assertEqual(pool.ring_owner.dtype, torch.int32)
        self.assertTrue((pool.ring_owner == -1).all())
        self.assertEqual(
            pool.get_kv_ring_owner().data_ptr(), pool.ring_owner.data_ptr()
        )
        rb = pool.get_kv_ring_buffer(1)
        self.assertEqual(
            [t.data_ptr() for t in rb],
            [
                pool.ring_k[1].data_ptr(),
                pool.ring_v[1].data_ptr(),
                pool.ring_ks[1].data_ptr(),
                pool.ring_vs[1].data_ptr(),
            ],
        )
        self.assertEqual(pool.k_buffer[0].shape, (rows, HEADS, DH))
        self.assertEqual(pool.k_scale_buffer[0].shape, (rows, HEADS, NG4))
        descs = [
            (d.name, d.shape, d.row_bytes, d.tokens_per_row)
            for d in pool._kv_buffer_descs
        ]
        self.assertEqual(
            descs, ref_descs, "descs must equal the int4 pool (ring not a desc)"
        )
        kb, vb = pool.get_kv_size_bytes()
        ring_k_bytes = 2 * R * HEADS * (DIM + NG8 * 2)
        self.assertEqual(kb, ref_kb + ring_k_bytes + R * 4, "ring (+ owner) in K bytes")
        self.assertEqual(vb, ref_vb + ring_k_bytes, "ring in V bytes")
        if lazy:
            o = pool._post_capture_owner
            self.assertIsNotNone(o)
            self.assertEqual(len(o.tensors), 8)
            self.assertEqual(o.bytes_per_token(), 2 * (256 + 256 + 32 + 32))
            lo = min(t.data_ptr() for t in o.tensors)
            hi = max(t.data_ptr() + t.numel() * t.element_size() for t in o.tensors)
            for t in (
                pool.ring_k
                + pool.ring_v
                + pool.ring_ks
                + pool.ring_vs
                + [pool.ring_owner]
            ):
                self.assertFalse(
                    lo <= t.data_ptr() < hi, "ring must live outside the owner"
                )
            self.assertTrue((pool.k_scale_buffer[1][: self.page] == 0).all())
        else:
            self.assertTrue((pool.k_scale_buffer[0] == 0).all())
            self.assertTrue((pool.k_buffer[0] == 0).all())
        # alias-free loc mod R inside the backed floor (the pool only enforces N <= R)
        loc = (torch.randperm(400, device=dev)[:50] + 64).sort().values
        seen, keep = set(), []
        for s in loc.tolist():
            if (s & MASK) not in seen:
                seen.add(s & MASK)
                keep.append(s)
        loc = torch.tensor(keep, dtype=torch.int64, device=dev)
        n = loc.numel()
        self.assertGreaterEqual(n, 30)
        xk_p = torch.randn(n, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        xv_p = torch.randn(n, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        pool.set_kv_buffer(None, loc, xk_p, xv_p, 1.0, 1.0, layer_id_override=1)
        pool.set_kv_buffer(
            None, loc, xk_p.reshape(n, -1), xv_p.reshape(n, -1), layer_id_override=0
        )
        torch.cuda.synchronize()
        pk_p, sk_p = ref_quant4(xk_p)
        pv_p, sv_p = ref_quant4(xv_p)
        q8_p, s8_p = ref_quant8(xk_p)
        q8v_p, s8v_p = ref_quant8(xv_p)
        for layer in (0, 1):
            ksb, vsb = pool.get_kv_scale_buffer(layer)
            self.assertTrue(torch.equal(pool.k_buffer[layer][loc], pk_p))
            self.assertTrue(torch.equal(ksb[loc], sk_p))
            self.assertTrue(torch.equal(pool.v_buffer[layer][loc], pv_p))
            self.assertTrue(torch.equal(vsb[loc], sv_p))
            rk_p, rv_p, rks_p, rvs_p = pool.get_kv_ring_buffer(layer)
            self.assertTrue(torch.equal(rk_p[loc & MASK], q8_p))
            self.assertTrue(torch.equal(rks_p[loc & MASK], s8_p))
            self.assertTrue(torch.equal(rv_p[loc & MASK], q8v_p))
            self.assertTrue(torch.equal(rvs_p[loc & MASK], s8v_p))
        self.assertTrue(torch.equal(pool.ring_owner.long()[loc & MASK], loc))
        self.assertEqual(int((pool.ring_owner == -1).sum()), R - n)
        zeros = torch.zeros(R + 1, HEADS, DIM, device=dev, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):  # N > R
            pool.set_kv_buffer(
                None,
                torch.arange(R + 1, device=dev, dtype=torch.int64),
                zeros,
                zeros,
                layer_id_override=0,
            )
        for bad_kw in (
            dict(k_scale=0.5),
            dict(v_scale=torch.ones(1, device=dev)),
            dict(dcp_kv_mask=torch.ones(n, device=dev)),
        ):
            with self.assertRaises((ValueError, NotImplementedError)):
                pool.set_kv_buffer(None, loc, xk_p, xv_p, layer_id_override=0, **bad_kw)
        pool.lazy_release()
        self.assertTrue((pool.ring_owner == -1).all(), "lazy_release resets the owner")
        smk, _ = pool.get_kv_smooth_buffer(1)
        self.assertEqual(smk.shape, (HEADS, DIM))
        self.assertTrue(bool((smk == 1).all()))
        self.assertTrue(torch.equal(smk, ones))

    def _run_pool_case(self, lazy):
        pool, ref_descs, ref_kb, ref_vb = self._make_pools(lazy)
        cleared = False
        try:
            self._check_pool(pool, ref_descs, ref_kb, ref_vb, lazy)
            pool._clear_buffers()
            cleared = True
            self.assertIsNone(pool.ring_k)
            self.assertIsNone(pool.ring_owner)
        finally:
            if not cleared:
                pool._clear_buffers()

    def test_pool_eager(self):
        self._run_pool_case(lazy=False)

    def test_pool_lazy_vmm(self):
        self._run_pool_case(lazy=True)


if __name__ == "__main__":
    unittest.main()
