"""Unit tests for the int4-g32 QSA KV path: the ``qsa/sparse_attn.py`` kernels
and ``MHATokenToKVPoolInt4`` (~30 MB VRAM, no server, no model).

Torch reference: per-token-per-head-per-32-group absmax/7, fp16 scale clamped
to fp16 max, q = rint(x / s) clamped to [-7, 7], nibble packing low = even
channel / high = odd channel with offset +8, dequant (nibble - 8) * s in fp32
-> bf16.
  0. pack / unpack round trip: every q in [-7, 7] survives pack -> unpack;
  1. ``quant_store_kv_int4``: quantize + scatter into random slots (int32 and
     int64 loc); payload and scales bit-exact vs the reference; a row that hits
     every value -7..7 (s = 1); untouched slots keep their sentinel; zero rows
     and a zero group get s = 1 and nibble 8;
  2. scale index arithmetic for GROUP=32 with 2 heads: flat index
     (slot*2 + h)*8 + g; nibble order;
  3. compact gather (decode / verify path): 3 requests (300 / 1200 / 4000
     tokens, permuted req_to_token), bit-exact dequant vs torch, rows beyond the
     packed range untouched, an invalid position inside the packed region is
     neither read nor written; the trtllm strided layout (cu_k = arange * stride
     with stride = 64 > topk = 40, topk not a multiple of BLOCK_TOPK, -1 padded
     indices, a position >= seq_len): valid rows bit-exact, every padding /
     invalid row of the strided scratch untouched; a bf16 scratch of the packed
     width is rejected;
  4. prefix row gather (chunk-prefill path): 3 requests with different lengths
     (partial last row block, non-identity req_indices) packed with a 32-row
     gap after EACH request, bit-exact, every gap row and the tail untouched;
  5. end-to-end relative RMS error of gather-dequant(quant(x)) for x ~ N(0, 3):
     8-11.5 % (g32);
  6. fp16-max scale clamp: groups with absmax > 7 * 65504 (1e6, bf16 max) store
     s = 65504 (never inf), bit-exact vs the reference, and both gather kernels
     dequantize them to finite +/- 7 * 65504;
  7. ``MHATokenToKVPoolInt4``: eager and lazy-VMM (SGLANG_KV_LAZY=1) paths,
     ``set_kv_buffer`` through the pool, payload / scale descs (row_bytes
     256 / 32), bytes per token 1152 for 2 layers, first-page scale rows zeroed,
     kv_bits = 4 (and 8 on the int8 pool).
"""

import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.qsa.sparse_attn import KV_INT4_GROUP as G
from sglang.srt.layers.attention.qsa.sparse_attn import (
    quant_store_kv_int4,
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_prefix_gather_dequant_int4,
)
from sglang.srt.mem_cache.int4_kv_pool import MHATokenToKVPoolInt4
from sglang.srt.mem_cache.int8_kv_pool import MHATokenToKVPoolInt8
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

SLOTS, HEADS, DIM, BATCH, TOPK = 4096, 2, 256, 3, 64
NG, DH = DIM // G, DIM // 2


def ref_pack(q):
    """q int [.., D] in [-7, 7] -> uint8 [.., D // 2]: low nibble = even channel,
    high = odd, offset +8."""
    return ((q[..., 0::2] + 8) | ((q[..., 1::2] + 8) << 4)).to(torch.uint8)


def ref_unpack(p):
    """uint8 [.., D // 2] -> int32 [.., D]."""
    b = p.to(torch.int32)
    return torch.stack([(b & 15) - 8, (b >> 4) - 8], dim=-1).reshape(*p.shape[:-1], -1)


def ref_quant(x):
    """x [N, H, D] -> (packed uint8 [N, H, D // 2], s fp16 [N, H, NG], q int32 [N, H, D])."""
    xf = x.float().reshape(*x.shape[:-1], NG, G)
    a = xf.abs().amax(-1)
    # never inf (the kernel clamps too)
    s = torch.where(a > 0, a / 7.0, torch.ones_like(a)).clamp(max=65504.0).half()
    q = (
        torch.clamp(torch.round(xf / s.float()[..., None]), -7, 7)
        .to(torch.int32)
        .reshape(x.shape)
    )
    return ref_pack(q), s, q


def ref_dequant(p, s):
    return (ref_unpack(p).float() * s.float().repeat_interleave(G, dim=-1)).to(
        torch.bfloat16
    )


def relerr(a, b):
    return float(
        ((a.float() - b.float()).pow(2).mean() / b.float().pow(2).mean()).sqrt()
    )


@unittest.skipUnless(torch.cuda.is_available(), "Triton KV kernels require CUDA")
class TestInt4KVKernels(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        dev = cls.dev = "cuda"
        cls.ones = torch.ones(HEADS, DIM, dtype=torch.float16, device=dev)
        cls.qa = torch.arange(-7, 8, device=dev, dtype=torch.int32)  # 15 values
        # scatter inputs
        cls.N = 1000
        cls.xk = torch.randn(cls.N, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        cls.xv = torch.randn(cls.N, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        cls.xk[5] = 0  # zero row -> s = 1, nibble 8
        cls.xv[7, 1, 64:96] = 0  # one zero group (g = 2) in one head
        full = torch.tensor(
            [-7, 7, -6, 6, -5, 5, -4, 4, -3, 3, -2, 2, -1, 1, 0, 7] * 16,
            device=dev,
            dtype=torch.bfloat16,
        )
        cls.xk[3, 0] = full  # absmax 7 -> s = 1 -> q hits every value in [-7, 7]
        cls.xv[3, 1] = -full
        cls.pk_ref, cls.sk_ref, cls.qk_ref = ref_quant(cls.xk)
        cls.pv_ref, cls.sv_ref, cls.qv_ref = ref_quant(cls.xv)
        # full pool for the gather tests
        cls.k16 = torch.randn(SLOTS, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        cls.v16 = torch.randn(SLOTS, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        cls.k4 = torch.empty(SLOTS, HEADS, DH, dtype=torch.uint8, device=dev)
        cls.v4 = torch.empty_like(cls.k4)
        cls.ks4 = torch.empty(SLOTS, HEADS, NG, dtype=torch.float16, device=dev)
        cls.vs4 = torch.empty_like(cls.ks4)
        quant_store_kv_int4(
            cls.k16,
            cls.v16,
            torch.arange(SLOTS, device=dev, dtype=torch.int32),
            cls.k4,
            cls.v4,
            cls.ks4,
            cls.vs4,
            cls.ones,
            cls.ones,
        )
        torch.cuda.synchronize()
        pk, sk, _ = ref_quant(cls.k16)
        pv, sv, _ = ref_quant(cls.v16)
        cls.full_ref = (pk, sk, pv, sv)
        cls.dk_all = ref_dequant(pk, sk)  # reference dequant of every slot
        cls.dv_all = ref_dequant(pv, sv)
        cls.seq_lens = torch.tensor([300, 1200, 4000], dtype=torch.int32, device=dev)
        cls.req_to_token = torch.stack(
            [torch.randperm(SLOTS, device=dev) for _ in range(BATCH)]
        ).to(torch.int32)
        cls.req_indices = torch.arange(BATCH, dtype=torch.int32, device=dev)
        cls.indices = torch.stack(
            [
                torch.randperm(int(n), device=dev)[:TOPK].sort().values
                for n in cls.seq_lens
            ]
        ).to(torch.int32)

    # ------------------------------------------------------------ helpers
    def _scatter(self, loc_dtype):
        dev = self.dev
        loc = torch.randperm(SLOTS, device=dev)[: self.N].to(loc_dtype)
        k_buf = torch.full((SLOTS, HEADS, DH), 0x77, dtype=torch.uint8, device=dev)
        v_buf = torch.full((SLOTS, HEADS, DH), 0x99, dtype=torch.uint8, device=dev)
        ks = torch.full((SLOTS, HEADS, NG), 3.0, dtype=torch.float16, device=dev)
        vs = torch.full((SLOTS, HEADS, NG), 5.0, dtype=torch.float16, device=dev)
        quant_store_kv_int4(
            self.xk, self.xv, loc, k_buf, v_buf, ks, vs, self.ones, self.ones
        )
        torch.cuda.synchronize()
        return loc, k_buf, v_buf, ks, vs

    def _run_compact(self, idx, k4=None, v4=None, ks4=None, vs4=None):
        dev = self.dev
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
            self.k4 if k4 is None else k4,
            self.v4 if v4 is None else v4,
            self.req_to_token,
            self.req_indices,
            idx,
            self.seq_lens,
            cu_k,
            out_k,
            out_v,
            BATCH,
            TOPK,
            k_scale=self.ks4 if ks4 is None else ks4,
            v_scale=self.vs4 if vs4 is None else vs4,
            sm_k=self.ones,
            sm_v=self.ones,
            kv_bits=4,
        )
        torch.cuda.synchronize()
        return cu_k, n, out_k, out_v

    def _prefix_gather(self):
        dev = self.dev
        lens = torch.tensor([300, 1203, 4001], dtype=torch.int32, device=dev)
        req_idx = torch.tensor([2, 0, 1], dtype=torch.int64, device=dev)
        gap = 32  # > BLOCK_T (16): a stray row block lands in the gap
        cu = F.pad((lens + gap).cumsum(0), (1, 0)).to(torch.int32).contiguous()
        total = int(cu[-1])
        pk = torch.full(
            (total + 100, HEADS, DIM), float("nan"), device=dev, dtype=torch.bfloat16
        )
        pv = torch.full_like(pk, float("nan"))
        qwen_sparse_prefix_gather_dequant_int4(
            self.k4,
            self.v4,
            self.ks4,
            self.vs4,
            self.ones,
            self.ones,
            self.req_to_token,
            req_idx,
            lens,
            cu,
            pk,
            pv,
            BATCH,
            int(lens.max()),
        )
        torch.cuda.synchronize()
        return lens, req_idx, gap, cu, pk, pv

    # ------------------------------------------------------------ tests
    def test_pack_unpack_round_trip(self):
        dev = self.dev
        qq = torch.cartesian_prod(self.qa, self.qa).reshape(
            -1
        )  # every (even, odd) pair
        pad = torch.zeros(DIM - qq.numel() % DIM, device=dev, dtype=torch.int32)
        qq = torch.cat([qq, pad]).reshape(-1, DIM)
        pp = ref_pack(qq)
        self.assertEqual(pp.dtype, torch.uint8)
        self.assertGreaterEqual(int(pp.min()), 0x11)
        self.assertLessEqual(int(pp.max()), 0xFF)
        self.assertTrue(torch.equal(ref_unpack(pp), qq))
        self.assertEqual(
            int(ref_pack(torch.tensor([[-7, 7]], device=dev, dtype=torch.int32))[0, 0]),
            0x01 | (0x0F << 4),
        )
        self.assertEqual(
            int(ref_pack(torch.tensor([[0, 0]], device=dev, dtype=torch.int32))[0, 0]),
            0x88,
        )

    def test_quant_store_scatter_bit_exact(self):
        self.assertTrue(torch.equal(self.qk_ref[3, 0].unique(), self.qa))
        self.assertTrue(torch.equal(self.qv_ref[3, 1].unique(), self.qa))
        for loc_dtype in (torch.int32, torch.int64):
            with self.subTest(loc_dtype=loc_dtype):
                loc, k_buf, v_buf, ks, vs = self._scatter(loc_dtype)
                ll = loc.long()
                self.assertTrue(torch.equal(k_buf[ll], self.pk_ref), "K payload")
                self.assertTrue(torch.equal(v_buf[ll], self.pv_ref), "V payload")
                self.assertTrue(torch.equal(ks[ll], self.sk_ref), "K scales")
                self.assertTrue(torch.equal(vs[ll], self.sv_ref), "V scales")
                untouched = torch.ones(SLOTS, dtype=torch.bool, device=self.dev)
                untouched[ll] = False
                self.assertTrue((k_buf[untouched] == 0x77).all())
                self.assertTrue((v_buf[untouched] == 0x99).all())
                self.assertTrue((ks[untouched] == 3.0).all())
                self.assertTrue((vs[untouched] == 5.0).all())
                # zero row -> s = 1, nibbles 8; zero group -> s = 1, nibbles 8
                self.assertTrue((ks[ll[5]] == 1.0).all())
                self.assertTrue((k_buf[ll[5]] == 0x88).all())
                self.assertEqual(float(vs[ll[7], 1, 2]), 1.0)
                self.assertTrue((v_buf[ll[7], 1, 32:48] == 0x88).all())
                # full-range rows
                self.assertTrue((ks[ll[3], 0] == 1.0).all())
                self.assertTrue(
                    torch.equal(ref_unpack(k_buf[ll[3], 0]).unique(), self.qa)
                )
                self.assertTrue((vs[ll[3], 1] == 1.0).all())
                self.assertTrue(
                    torch.equal(ref_unpack(v_buf[ll[3], 1]).unique(), self.qa)
                )
                self.assertEqual(int(ref_unpack(k_buf[ll]).abs().max()), 7)

    def test_scale_index_and_nibble_order(self):
        loc, k_buf, _, ks, _ = self._scatter(torch.int64)
        flat = ks.view(-1)
        for _ in range(64):
            t = int(torch.randint(self.N, (1,)))
            h = int(torch.randint(HEADS, (1,)))
            g = int(torch.randint(NG, (1,)))
            slot = int(loc[t])
            self.assertEqual(
                float(flat[(slot * HEADS + h) * NG + g]), float(self.sk_ref[t, h, g])
            )
            c = int(torch.randint(DH, (1,)))
            b = int(k_buf[slot, h, c])
            self.assertEqual((b & 15) - 8, int(self.qk_ref[t, h, 2 * c]))
            self.assertEqual((b >> 4) - 8, int(self.qk_ref[t, h, 2 * c + 1]))
        self.assertFalse(torch.equal(self.sk_ref[:, 0], self.sk_ref[:, 1]))

    def test_full_pool_quant_bit_exact(self):
        pk, sk, pv, sv = self.full_ref
        self.assertTrue(torch.equal(self.k4, pk))
        self.assertTrue(torch.equal(self.ks4, sk))
        self.assertTrue(torch.equal(self.v4, pv))
        self.assertTrue(torch.equal(self.vs4, sv))

    def test_compact_gather_bit_exact(self):
        cu_k, n, out_k, out_v = self._run_compact(self.indices)
        self.assertEqual(n, BATCH * TOPK)
        for b in range(BATCH):
            rows = self.req_to_token[b, self.indices[b].long()].long()
            a, e = int(cu_k[b]), int(cu_k[b + 1])
            self.assertTrue(torch.equal(out_k[a:e], self.dk_all[rows]), f"req {b} K")
            self.assertTrue(torch.equal(out_v[a:e], self.dv_all[rows]), f"req {b} V")
        self.assertTrue(torch.isnan(out_k[n:]).all(), "rows beyond the packed range")
        self.assertTrue(torch.isnan(out_v[n:]).all(), "rows beyond the packed range")

    def test_compact_gather_invalid_position_untouched(self):
        cu_k, n, out_k, _ = self._run_compact(self.indices)
        bad = self.indices.clone()
        bad[1, 0] = int(self.seq_lens[1]) + 5  # invalid position inside request 1
        cu_b, n_b, ok_b, ov_b = self._run_compact(bad)
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
        # cu_k = arange(batch + 1) * stride with stride = ceil(topk / page) * page > topk,
        # so valid_count (= stride) never masks anything and the `cols < topk` load
        # mask (other = -1) keeps the page-padding columns [topk, stride) untouched.
        dev = self.dev
        page_s, topk_s = 64, 40
        stride_s = -(-topk_s // page_s) * page_s
        self.assertGreater(stride_s, topk_s)
        self.assertNotEqual(topk_s % 16, 0)
        cu_s = torch.arange(BATCH + 1, dtype=torch.int32, device=dev) * stride_s
        nsel = [topk_s, 25, 33]  # valid selections per request; rest -1
        idx_s = torch.full((BATCH, topk_s), -1, dtype=torch.int32, device=dev)
        for b in range(BATCH):
            idx_s[b, : nsel[b]] = (
                torch.randperm(int(self.seq_lens[b]), device=dev)[: nsel[b]]
                .sort()
                .values.to(torch.int32)
            )
        idx_s[2, 5] = int(self.seq_lens[2]) + 3  # >= seq_len inside request 2
        ok_s = torch.full(
            (BATCH * stride_s, HEADS, DIM),
            float("nan"),
            device=dev,
            dtype=torch.bfloat16,
        )
        ov_s = torch.full_like(ok_s, float("nan"))
        qwen_sparse_kv_extraction_compact_triton(
            self.k4,
            self.v4,
            self.req_to_token,
            self.req_indices,
            idx_s,
            self.seq_lens,
            cu_s,
            ok_s,
            ov_s,
            BATCH,
            topk_s,
            k_scale=self.ks4,
            v_scale=self.vs4,
            sm_k=self.ones,
            sm_v=self.ones,
            kv_bits=4,
        )
        torch.cuda.synchronize()
        written = torch.zeros(BATCH * stride_s, dtype=torch.bool, device=dev)
        for b in range(BATCH):
            for c in range(topk_s):
                pos = int(idx_s[b, c])
                if 0 <= pos < int(self.seq_lens[b]):
                    r = b * stride_s + c
                    slot = int(self.req_to_token[b, pos])
                    self.assertTrue(torch.equal(ok_s[r], self.dk_all[slot]), (b, c))
                    self.assertTrue(torch.equal(ov_s[r], self.dv_all[slot]), (b, c))
                    written[r] = True
        self.assertEqual(int(written.sum()), sum(nsel) - 1)
        self.assertTrue(torch.isnan(ok_s[~written]).all())
        self.assertTrue(torch.isnan(ov_s[~written]).all())
        self.assertFalse(written.view(BATCH, stride_s)[:, topk_s:].any())

    def test_packed_width_scratch_rejected(self):
        # the backend derives the logical head_dim from the pool
        cu_k, n, _, _ = self._run_compact(self.indices)
        narrow = torch.empty(n, HEADS, DH, device=self.dev, dtype=torch.bfloat16)
        with self.assertRaises(AssertionError):
            qwen_sparse_kv_extraction_compact_triton(
                self.k4,
                self.v4,
                self.req_to_token,
                self.req_indices,
                self.indices,
                self.seq_lens,
                cu_k,
                narrow,
                narrow.clone(),
                BATCH,
                TOPK,
                k_scale=self.ks4,
                v_scale=self.vs4,
                sm_k=self.ones,
                sm_v=self.ones,
                kv_bits=4,
            )

    def test_prefix_gather_bit_exact_gaps_untouched(self):
        lens, req_idx, gap, cu, pk, pv = self._prefix_gather()
        written = torch.zeros(pk.shape[0], dtype=torch.bool, device=self.dev)
        for b in range(BATCH):
            rows = self.req_to_token[int(req_idx[b]), : int(lens[b])].long()
            a, e = int(cu[b]), int(cu[b]) + int(lens[b])
            self.assertTrue(torch.equal(pk[a:e], self.dk_all[rows]), f"req {b} K")
            self.assertTrue(torch.equal(pv[a:e], self.dv_all[rows]), f"req {b} V")
            written[a:e] = True
            self.assertTrue(torch.isnan(pk[e : e + gap]).all(), f"gap after req {b}")
            self.assertTrue(torch.isnan(pv[e : e + gap]).all(), f"gap after req {b}")
        self.assertTrue(
            torch.isnan(pk[~written]).all() and torch.isnan(pv[~written]).all()
        )
        self.assertEqual(int(written.sum()), int(lens.sum()))

    def test_end_to_end_relative_error(self):
        lens, req_idx, _, cu, pk, pv = self._prefix_gather()
        written = torch.zeros(pk.shape[0], dtype=torch.bool, device=self.dev)
        for b in range(BATCH):
            written[int(cu[b]) : int(cu[b]) + int(lens[b])] = True
        gathered_k = torch.cat(
            [
                self.k16[self.req_to_token[int(req_idx[b]), : int(lens[b])].long()]
                for b in range(BATCH)
            ]
        )
        gathered_v = torch.cat(
            [
                self.v16[self.req_to_token[int(req_idx[b]), : int(lens[b])].long()]
                for b in range(BATCH)
            ]
        )
        ek, ev = relerr(pk[written], gathered_k), relerr(pv[written], gathered_v)
        # g32 on N(0, 3): ~9-10 % (int8_g64 ~0.9 %, e4m3 ~2.7 %)
        self.assertTrue(0.08 < ek < 0.115, ek)
        self.assertTrue(0.08 < ev < 0.115, ev)

    def test_fp16_max_scale_clamp(self):
        dev = self.dev
        k4, v4, ks4, vs4 = (t.clone() for t in (self.k4, self.v4, self.ks4, self.vs4))
        big = torch.zeros(1, HEADS, DIM, device=dev, dtype=torch.bfloat16)
        big[0, 0, :32] = torch.tensor(
            [1e6, -1e6, 5e5, -5e5, 4.6e5, -4.6e5, 1e5, -1e5] * 4, device=dev
        ).to(torch.bfloat16)
        big[0, 1, 64:96] = torch.finfo(torch.bfloat16).max  # every channel of group 2
        bigv = -big
        pos7 = 17
        slot7 = int(self.req_to_token[0, pos7])
        quant_store_kv_int4(
            big,
            bigv,
            torch.tensor([slot7], dtype=torch.int32, device=dev),
            k4,
            v4,
            ks4,
            vs4,
            self.ones,
            self.ones,
        )
        torch.cuda.synchronize()
        pk_b, sk_b, q_b = ref_quant(big)
        pv_b, sv_b, _ = ref_quant(bigv)
        self.assertTrue(torch.isfinite(sk_b).all())
        self.assertEqual(float(sk_b[0, 0, 0]), 65504.0)
        self.assertEqual(float(sk_b[0, 1, 2]), 65504.0)
        self.assertTrue(torch.isfinite(ks4[slot7]).all(), "kernel stored an inf scale")
        self.assertTrue(torch.isfinite(vs4[slot7]).all(), "kernel stored an inf scale")
        self.assertTrue(torch.equal(k4[slot7], pk_b[0]))
        self.assertTrue(torch.equal(ks4[slot7], sk_b[0]))
        self.assertTrue(torch.equal(v4[slot7], pv_b[0]))
        self.assertTrue(torch.equal(vs4[slot7], sv_b[0]))
        sat = torch.tensor(7 * 65504.0, device=dev).to(torch.bfloat16)
        d_b, dv_b = ref_dequant(pk_b, sk_b)[0], ref_dequant(pv_b, sv_b)[0]
        self.assertTrue(torch.isfinite(d_b).all())
        self.assertEqual(float(d_b[0, 0]), float(sat))
        self.assertEqual(float(d_b[0, 1]), -float(sat))
        self.assertTrue((d_b[1, 64:96] == sat).all() and (dv_b[1, 64:96] == -sat).all())
        self.assertEqual(int(q_b[0, 0, :32].abs().max()), 7)
        self.assertLess(int(q_b[0, 0, 6].abs()), 7)  # 1e5 / 65504 -> 2
        # compact gather of the clamped row
        idx7 = self.indices.clone()
        idx7[0, 0] = pos7
        cu7, n7, ok7, ov7 = self._run_compact(idx7, k4, v4, ks4, vs4)
        r7 = int(cu7[0])
        self.assertTrue(
            torch.isfinite(ok7[:n7]).all() and torch.isfinite(ov7[:n7]).all()
        )
        self.assertTrue(torch.equal(ok7[r7], d_b) and torch.equal(ov7[r7], dv_b))
        # prefix gather of the clamped row
        pk7 = torch.full(
            (pos7 + 8, HEADS, DIM), float("nan"), device=dev, dtype=torch.bfloat16
        )
        pv7 = torch.full_like(pk7, float("nan"))
        qwen_sparse_prefix_gather_dequant_int4(
            k4,
            v4,
            ks4,
            vs4,
            self.ones,
            self.ones,
            self.req_to_token,
            torch.tensor([0], dtype=torch.int64, device=dev),
            torch.tensor([pos7 + 8], dtype=torch.int32, device=dev),
            torch.tensor([0, pos7 + 8], dtype=torch.int32, device=dev),
            pk7,
            pv7,
            1,
            pos7 + 8,
        )
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(pk7).all() and torch.isfinite(pv7).all())
        self.assertTrue(torch.equal(pk7[pos7], d_b) and torch.equal(pv7[pos7], dv_b))


@unittest.skipUnless(torch.cuda.is_available(), "KV pools require CUDA")
class TestInt4KVPool(CustomTestCase):
    size, page = 1024, 64

    def _make_pool(self, lazy):
        env = dict(os.environ)
        env.pop("SGLANG_KV_LAZY", None)
        if lazy:
            env["SGLANG_KV_LAZY"] = "1"
            env["SGLANG_KV_LAZY_FLOOR"] = "512"
        with patch.dict(os.environ, env, clear=True):
            return MHATokenToKVPoolInt4(
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

    def _check_pool(self, pool, lazy):
        dev = "cuda"
        rows = self.size + self.page
        self.assertEqual(pool.kv_bits, 4)
        self.assertEqual(pool.dtype, torch.uint8)
        self.assertEqual(pool.store_dtype, torch.uint8)
        self.assertEqual(pool.head_dim, DIM)
        for buf in (
            pool.k_buffer,
            pool.v_buffer,
            pool.k_scale_buffer,
            pool.v_scale_buffer,
        ):
            self.assertEqual(len(buf), 2)
        self.assertEqual(pool.k_buffer[0].shape, (rows, HEADS, DH))
        self.assertEqual(pool.k_buffer[0].dtype, torch.uint8)
        self.assertEqual(pool.v_buffer[1].shape, (rows, HEADS, DH))
        self.assertEqual(pool.v_buffer[1].dtype, torch.uint8)
        self.assertEqual(pool.k_scale_buffer[0].shape, (rows, HEADS, NG))
        self.assertEqual(pool.k_scale_buffer[0].dtype, torch.float16)
        self.assertEqual(pool.get_key_buffer(1).dtype, torch.uint8)
        self.assertEqual(pool.get_key_buffer(1).data_ptr(), pool.k_buffer[1].data_ptr())
        descs = pool._kv_buffer_descs
        self.assertEqual(
            [d.name for d in descs],
            ["k0", "k1", "v0", "v1", "ks0", "ks1", "vs0", "vs1"],
        )
        self.assertTrue(
            all(
                d.row_bytes == HEADS * DH
                and d.shape == (rows, HEADS, DH)
                and d.tokens_per_row == 1
                for d in descs[:4]
            )
        )
        self.assertTrue(
            all(d.row_bytes == 32 and d.shape == (rows, 32) for d in descs[4:])
        )
        kb, vb = pool.get_kv_size_bytes()
        self.assertEqual(kb, 2 * rows * (HEADS * DH + HEADS * NG * 2))
        self.assertEqual(vb, kb)
        if lazy:
            o = pool._post_capture_owner
            self.assertIsNotNone(o)
            self.assertEqual(len(o.tensors), 8)
            self.assertEqual(o.bytes_per_token(), 2 * (256 + 256 + 32 + 32))
            self.assertEqual(pool.k_scale_buffer[0].data_ptr(), o.tensors[4].data_ptr())
            self.assertTrue((pool.k_scale_buffer[1][: self.page] == 0).all())
            self.assertTrue((pool.v_scale_buffer[0][: self.page] == 0).all())
        else:
            self.assertTrue((pool.k_scale_buffer[0] == 0).all())
            self.assertTrue((pool.k_buffer[0] == 0).all())
        n = 200
        loc = torch.randperm(512, device=dev)[:n].to(
            torch.int64
        )  # inside the backed floor
        xk = torch.randn(n, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        xv = torch.randn(n, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        # HybridLinearKVPool style (unit scales) and the [N, H*D] form
        pool.set_kv_buffer(None, loc, xk, xv, 1.0, 1.0, layer_id_override=1)
        pool.set_kv_buffer(
            None, loc, xk.reshape(n, -1), xv.reshape(n, -1), layer_id_override=0
        )
        torch.cuda.synchronize()
        pk_r, sk_r, _ = ref_quant(xk)
        pv_r, sv_r, _ = ref_quant(xv)
        for layer in (0, 1):
            ksb, vsb = pool.get_kv_scale_buffer(layer)
            self.assertTrue(torch.equal(pool.k_buffer[layer][loc], pk_r))
            self.assertTrue(torch.equal(ksb[loc], sk_r))
            self.assertTrue(torch.equal(pool.v_buffer[layer][loc], pv_r))
            self.assertTrue(torch.equal(vsb[loc], sv_r))
        smk, smv = pool.get_kv_smooth_buffer(1)
        self.assertEqual(smk.shape, (HEADS, DIM))
        self.assertTrue(bool((smk == 1).all()) and bool((smv == 1).all()))
        for bad_kw in (
            dict(k_scale=0.5),
            dict(v_scale=torch.ones(1, device=dev)),
            dict(dcp_kv_mask=torch.ones(n, device=dev)),
        ):
            with self.assertRaises((ValueError, NotImplementedError)):
                pool.set_kv_buffer(None, loc, xk, xv, layer_id_override=0, **bad_kw)
        with self.assertRaises(NotImplementedError):
            pool.get_cpu_copy(loc)
        with self.assertRaises(NotImplementedError):
            pool.load_cpu_copy(None, loc)
        with self.assertRaises(NotImplementedError):
            pool.set_kv_buffer_prefix_valid()

    def test_kv_bits_dispatch_keys(self):
        self.assertEqual(MHATokenToKVPoolInt4.kv_bits, 4)
        self.assertEqual(MHATokenToKVPoolInt8.kv_bits, 8)

    def test_pool_eager(self):
        pool = self._make_pool(lazy=False)
        try:
            self._check_pool(pool, lazy=False)
        finally:
            pool._clear_buffers()

    def test_pool_lazy_vmm(self):
        pool = self._make_pool(lazy=True)
        try:
            self._check_pool(pool, lazy=True)
        finally:
            pool._clear_buffers()


if __name__ == "__main__":
    unittest.main()
