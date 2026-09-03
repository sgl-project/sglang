"""Unit tests for the int8-g64 QSA KV path: the ``qsa/sparse_attn.py`` kernels
and ``MHATokenToKVPoolInt8`` (~50 MB VRAM, no server, no model).

Torch reference: per-token-per-head-per-64-group absmax/127, fp16 scale,
q = rint(x / s) clamped to [-127, 127], dequant q * s in fp32 -> bf16.
  1. ``quant_store_kv_int8``: quantize + scatter into random slots (int32 and
     int64 loc); payload and scales bit-exact vs the reference; untouched slots
     keep their sentinel; zero rows get s = 1;
  2. scale index arithmetic for GROUP=64 with 2 heads: flat index
     (slot*2 + h)*4 + g;
  3. compact gather (decode / verify path): 3 requests (300 / 1200 / 4000
     tokens, permuted req_to_token), bit-exact dequant vs torch, rows beyond the
     packed range untouched, an invalid position inside the packed region is
     neither read nor written (same store mask as ``_compact_kv`` /
     ``_compact_kv_fp8``: on the trtllm strided tables cu_k spans the page
     stride, so zero-filling would write every unused column);
  4. prefix row gather (chunk-prefill path): 3 requests with different lengths
     (partial last row block, non-identity req_indices) packed with a 32-row
     gap after EACH request, bit-exact, every gap row and the tail untouched;
  5. end-to-end relative RMS error of gather-dequant(quant(x)) for x ~ N(0, 3)
     below 1.2 %;
  6. ``MHATokenToKVPoolInt8``: eager path and lazy-VMM path (SGLANG_KV_LAZY=1),
     ``set_kv_buffer`` through the pool, scale descs / bytes per token,
     first-page scale rows zeroed, guards.
"""

import os
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.qsa.sparse_attn import KV_INT8_GROUP as G
from sglang.srt.layers.attention.qsa.sparse_attn import (
    quant_store_kv_int8,
    qwen_sparse_fa2_cu_seqlens_triton,
    qwen_sparse_kv_extraction_compact_triton,
    qwen_sparse_prefix_gather_dequant_int8,
)
from sglang.srt.mem_cache.int8_kv_pool import MHATokenToKVPoolInt8
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")

SLOTS, HEADS, DIM, BATCH, TOPK = 4096, 2, 256, 3, 64
NG = DIM // G


def ref_quant(x):
    """x [N, H, D] -> (q int8 [N, H, D], s fp16 [N, H, NG])."""
    xf = x.float().reshape(*x.shape[:-1], NG, G)
    a = xf.abs().amax(-1)
    s = torch.where(a > 0, a / 127.0, torch.ones_like(a)).half()
    q = (
        torch.clamp(torch.round(xf / s.float()[..., None]), -127, 127)
        .to(torch.int8)
        .reshape(x.shape)
    )
    return q, s


def ref_dequant(q, s):
    return (q.float() * s.float().repeat_interleave(G, dim=-1)).to(torch.bfloat16)


def relerr(a, b):
    return float(
        ((a.float() - b.float()).pow(2).mean() / b.float().pow(2).mean()).sqrt()
    )


@unittest.skipUnless(torch.cuda.is_available(), "Triton KV kernels require CUDA")
class TestInt8KVKernels(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        dev = cls.dev = "cuda"
        cls.ones = torch.ones(HEADS, DIM, dtype=torch.float16, device=dev)
        # scatter inputs
        cls.N = 1000
        cls.xk = torch.randn(cls.N, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        cls.xv = torch.randn(cls.N, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        cls.xk[5] = 0  # zero row -> s = 1, q = 0
        cls.xv[7, 1, 64:128] = 0  # one zero group in one head
        cls.qk_ref, cls.sk_ref = ref_quant(cls.xk)
        cls.qv_ref, cls.sv_ref = ref_quant(cls.xv)
        # full pool for the gather tests
        cls.k16 = torch.randn(SLOTS, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        cls.v16 = torch.randn(SLOTS, HEADS, DIM, device=dev, dtype=torch.bfloat16) * 3
        cls.k8 = torch.empty(SLOTS, HEADS, DIM, dtype=torch.int8, device=dev)
        cls.v8 = torch.empty_like(cls.k8)
        cls.ks8 = torch.empty(SLOTS, HEADS, NG, dtype=torch.float16, device=dev)
        cls.vs8 = torch.empty_like(cls.ks8)
        quant_store_kv_int8(
            cls.k16,
            cls.v16,
            torch.arange(SLOTS, device=dev, dtype=torch.int32),
            cls.k8,
            cls.v8,
            cls.ks8,
            cls.vs8,
            cls.ones,
            cls.ones,
        )
        torch.cuda.synchronize()
        qk, sk = ref_quant(cls.k16)
        qv, sv = ref_quant(cls.v16)
        cls.full_ref = (qk, sk, qv, sv)
        cls.dk_all = ref_dequant(qk, sk)  # reference dequant of every slot
        cls.dv_all = ref_dequant(qv, sv)
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
        k_buf = torch.full((SLOTS, HEADS, DIM), 7, dtype=torch.int8, device=dev)
        v_buf = torch.full((SLOTS, HEADS, DIM), -7, dtype=torch.int8, device=dev)
        ks = torch.full((SLOTS, HEADS, NG), 3.0, dtype=torch.float16, device=dev)
        vs = torch.full((SLOTS, HEADS, NG), 5.0, dtype=torch.float16, device=dev)
        quant_store_kv_int8(
            self.xk, self.xv, loc, k_buf, v_buf, ks, vs, self.ones, self.ones
        )
        torch.cuda.synchronize()
        return loc, k_buf, v_buf, ks, vs

    def _run_compact(self, idx):
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
            self.k8,
            self.v8,
            self.req_to_token,
            self.req_indices,
            idx,
            self.seq_lens,
            cu_k,
            out_k,
            out_v,
            BATCH,
            TOPK,
            k_scale=self.ks8,
            v_scale=self.vs8,
            sm_k=self.ones,
            sm_v=self.ones,
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
        qwen_sparse_prefix_gather_dequant_int8(
            self.k8,
            self.v8,
            self.ks8,
            self.vs8,
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
    def test_quant_store_scatter_bit_exact(self):
        for loc_dtype in (torch.int32, torch.int64):
            with self.subTest(loc_dtype=loc_dtype):
                loc, k_buf, v_buf, ks, vs = self._scatter(loc_dtype)
                ll = loc.long()
                self.assertTrue(torch.equal(k_buf[ll], self.qk_ref), "K payload")
                self.assertTrue(torch.equal(v_buf[ll], self.qv_ref), "V payload")
                self.assertTrue(torch.equal(ks[ll], self.sk_ref), "K scales")
                self.assertTrue(torch.equal(vs[ll], self.sv_ref), "V scales")
                untouched = torch.ones(SLOTS, dtype=torch.bool, device=self.dev)
                untouched[ll] = False
                self.assertTrue((k_buf[untouched] == 7).all())
                self.assertTrue((v_buf[untouched] == -7).all())
                self.assertTrue((ks[untouched] == 3.0).all())
                self.assertTrue((vs[untouched] == 5.0).all())
                # zero row must give s = 1, q = 0; zero group s = 1
                self.assertTrue((ks[ll[5]] == 1.0).all())
                self.assertTrue((k_buf[ll[5]] == 0).all())
                self.assertEqual(float(vs[ll[7], 1, 1]), 1.0)
                self.assertEqual(int(self.qk_ref.abs().max()), 127)
                self.assertEqual(int(k_buf[ll].abs().max()), 127)

    def test_scale_index_arithmetic(self):
        loc, _, _, ks, _ = self._scatter(torch.int64)
        flat = ks.view(-1)
        for _ in range(64):
            t = int(torch.randint(self.N, (1,)))
            h = int(torch.randint(HEADS, (1,)))
            g = int(torch.randint(NG, (1,)))
            slot = int(loc[t])
            self.assertEqual(
                float(flat[(slot * HEADS + h) * NG + g]), float(self.sk_ref[t, h, g])
            )
        # heads must carry distinct scales for the probe to be meaningful
        self.assertFalse(torch.equal(self.sk_ref[:, 0], self.sk_ref[:, 1]))

    def test_full_pool_quant_bit_exact(self):
        qk, sk, qv, sv = self.full_ref
        self.assertTrue(torch.equal(self.k8, qk))
        self.assertTrue(torch.equal(self.ks8, sk))
        self.assertTrue(torch.equal(self.v8, qv))
        self.assertTrue(torch.equal(self.vs8, sv))

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
        self.assertLess(relerr(pk[written], gathered_k), 0.012)
        self.assertLess(relerr(pv[written], gathered_v), 0.012)


@unittest.skipUnless(torch.cuda.is_available(), "KV pools require CUDA")
class TestInt8KVPool(CustomTestCase):
    size, page = 1024, 64

    def _make_pool(self, lazy):
        env = dict(os.environ)
        env.pop("SGLANG_KV_LAZY", None)
        if lazy:
            env["SGLANG_KV_LAZY"] = "1"
            env["SGLANG_KV_LAZY_FLOOR"] = "512"
        with patch.dict(os.environ, env, clear=True):
            return MHATokenToKVPoolInt8(
                size=self.size,
                page_size=self.page,
                dtype=torch.int8,
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
        self.assertEqual(pool.dtype, torch.int8)
        self.assertEqual(pool.store_dtype, torch.int8)
        for buf in (
            pool.k_buffer,
            pool.v_buffer,
            pool.k_scale_buffer,
            pool.v_scale_buffer,
        ):
            self.assertEqual(len(buf), 2)
        self.assertEqual(pool.k_buffer[0].shape, (rows, HEADS, DIM))
        self.assertEqual(pool.k_buffer[0].dtype, torch.int8)
        self.assertEqual(pool.k_scale_buffer[0].shape, (rows, HEADS, NG))
        self.assertEqual(pool.k_scale_buffer[0].dtype, torch.float16)
        self.assertEqual(pool.get_key_buffer(1).dtype, torch.int8)
        self.assertEqual(pool.get_key_buffer(1).data_ptr(), pool.k_buffer[1].data_ptr())
        descs = pool._kv_buffer_descs
        self.assertEqual(
            [d.name for d in descs],
            ["k0", "k1", "v0", "v1", "ks0", "ks1", "vs0", "vs1"],
        )
        self.assertTrue(
            all(d.row_bytes == 16 and d.shape == (rows, 16) for d in descs[4:])
        )
        kb, vb = pool.get_kv_size_bytes()
        self.assertEqual(kb, 2 * rows * (HEADS * DIM + HEADS * NG * 2))
        self.assertEqual(vb, kb)
        if lazy:
            o = pool._post_capture_owner
            self.assertIsNotNone(o)
            self.assertEqual(len(o.tensors), 8)
            self.assertEqual(o.bytes_per_token(), 2 * (512 + 512 + 16 + 16))
            self.assertEqual(pool.k_scale_buffer[0].data_ptr(), o.tensors[4].data_ptr())
            self.assertTrue((pool.k_scale_buffer[1][: self.page] == 0).all())
            self.assertTrue((pool.v_scale_buffer[0][: self.page] == 0).all())
        else:
            self.assertTrue((pool.k_scale_buffer[0] == 0).all())
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
        qk_r, sk_r = ref_quant(xk)
        qv_r, sv_r = ref_quant(xv)
        for layer in (0, 1):
            ksb, vsb = pool.get_kv_scale_buffer(layer)
            self.assertTrue(torch.equal(pool.k_buffer[layer][loc], qk_r))
            self.assertTrue(torch.equal(ksb[loc], sk_r))
            self.assertTrue(torch.equal(pool.v_buffer[layer][loc], qv_r))
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

    def test_kv_bits_dispatch_key(self):
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
