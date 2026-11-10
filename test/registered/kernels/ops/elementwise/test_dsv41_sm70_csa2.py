"""SM70 DSV4.1 CSA2 CUDA pack / indexer / sparse decode vs the torch oracle.

Not registered for GPU CI (V100 worktree only).
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.dsv4 import sm70_csa2_reference as R
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(
    est_time=90,
    stage="base-b-kernel-unit",
    runner_config="1-gpu-large",
    disabled="SM70 V100 worktree only; do not register GPU CI",
)

FP16 = torch.float16
LOGIT_RTOL = 2**-10
ATTN_ATOL = 1e-2


def _rows_as_sets(idx):
    return [set(r[r >= 0].tolist()) for r in idx]


def assert_topk_equivalent(tc, idx_a, idx_b, logits, rel_tie=1e-5, msg=""):
    """Selections equal up to ties (same contract as the CSA2 CPU tests)."""
    tc.assertEqual(idx_a.shape, idx_b.shape, msg)
    for r, (a, b) in enumerate(zip(_rows_as_sets(idx_a), _rows_as_sets(idx_b))):
        if a == b:
            continue
        tc.assertEqual(len(a), len(b), f"{msg} row {r}: different selection sizes")
        row = logits[r].double()
        kth = row.topk(len(a)).values[-1].item()
        scale = max(abs(kth), 1.0)
        for j in a ^ b:
            tc.assertLessEqual(
                abs(row[j].item() - kth),
                rel_tie * scale,
                f"{msg} row {r}: position {j} is not a tie at the boundary {kth}",
            )


def _require_sm70():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA required")
    if torch.cuda.get_device_capability()[0] != 7:
        raise unittest.SkipTest("SM70 required")


def _csa2():
    from sglang.kernels.ops.attention import sm70_dsv41_csa2 as csa2

    return csa2


def _cpu_freqs(n: int, rope_dim: int = 64, seed: int = 0) -> torch.Tensor:
    table = R.precompute_freqs_cis(rope_dim, n + 8, 0, 10000.0, 1.0, 32.0, 1.0)
    return table[:n].contiguous()


class TestSm70Dsv41Csa2Gpu(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        _require_sm70()
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        torch.set_num_threads(4)
        cls.csa2 = _csa2()
        cls.dev = torch.device("cuda")

    def test_pack_kv_bit_exact_vs_oracle(self):
        g = torch.Generator(device="cpu").manual_seed(0)
        x = torch.randn(17, 512, generator=g, dtype=torch.float32).to(FP16)
        pay_ref, sc_ref = R.pack_kv_fp4_e4m3(x)
        packed = self.csa2.pack_kv_fp4(x.to(self.dev))
        self.assertTrue(torch.equal(packed[:, :256].cpu(), pay_ref))
        self.assertTrue(torch.equal(packed[:, 256:].cpu(), sc_ref))

    def test_pack_kv_fused_rope_bit_exact(self):
        g = torch.Generator(device="cpu").manual_seed(11)
        x = torch.randn(9, 512, generator=g, dtype=torch.float32).to(FP16)
        freqs = _cpu_freqs(9)
        rotated = R.rope_tail(x, freqs, 64)
        pay_ref, sc_ref = R.pack_kv_fp4_e4m3(rotated)
        packed = self.csa2.pack_kv_fp4(x.to(self.dev), freqs.to(self.dev), 64)
        self.assertTrue(torch.equal(packed[:, :256].cpu(), pay_ref))
        self.assertTrue(torch.equal(packed[:, 256:].cpu(), sc_ref))

    def test_pack_kv_zero_block_keeps_min_scale(self):
        x = torch.zeros(1, 512, dtype=FP16, device=self.dev)
        packed = self.csa2.pack_kv_fp4(x)
        pay_ref, sc_ref = R.pack_kv_fp4_e4m3(x.cpu())
        self.assertTrue(torch.equal(packed[:, :256].cpu(), pay_ref))
        self.assertTrue(torch.equal(packed[:, 256:].cpu(), sc_ref))

    def test_pack_index_bit_exact_vs_oracle(self):
        g = torch.Generator(device="cpu").manual_seed(1)
        x = torch.randn(9, 128, generator=g, dtype=torch.float32).to(FP16)
        pay_ref, exp_ref = R.pack_index_fp4_ue8m0(x)
        packed = self.csa2.pack_index_k(x.to(self.dev))
        self.assertTrue(torch.equal(packed[:, :64].cpu(), pay_ref))
        self.assertTrue(torch.equal(packed[:, 64:].cpu(), exp_ref))

    def test_pack_index_fused_rope_bit_exact(self):
        g = torch.Generator(device="cpu").manual_seed(12)
        x = torch.randn(5, 128, generator=g, dtype=torch.float32).to(FP16)
        freqs = _cpu_freqs(5)
        rotated = R.rope_tail(x, freqs, 64)
        pay_ref, exp_ref = R.pack_index_fp4_ue8m0(rotated)
        packed = self.csa2.pack_index_k(x.to(self.dev), freqs.to(self.dev), 64)
        self.assertTrue(torch.equal(packed[:, :64].cpu(), pay_ref))
        self.assertTrue(torch.equal(packed[:, 64:].cpu(), exp_ref))

    def test_pack_swa_bit_exact_vs_oracle(self):
        g = torch.Generator(device="cpu").manual_seed(13)
        x = torch.randn(7, 512, generator=g, dtype=torch.float32).to(FP16)
        pay_ref, exp_ref = R.pack_swa_fp8_ue8m0(x)
        packed = self.csa2.pack_swa_fp8(x.to(self.dev))
        self.assertTrue(torch.equal(packed[:, :512].cpu(), pay_ref))
        self.assertTrue(torch.equal(packed[:, 512:].cpu(), exp_ref))

    def test_index_logits_vs_oracle(self):
        g = torch.Generator(device="cpu").manual_seed(2)
        t, h, d, n, ratio = 3, 8, 128, 64, 2
        q = torch.randn(t, h, d, generator=g, dtype=torch.float32).to(FP16)
        k = torch.randn(n, d, generator=g, dtype=torch.float32).to(FP16)
        w = torch.randn(t, h, generator=g, dtype=torch.float32).to(FP16)
        query_pos = torch.tensor([10, 20, 30], dtype=torch.int32)
        kq = R.fake_quant_fp4_ue8m0(k)
        pay, exp = R.pack_index_fp4_ue8m0(k)
        rows = torch.cat([pay, exp], dim=-1)
        ref = R.index_scores(q, kq, w, torch.float32)
        vis = (query_pos + 1) // ratio
        reach = torch.arange(n)[None, :] < vis[:, None]
        ref = ref.masked_fill(~reach, -torch.inf)
        got = self.csa2.index_logits(
            q.to(self.dev),
            w.to(self.dev),
            rows.to(self.dev),
            query_pos.to(self.dev),
            ratio,
        )
        finite = torch.isfinite(ref)
        self.assertTrue(torch.equal(torch.isfinite(got.cpu()), finite))
        scale = ref[finite].abs().amax().clamp_min(1e-4)
        self.assertLessEqual(
            (got.cpu()[finite] - ref[finite]).abs().amax().item(),
            LOGIT_RTOL * scale.item(),
        )

    def test_index_topk_tie_aware(self):
        g = torch.Generator(device="cpu").manual_seed(4)
        t, h, n, ratio, topk = 4, 8, 48, 1, 16
        q = torch.randn(t, h, 128, generator=g, dtype=torch.float32).to(FP16)
        k = torch.randn(n, 128, generator=g, dtype=torch.float32).to(FP16)
        w = torch.randn(t, h, generator=g, dtype=torch.float32).to(FP16)
        query_pos = torch.arange(n, n + t, dtype=torch.int32)
        kq = R.fake_quant_fp4_ue8m0(k)
        pay, exp = R.pack_index_fp4_ue8m0(k)
        rows = torch.cat([pay, exp], dim=-1)
        ref = R.index_scores(q, kq, w, torch.float32)
        vis = (query_pos + 1) // ratio
        reach = torch.arange(n)[None, :] < vis[:, None]
        ref = ref.masked_fill(~reach, -torch.inf)
        got = self.csa2.index_logits(
            q.to(self.dev),
            w.to(self.dev),
            rows.to(self.dev),
            query_pos.to(self.dev),
            ratio,
        )
        compress_lens = vis[:, None]
        idx_ref = R.topk_positions(ref, compress_lens, topk)
        idx_got = R.topk_positions(got.cpu(), compress_lens, topk)
        assert_topk_equivalent(self, idx_ref, idx_got, ref)

    def test_sparse_decode_vs_oracle(self):
        g = torch.Generator(device="cpu").manual_seed(3)
        h, w, ksel = 4, 128, 32
        q = torch.randn(h, 512, generator=g, dtype=torch.float32).to(FP16) * 0.1
        swa = torch.randn(w, 512, generator=g, dtype=torch.float32).to(FP16) * 0.1
        kv = torch.randn(ksel, 512, generator=g, dtype=torch.float32).to(FP16) * 0.1
        swa_valid = torch.ones(w, dtype=torch.bool)
        swa_valid[:8] = False
        kv_valid = torch.ones(ksel, dtype=torch.bool)
        kv_valid[0] = False
        sink = torch.randn(h, generator=g, dtype=torch.float32)
        scale = 512**-0.5
        swa_pay, swa_exp = R.pack_swa_fp8_ue8m0(swa)
        kv_pay, kv_sc = R.pack_kv_fp4_e4m3(kv)
        swa_rows = torch.cat([swa_pay, swa_exp], dim=-1)
        kv_rows = torch.cat([kv_pay, kv_sc], dim=-1)
        swa_u = R.unpack_swa_fp8_ue8m0(swa_pay, swa_exp, FP16)
        kv_u = R.unpack_kv_fp4_e4m3(kv_pay, kv_sc, FP16)
        keys = torch.cat([swa_u, kv_u], dim=0).unsqueeze(0)
        valid = torch.cat([swa_valid, kv_valid], dim=0).unsqueeze(0)
        ref = R.sparse_attention_rows(q.unsqueeze(0), keys, valid, sink, scale)
        got = self.csa2.sparse_decode(
            q.to(self.dev),
            swa_rows.to(self.dev),
            kv_rows.to(self.dev),
            swa_valid.to(self.dev),
            kv_valid.to(self.dev),
            sink.to(self.dev),
            scale,
        )
        self.assertLessEqual(
            (got.cpu().float() - ref[0].float()).abs().amax().item(), ATTN_ATOL
        )

    def test_sparse_decode_empty_row_is_zero(self):
        q = torch.randn(2, 512, dtype=FP16, device=self.dev)
        sink = torch.zeros(2, dtype=torch.float32, device=self.dev)
        got = self.csa2.sparse_decode(
            q,
            q.new_empty((0, 528), dtype=torch.uint8),
            q.new_empty((0, 288), dtype=torch.uint8),
            q.new_empty((0,), dtype=torch.uint8),
            q.new_empty((0,), dtype=torch.uint8),
            sink,
            512**-0.5,
        )
        self.assertTrue(torch.equal(got.cpu(), torch.zeros(2, 512, dtype=FP16)))

    def test_sparse_prefill_vs_oracle(self):
        g = torch.Generator(device="cpu").manual_seed(11)
        t, h, w, ksel = 5, 4, 16, 8
        q = torch.randn(t, h, 512, generator=g, dtype=torch.float32).to(FP16) * 0.1
        swa = torch.randn(t, w, 512, generator=g, dtype=torch.float32).to(FP16) * 0.1
        kv = torch.randn(t, ksel, 512, generator=g, dtype=torch.float32).to(FP16) * 0.1
        swa_valid = torch.ones(t, w, dtype=torch.bool)
        swa_valid[:, :2] = False
        kv_valid = torch.ones(t, ksel, dtype=torch.bool)
        kv_valid[:, 0] = False
        sink = torch.randn(h, generator=g, dtype=torch.float32)
        scale = 512**-0.5
        swa_pay, swa_exp = R.pack_swa_fp8_ue8m0(swa)
        kv_pay, kv_sc = R.pack_kv_fp4_e4m3(kv)
        swa_rows = torch.cat([swa_pay, swa_exp], dim=-1)
        kv_rows = torch.cat([kv_pay, kv_sc], dim=-1)
        swa_u = R.unpack_swa_fp8_ue8m0(swa_pay, swa_exp, FP16)
        kv_u = R.unpack_kv_fp4_e4m3(kv_pay, kv_sc, FP16)
        keys = torch.cat([swa_u, kv_u], dim=1)
        valid = torch.cat([swa_valid, kv_valid], dim=1)
        ref = R.sparse_attention_rows(q, keys, valid, sink, scale)
        got = self.csa2.sparse_prefill(
            q.to(self.dev),
            swa_rows.to(self.dev),
            kv_rows.to(self.dev),
            swa_valid.to(self.dev),
            kv_valid.to(self.dev),
            sink.to(self.dev),
            scale,
        )
        self.assertLessEqual(
            (got.cpu().float() - ref.float()).abs().amax().item(), ATTN_ATOL
        )

    def test_sparse_prefill_swa_only_and_empty(self):
        g = torch.Generator(device="cpu").manual_seed(12)
        t, h, w = 3, 2, 8
        q = torch.randn(t, h, 512, generator=g, dtype=torch.float32).to(FP16) * 0.1
        swa = torch.randn(t, w, 512, generator=g, dtype=torch.float32).to(FP16) * 0.1
        swa_valid = torch.ones(t, w, dtype=torch.bool)
        sink = torch.zeros(h, dtype=torch.float32)
        scale = 512**-0.5
        swa_pay, swa_exp = R.pack_swa_fp8_ue8m0(swa)
        swa_rows = torch.cat([swa_pay, swa_exp], dim=-1)
        swa_u = R.unpack_swa_fp8_ue8m0(swa_pay, swa_exp, FP16)
        ref = R.sparse_attention_rows(q, swa_u, swa_valid, sink, scale)
        got = self.csa2.sparse_prefill(
            q.to(self.dev),
            swa_rows.to(self.dev),
            q.new_empty((t, 0, 288), dtype=torch.uint8),
            swa_valid.to(self.dev),
            q.new_empty((t, 0), dtype=torch.uint8),
            sink.to(self.dev),
            scale,
        )
        self.assertLessEqual(
            (got.cpu().float() - ref.float()).abs().amax().item(), ATTN_ATOL
        )
        z = self.csa2.sparse_prefill(
            q.to(self.dev),
            q.new_empty((t, 0, 528), dtype=torch.uint8),
            q.new_empty((t, 0, 288), dtype=torch.uint8),
            q.new_empty((t, 0), dtype=torch.uint8),
            q.new_empty((t, 0), dtype=torch.uint8),
            sink.to(self.dev),
            scale,
        )
        self.assertTrue(torch.equal(z.cpu(), torch.zeros(t, h, 512, dtype=FP16)))

    def test_decode_bs1_orchestration_vs_oracle(self):
        from sglang.srt.layers.attention.dsv4.sm70_csa2 import (
            sm70_forward_low_ratio_sources,
            sm70_forward_sparse,
        )

        g = torch.Generator(device="cpu").manual_seed(5)
        hidden, heads, t = 64, 4, 8
        dev = self.dev
        x = (torch.randn(t, hidden, generator=g, dtype=torch.float32).to(FP16) * 0.05).to(dev)
        positions = torch.arange(t, dtype=torch.int64, device=dev)
        freqs = _cpu_freqs(t + 4).to(dev)
        wkv = (torch.randn(512, hidden, generator=g, dtype=torch.float32).to(FP16) * 0.02).to(dev)
        kv_norm = torch.ones(512, dtype=FP16, device=dev)
        wq_b = (torch.randn(8 * 128, 32, generator=g, dtype=torch.float32).to(FP16) * 0.02).to(dev)
        w_proj = (torch.randn(8, hidden, generator=g, dtype=torch.float32).to(FP16) * 0.02).to(dev)
        wk = (torch.randn(128, 512, generator=g, dtype=torch.float32).to(FP16) * 0.02).to(dev)
        k_norm = torch.ones(128, dtype=FP16, device=dev)
        wkv_c = (torch.randn(512, hidden, generator=g, dtype=torch.float32).to(FP16) * 0.02).to(dev)
        norm_c = torch.ones(512, dtype=FP16, device=dev)
        q_lora = (torch.randn(t, 32, generator=g, dtype=torch.float32).to(FP16) * 0.05).to(dev)
        sink = torch.zeros(heads, dtype=torch.float32, device=dev)
        q = (torch.randn(t, heads, 512, generator=g, dtype=torch.float32).to(FP16) * 0.1).to(dev)

        def lin(v, weight):
            return (v.float() @ weight.float().t()).to(v.dtype)

        indexer = SimpleNamespace(
            owns_k=True,
            is_candidate_source=False,
            uses_candidates=False,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
            index_topk=16,
            wk=SimpleNamespace(weight=wk),
            k_norm=SimpleNamespace(weight=k_norm),
            queries=lambda ql, fr: R.fake_quant_fp4_ue8m0(
                R.rope_tail(lin(ql, wq_b).view(ql.shape[0], 8, 128), fr, 64)
            ),
            head_weights=lambda xx: lin(xx, w_proj) * (128**-0.5 * 8**-0.5),
        )
        compressor = SimpleNamespace(
            wkv=SimpleNamespace(weight=wkv_c),
            norm=SimpleNamespace(weight=norm_c),
        )
        layer = SimpleNamespace(
            layer_id=20,
            compress_ratio=1,
            head_dim=512,
            qk_rope_head_dim=64,
            eps=1e-20,
            sliding_window=128,
            softmax_scale=512**-0.5,
            freqs_cis=freqs,
            wkv=lambda v: (lin(v, wkv), None),
            kv_norm=SimpleNamespace(weight=kv_norm),
            compressor=compressor,
            indexer=indexer,
            attn_sink=sink,
        )
        backend = SimpleNamespace(model_runner=None)
        fb_prefill = SimpleNamespace(
            batch_size=1,
            forward_mode=SimpleNamespace(is_decode=lambda: False),
        )
        sm70_forward_low_ratio_sources(
            backend,
            layer,
            x[:-1],
            q_lora[:-1],
            positions[:-1],
            fb_prefill,
        )
        fb_dec = SimpleNamespace(
            batch_size=1,
            forward_mode=SimpleNamespace(is_decode=lambda: True),
        )
        sm70_forward_low_ratio_sources(
            backend,
            layer,
            x[-1:],
            q_lora[-1:],
            positions[-1:],
            fb_dec,
        )
        out = sm70_forward_sparse(
            backend,
            q[-1:],
            layer,
            fb_dec,
            1,
            sink,
        )
        self.assertEqual(tuple(out.shape), (1, heads, 512))
        self.assertFalse(torch.isnan(out).any().item())

    def test_backend_sm70_dispatch_is_wired(self):
        repo = Path(__file__).resolve().parents[5]
        src = (repo / "python/sglang/srt/layers/attention/deepseek_v4_backend.py").read_text()
        self.assertIn("sm70_forward_low_ratio_sources", src)
        self.assertNotIn("not implemented on SM70 yet", src)
        self.assertIn("sm70_forward_sparse", src)
        self.assertIn("compress_ratio in (1, 2)", src)
        model = (repo / "python/sglang/srt/models/deepseek_v4.py").read_text()
        self.assertIn("if self.compress_ratio in (1, 2):", model)
        self.assertIn("Skip FlashMLA fused store", model)

    def test_pack_at_matches_index_copy(self):
        g = torch.Generator(device="cpu").manual_seed(21)
        x = torch.randn(3, 512, generator=g, dtype=torch.float32).to(FP16)
        freqs = _cpu_freqs(16)
        pos = torch.tensor([2, 7, 15], dtype=torch.int64)
        table = self.csa2.freqs_cis_interleaved_table(freqs.to(self.dev))
        gfreq = freqs[pos]
        packed = self.csa2.pack_swa_fp8(x.to(self.dev), gfreq.to(self.dev), 64)
        dst = torch.zeros((16, 528), dtype=torch.uint8, device=self.dev)
        self.csa2.pack_swa_fp8_at(
            dst,
            x.to(self.dev),
            pos.to(self.dev),
            freqs_table=table,
            rope_dim=64,
            dst_mod=16,
        )
        ref = torch.zeros_like(dst)
        ref.index_copy_(0, pos.to(self.dev), packed)
        self.assertTrue(torch.equal(dst.cpu(), ref.cpu()))

        xk = torch.randn(3, 512, generator=g, dtype=torch.float32).to(FP16)
        packed_kv = self.csa2.pack_kv_fp4(xk.to(self.dev), freqs.to(self.dev)[:3], 64)
        dst_kv = torch.zeros((8, 288), dtype=torch.uint8, device=self.dev)
        pos2 = torch.tensor([1, 3, 5], dtype=torch.int64)
        self.csa2.pack_kv_fp4_at(
            dst_kv,
            xk.to(self.dev),
            pos2.to(self.dev),
            freqs_table=table,
            rope_dim=64,
            row_div=2,
            freq_mul=2,
        )
        gfreq = freqs[(pos2 // 2) * 2]
        packed_r2 = self.csa2.pack_kv_fp4(xk.to(self.dev), gfreq.to(self.dev), 64)
        ref_kv = torch.zeros_like(dst_kv)
        ref_kv.index_copy_(0, (pos2 // 2).to(self.dev), packed_r2)
        self.assertTrue(torch.equal(dst_kv.cpu(), ref_kv.cpu()))

    def test_sparse_decode_indexed_matches_gather(self):
        g = torch.Generator(device="cpu").manual_seed(22)
        h, w, cap, ksel = 4, 128, 64, 32
        q = (torch.randn(h, 512, generator=g, dtype=torch.float32).to(FP16) * 0.1).to(
            self.dev
        )
        swa = torch.randn(w, 512, generator=g, dtype=torch.float32).to(FP16) * 0.1
        kv = torch.randn(cap, 512, generator=g, dtype=torch.float32).to(FP16) * 0.1
        sink = torch.randn(h, generator=g, dtype=torch.float32).to(self.dev)
        scale = 512**-0.5
        swa_pay, swa_exp = R.pack_swa_fp8_ue8m0(swa)
        kv_pay, kv_sc = R.pack_kv_fp4_e4m3(kv)
        swa_rows = torch.cat([swa_pay, swa_exp], dim=-1).to(self.dev)
        kv_table = torch.cat([kv_pay, kv_sc], dim=-1).to(self.dev)
        kv_idx = torch.full((ksel,), -1, dtype=torch.int32, device=self.dev)
        kv_idx[0] = 3
        kv_idx[1] = 0
        kv_idx[4] = 17
        pos = torch.tensor([5], dtype=torch.int64, device=self.dev)
        slots = torch.arange(w, device=self.dev, dtype=pos.dtype)
        valid_s = (torch.remainder(pos[:, None] - slots[None, :], w) <= pos[:, None])[
            0
        ].to(torch.uint8)
        gathered = kv_table[kv_idx.clamp_min(0).to(torch.int64)]
        valid_k = (kv_idx >= 0).to(torch.uint8)
        ref = self.csa2.sparse_decode(
            q, swa_rows, gathered, valid_s, valid_k, sink, scale
        )
        got = self.csa2.sparse_decode_indexed(
            q, swa_rows, kv_table, kv_idx, pos, sink, scale
        )
        self.assertLessEqual((got.float() - ref.float()).abs().amax().item(), ATTN_ATOL)


if __name__ == "__main__":
    unittest.main()
