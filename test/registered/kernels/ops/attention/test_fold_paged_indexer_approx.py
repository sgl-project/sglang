"""The opt-in fold path must match ITS OWN spec exactly.

The fold is a documented lossy approximation of the C4 indexer scoring — its
spec is the LINEAR operator (no per-head ReLU). This test pins the
implementation to that spec against an fp64 reference, and pins the
non-shared-page-table fallback bitwise to the exact fused path. It does NOT
test closeness to the real operator: the divergence is the documented cost
(see the implementation's docstring and the #33271 discussion).
"""

from __future__ import annotations

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.fused_paged_indexer import (
    fused_paged_mqa_logits,
)
from sglang.srt.layers.attention.dsv4.indexer import fold_paged_mqa_logits_approx
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

_HAS_CUDA = torch.cuda.is_available()


def _case(batch, pool_pages, max_pages, max_seq_len, seed, shared):
    g = torch.Generator(device="cuda").manual_seed(seed)
    vals = (torch.randn(pool_pages, 64, 128, generator=g, device="cuda") * 0.3).to(
        torch.float8_e4m3fn
    )
    scales = torch.rand(pool_pages, 64, generator=g, device="cuda") * 0.5 + 0.75
    page = torch.cat(
        [
            vals.view(torch.uint8).reshape(pool_pages, 8192),
            scales.view(torch.uint8).reshape(pool_pages, 256),
        ],
        dim=1,
    )
    kvcache = page.reshape(pool_pages, 64, 1, 132).contiguous()
    q = (torch.randn(batch, 1, 64, 128, generator=g, device="cuda") * 0.3).to(
        torch.float8_e4m3fn
    )
    weight = torch.randn(batch, 64, generator=g, device="cuda")
    if shared:
        row = torch.randperm(pool_pages, device="cuda")[:max_pages].to(torch.int32)
        page_table = row[None, :].expand(batch, max_pages).contiguous()
    else:
        page_table = torch.randint(
            0,
            pool_pages,
            (batch, max_pages),
            generator=g,
            device="cuda",
            dtype=torch.int32,
        )
    base = max(1, max_pages * 64 - batch)
    seq_lens = (
        (base + torch.arange(batch, device="cuda", dtype=torch.int32) + 1)
        .clamp(max=max_pages * 64)
        .reshape(batch, 1)
    )
    return q, kvcache, weight, seq_lens, page_table, max_seq_len


def _linear_op_fp64(q, kvcache, weight, page_table, max_seq_len):
    pool = kvcache.shape[0]
    flat = kvcache.reshape(pool, 8448)
    vals = flat[:, :8192].contiguous().view(torch.float8_e4m3fn).reshape(pool, 64, 128)
    scales = flat[:, 8192:].contiguous().view(torch.float32).reshape(pool, 64)
    pages = page_table[0].clamp(min=0).long()
    kv = vals[pages].to(torch.float64).reshape(-1, 128)
    sc = scales[pages].reshape(-1).double()
    n = min(kv.shape[0], max_seq_len)
    q_eff = (weight.double().unsqueeze(-1) * q[:, 0].double()).sum(dim=1)
    return (q_eff @ kv[:n].t()) * sc[:n].unsqueeze(0)


@unittest.skipUnless(_HAS_CUDA, "CUDA required")
class TestFoldPagedIndexerApprox(CustomTestCase):
    def test_matches_linear_operator_spec(self):
        q, kv, w, sl, pt, msl = _case(256, 256, 32, 2048, 0, shared=True)
        got = fold_paged_mqa_logits_approx(
            q, kv, w, sl, pt, None, msl, clean_logits=False
        )
        ref = _linear_op_fp64(q, kv, w, pt, msl)
        valid = (torch.arange(msl, device="cuda")[None, :] < sl.reshape(-1)[:, None])[
            :, : ref.shape[1]
        ]
        err = (got[:, : ref.shape[1]].double() - ref).abs()[valid].max().item()
        scale = ref.abs().max().item()
        self.assertLess(err, 1e-4 * max(scale, 1.0))

    def test_invalid_region_is_neg_inf(self):
        q, kv, w, sl, pt, msl = _case(64, 128, 16, 1024, 1, shared=True)
        got = fold_paged_mqa_logits_approx(
            q, kv, w, sl, pt, None, msl, clean_logits=False
        )
        invalid = torch.arange(msl, device="cuda")[None, :] >= sl.reshape(-1)[:, None]
        self.assertTrue(torch.isneginf(got[invalid]).all())

    def test_non_shared_rows_fall_back_to_exact_bitwise(self):
        q, kv, w, sl, pt, msl = _case(64, 128, 16, 1024, 2, shared=False)
        got = fold_paged_mqa_logits_approx(
            q, kv, w, sl, pt, None, msl, clean_logits=False
        )
        exact = fused_paged_mqa_logits(q, kv, w, sl, pt, None, msl, clean_logits=False)
        self.assertTrue(torch.equal(got, exact))

    def test_peak_memory_stays_near_output_size(self):
        # The folded GEMM walks the sequence axis in blocks of
        # ~256 MB / (B*4) columns; computed whole it materialises two [B, n]
        # fp32 transients on top of the output (measured +1.66 GB at 128K,
        # B=8192), which is how the flag once took a needle run down. The
        # shape here makes n exceed one block (B=4096 -> 16384-column steps,
        # n=18432), so the chunked walk is actually exercised. Budget: output
        # + one ~256 MB block + decoded KV + slack; the whole-tensor form
        # needs output + 2n*B*4 (= +604 MB here) and must trip this.
        q, kv, w, sl, pt, msl = _case(4096, 320, 288, 18432, 3, shared=True)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out = fold_paged_mqa_logits_approx(
            q, kv, w, sl, pt, None, msl, clean_logits=False
        )
        peak_over_base = torch.cuda.max_memory_allocated() - base
        out_bytes = out.numel() * out.element_size()
        self.assertLess(peak_over_base, out_bytes + (320 << 20))


if __name__ == "__main__":
    unittest.main()
