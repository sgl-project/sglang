"""Fused paged FP8 indexer scoring vs the torch reference and an fp64 oracle.

Two design points, both consequences of how a defective verification slipped
through on this same operator once before (#33271 discussion):

- The reference the kernel is compared against is the in-tree torch path
  (``fp8_paged_mqa_logits_torch_sm120``), not a re-derivation of it, and both
  are additionally compared against an fp64 oracle transcribed line-by-line
  from that torch path. The kernel is required to track the oracle's top-k at
  least as well as the torch path does — comparing against the torch path
  alone caps the criterion at the reference's own bf16 rounding noise.
- A deliberately broken implementation (per-head ReLU dropped, which is the
  linear operator a head-dimension fold computes) must FAIL the same
  criterion. A gate that cannot catch that is not a gate.

Cache layout (page-level segments, from ``indexer.py``): each 8448-byte page
is [8192 B: 64 tokens x 128 FP8 values][256 B: 64 fp32 scales].
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsv4.fused_paged_indexer import (
    fused_paged_mqa_logits,
)
from sglang.srt.layers.attention.dsv4.indexer import (
    fp8_paged_mqa_logits_torch_sm120,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")

_HAS_CUDA = torch.cuda.is_available()


def _synth_case(batch, pool_pages, max_pages, max_seq_len, seed):
    """Paged data with the real layout; FP8 encoded from randn so no NaN
    bit patterns occur (the bit decoder maps 0x7F/0xFF to +-480 by design).
    The page table must cover max_seq_len — the torch path assumes it."""
    assert max_pages * 64 >= max_seq_len
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
        .reshape(batch, 1)  # the live caller passes [B, 1]
    )
    return q, kvcache, weight, seq_lens, page_table, max_seq_len


def _oracle_fp64(q, kvcache, weight, page_table, max_seq_len):
    """The torch path's arithmetic, transcribed, in fp64."""
    batch = q.shape[0]
    pool = kvcache.shape[0]
    flat = kvcache.reshape(pool, 8448)
    vals = flat[:, :8192].contiguous().view(torch.float8_e4m3fn).reshape(pool, 64, 128)
    scales = flat[:, 8192:].contiguous().view(torch.float32).reshape(pool, 64)
    pages = page_table.clamp(min=0).long()
    n = pages.shape[1] * 64
    kv64 = vals[pages].to(torch.float64).reshape(batch, n, 128)
    sc = scales[pages].reshape(batch, n).double()
    q64 = q[:, 0].to(torch.float64)
    s = F.relu(torch.bmm(kv64, q64.transpose(1, 2))) * weight.double().unsqueeze(1)
    out = (s.sum(dim=2) * sc)[:, : min(n, max_seq_len)]
    if out.shape[1] < max_seq_len:
        out = F.pad(out, (0, max_seq_len - out.shape[1]), value=float("-inf"))
    return out


def _no_relu_bf16(q, kvcache, weight, page_table, max_seq_len):
    """The linear operator a head-dimension fold computes — must be caught."""
    batch = q.shape[0]
    pool = kvcache.shape[0]
    flat = kvcache.reshape(pool, 8448)
    vals = flat[:, :8192].contiguous().view(torch.float8_e4m3fn).reshape(pool, 64, 128)
    scales = flat[:, 8192:].contiguous().view(torch.float32).reshape(pool, 64)
    pages = page_table.clamp(min=0).long()
    n = pages.shape[1] * 64
    kvb = vals[pages].to(torch.bfloat16).reshape(batch, n, 128)
    sc = scales[pages].reshape(batch, n)
    qb = q[:, 0].to(torch.bfloat16)
    s = torch.bmm(kvb, qb.transpose(1, 2)) * weight.unsqueeze(1)
    out = (s.sum(dim=2) * sc)[:, : min(n, max_seq_len)].float()
    if out.shape[1] < max_seq_len:
        out = F.pad(out, (0, max_seq_len - out.shape[1]), value=float("-inf"))
    return out


def _topk_overlap(a, b, seq_lens, max_seq_len, k=512):
    overlaps = []
    lens = seq_lens.reshape(-1).long()
    for row in range(a.shape[0]):
        n = min(int(lens[row]), max_seq_len)
        kk = min(k, n)
        if kk < 16:
            continue
        ia = torch.topk(a[row, :n], kk).indices
        ib = torch.topk(b[row, :n], kk).indices
        overlaps.append(len(set(ia.tolist()) & set(ib.tolist())) / kk)
    return min(overlaps), sum(overlaps) / len(overlaps)


@unittest.skipUnless(_HAS_CUDA, "CUDA required")
class TestFusedPagedIndexer(CustomTestCase):
    CASES = [(64, 128, 16, 1024, 0), (256, 512, 64, 4096, 1), (37, 200, 33, 2050, 2)]

    def _run(self, case):
        q, kv, w, sl, pt, msl = _synth_case(*case)
        ref = fp8_paged_mqa_logits_torch_sm120(
            q, kv, w, sl, pt, None, msl, clean_logits=False
        ).float()
        got = fused_paged_mqa_logits(q, kv, w, sl, pt, None, msl, clean_logits=False)
        return q, kv, w, sl, pt, msl, ref, got

    def test_invalid_region_matches_torch_bitwise(self):
        for case in self.CASES:
            q, kv, w, sl, pt, msl, ref, got = self._run(case)
            valid = torch.arange(msl, device="cuda")[None, :] < sl.reshape(-1)[:, None]
            self.assertTrue(torch.equal(ref[~valid], got[~valid]))

    def test_tracks_fp64_oracle_at_least_as_well_as_torch(self):
        for case in self.CASES:
            q, kv, w, sl, pt, msl, ref, got = self._run(case)
            oracle = _oracle_fp64(q, kv, w, pt, msl)
            valid = torch.arange(msl, device="cuda")[None, :] < sl.reshape(-1)[:, None]
            err_ref = (ref.double() - oracle).abs()[valid].max().item()
            err_got = (got.double() - oracle).abs()[valid].max().item()
            # fp32 accumulation with no bf16 materialisation: the fused kernel
            # should sit orders of magnitude closer to the operator than the
            # torch path it replaces (measured ~1e4x; require 1e2x headroom).
            self.assertLess(err_got * 100, err_ref)
            k_min, _ = _topk_overlap(got, oracle.float(), sl, msl)
            r_min, _ = _topk_overlap(ref, oracle.float(), sl, msl)
            self.assertGreaterEqual(k_min, r_min)

    def test_missing_relu_is_caught_by_the_same_criterion(self):
        q, kv, w, sl, pt, msl, ref, _ = self._run(self.CASES[1])
        broken = _no_relu_bf16(q, kv, w, pt, msl)
        _, mean_overlap = _topk_overlap(ref, broken, sl, msl)
        self.assertLess(mean_overlap, 0.95)


if __name__ == "__main__":
    unittest.main()
