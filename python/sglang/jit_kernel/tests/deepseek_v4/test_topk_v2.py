"""Correctness tests for the DeepSeek-V4 (DSA indexer) JIT top-k transform v2.

The v2 kernel selects the per-row top-k of ``scores`` (ragged ``seq_lens``) and
writes the page-table transform of the selected raw indices into the output. We
validate against ``torch.topk`` with a small tolerance for boundary ties (the
fp16 coarse histogram can swap elements of equal score).

Covers every dispatch path: trivial (seq<=k), register (seq<=8192), streaming
(seq>8192 non-cluster), and cluster (seq>=64K with batch<=128), plus identity &
permutation page tables and ragged lengths.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.jit_kernel.dsv4.topk import plan_topk_v2, topk_transform_512_v2
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-large")

PAGE_SIZE = 64  # c4 page size = 256 // 4
PAGE_BITS = PAGE_SIZE.bit_length() - 1
PAGE_MASK = PAGE_SIZE - 1
MAX_PERMIT_ERROR = 5


def _assert_topk_close(scores, ref_raw, our_raw, bs, seq_lens, k):
    """Set-compare our top-k raw indices vs torch's, tolerating equal-score ties."""
    bad = 0
    for i in range(bs):
        L = int(seq_lens[i])
        ref = set(ref_raw[i])
        our = set(our_raw[i])
        more, less = our - ref, ref - our
        if more or less:
            mv = sorted(scores[i, list(more)].tolist())
            lv = sorted(scores[i, list(less)].tolist())
            if mv != lv:  # not merely a tie swap -> genuine error
                bad += len(more)
                print(f"b={i} L={L} k={k}: more={list(more)[:4]} less={list(less)[:4]} mv={mv[:3]} lv={lv[:3]}")
        # exactly min(k, L) valid (non -1) selections expected
        assert len(our) == min(k, L), f"b={i}: {len(our)} valid != {min(k, L)}"
    assert bad <= MAX_PERMIT_ERROR, f"{bad=} > {MAX_PERMIT_ERROR}"


def _make_page_table(batch, num_pages, mode, device):
    if mode == "identity":
        pt = torch.arange(num_pages, dtype=torch.int32, device=device)
        inv = pt
    else:  # permutation
        pt = torch.randperm(num_pages, device=device).to(torch.int32)
        inv = torch.empty_like(pt)
        inv[pt.long()] = torch.arange(num_pages, dtype=torch.int32, device=device)
    return pt.unsqueeze(0).expand(batch, -1).contiguous(), inv.cpu()


def _invert(out_row, inv_cpu):
    """Undo page_to_indices for a list of page indices (drop -1 padding)."""
    raw = []
    for v in out_row:
        if v == -1:
            continue
        raw.append((int(inv_cpu[v >> PAGE_BITS]) << PAGE_BITS) | (v & PAGE_MASK))
    return raw


@pytest.mark.parametrize("page_mode", ["identity", "perm"])
@pytest.mark.parametrize(
    "batch,seq",
    [
        (8, 512),       # trivial (seq == k)
        (8, 4096),      # register
        (200, 16384),   # streaming (batch > 128 => non-cluster)
        (4, 16384),     # streaming (seq < floor => non-cluster, small batch)
        (100, 65536),   # streaming (seq == floor)
        (256, 131072),  # batch > 128 => streaming even at long ctx
        (2, 131072),    # persistent cluster, pool >= N
        (40, 262144),   # persistent cluster round-robin (N > pool of 30)
        (64, 262144),   # persistent cluster round-robin (N > pool of 30)
    ],
)
@pytest.mark.parametrize("k", [512, 1024])
@torch.inference_mode()
def test_topk_v2(batch: int, seq: int, k: int, page_mode: str) -> None:
    if k > seq:
        pytest.skip("k cannot exceed seq")
    torch.manual_seed(batch * 1000 + seq + k)
    device = "cuda"
    scores = torch.randn(batch, seq, dtype=torch.float32, device=device)
    seq_lens = torch.full((batch,), seq, dtype=torch.int32, device=device)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(batch, num_pages, page_mode, device)
    out = torch.full((batch, k), -1, dtype=torch.int32, device=device)

    metadata = plan_topk_v2(seq_lens)
    topk_transform_512_v2(scores, seq_lens, page_table, out, PAGE_SIZE, metadata)
    torch.cuda.synchronize()

    out_cpu = out.cpu().tolist()
    our_raw = [_invert(out_cpu[i], inv_cpu) for i in range(batch)]
    ref_raw = [
        torch.topk(scores[i, :seq], k, sorted=False).indices.cpu().tolist()
        for i in range(batch)
    ]
    _assert_topk_close(scores.cpu(), ref_raw, our_raw, batch, seq_lens.cpu(), k)


@pytest.mark.parametrize("k", [512, 1024, 2048])
@torch.inference_mode()
def test_topk_v2_ragged(k: int) -> None:
    """Ragged per-batch lengths spanning trivial..cluster paths in one launch."""
    torch.manual_seed(2024 + k)
    device = "cuda"
    batch, seq = 64, 131072
    scores = torch.randn(batch, seq, dtype=torch.float32, device=device)
    # mix of short (trivial/register), medium and long (cluster) rows
    lengths = torch.randint(k, seq + 1, (batch,), dtype=torch.int32, device=device)
    lengths[0] = max(1, k // 2)  # force a trivial row (seq < k)
    num_pages = (seq + PAGE_SIZE - 1) // PAGE_SIZE
    page_table, inv_cpu = _make_page_table(batch, num_pages, "perm", device)
    out = torch.full((batch, k), -1, dtype=torch.int32, device=device)

    metadata = plan_topk_v2(lengths)
    topk_transform_512_v2(scores, lengths, page_table, out, PAGE_SIZE, metadata)
    torch.cuda.synchronize()

    out_cpu = out.cpu().tolist()
    sl = lengths.cpu()
    scores_cpu = scores.cpu()
    our_raw, ref_raw = [], []
    for i in range(batch):
        L = int(sl[i])
        our_raw.append(_invert(out_cpu[i], inv_cpu))
        if L <= k:
            ref_raw.append(list(range(L)))  # trivial: all positions
        else:
            ref_raw.append(torch.topk(scores_cpu[i, :L], k, sorted=False).indices.tolist())
    _assert_topk_close(scores_cpu, ref_raw, our_raw, batch, sl, k)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
