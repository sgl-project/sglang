"""The fp4 DSA prefill indexer against DeepGEMM's `fp8_fp4_mqa_logits`: every row's candidate
slab must hold the exact top-k of the reference logits, with the reference's own scores."""

import itertools

import pytest
import torch

from sglang.kernels.ops.attention.dsa.litetopk_prefill import fp4_indexer_topk
from sglang.kernels.ops.attention.dsa.litetopk_prefill.fp4_indexer import (
    FP4_INDEXER_CAP,
    fp4_indexer_candidates,
    is_fp4_indexer_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not is_fp4_indexer_supported(), reason="the fp4 prefill indexer needs SM100"
)


def build_rows(requests, ratio):
    """Rows as sglang's prefill plan lays them out: request by request, row at token
    position p sees (p + 1) // ratio compressed keys from its request's first column."""
    lens = [ctx // ratio for ctx, _ in requests]
    starts = list(itertools.accumulate(lens, initial=0))[:-1]
    ks, ke = [], []
    for (ctx, rows), s in zip(requests, starts):
        for i in range(rows):
            p = ctx - rows + i
            ks.append(s)
            ke.append(s + (p + 1) // ratio)
    return (
        torch.tensor(ks, dtype=torch.int32, device="cuda"),
        torch.tensor(ke, dtype=torch.int32, device="cuda"),
        sum(lens),
    )


def make_inputs(heads, requests, ratio, seed):
    from deep_gemm.utils.math import per_token_cast_to_fp4

    torch.manual_seed(seed)
    ks, ke, num_kv = build_rows(requests, ratio)
    rows = ks.shape[0]
    q = torch.randn(rows, heads, 128, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((num_kv + 3) // 4 * 4, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(rows, heads, device="cuda", dtype=torch.bfloat16)
    q_fp4, q_sf = per_token_cast_to_fp4(
        q.view(-1, 128), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    kv_fp4, kv_sf = per_token_cast_to_fp4(
        kv, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    return dict(
        q_fp4=q_fp4.view(rows, heads, 64).contiguous(),
        q_sf=q_sf.view(rows, heads).contiguous(),
        weights=weights,
        kv_fp4=kv_fp4.contiguous(),
        kv_sf=kv_sf.view(-1).contiguous(),
        ks=ks,
        ke=ke,
        width=(max(ke - ks).item() + 63) // 64 * 64,
    )


def reference_logits(inp):
    from deep_gemm import fp8_fp4_mqa_logits

    return fp8_fp4_mqa_logits(
        (inp["q_fp4"], inp["q_sf"]),
        (inp["kv_fp4"], inp["kv_sf"]),
        inp["weights"],
        inp["ks"],
        inp["ke"],
        False,
        inp["width"],
        logits_dtype=torch.bfloat16,
    ).float()


@pytest.mark.parametrize(
    "heads,topk,ratio,requests",
    [
        (64, 1024, 4, [(65536, 256)]),
        (64, 1024, 4, [(262144, 96), (65536, 96), (16384, 64)]),
        (32, 512, 2, [(65536, 256)]),
        (32, 512, 1, [(131072, 128), (40960, 128)]),
        # rows that see at most topk keys produce every key
        (64, 1024, 4, [(4096, 128)]),
        # 32-head dense blocks (every row <= 6144 keys) with distinct row lengths and a request boundary inside a
        # block: each row keeps its own count (nvcc 13.0 miscompiled a lane select there; odd rows got a neighbour's)
        (32, 512, 1, [(3000, 66), (40, 7)]),
    ],
)
def test_fp4_indexer_candidates(heads, topk, ratio, requests):
    inp = make_inputs(heads, requests, ratio, seed=heads + ratio)
    run = lambda: fp4_indexer_candidates(
        inp["q_fp4"],
        inp["q_sf"],
        inp["weights"],
        inp["kv_fp4"],
        inp["kv_sf"],
        inp["ks"],
        inp["ke"],
        topk,
    )
    out, again = run(), run()
    torch.cuda.synchronize()
    lens = (inp["ke"] - inp["ks"]).to(torch.int64)
    count = out.counts.to(torch.int64)
    kept = count.clamp_max(FP4_INDEXER_CAP)
    sel = torch.arange(FP4_INDEXER_CAP, device="cuda")[None, :] < kept[:, None]
    assert int(out.overflow.sum()) == 0
    # the candidate set is deterministic (its order within the slab is not)
    assert torch.equal(out.counts, again.counts) and torch.equal(
        out.thresholds, again.thresholds
    )
    # T = -inf rows hold every key in key order: the index of slot s is s (not stored)
    ident = (out.thresholds == float("-inf"))[:, None]
    slot = torch.arange(FP4_INDEXER_CAP, device="cuda", dtype=torch.int32)[None, :]
    out_indices = torch.where(ident, slot, out.indices)
    again_indices = torch.where(ident, slot, again.indices)
    for a, b in (
        (out.scores.float(), again.scores.float()),
        (out_indices, again_indices),
    ):
        va = torch.where(sel, a, torch.zeros_like(a)).sort(dim=1).values
        vb = torch.where(sel, b, torch.zeros_like(b)).sort(dim=1).values
        assert torch.equal(va, vb)
    logits = reference_logits(inp)
    col = torch.arange(logits.shape[1], device="cuda")
    valid = col[None, :] < lens[:, None]
    thr = out.thresholds
    # count == #(logits >= T), and the stored scores are the reference's own
    assert torch.equal((valid & (logits >= thr[:, None])).sum(1), count)
    idx = out_indices.to(torch.int64).clamp(0, logits.shape[1] - 1)
    diff = torch.where(
        sel, logits.gather(1, idx) - out.scores.float(), torch.zeros_like(logits[:, :1])
    )
    assert float(diff.abs().max()) == 0.0
    assert not bool(
        (
            sel & ((out_indices < 0) | (out_indices >= lens[:, None].to(torch.int32)))
        ).any()
    )
    # every logit strictly above the k-th is a candidate; T never exceeds the k-th score
    masked = logits.masked_fill(~valid, float("-inf"))
    k = min(topk, logits.shape[1])
    kth = masked.topk(k, dim=1).values[:, -1]
    assert not bool((valid & (masked > kth[:, None]) & (logits < thr[:, None])).any())
    assert not bool(((thr > kth) & (lens > k)).any())
    short = lens <= topk
    assert torch.equal(count[short], lens[short])


@pytest.mark.parametrize(
    "heads,topk,ratio,requests",
    [
        (64, 1024, 4, [(262144, 96), (65536, 96), (4096, 64)]),
        (32, 512, 1, [(131072, 128), (40960, 128)]),
    ],
)
def test_fp4_indexer_topk(heads, topk, ratio, requests):
    """The fp4 indexer + bf16 top-k pair returns exactly the top-k of DeepGEMM's bf16 logits (any
    copies of the k-th value), offset by `out_offsets`, -1 past min(topk, len)."""
    inp = make_inputs(heads, requests, ratio, seed=heads + ratio + 1)
    ks, ke = inp["ks"], inp["ke"]
    out = fp4_indexer_topk(
        inp["q_fp4"],
        inp["q_sf"],
        inp["weights"],
        inp["kv_fp4"],
        inp["kv_sf"],
        ks,
        ke,
        topk,
        out_offsets=ks,
    )
    torch.cuda.synchronize()
    logits = reference_logits(inp)
    lens = (ke - ks).to(torch.int64)
    n = lens.clamp(max=topk)
    valid = torch.arange(topk, device="cuda")[None, :] < n[:, None]
    assert (out[~valid] == -1).all(), "padding past min(topk, len) is not -1"
    rel = torch.where(
        valid,
        out.to(torch.int64) - ks[:, None],
        torch.zeros_like(out, dtype=torch.int64),
    )
    assert ((rel >= 0) & (rel < lens[:, None])).all(), (
        "selected key outside the row's range"
    )
    srt = torch.where(valid, rel, torch.full_like(rel, -1)).sort(dim=1).values
    assert not ((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)).any(), (
        "duplicate selection"
    )
    col = torch.arange(logits.shape[1], device="cuda")
    masked = logits.masked_fill(col[None, :] >= lens[:, None], float("-inf"))
    got = torch.where(
        valid, masked.gather(1, rel), torch.full_like(masked[:, :1], float("-inf"))
    )
    ref = masked.topk(min(topk, masked.shape[1]), dim=1).values
    ref = torch.where(
        valid[:, : ref.shape[1]], ref, torch.full_like(ref, float("-inf"))
    )
    assert torch.equal(got.sort(dim=1, descending=True).values[:, : ref.shape[1]], ref)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
