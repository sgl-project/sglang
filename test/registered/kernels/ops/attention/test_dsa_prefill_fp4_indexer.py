"""The fp4 DSA prefill indexer against DeepGEMM's `fp8_fp4_mqa_logits`: every row's candidate
slab must hold the exact top-k of the reference logits, with the reference's own scores."""

import itertools
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.dsa.litetopk_prefill import fp4_indexer_topk
from sglang.kernels.ops.attention.dsa.litetopk_prefill.fp4_indexer import (
    FP4_INDEXER_CAP,
    FP4_INDEXER_DENSE_LEN,
    FP4_INDEXER_MAX_SPAN,
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
        # long rows: 512K and 1M keys take the 129- and 257-key sample strides
        (32, 512, 2, [(1048576, 128)]),
        (32, 512, 1, [(1048576, 64)]),
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
        (32, 512, 2, [(1048576, 128)]),
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


@pytest.mark.parametrize("heads", [32, 64])
def test_fp4_indexer_max_span(heads):
    """A block spanning FP4_INDEXER_MAX_SPAN[heads] keys samples its farthest rows; rows past that span get none."""
    from deep_gemm.utils.math import per_token_cast_to_fp4

    # One block of 128 // heads rows: the last `rows` rows of each request.
    far, rows = 1024, 64 // heads
    for near, sampled in (
        (FP4_INDEXER_MAX_SPAN[heads] - far, True),
        (FP4_INDEXER_MAX_SPAN[heads], False),
    ):
        ke = [near - rows + 1 + i for i in range(rows)]
        ke += [near + far - rows + 1 + i for i in range(rows)]
        ks = torch.tensor([0] * rows + [near] * rows, dtype=torch.int32, device="cuda")
        ke = torch.tensor(ke, dtype=torch.int32, device="cuda")
        torch.manual_seed(3)
        q = torch.randn(2 * rows, heads, 128, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(near + far, 128, device="cuda", dtype=torch.bfloat16)
        q_fp4, q_sf = per_token_cast_to_fp4(
            q.view(-1, 128), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
        )
        kv_fp4, kv_sf = per_token_cast_to_fp4(
            kv, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
        )
        weights = torch.randn(2 * rows, heads, device="cuda", dtype=torch.bfloat16)
        res = fp4_indexer_candidates(
            q_fp4.view(2 * rows, heads, 64).contiguous(),
            q_sf.view(2 * rows, heads).contiguous(),
            weights,
            kv_fp4.contiguous(),
            kv_sf.view(-1).contiguous(),
            ks,
            ke,
            512,
        )
        far_samples = res.samples[rows:]
        if sampled:
            assert bool((far_samples > 0).all()), far_samples
        else:
            assert bool((far_samples == 0).all()), far_samples


@pytest.mark.parametrize("heads", [32, 64])
def test_full_topk_lite_dispatch_at_max_span(monkeypatch, heads):
    """DeepSeek-V4.1 full top-k layers take the fp4 lite path while the gathered index keys fit FP4_INDEXER_MAX_SPAN,
    or every row is dense; otherwise they take the dense path."""
    from sglang.srt.layers.attention.dsv4.v41_indexer import full_topk

    class LitePath(Exception):
        pass

    class DensePath(Exception):
        pass

    def lite_path(*args, **kwargs):
        raise LitePath

    def dense_path(*args, **kwargs):
        raise DensePath

    monkeypatch.setattr(full_topk.FullTopKIndexer, "_prefill_lite_topk", lite_path)
    indexer = full_topk.FullTopKIndexer(
        token_to_kv_pool=SimpleNamespace(get_low_ratio_index_k_fp4=dense_path),
        req_to_token=None,
        use_deep_gemm_prefill=True,
        use_deep_gemm_decode=True,
        use_prefill_lite_topk=True,
    )
    rows = 1024
    inputs = SimpleNamespace(
        reset_outputs=lambda: None,
        out_raw_indices=torch.empty(rows, 512, dtype=torch.int32),
        row_shard=None,
        layer_id=0,
    )
    span, dense = FP4_INDEXER_MAX_SPAN[heads], FP4_INDEXER_DENSE_LEN
    for lens, path in (
        ([span - 65536, 65536], LitePath),
        # The longest request fits; only the batch's gathered keys pass the span.
        ([span - 65536, 65540], DensePath),
        # Every block is dense and samples nothing, however many keys the batch gathers.
        ([dense] * (span // dense + 1), LitePath),
    ):
        data = SimpleNamespace(
            num_rows=rows,
            lens_per_request=lens,
            k_slots=torch.empty(sum(lens), dtype=torch.int64),
            q_sf=torch.empty(rows, heads, dtype=torch.int32),
        )
        monkeypatch.setattr(
            full_topk, "get_deep_gemm_prefill_data", lambda _, d=data: d
        )
        with pytest.raises(path):
            indexer.topk_prefill(inputs)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
