"""Correctness of the FP8 DSA prefill indexer candidate filter against
DeepGEMM `fp8_mqa_logits` + an exact torch top-k."""

import math
import random
from typing import List, Sequence, Tuple

import pytest
import torch

from sglang.kernels.ops.attention.dsa.litetopk_prefill import fp8_indexer_topk
from sglang.kernels.ops.attention.dsa.litetopk_prefill.fp8_indexer import (
    fp8_indexer_candidates,
    is_fp8_indexer_supported,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HEAD_DIM = 128
HEADS = (32, 64)
TOPK = 2048
CAP = 12288
# Kernel constants mirrored for the threshold check (fp8_indexer.cuh): single-key samples on an odd stride chosen
# from the block's span, at most MAX_SAMPLE_TILES tiles of 256.
TILE_KEYS, MAX_SAMPLE_TILES, ALPHA = 256, 32, 4.0


def sample_stride(span: int) -> int:
    return 65 if span <= 524288 else 129 if span <= 1048576 else 257


def make_ragged(
    requests: Sequence[Tuple[int, int, str]], device="cuda"
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """requests: (ctx, rows, where) with where in {"last", "mid"}; rows of
    request b at positions p see keys [cum_b, cum_b + p + 1) (sglang ragged
    prefill convention). Returns ks, ke (int32) and the key count."""
    ks: List[int] = []
    ke: List[int] = []
    cum = 0
    for ctx, rows, where in requests:
        assert rows <= ctx
        first = ctx - rows if where == "last" else max(0, ctx // 2 - rows // 2)
        for p in range(first, first + rows):
            ks.append(cum)
            ke.append(cum + p + 1)
        cum += ctx
    return (
        torch.tensor(ks, dtype=torch.int32, device=device),
        torch.tensor(ke, dtype=torch.int32, device=device),
        cum,
    )


def gen_inputs(rows: int, keys: int, seed: int = 0, device="cuda", heads: int = 32):
    """Random fp8 inputs of DeepGEMM's test magnitude; positive weights."""
    from deep_gemm.utils import per_custom_dims_cast_to_fp8

    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        rows, heads, HEAD_DIM, device=device, dtype=torch.bfloat16, generator=g
    )
    kv = torch.randn(keys, HEAD_DIM, device=device, dtype=torch.bfloat16, generator=g)
    weights = (
        torch.rand(rows, heads, device=device, dtype=torch.float32, generator=g) + 0.1
    )
    q_fp8 = q.to(torch.float8_e4m3fn)
    kv_fp8, kv_scale = per_custom_dims_cast_to_fp8(kv, (0,), False)
    return q_fp8, weights, kv_fp8.contiguous(), kv_scale.contiguous()


def ref_logits(q_fp8, weights, kv_fp8, kv_scale, ks, ke) -> torch.Tensor:
    """[rows, keys] fp32, -inf outside each row's range (DeepGEMM kernel)."""
    import deep_gemm

    return deep_gemm.fp8_mqa_logits(
        q_fp8, (kv_fp8, kv_scale), weights, ks, ke, clean_logits=True
    )


def f16_round_down(x: torch.Tensor) -> torch.Tensor:
    """fp16 toward -inf (the kernel's `cvt.rm.f16.f32`), as fp32."""
    h = x.to(torch.float16)  # round to nearest
    hf = h.to(torch.float32)
    # Where nearest rounded up, step one fp16 ulp down.
    down = torch.nextafter(h, torch.full_like(h, -math.inf)).to(torch.float32)
    return torch.where(hf > x, down, hf)


def expected_thresholds(
    ks_l, ke_l, logits_rows, blk, dense_bound, topk, num_keys, rows, rpc=8
):
    """Replicates the kernel's sample grid for block `blk` (rows rpc*blk ..
    rpc*blk + rpc - 1) and returns the expected T of each row from the fp32
    logits."""
    lo, hi, need = None, 0, False
    lens = []
    for i in range(rpc):
        r = blk * rpc + i
        if r < rows:
            s, e = min(max(ks_l[r], 0), num_keys), min(max(ke_l[r], 0), num_keys)
            e = max(e, s)
        else:
            s = e = 0
        lens.append((s, e - s))
        if e > s:
            lo = s if lo is None else min(lo, s)
            hi = max(hi, e)
            need |= (e - s) > dense_bound
    out = []
    if lo is None:
        return [math.inf] * rpc
    lo = lo // 4 * 4
    stride = sample_stride(hi - lo)
    base = lo + ((blk * 2654435761) % 2**32 >> 8) % stride
    num = -(-(hi - base) // stride) if base < hi else 0
    tiles = min(-(-num // TILE_KEYS), MAX_SAMPLE_TILES) if need else 0
    idx = torch.arange(tiles * TILE_KEYS, device=logits_rows.device)
    keys = base + idx * stride
    keys = keys[keys < num_keys // stride * stride]
    for i in range(rpc):
        s, m = lens[i]
        if m == 0:
            out.append(math.inf)
            continue
        if m <= dense_bound:
            out.append(-math.inf)
            continue
        valid = keys[(keys >= s) & (keys < s + m) & (keys < num_keys)]
        S = valid.numel()
        if S == 0:
            out.append(-math.inf)
            continue
        j = topk * S / m
        jp = math.ceil((1 + ALPHA * math.sqrt(2 / j)) * j)
        if jp > S or jp == 0:
            out.append(-math.inf)
            continue
        samp = logits_rows[i, valid]
        kth = torch.topk(samp, jp).values[-1]
        out.append(f16_round_down(kth.reshape(1)).item())
    return out


def check_cell(
    requests,
    heads: int = 32,
    rows_chunk: int = 1024,
    seed: int = 0,
    dense_bound=TOPK,
    max_blocks_checked=256,
    **kw,
):
    """Checks every row of the cell against DeepGEMM logits. The scores carry
    the e4m3 split error of q' (measured <= 6e-7 relative), so the score checks
    are tolerance checks and the threshold / count checks allow logits within
    that band of T."""
    from sglang.kernels.ops.attention.dsa.litetopk_prefill.fp8_indexer import (
        FP8_INDEXER_BLOCK_ROWS,
    )

    rpc = FP8_INDEXER_BLOCK_ROWS[heads]
    # The q' split error is absolute (a fraction of the row's score scale):
    # measured <= 1e-4 on scores of ~100; the band is 1e-5 x the row's max |logit|.
    rtol = 1e-5
    ks, ke, keys = make_ragged(requests)
    rows = ks.numel()
    q, w, kv, sf = gen_inputs(rows, keys, seed, heads=heads)
    res = fp8_indexer_candidates(
        q, w, kv, sf, ks, ke, TOPK, cap=CAP, dense_bound=dense_bound, **kw
    )
    res2 = fp8_indexer_candidates(
        q, w, kv, sf, ks, ke, TOPK, cap=CAP, dense_bound=dense_bound, **kw
    )
    torch.cuda.synchronize()
    assert torch.equal(res.counts, res2.counts) and torch.equal(
        res.thresholds, res2.thresholds
    ), "nondeterministic"
    assert int(res.overflow.sum()) == 0, "overflow flagged"
    assert bool((res.counts <= CAP).all())
    ks_l, ke_l = ks.tolist(), ke.tolist()
    lens = (ke - ks).clamp(min=0)
    rng = random.Random(seed)
    num_blocks = (rows + rpc - 1) // rpc
    checked_blocks = set(
        rng.sample(range(num_blocks), min(num_blocks, max_blocks_checked))
    )
    stats = {"count": res.counts.clone()}
    for r0 in range(0, rows, rows_chunk):
        r1 = min(rows, r0 + rows_chunk)
        logits = ref_logits(q[r0:r1], w[r0:r1], kv, sf, ks[r0:r1], ke[r0:r1])
        logits = logits[:, :keys]
        cnt = res.counts[r0:r1]
        thr = res.thresholds[r0:r1]
        n = r1 - r0
        ar = torch.arange(CAP, device="cuda")
        valid = ar[None, :] < cnt[:, None]
        idx = res.indices[r0:r1].clamp(min=0).long()
        gidx = (ks[r0:r1, None].long() + idx).clamp(max=keys - 1)
        ref_at = logits.gather(1, gidx)
        # (2) stored scores equal the reference logits within the band
        got = res.scores[r0:r1]
        rowscale = (
            logits.abs()
            .nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            .amax(dim=1)
            .clamp(min=1.0)
        )
        bad = valid & ((got - ref_at).abs() > rtol * rowscale[:, None])
        assert not bad.any(), (
            f"score mismatch rows {r0}..{r1}: {bad.sum().item()} entries"
        )
        # indices inside the row's range and unique
        assert bool(((idx < lens[r0:r1, None]) | ~valid).all()), "index out of range"
        # (3) count == number of logits >= T in range (T = -inf rows count only their range)
        col = torch.arange(keys, device="cuda")
        inrange = (col[None, :] >= ks[r0:r1, None]) & (col[None, :] < ke[r0:r1, None])
        band = rtol * rowscale
        ge_hi = (inrange & (logits >= (thr - band)[:, None])).sum(1)
        ge_lo = (inrange & (logits >= (thr + band)[:, None])).sum(1)
        ok = (cnt >= ge_lo.int()) & (cnt <= ge_hi.int())
        assert bool(ok.all()), (
            f"count outside the tolerance band rows {r0}..{r1}: {(~ok).sum().item()}"
        )
        del inrange
        # (1) the top-k is contained: T <= the k-th largest in range
        k = int(min(TOPK, lens[r0:r1].max().item()))
        if k > 0:
            top = torch.topk(logits, k, dim=1).values
            kk = (lens[r0:r1].clamp(max=TOPK) - 1).clamp(min=0)
            kth = top.gather(1, kk[:, None].long()).squeeze(1)
            ok = (thr <= kth + rtol * rowscale) | (lens[r0:r1] == 0)
            assert bool(ok.all()), (
                f"threshold above k-th value in rows {r0}..{r1}: {(~ok).sum().item()}"
            )
        # short rows keep everything
        short = (lens[r0:r1] <= dense_bound) & (lens[r0:r1] > 0)
        assert torch.equal(cnt[short], lens[r0:r1][short].int())
        assert bool(torch.isneginf(thr[short]).all())
        assert bool((cnt[lens[r0:r1] == 0] == 0).all())
        # (4) T equals the fp16-rounded-down j'-th largest sample (sampled blocks only)
        for blk in range(r0 // rpc, (r1 + rpc - 1) // rpc):
            if blk not in checked_blocks:
                continue
            exp = expected_thresholds(
                ks_l,
                ke_l,
                logits[blk * rpc - r0 : blk * rpc - r0 + rpc],
                blk,
                dense_bound,
                TOPK,
                keys,
                rows,
                rpc,
            )
            for i in range(rpc):
                r = blk * rpc + i
                if r >= rows:
                    break
                got_t = thr[r - r0].item()
                if math.isfinite(exp[i]):
                    assert (
                        abs(got_t - exp[i])
                        <= 2 * rtol * rowscale[r - r0].item() + 0.0625
                    ), f"row {r}: T {got_t} vs {exp[i]}"
                else:
                    assert got_t == exp[i] or (
                        math.isnan(got_t) and math.isnan(exp[i])
                    ), f"row {r}: T {got_t} != expected {exp[i]}"
        del logits
    return res, stats


CELLS = {
    "small_last_64k_r1024": [(64 * 1024, 1024, "last")],
    "short_rows_mixed": [
        (4096, 512, "last"),
        (1024, 1024, "last"),
        (16384, 512, "mid"),
    ],
    # single requests at the 129- and 257-key sample strides
    "long_1m_r256": [(1024 * 1024 - 4096, 256, "last")],
    "long_1280k_r128": [(1280 * 1024, 128, "last")],
    "mixed4_r1024": [
        (1024 * 1024, 256, "last"),
        (256 * 1024, 256, "last"),
        (128 * 1024, 256, "last"),
        (64 * 1024, 256, "last"),
    ],
}


@pytest.mark.skipif(not is_fp8_indexer_supported(), reason="SM100 only")
@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("cell", list(CELLS))
def test_fp8_indexer_candidates(cell, heads):
    check_cell(CELLS[cell], heads)


@pytest.mark.skipif(not is_fp8_indexer_supported(), reason="SM100 only")
@pytest.mark.parametrize("heads", HEADS)
def test_fp8_indexer_tail_split(heads):
    """The blocks of the last, partial wave are scanned in pieces on several CTAs: any grid size and piece count
    gives the same candidates, counts, thresholds and overflow flags as whole blocks on one CTA."""
    ks, ke, keys = make_ragged(CELLS["mixed4_r1024"])
    q, w, kv, sf = gen_inputs(ks.numel(), keys, 3, heads=heads)
    ref = None
    for num_ctas, tail_chunks in (
        (1, 1),
        (37, 2),
        (37, 8),
        (100, 4),
        (100, 32),
        (148, 1),
    ):
        r = fp8_indexer_candidates(
            q,
            w,
            kv,
            sf,
            ks,
            ke,
            TOPK,
            cap=CAP,
            num_ctas=num_ctas,
            tail_chunks=tail_chunks,
        )
        valid = torch.arange(CAP, device="cuda")[None, :] < r.counts[:, None]
        idx = torch.where(valid, r.indices, torch.full_like(r.indices, -1)).sort(dim=1)
        got = (r.counts, r.thresholds, r.overflow, idx.values)
        if ref is None:
            ref = got
        else:
            for a, b in zip(ref, got):
                assert torch.equal(a, b), f"num_ctas={num_ctas}"


@pytest.mark.skipif(not is_fp8_indexer_supported(), reason="SM100 only")
@pytest.mark.parametrize("heads", HEADS)
@pytest.mark.parametrize("cell", ["short_rows_mixed", "mixed4_r1024"])
def test_fp8_indexer_topk(cell, heads):
    """The fp8 indexer + fp32 top-k pair returns the top-k keys of DeepGEMM's logits, offset by
    `out_offsets`, -1 past min(topk, len); keys whose logits lie within the scores' split error
    (1e-5 of the row's max |logit|) of the k-th may swap."""
    ks, ke, keys = make_ragged(CELLS[cell])
    q, w, kv, sf = gen_inputs(ks.numel(), keys, 5, heads=heads)
    out = fp8_indexer_topk(q, w, kv, sf, ks, ke, TOPK, out_offsets=ks)
    torch.cuda.synchronize()
    lens = (ke - ks).long()
    for r0 in range(0, ks.numel(), 256):
        r1 = min(r0 + 256, ks.numel())
        logits = ref_logits(q[r0:r1], w[r0:r1], kv, sf, ks[r0:r1], ke[r0:r1])
        o = out[r0:r1].long()
        n = lens[r0:r1].clamp(max=TOPK)
        valid = torch.arange(TOPK, device="cuda")[None, :] < n[:, None]
        assert (o[~valid] == -1).all(), "padding past min(topk, len) is not -1"
        inside = (o >= ks[r0:r1, None]) & (o < ke[r0:r1, None])
        assert (inside | ~valid).all(), "selected key outside the row's range"
        srt = torch.where(valid, o, torch.full_like(o, -1)).sort(dim=1).values
        assert not ((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] >= 0)).any(), (
            "duplicate selection"
        )
        got = torch.where(
            valid,
            logits.gather(1, o.clamp(min=0)),
            torch.full_like(logits[:, :1], float("inf")),
        )
        kth = logits.topk(TOPK, dim=1).values.gather(1, (n - 1).clamp(min=0)[:, None])
        tol = 1e-5 * logits.masked_fill(~torch.isfinite(logits), 0).abs().amax(
            1, keepdim=True
        )
        assert (got >= kth - tol).all(), "selected key below the k-th logit"
        above = (logits > kth + tol).sum(1)
        assert torch.equal(((got > kth + tol) & valid).sum(1), above), (
            "key above the k-th logit missed"
        )
        del logits


def test_sample_grid_cap():
    """Up to 2M keys of span the strided sample fits the 32 sample tiles (8192 samples per row) that the per-CTA
    sample-code scratch (kMaxSampleTiles) holds, so no sample is dropped."""
    for span in list(range(1, 70000, 977)) + list(range(2**19 - 2048, 2**21, 509)):
        stride = sample_stride(span)
        assert stride % 2 == 1
        assert -(-span // stride) <= MAX_SAMPLE_TILES * TILE_KEYS, (span, stride)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
