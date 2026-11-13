"""Correctness tests for the prefill DSA candidate top-k (``dsa/litetopk_prefill/topk_candidates_fp32.cuh``
and ``dsa/litetopk_prefill/topk_candidates_bf16.cuh``, both behind ``topk_candidates``).

Rows are synthetic candidate slabs: a long score row, a threshold T taken as the
fp16 floor of the (1 + alpha) k-th largest score, the candidates ``>= T`` in key
order. The exact fp32 top-k set must come back (ties at the k-th value may be
swapped), ``-1`` exactly past ``min(k, count)``.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsa.litetopk_prefill.topk_candidates import (
    TOPK_CANDIDATES_CAP,
    topk_candidates,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

INT_MAX = 2**31 - 1


def _requires_sm100():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        pytest.skip(
            "Blackwell-only kernel (cp.async.bulk + 16-bit packed histogram tuning)"
        )


def make_rows(rows, k, counts, gen, cap=TOPK_CANDIDATES_CAP, thr_mode="band"):
    """Slab rows with the given candidate counts; scores ~ N(-60, 5) like GLM logits."""
    scores = torch.full((rows, cap), -float("inf"), device="cuda")
    indices = torch.full((rows, cap), -1, dtype=torch.int32, device="cuda")
    thr = torch.empty(rows, device="cuda")
    for r in range(rows):
        n = int(counts[r])
        s = torch.randn(n, device="cuda", generator=gen) * 5 - 60
        if thr_mode == "band" and n > 0:
            t = s.min().half().float()  # fp16-representable, <= every candidate
        else:  # "neg_inf"
            t = -float("inf")
        scores[r, :n] = s
        indices[r, :n] = torch.sort(
            torch.randperm(1 << 20, device="cuda", generator=gen)[:n]
        ).values.to(torch.int32)
        thr[r] = t
    count = torch.tensor(counts, dtype=torch.int32, device="cuda")
    return scores, indices, count, thr


def positions_of(indices, count, sel, valid):
    """Key indices -> positions in the live prefix of each row (-1 when absent)."""
    R, C = indices.shape
    live = torch.arange(C, device=sel.device)[None, :] < count[:, None]
    pad = torch.where(live, indices, torch.full_like(indices, INT_MAX))
    sorted_idx, perm = pad.sort(dim=1)
    q = torch.where(valid, sel, torch.zeros_like(sel))
    loc = torch.searchsorted(sorted_idx, q.contiguous()).clamp(max=C - 1)
    found = (sorted_idx.gather(1, loc) == q) & valid
    return torch.where(found, perm.gather(1, loc), torch.full_like(loc, -1))


def check(scores, indices, count, k, out):
    """Tie-aware exactness: the selected scores, sorted, equal the exact top-k scores, sorted."""
    R, C = scores.shape
    n = count.clamp(max=k)
    col = torch.arange(k, device=out.device)[None, :]
    valid = col < n[:, None]
    assert not (out[valid] == -1).any(), "-1 inside the selected prefix"
    assert (out[~valid] == -1).all(), "padding past min(k, count) is not -1"
    pos = positions_of(indices, count, out, valid)
    assert not (valid & (pos < 0)).any(), (
        "selected index not among the row's candidates"
    )
    ps, _ = torch.where(valid, pos, torch.full_like(pos, INT_MAX)).sort(dim=1)
    assert not ((ps[:, 1:] == ps[:, :-1]) & (ps[:, 1:] != INT_MAX)).any(), (
        "duplicate selection"
    )
    got = torch.where(
        valid,
        scores.gather(1, pos.clamp(min=0)),
        torch.full_like(pos, 0, dtype=scores.dtype) - float("inf"),
    )
    ref = torch.topk(scores, k, dim=1).values
    ref = torch.where(valid, ref, torch.full_like(ref, -float("inf")))
    torch.testing.assert_close(
        got.sort(dim=1, descending=True).values, ref, rtol=0, atol=0
    )


def run_all(scores, indices, count, thr, k, with_offsets=True):
    R = scores.shape[0]
    off = (
        (torch.arange(R, dtype=torch.int32, device="cuda") * 1000)
        if with_offsets
        else None
    )
    out = torch.full((R, k), -7, dtype=torch.int32, device="cuda")
    topk_candidates(scores, indices, count, thr, out, out_offsets=off)
    torch.cuda.synchronize()
    if off is not None:
        out = torch.where(out >= 0, out - off[:, None], out)
    check(scores, indices, count, k, out)


def bf16_view(scores, thr):
    """The same rows as a bf16 slab: RNE scores, T rounded toward -inf so no candidate is lost."""
    sb = scores.to(torch.bfloat16)
    tb = thr.to(torch.bfloat16).float()
    down = torch.nextafter(
        tb.to(torch.bfloat16),
        torch.tensor(-float("inf"), dtype=torch.bfloat16, device=tb.device),
    ).float()
    return sb, torch.where(tb > thr, down, tb)


def run_bf16(scores, indices, count, thr, k, with_offsets=True):
    R = scores.shape[0]
    sb, tb = bf16_view(scores, thr)
    # a bf16 slab's T = -inf row holds every key in key order: the index of slot s is s (the slab is not read)
    slot = torch.arange(indices.shape[1], dtype=torch.int32, device="cuda")[None, :]
    indices = torch.where((tb == -float("inf"))[:, None], slot, indices)
    off = (
        (torch.arange(R, dtype=torch.int32, device="cuda") * 1000)
        if with_offsets
        else None
    )
    out = torch.full((R, k), -7, dtype=torch.int32, device="cuda")
    topk_candidates(sb, indices, count, tb, out, out_offsets=off)
    torch.cuda.synchronize()
    if off is not None:
        out = torch.where(out >= 0, out - off[:, None], out)
    try:
        check(sb.float(), indices, count, k, out)
    except AssertionError as e:
        raise AssertionError(f"bf16: {e}") from e


@pytest.mark.parametrize("k", [512, 1024, 2048])
def test_random_rows(k):
    _requires_sm100()
    gen = torch.Generator(device="cuda").manual_seed(k)
    counts = torch.randint(
        k + 1,
        TOPK_CANDIDATES_CAP + 1,
        (48,),
        generator=torch.Generator().manual_seed(k),
    ).tolist()
    counts[:4] = [k + 1, 4096, 8192, 8193]  # stage boundaries of the 2-buffer ring
    rows = make_rows(48, k, counts, gen)
    run_all(*rows, k)
    run_bf16(
        *rows, k
    )  # bf16 rounding makes wide exact ties: the bf16 kernel must order them


@pytest.mark.parametrize("k", [512, 2048])
def test_edges(k):
    _requires_sm100()
    gen = torch.Generator(device="cuda").manual_seed(7)
    cap = TOPK_CANDIDATES_CAP
    counts = [0, 1, k - 1, k, k + 1, cap, cap - 1, 8192, 8193, 4096]
    scores, indices, count, thr = make_rows(len(counts), k, counts, gen)
    run_all(scores, indices, count, thr, k)
    run_bf16(scores, indices, count, thr, k)
    # T = -inf rows (short rows keep every key)
    scores, indices, count, thr = make_rows(
        4, k, [k + 5, 5000, cap, 1], gen, thr_mode="neg_inf"
    )
    run_all(scores, indices, count, thr, k)
    run_bf16(scores, indices, count, thr, k)


@pytest.mark.parametrize("k", [512, 2048])
def test_ties(k):
    """Copies of the k-th value straddling the boundary, within the 256-slot tie buffer: the warp
    ballot path (<= 64 ties) and the radix path past it, and +0 / -0 copies (one value, two order
    keys)."""
    _requires_sm100()
    gen = torch.Generator(device="cuda").manual_seed(11)
    counts = [8000, 8000, 9000, 3000]
    scores, indices, count, thr = make_rows(len(counts), k, counts, gen)
    for r, m in ((0, 40), (1, 200), (2, 120)):
        s = scores[r, : counts[r]]
        vk = s.sort(descending=True).values[k - 1]
        s[s.argsort(descending=True)[k - m // 2 : k - m // 2 + m]] = vk
        thr[r] = s.min().half().float()
    s = scores[
        3, :3000
    ]  # k - 100 scores above, then 200 zeros of both signs, the rest below
    s[: k - 100] = 5.0
    s[k - 100 : k + 100] = 0.0
    s[k - 100 : k + 100 : 2] = -0.0
    thr[3] = s.min().half().float()
    run_all(scores, indices, count, thr, k)
    run_bf16(scores, indices, count, thr, k)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
