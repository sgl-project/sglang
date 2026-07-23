"""Tests for the CuTe DSL deterministic DSA top-k kernel."""

import sys
import zlib

import pytest
import torch

from sglang.jit_kernel.utils import get_jit_cuda_arch, is_hip_runtime
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from sglang.jit_kernel.cutedsl_topk import (  # noqa: E402
    HAVE_CUTE_TOPK,
    cute_topk_func,
    supports,
)

NEG_INF = float("-inf")


def _skip_if_unsupported():
    if is_hip_runtime() or get_jit_cuda_arch().major < 9 or not HAVE_CUTE_TOPK:
        pytest.skip("SM90+ with nvidia-cutlass-dsl required")


def _make_case(kind, batch, width, topk, seed, with_row_starts):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if kind == "randn":
        score = torch.randn(batch, width, device="cuda", generator=g)
    elif kind == "relu":
        # Realistic DSA indexer logits: sum_h relu(q.k) * w produces an
        # exact-0.0 tie plateau covering most of the row.
        score = (
            torch.relu(torch.randn(batch, width, device="cuda", generator=g) - 1.28)
            * 11.3
        )
    elif kind == "quant":
        score = torch.randint(0, 64, (batch, width), device="cuda", generator=g).float()
    elif kind == "neginf":
        score = torch.randn(batch, width, device="cuda", generator=g)
        mask = torch.rand(batch, width, device="cuda", generator=g) < 0.3
        score = score.masked_fill(mask, NEG_INF)
        score[0, :] = NEG_INF
        if batch > 3:
            # all -inf with a nonzero window (row 0's length is zeroed below):
            # n_valid == 0 -> all -1 output
            score[3, :] = NEG_INF
    else:
        raise ValueError(kind)
    if with_row_starts:
        row_starts = torch.randint(
            0, max(width // 4, 1), (batch,), device="cuda", generator=g
        ).to(torch.int32)
    else:
        row_starts = None
    max_len = width - (
        row_starts.to(torch.int64)
        if row_starts is not None
        else torch.zeros(batch, dtype=torch.int64, device="cuda")
    )
    lengths = (torch.rand(batch, device="cuda", generator=g) * max_len.float()).to(
        torch.int32
    )
    # edge lengths
    lengths[0] = 0
    if batch > 1:
        lengths[1] = torch.minimum(
            torch.tensor(topk, dtype=torch.int32, device="cuda"),
            max_len[1].to(torch.int32),
        )
    if batch > 2:
        lengths[2] = max_len[2].to(torch.int32)
    return score, lengths, row_starts


def _check_contract(score, lengths, topk, idx, row_starts):
    """Tie-robust exact top-k check.

    Per row: index range/uniqueness, count == min(topk, n_valid), selected
    score multiset bitwise-equals the reference top multiset (unique answer
    even under exact fp32 ties), canonical order (strictly ascending
    indices), -1 tail-packed.
    """
    batch, _ = score.shape
    starts = (
        torch.zeros(batch, dtype=torch.int64, device=score.device)
        if row_starts is None
        else row_starts.to(torch.int64)
    )
    for r in range(batch):
        ln = int(lengths[r])
        st = int(starts[r])
        row = idx[r].to(torch.int64)
        sel = row[row >= 0]
        n_out = sel.numel()
        assert bool((row >= -1).all()) and bool((row < ln).all())
        assert torch.unique(sel).numel() == n_out
        w = score[r, st : st + ln] if ln > 0 else score[r, :0]
        finite = w != NEG_INF
        n_valid = int(finite.sum())
        assert n_out == min(topk, n_valid)
        neg = row == -1
        if bool(neg.any()):
            first = int(torch.argmax(neg.int()))
            assert bool(neg[first:].all())  # -1 strictly tail-packed
        if n_out == 0:
            continue
        got = torch.sort(w[sel], descending=True).values
        ref = torch.sort(w[finite], descending=True).values[:n_out]
        assert torch.equal(got.view(torch.int32), ref.view(torch.int32))
        # canonical order: strictly ascending indices (the scan-emit output
        # stage's slot order; part of the determinism contract)
        assert bool((row[1:n_out] > row[: n_out - 1]).all())
        # smallest-index tie subset at the k-th boundary: among candidates
        # whose score bitwise-equals the k-th (minimum selected) score, the
        # selected ones must be exactly the smallest indices
        sb = w[sel].view(torch.int32).to(torch.int64)
        uk = torch.where(sb < 0, ~sb & 0xFFFFFFFF, sb | 0x80000000)
        kth_bits = sb[torch.argmin(uk)]
        wb = w.view(torch.int32).to(torch.int64)
        finite_idx = finite.nonzero().flatten()
        tie_all = finite_idx[wb[finite_idx] == kth_bits]
        tie_sel = sel[sb == kth_bits]
        assert torch.equal(torch.sort(tie_sel).values, tie_all[: tie_sel.numel()])


@pytest.mark.parametrize("kind", ["randn", "relu", "quant", "neginf"])
@pytest.mark.parametrize("with_row_starts", [False, True])
def test_cutedsl_topk_contract(kind, with_row_starts):
    _skip_if_unsupported()
    topk = 2048
    score, lengths, row_starts = _make_case(
        kind,
        64,
        16384,
        topk,
        seed=zlib.crc32(f"{kind}:{with_row_starts}".encode()) & 0xFFFF,
        with_row_starts=with_row_starts,
    )
    assert supports(score, topk)
    out = cute_topk_func(score, lengths, topk, row_starts)
    assert out.shape == (64, topk) and out.dtype == torch.int32
    _check_contract(score, lengths, topk, out, row_starts)


@pytest.mark.parametrize("topk", [512, 2048])
def test_cutedsl_topk_deterministic(topk):
    _skip_if_unsupported()
    # exact-tie-heavy input: the regime where arrival-order kernels return a
    # different index set run to run. NB: topk=512 exercises the index-sort
    # output stage, topk=2048 the scan-emit stage (see _pick_scanemit).
    score, lengths, _ = _make_case(
        "relu", 64, 16384, topk, seed=7, with_row_starts=False
    )
    ref = cute_topk_func(score, lengths, topk)
    for _ in range(10):
        out = cute_topk_func(score, lengths, topk)
        assert torch.equal(out, ref)  # bitwise, including slot order


@pytest.mark.parametrize("kind", ["relu", "quant"])
def test_cutedsl_topk_stage_equivalence(kind, monkeypatch):
    """The two output stages emit the identical canonical output
    bit-for-bit (ascending index), including on tie-heavy inputs and
    short/empty rows."""
    _skip_if_unsupported()
    topk = 2048
    score, lengths, row_starts = _make_case(
        kind, 64, 16384, topk, seed=11, with_row_starts=True
    )
    monkeypatch.setenv("SGLANG_DSA_TOPK_CUTEDSL_STAGE", "scanemit")
    a = cute_topk_func(score, lengths, topk, row_starts)
    monkeypatch.setenv("SGLANG_DSA_TOPK_CUTEDSL_STAGE", "sort")
    b = cute_topk_func(score, lengths, topk, row_starts)
    assert torch.equal(a, b)
    _check_contract(score, lengths, topk, a, row_starts)


def test_cutedsl_topk_cuda_graph():
    _skip_if_unsupported()
    topk = 2048
    score, lengths, _ = _make_case(
        "randn", 32, 8192, topk, seed=3, with_row_starts=False
    )
    ref_warm = cute_topk_func(score, lengths, topk)  # eager warmup (compiles)
    torch.cuda.synchronize()
    del ref_warm
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        cute_topk_func(score, lengths, topk)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        out = cute_topk_func(score, lengths, topk)
    g = torch.Generator(device="cuda").manual_seed(9)
    score.copy_(torch.randn(score.shape, device="cuda", generator=g))
    graph.replay()
    torch.cuda.synchronize()
    ref = cute_topk_func(score, lengths, topk)
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def test_cutedsl_topk_empty():
    _skip_if_unsupported()
    out = cute_topk_func(
        torch.empty(0, 128, device="cuda"),
        torch.empty(0, dtype=torch.int32, device="cuda"),
        2048,
    )
    assert out.shape == (0, 2048)


def test_cutedsl_topk_survcap_fallback(monkeypatch):
    """c1 > SURV_CAP forces the full-row-rescan refinement fallback; the
    result must be bit-identical to the staged path on both output stages."""
    _skip_if_unsupported()
    topk = 2048
    # quantized scores: every coarse histogram bucket holds thousands of
    # candidates, so SURVCAP=64 cannot stage bucket b* -> fallback path
    score, lengths, _ = _make_case(
        "quant", 8, 16384, topk, seed=5, with_row_starts=False
    )
    ref = cute_topk_func(score, lengths, topk)
    monkeypatch.setenv("SGLANG_DSA_TOPK_CUTEDSL_SURVCAP", "64")
    for stage in ("scanemit", "sort"):
        monkeypatch.setenv("SGLANG_DSA_TOPK_CUTEDSL_STAGE", stage)
        out = cute_topk_func(score, lengths, topk)
        assert torch.equal(out, ref)


def test_cutedsl_topk_envelope_errors(monkeypatch):
    """The wrapper enforces the documented envelope with actionable errors,
    including on empty shapes, and validates its env overrides."""
    _skip_if_unsupported()
    score = torch.randn(2, 128, device="cuda")
    lengths = torch.full((2,), 128, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError):
        cute_topk_func(score, lengths, 4097)  # topk > cap
    with pytest.raises(ValueError):
        cute_topk_func(score, lengths, -1)  # negative topk
    with pytest.raises(ValueError):  # L >= 2**24 rejected even at B=0
        cute_topk_func(
            torch.empty(0, 1 << 24, device="cuda"),
            torch.empty(0, dtype=torch.int32, device="cuda"),
            2048,
        )
    with pytest.raises(ValueError):
        cute_topk_func(score, lengths[:1], 128)  # undersized lengths
    with pytest.raises(ValueError):
        cute_topk_func(score, lengths, 128, row_starts=lengths[:1])
    with pytest.raises(ValueError):
        cute_topk_func(score.half(), lengths, 128)  # wrong dtype
    monkeypatch.setenv("SGLANG_DSA_TOPK_CUTEDSL_SURVCAP", "zero")
    with pytest.raises(ValueError):
        cute_topk_func(score, lengths, 128)
    monkeypatch.delenv("SGLANG_DSA_TOPK_CUTEDSL_SURVCAP")
    monkeypatch.setenv("SGLANG_DSA_TOPK_CUTEDSL_STAGE", "bogus")
    with pytest.raises(ValueError):
        cute_topk_func(score, lengths, 128)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
