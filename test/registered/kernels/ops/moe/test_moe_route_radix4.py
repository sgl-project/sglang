# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ROCm radix-4 K3 router.

Two references, because the kernel owes two different things. A pure-torch fp32
oracle spells out the contract -- what the weights are, where a NaN ranks, how a
row is ordered -- and covers the shapes and edge cases. aiter is the router this
one stands in for on decode-sized batches while larger ones still go to it, so
the test_*_aiter cases hold the kernel to matching it column for column, ties
included; anything less and a token's routing would depend on the batch size.

Run:  pytest test/registered/kernels/ops/moe/test_moe_route_radix4.py -v
"""

import sys

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

if not is_hip():
    pytest.skip("The radix-4 router is the ROCm path.", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("Requires a GPU.", allow_module_level=True)

from sglang.kernels.ops.moe import moe_route_radix4

if not moe_route_radix4.supported_hardware():
    pytest.skip("The kernel targets gfx942/gfx950.", allow_module_level=True)

# Deliberately not gated on available(): that reports a kernel which failed to
# build as merely unavailable, so serving can fall back to aiter, and a file that
# skipped on it would report a kernel that stopped compiling as a green run. On
# hardware the kernel targets it has to build, so build it here and let the
# toolchain error reach the report.
moe_route_radix4.build()

try:
    from aiter import biased_grouped_topk as aiter_biased_grouped_topk
except ImportError:
    aiter_biased_grouped_topk = None

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

NUM_EXPERTS = 896
TOPK = 16
# The kernel keys a NaN to 0, which is below every number including -inf, so -inf
# stands in for it here. The two part company only on a row that ranks an expert
# at -inf as well, which needs an infinite bias and is outside the contract.
NAN_RANK = float("-inf")

# kAiterTieLaneRank from route_radix4_hip.cuh. Kept as a second copy on purpose:
# test_route_radix4_tie_order_is_aiters checks it against aiter directly, so a
# change on aiter's side fails here instead of drifting silently in the kernel.
_TIE_LANE_RANK = [
    56, 57, 58, 59, 63, 62, 61, 60, 52, 53, 54, 55, 51, 50, 49, 48,
    40, 41, 42, 43, 47, 46, 45, 44, 36, 37, 38, 39, 35, 34, 33, 32,
    24, 25, 26, 27, 31, 30, 29, 28, 20, 21, 22, 23, 19, 18, 17, 16,
    8, 9, 10, 11, 15, 14, 13, 12, 4, 5, 6, 7, 3, 2, 1, 0,
]  # fmt: skip


def _tie_priority(expert):
    """Where aiter's wave64 walk reaches an expert; a bijection onto [0, 896)."""
    group = expert >> 2
    lane, bank = group & 63, group >> 6
    rank = _TIE_LANE_RANK[lane]
    group_rank = rank * 3 + bank if rank < 32 else 96 + (rank - 32) * 4 + bank
    return (group_rank << 2) + (expert & 3)


TIE_PRIORITY = [_tie_priority(e) for e in range(NUM_EXPERTS)]
assert len(set(TIE_PRIORITY)) == NUM_EXPERTS, "tie priority is not a permutation"
# Experts in tie order, so a stable descending sort over these columns breaks ties
# the way the kernel does.
_BY_PRIORITY = torch.tensor(
    sorted(range(NUM_EXPERTS), key=TIE_PRIORITY.__getitem__), device="cuda"
)


def _oracle(scores, bias, renormalize, scaling):
    """Contract, from route_radix4_hip.cuh: bias ranks only and the emitted
    weight stays bias-free, a NaN ranks below every number so it can never win,
    ties follow aiter's walk, renormalize divides by the winners' sum (guarded to
    1 when that sum is non-positive) before scaling. Winners are emitted highest
    ranking value first, equal values in that same tie order."""
    s = torch.sigmoid(scores.float())
    biased = s + bias.float()
    biased = torch.where(torch.isnan(biased), torch.full_like(biased, NAN_RANK), biased)
    # Reorder the columns into tie order first, so that a stable descending sort
    # leaves equal values in it, then map the winners back to expert ids.
    by_priority = biased[:, _BY_PRIORITY]
    picked = torch.argsort(by_priority, dim=-1, descending=True, stable=True)[:, :TOPK]
    ranked = _BY_PRIORITY[picked]
    w = s.gather(1, ranked)
    if renormalize:
        total = w.sum(-1, keepdim=True)
        w = w / torch.where(total > 0, total, torch.ones_like(total))
    return w * scaling, ranked.to(torch.int32)


def _assert_matches_oracle(scores, bias, renormalize=True, scaling=2.5):
    ref_w, ref_ids = _oracle(scores, bias, renormalize, scaling)
    w, ids = moe_route_radix4.route_radix4(scores, bias, TOPK, renormalize, scaling)
    # Column by column: the position a winner lands in is part of the contract.
    assert torch.equal(ids, ref_ids)
    # The kernel's sigmoid is an approximate hardware sequence (matched to
    # aiter's), whose last bits differ from torch's exact sigmoid.
    torch.testing.assert_close(w, ref_w, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("m", [1, 3, 32, 255, 256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("renormalize", [False, True])
def test_route_radix4_matches_oracle(m, dtype, renormalize):
    generator = torch.Generator(device="cuda").manual_seed(1000 + m)
    # A padded row, so the kernel is exercised on a non-contiguous stride too.
    backing = torch.randn(
        (m, NUM_EXPERTS + 37), dtype=dtype, device="cuda", generator=generator
    )
    scores = backing[:, 19 : 19 + NUM_EXPERTS]
    bias = torch.randn(NUM_EXPERTS, dtype=dtype, device="cuda", generator=generator)
    assert scores.stride(0) == NUM_EXPERTS + 37
    _assert_matches_oracle(scores, bias, renormalize)


@pytest.mark.parametrize("renormalize", [False, True])
def test_route_radix4_ties(renormalize):
    """Only the tie rule decides these, so an arrival-order compaction or a
    mis-ranked tie shows up immediately."""
    bias = torch.zeros(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")

    # Every key identical: the search never runs a round and the whole field is
    # a tie, which is the case an early-exit shortcut is most likely to miss.
    _assert_matches_oracle(
        torch.zeros(4, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda"),
        bias,
        renormalize,
    )

    plateau = torch.full((4, NUM_EXPERTS), 0.25, dtype=torch.bfloat16, device="cuda")
    plateau[:, 7] = 2.0
    plateau[:, 300] = 2.0
    plateau[:, 800] = 1.5
    _assert_matches_oracle(plateau, bias, renormalize)

    # A tied set too small to fill the quota on its own, spread across the
    # block's waves and across each thread's register slots.
    for seed in range(4):
        generator = torch.Generator().manual_seed(seed)
        picks = torch.randperm(NUM_EXPERTS, generator=generator)[:40]
        scores = torch.full((1, NUM_EXPERTS), -4.0, dtype=torch.bfloat16, device="cuda")
        scores[0, picks] = 4.0
        _assert_matches_oracle(scores, bias, renormalize)


def test_route_radix4_nan_never_wins():
    bias = torch.zeros(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    scores = torch.randn(4, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    scores[:, 100] = float("nan")
    scores[:, 500] = float("nan")
    # NaN sorts above every finite key under a raw monotone bit map, so without
    # the floor these two would be picked before the real winner.
    scores[:, 101] = 5.0
    ids = moe_route_radix4.route_radix4(scores, bias, TOPK, True, 2.5)[1]
    assert not bool(((ids == 100) | (ids == 500)).any())
    _assert_matches_oracle(scores, bias)


def test_route_radix4_extremes():
    """Saturated sigmoids: the low key bits carry no information, so the search
    has to lean on the shared-prefix skip and on the tie rule."""
    extreme = torch.linspace(
        -90, 90, NUM_EXPERTS, dtype=torch.float32, device="cuda"
    ).repeat(4, 1)
    for dtype in (torch.bfloat16, torch.float32):
        bias = torch.zeros(NUM_EXPERTS, dtype=dtype, device="cuda")
        _assert_matches_oracle(extreme.to(dtype), bias)


def test_route_radix4_saturated_row():
    """Every sigmoid underflows to zero, so the renorm divisor is zero and only
    the guard keeps the row from coming back as inf or NaN."""
    bias = torch.zeros(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    scores = torch.full((2, NUM_EXPERTS), -200.0, dtype=torch.float32, device="cuda")
    scores[0, :16] = -120.0
    w, _ = moe_route_radix4.route_radix4(scores, bias, TOPK, True, 2.5)
    assert torch.equal(w, torch.zeros_like(w))
    _assert_matches_oracle(scores, bias)


def test_route_radix4_reproducible():
    """Nothing the compaction races over reaches the output: the winner set is
    settled by rank, the emitted order is settled by rank in the epilogue, and
    the renorm divisor is reduced over threads. So a whole row repeats exactly,
    column positions included."""
    scores = torch.randn(512, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")

    first_w, first_ids = moe_route_radix4.route_radix4(scores, bias, TOPK, True, 2.5)
    for _ in range(16):
        w, ids = moe_route_radix4.route_radix4(scores, bias, TOPK, True, 2.5)
        assert torch.equal(ids, first_ids)
        assert torch.equal(w, first_w)


def _aiter_route(scores, bias, renormalize=True, scaling=2.5):
    w = torch.empty((scores.shape[0], TOPK), dtype=torch.float32, device="cuda")
    ids = torch.empty((scores.shape[0], TOPK), dtype=torch.int32, device="cuda")
    aiter_biased_grouped_topk(scores, bias, w, ids, 1, 1, renormalize, scaling)
    return w, ids


requires_aiter = pytest.mark.skipif(
    aiter_biased_grouped_topk is None, reason="aiter is not installed"
)


@requires_aiter
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("renormalize", [False, True])
def test_route_radix4_matches_aiter(dtype, renormalize):
    """The kernel stands in for the aiter router on decode-sized batches while
    larger ones keep going to aiter, so a row has to agree with it column by
    column, ties included -- otherwise a token's routing would depend on the batch
    size. bf16 quantization makes exact ties common enough that plain random
    scores reach them."""
    generator = torch.Generator(device="cuda").manual_seed(7)
    scores = torch.randn(
        1024, NUM_EXPERTS, dtype=dtype, device="cuda", generator=generator
    )
    bias = torch.randn(NUM_EXPERTS, dtype=dtype, device="cuda", generator=generator)

    ref_w, ref_ids = _aiter_route(scores, bias, renormalize)
    w, ids = moe_route_radix4.route_radix4(scores, bias, TOPK, renormalize, 2.5)
    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(w, ref_w, rtol=1e-5, atol=1e-6)


@requires_aiter
@pytest.mark.parametrize("pool", [17, 24, 40, 128, 600, NUM_EXPERTS])
def test_route_radix4_tie_order_is_aiters(pool):
    """kAiterTieLaneRank is read off aiter rather than derived, so pin it: rows
    whose whole top-k is settled by the tie rule, at pool sizes that put the cutoff
    inside the tied set. A wrong entry either moves an expert across that cutoff or
    swaps two columns, and both land here."""
    bias = torch.zeros(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    generator = torch.Generator().manual_seed(pool)
    scores = torch.full((4, NUM_EXPERTS), -4.0, dtype=torch.float32, device="cuda")
    for row in range(4):
        scores[row, torch.randperm(NUM_EXPERTS, generator=generator)[:pool]] = 4.0

    ref_w, ref_ids = _aiter_route(scores, bias)
    w, ids = moe_route_radix4.route_radix4(scores, bias, TOPK, True, 2.5)
    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(w, ref_w, rtol=1e-5, atol=1e-6)


@requires_aiter
def test_route_radix4_tie_straddling_the_cutoff_matches_aiter():
    """Unambiguous winners take the first columns and a tied set fights over what
    is left, so the tie rule settles both which experts get in and where they go."""
    bias = torch.zeros(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    scores = torch.full((8, NUM_EXPERTS), -5.0, dtype=torch.float32, device="cuda")
    generator = torch.Generator().manual_seed(11)
    for row in range(8):
        perm = torch.randperm(NUM_EXPERTS, generator=generator)
        for j, e in enumerate(perm[:10].tolist()):
            scores[row, e] = 8.0 - j * 0.25
        scores[row, perm[10:30]] = 1.0

    ref_w, ref_ids = _aiter_route(scores, bias)
    w, ids = moe_route_radix4.route_radix4(scores, bias, TOPK, True, 2.5)
    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(w, ref_w, rtol=1e-5, atol=1e-6)


def test_route_radix4_graph_replay():
    scores = torch.randn(32, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    ref_w, ref_ids = _oracle(scores, bias, True, 2.5)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w, ids = moe_route_radix4.route_radix4(scores, bias, TOPK, True, 2.5)
    graph.replay()
    torch.cuda.synchronize()

    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(w, ref_w, rtol=1e-5, atol=1e-6)


def test_route_radix4_coverage():
    scores = torch.empty(32, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    bias = torch.empty(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    assert moe_route_radix4.covered(scores, bias, TOPK, 1, 1)
    assert not moe_route_radix4.covered(scores, bias, TOPK, 8, 4)
    assert not moe_route_radix4.covered(scores, bias, TOPK - 1, 1, 1)
    assert not moe_route_radix4.covered(scores[:, :-1], bias[:-1], TOPK, 1, 1)
    assert not moe_route_radix4.covered(
        scores.new_empty((1536, NUM_EXPERTS)), bias, TOPK, 1, 1
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
