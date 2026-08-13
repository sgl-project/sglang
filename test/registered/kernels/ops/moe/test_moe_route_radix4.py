# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ROCm radix-4 K3 router.

The reference is a pure-torch fp32 oracle rather than aiter. aiter is what the
kernel replaces, so it agrees on every input whose top-k is unambiguous, but it
breaks exact ties in its own wave64 traversal order, which is neither the
kernel's contract nor a rule the kernel could cheaply reproduce. Comparing
against the oracle instead pins the contract the kernel actually promises.

Run:  pytest test/registered/kernels/ops/moe/test_moe_route_radix4.py -v
"""

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

if not is_hip():
    pytest.skip("The radix-4 router is the ROCm path.", allow_module_level=True)
if not torch.cuda.is_available():
    pytest.skip("Requires a GPU.", allow_module_level=True)

from sglang.kernels.ops.moe import moe_route_radix4

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")

NUM_EXPERTS = 896
TOPK = 16
NAN_FLOOR = -1e30

pytestmark = pytest.mark.skipif(
    not moe_route_radix4.available(), reason="kernel does not build on this toolchain"
)


def _oracle(scores, bias, renormalize, scaling):
    """Contract, from route_radix4_hip.cuh: bias ranks only and the emitted
    weight stays bias-free, NaN is floored so it can never win, ties go to the
    lower expert id, renormalize divides by the winners' sum (guarded to 1 when
    that sum is non-positive) before scaling."""
    s = torch.sigmoid(scores.float())
    biased = s + bias.float()
    biased = torch.where(
        torch.isnan(biased), torch.full_like(biased, NAN_FLOOR), biased
    )
    # stable + descending: equal biased values keep ascending-id order
    ranked = torch.argsort(biased, dim=-1, descending=True, stable=True)[:, :TOPK]
    ranked, _ = ranked.sort(dim=-1)
    w = s.gather(1, ranked)
    if renormalize:
        total = w.sum(-1, keepdim=True)
        w = w / torch.where(total > 0, total, torch.ones_like(total))
    return w * scaling, ranked.to(torch.int32)


def _assert_matches_oracle(scores, bias, renormalize=True, scaling=2.5):
    ref_w, ref_ids = _oracle(scores, bias, renormalize, scaling)
    w, ids = moe_route_radix4.route_radix4(scores, bias, TOPK, renormalize, scaling)
    # Winners come out unordered; the MoE sorting stage downstream is what fixes
    # the order, so compare the sets.
    order = ids.argsort(dim=-1)
    ids, w = ids.gather(1, order), w.gather(1, order)
    assert torch.equal(ids, ref_ids)
    # The kernel's sigmoid uses __expf, whose last bits differ from torch's.
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
    """Only the lowest-id rule decides these, so an arrival-order compaction or
    a mis-ranked tie shows up immediately."""
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
    """The winner set is settled by rank and the renorm divisor is reduced over
    threads, so neither depends on the order the compaction happened to run in.
    The order the winners appear in a row does, and is not part of the contract."""
    scores = torch.randn(512, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    bias = torch.zeros(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")

    def route():
        w, ids = moe_route_radix4.route_radix4(scores, bias, TOPK, True, 2.5)
        order = ids.argsort(dim=-1)
        return w.gather(1, order), ids.gather(1, order)

    first_w, first_ids = route()
    for _ in range(4):
        w, ids = route()
        assert torch.equal(ids, first_ids)
        assert torch.equal(w, first_w)


def test_route_radix4_graph_replay():
    scores = torch.randn(32, NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    ref_w, ref_ids = _oracle(scores, bias, True, 2.5)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        w, ids = moe_route_radix4.route_radix4(scores, bias, TOPK, True, 2.5)
    graph.replay()
    torch.cuda.synchronize()

    order = ids.argsort(dim=-1)
    assert torch.equal(ids.gather(1, order), ref_ids)
    torch.testing.assert_close(w.gather(1, order), ref_w, rtol=1e-5, atol=1e-6)


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
