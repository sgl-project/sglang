# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the ROCm radix-4 K3 router.

Two references, because the kernel owes two different things. A pure-torch fp32
oracle spells out the contract -- what the weights are, where a NaN ranks, how a
row is ordered -- and covers the shapes and edge cases. aiter is the router this
one stands in for on decode-sized batches while larger ones still go to it, so
the test_*_aiter cases hold the kernel to matching it column for column, ties
included; anything less and a token's routing would depend on the batch size.

Run:  SGLANG_USE_AITER=1 pytest test/registered/kernels/ops/moe/test_moe_route_radix4.py -v
"""

import sys
from collections import defaultdict

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


@pytest.mark.parametrize(
    "num_experts,topk,num_groups,topk_group,num_tokens",
    [
        (256, 8, 1, 1, 4),
        (256, 8, 8, 4, 4),
        (896, 16, 1, 1, 1024),
        (896, 16, 1, 1, 1025),
    ],
)
@pytest.mark.parametrize("radix4", [False, True])
@requires_aiter
def test_select_experts_preserves_offset_fp32_bias(
    num_experts, topk, num_groups, topk_group, num_tokens, radix4
):
    """Both router dispatches must preserve an fp32 bias that bf16 would collapse."""
    from sglang.srt.environ import envs
    from sglang.srt.layers.moe import topk as topk_module
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts
    from sglang.srt.runtime_context import get_context

    if not topk_module._use_aiter:
        pytest.skip("Requires SGLANG_USE_AITER=1 before importing the router.")

    scores = torch.zeros((num_tokens, num_experts), dtype=torch.bfloat16, device="cuda")
    bias = 16.0 + torch.arange(num_experts, dtype=torch.float32, device="cuda") * 1e-4
    bias = bias.roll(num_experts // 2)
    config = TopKConfig(
        top_k=topk,
        use_grouped_topk=True,
        num_expert_group=num_groups,
        topk_group=topk_group,
        correction_bias=bias,
        routed_scaling_factor=2.5,
    )
    with (
        get_context().override_server_args(),
        envs.SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK.override(False),
        envs.SGLANG_K3_RADIX4_TOPK.override(radix4),
    ):
        weights, ids, _ = select_experts(
            torch.zeros((num_tokens, 8), device="cuda"), scores, config
        )
    expected_ids = torch.arange(
        num_experts // 2 - topk, num_experts // 2, device="cuda"
    )
    torch.testing.assert_close(
        ids.sort(dim=-1).values,
        expected_ids.expand_as(ids).to(ids.dtype),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(weights, torch.full_like(weights, 2.5 / topk))


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


# ===== Fused BM=16 sort (route_radix4_with_sort) =====

BM = 16
MODEL_DIM = 3584


def _payload_from_route(ids, weights):
    bag = defaultdict(list)
    m = ids.shape[0]
    for token in range(m):
        for slot in range(TOPK):
            bag[int(ids[token, slot])].append(
                (token, slot, float(weights[token, slot]))
            )
    for expert in bag:
        bag[expert].sort()
    return bag


def _payload_from_sorted(
    sorted_token_ids, sorted_weights, sorted_expert_ids, n_blocks, m
):
    bag = defaultdict(list)
    for block in range(n_blocks):
        expert = int(sorted_expert_ids[block])
        start = block * BM
        live = []
        for row in range(BM):
            packed = int(sorted_token_ids[start + row])
            token = packed & 0x00FFFFFF
            slot = (packed >> 24) & 0xFF
            weight = float(sorted_weights[start + row])
            if token == m:
                assert weight == 0.0
                continue
            live.append((token, slot, weight))
        if live:
            bag[expert].extend(live)
    for expert in bag:
        bag[expert].sort()
    return bag


# 1 is the grid that skips the arrival handshake and 2..64 sort in the last CTA.
# Past 64 the last ceil(M / 16) CTAs share the sort: 65 is the smallest such
# tail (5 CTAs, a short last chunk), 300 has a short last chunk too, and 1024 is
# the router's largest batch (64 tail CTAs).
@pytest.mark.parametrize("m", [1, 2, 4, 8, 16, 32, 64, 65, 80, 128, 300, 512, 1024])
def test_fused_sort_matches_route_and_layout(m):
    generator = torch.Generator(device="cuda").manual_seed(7 + m)
    scores = torch.randn(
        (m, NUM_EXPERTS), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    bias = torch.randn(
        NUM_EXPERTS, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    moe_buf = torch.ones((m, MODEL_DIM), dtype=torch.bfloat16, device="cuda")

    ref_w, ref_ids = moe_route_radix4.route_radix4(scores, bias, TOPK, True, 2.5)
    w, ids, aux = moe_route_radix4.route_radix4_with_sort(
        scores, bias, TOPK, True, 2.5, moe_buf=moe_buf
    )

    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(w, ref_w, rtol=0, atol=0)
    assert torch.count_nonzero(moe_buf).item() == 0
    assert int(aux["num_valid_ids"][1]) == m

    expected = _payload_from_route(ref_ids.cpu(), ref_w.cpu())
    got = _payload_from_sorted(
        aux["sorted_token_ids"].cpu(),
        aux["sorted_weights"].cpu(),
        aux["sorted_expert_ids"].cpu(),
        int(aux["num_valid_ids"][0]) // BM,
        m,
    )
    assert got.keys() == expected.keys()
    for expert in expected:
        assert got[expert] == expected[expert]

    used = int(aux["num_valid_ids"][0])
    m_indices = aux["m_indices"].cpu()[:used]
    packed = aux["sorted_token_ids"].cpu()[:used]
    for i in range(used):
        token = int(packed[i]) & 0x00FFFFFF
        assert int(m_indices[i]) == token

    reverse = aux["reverse_sorted"].cpu()
    for token in range(m):
        for slot in range(TOPK):
            pos = int(reverse[token * TOPK + slot])
            packed_i = int(packed[pos])
            assert (packed_i & 0x00FFFFFF) == token
            assert ((packed_i >> 24) & 0xFF) == slot


# (12, 7) sorts in the last CTA. Past 64 tokens every padding row sends all 16
# routes to expert 0, which the tail's one-route-per-token ranking cannot see, so
# they are placed separately: none, one, a whole tail chunk, and all but one.
@pytest.mark.parametrize(
    "m, n_real", [(12, 7), (80, 80), (80, 79), (80, 61), (300, 1), (300, 0)]
)
def test_fused_sort_routes_padded_rows_like_the_mask(m, n_real):
    """select_experts masks CUDA-graph padding rows to expert 0 and weight 0 in
    place after routing. AITER matches the registered sort by tensor address,
    so the sort has to describe the masked rows, not the routed ones, and give
    each of their routes a slot of its own."""
    scores = torch.randn((m, NUM_EXPERTS), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    non_padded = torch.tensor(n_real, dtype=torch.int32, device="cuda")

    ref_w, ref_ids = moe_route_radix4.route_radix4(scores, bias, TOPK, True, 2.5)
    ref_ids[n_real:] = 0
    ref_w[n_real:] = 0.0
    w, ids, aux = moe_route_radix4.route_radix4_with_sort(
        scores, bias, TOPK, True, 2.5, num_token_non_padded=non_padded
    )

    assert torch.equal(ids, ref_ids)
    torch.testing.assert_close(w, ref_w, rtol=0, atol=0)
    expected = _payload_from_route(ref_ids.cpu(), ref_w.cpu())
    got = _payload_from_sorted(
        aux["sorted_token_ids"].cpu(),
        aux["sorted_weights"].cpu(),
        aux["sorted_expert_ids"].cpu(),
        int(aux["num_valid_ids"][0]) // BM,
        m,
    )
    assert got == expected
    reverse = aux["reverse_sorted"].cpu()
    assert len(set(reverse.tolist())) == m * TOPK


@pytest.mark.parametrize("m", [65, 300])
def test_fused_sort_tail_is_stable(m):
    """Past 64 tokens each expert's routes come out in token order, which is the
    layout AITER's own sort writes from there on, and the same on every launch."""
    generator = torch.Generator(device="cuda").manual_seed(m)
    scores = torch.randn(
        (m, NUM_EXPERTS), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    bias = torch.randn(
        NUM_EXPERTS, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    _, _, first = moe_route_radix4.route_radix4_with_sort(scores, bias, TOPK, True, 2.5)
    _, _, again = moe_route_radix4.route_radix4_with_sort(scores, bias, TOPK, True, 2.5)
    used = int(first["num_valid_ids"][0])
    for key in ("sorted_token_ids", "sorted_weights", "m_indices"):
        assert torch.equal(first[key][:used], again[key][:used]), key
    assert torch.equal(first["reverse_sorted"], again["reverse_sorted"])

    packed = first["sorted_token_ids"][:used].cpu()
    experts = first["sorted_expert_ids"][: used // BM].cpu().repeat_interleave(BM)
    live = (packed & 0x00FFFFFF) < m
    for expert in experts[live].unique().tolist():
        tokens = packed[live & (experts == expert)] & 0x00FFFFFF
        assert torch.all(tokens[1:] > tokens[:-1]), expert


def test_fused_sort_graph_replay():
    """Each launch has to leave the stream's arrival counters ready for the next
    one: the last CTA's sort resets its counter, and the tail sort's counters
    alternate, each launch clearing the one before. A counter left behind would
    elect no sorter, or every CTA, in the launches after it. Mixes the two sorts
    and an odd number of tail launches per replay, so replays alternate too."""
    sizes = [80, 300, 12, 80]
    inputs = [
        (
            torch.randn((m, NUM_EXPERTS), dtype=torch.bfloat16, device="cuda"),
            torch.randn(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda"),
            torch.ones((m, MODEL_DIM), dtype=torch.bfloat16, device="cuda"),
        )
        for m in sizes
    ]
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Allocates this stream's arrival counters outside the capture.
        moe_route_radix4.route_radix4_with_sort(*inputs[0][:2], TOPK, True, 2.5)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        outs = [
            moe_route_radix4.route_radix4_with_sort(
                scores, bias, TOPK, True, 2.5, moe_buf=moe_buf
            )
            for scores, bias, moe_buf in inputs
        ]
    expected = []
    for scores, bias, _ in inputs:
        ref_w, ref_ids = moe_route_radix4.route_radix4(scores, bias, TOPK, True, 2.5)
        expected.append((ref_ids, _payload_from_route(ref_ids.cpu(), ref_w.cpu())))
    for _ in range(3):
        for (_, _, moe_buf), (_, _, aux) in zip(inputs, outs):
            aux["sorted_token_ids"].fill_(-1)
            moe_buf.fill_(1.0)
        graph.replay()
        torch.cuda.synchronize()
        for m, (_, _, moe_buf), (_, ids, aux), (ref_ids, payload) in zip(
            sizes, inputs, outs, expected
        ):
            assert torch.equal(ids, ref_ids)
            assert torch.count_nonzero(moe_buf).item() == 0
            got = _payload_from_sorted(
                aux["sorted_token_ids"].cpu(),
                aux["sorted_weights"].cpu(),
                aux["sorted_expert_ids"].cpu(),
                int(aux["num_valid_ids"][0]) // BM,
                m,
            )
            assert got == payload, m


def test_fused_sort_rejects_a_moe_buf_it_cannot_zero():
    """aux tells AITER the buffer was zeroed, so a buffer with the wrong row
    count has to fail rather than be skipped."""
    m = 4
    scores = torch.randn((m, NUM_EXPERTS), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    moe_buf = torch.ones((m + 1, MODEL_DIM), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(Exception):
        moe_route_radix4.route_radix4_with_sort(
            scores, bias, TOPK, True, 2.5, moe_buf=moe_buf
        )


@pytest.mark.parametrize("m", [1, 12, 64, 80, 512])
def test_aiter_accepts_fused_sort(m):
    """AITER drops a registered sort whose aux lacks a key it matches on, such as
    topk_weights_ptr, logs it once and sorts again, so a missing key only shows
    up as a lost speedup."""
    try:
        from aiter.presorted_moe import match_presorted_moe
    except ImportError:
        pytest.skip("AITER without register_presorted_moe (ROCm/aiter#6484)")
    scores = torch.randn((m, NUM_EXPERTS), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(NUM_EXPERTS, dtype=torch.bfloat16, device="cuda")
    w, ids, aux = moe_route_radix4.route_radix4_with_sort(scores, bias, TOPK, True, 2.5)
    assert (
        match_presorted_moe(aux, ids, w, block_m=BM, num_experts=NUM_EXPERTS, topk=TOPK)
        is aux
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
