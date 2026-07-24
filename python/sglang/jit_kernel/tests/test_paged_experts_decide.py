"""Correctness + CUDA-graph-capture tests for the paged_experts_decide JIT kernel.

The kernel computes the per-decode-step Paged Experts residency plan on the GPU (no host sync). Validated
against a pure-Python reference of the same keep-warm/LRU and static-wave logic, and — the whole point of
moving the decision on-device — that the kernel is CUDA-graph-capturable (capture once, replay with the
inputs mutated in place, and the residency state evolves exactly as the eager run, since the step counter
is itself on-device).
"""

import sys

import pytest
import torch

from sglang.jit_kernel.paged_experts_decide import (
    paged_experts_decide,
    paged_experts_decide_bounded,
    paged_experts_decide_wave,
    paged_experts_gather,
    paged_experts_gather_multi,
    paged_experts_host_devptr,
    paged_experts_remap_mask,
    paged_experts_scatter_multi,
    paged_experts_scratch_split,
)


# ---------------------------------------------------------------------------
# Pure-Python references (mirror the kernel logic)
# ---------------------------------------------------------------------------
def _ref_decide(topk, slot_expert, expert_slot, slot_lastuse, freq, step, lfu):
    """In-place keep-warm + LRU/LFU decision; returns (src, dst)."""
    distinct = [int(e) for e in topk if e >= 0]
    for e in distinct:
        freq[e] += 1
        s = expert_slot[e]
        if s >= 0:
            slot_lastuse[s] = step
    K = len(slot_expert)
    src, dst = [], []
    for e in distinct:
        if expert_slot[e] >= 0:
            continue
        victim, best_f, best_lu = -1, None, None
        for s in range(K):
            se = slot_expert[s]
            if se in distinct:
                continue
            f = freq[se] if (lfu and se >= 0) else 0
            lu = slot_lastuse[s]
            if best_f is None or f < best_f or (f == best_f and lu < best_lu):
                best_f, best_lu, victim = f, lu, s
        if victim < 0:
            continue
        old = slot_expert[victim]
        if old >= 0:
            expert_slot[old] = -1
        slot_expert[victim] = e
        expert_slot[e] = victim
        slot_lastuse[victim] = step
        src.append(e)
        dst.append(victim)
    return src, dst


def _ref_wave(topk, E, K, w, slot_base=0):
    lo, hi = w * K, w * K + K
    idx = [(e - lo + slot_base) if lo <= e < hi else -1 for e in range(E)]
    src, dst = [], []
    for e in topk:
        if not (lo <= e < hi):
            continue
        if e not in src:
            src.append(e)
            dst.append(e - lo + slot_base)
    return src, dst, idx


def _i32(x):
    return torch.tensor(x, dtype=torch.int32, device="cuda")


def _new_state(E, K):
    """Fresh device residency state + output buffers, matching the in-tree pager's initial seeding
    (slots 0..K-1 hold experts 0..K-1)."""
    step_ctr = _i32([0])
    slot_expert = _i32(list(range(K)))
    expert_slot = _i32([-1] * E)
    expert_slot[:K] = torch.arange(K, dtype=torch.int32, device="cuda")
    slot_lastuse = _i32([0] * K)
    freq = _i32([0] * E)
    src = _i32([0] * K)
    dst = _i32([0] * K)
    n_out = _i32([0])
    idx = _i32([-1] * E)
    return step_ctr, slot_expert, expert_slot, slot_lastuse, freq, src, dst, n_out, idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for the decide kernel"
)


@requires_cuda
@pytest.mark.parametrize("lfu", [False, True])
def test_decide_matches_reference(lfu):
    E, K = 16, 6
    sc, se, es, lu, fq, src, dst, n_out, idx = _new_state(E, K)
    r_se = list(range(K))
    r_es = [-1] * E
    r_es[:K] = list(range(K))
    r_lu = [0] * K
    r_fq = [0] * E
    torch.manual_seed(0)
    for step in range(
        1, 25
    ):  # the kernel's on-device counter starts at 0 and ++s to 1 on the first call
        ndist = int(
            torch.randint(1, K + 1, (1,)).item()
        )  # 1..K distinct (keep-warm regime)
        experts = torch.randperm(E)[:ndist].tolist()
        paged_experts_decide(
            _i32(experts), sc, se, es, lu, fq, lfu, src, dst, n_out, idx
        )
        r_src, r_dst = _ref_decide(experts, r_se, r_es, r_lu, r_fq, step, lfu)
        n = int(n_out.item())
        assert int(sc.item()) == step, f"step {step}: counter"
        assert se.tolist() == r_se, f"step {step}: slot_expert"
        assert es.tolist() == r_es, f"step {step}: expert_slot"
        assert idx.tolist() == r_es, f"step {step}: idx == expert_slot snapshot"
        assert src[:n].tolist() == r_src, f"step {step}: src"
        assert dst[:n].tolist() == r_dst, f"step {step}: dst"


@requires_cuda
def test_decide_wave_matches_reference():
    E, K = 16, 6
    sc, se, es, lu, fq, src, dst, n_out, idx = _new_state(E, K)
    experts = [0, 3, 7, 8, 13, 1, 9]  # distinct > K -> served in waves
    topk = _i32(experts)
    nwaves = (E + K - 1) // K
    served = []
    for w in range(nwaves):
        paged_experts_decide_wave(topk, E, K, w, src, dst, n_out, idx)
        r_src, r_dst, r_idx = _ref_wave(experts, E, K, w)
        n = int(n_out.item())
        assert idx.tolist() == r_idx, f"wave {w}: idx"
        assert src[:n].tolist() == r_src, f"wave {w}: src"
        assert dst[:n].tolist() == r_dst, f"wave {w}: dst"
        served += src[:n].tolist()
    assert sorted(served) == sorted(
        experts
    )  # every active expert served in exactly one wave

    # banked (double-buffered) waves: half-K waves ping-pong between slot banks 0 and K//2 —
    # dst/idx carry the bank offset, and every active expert is still served in exactly one wave
    half = K // 2
    nwaves = (E + half - 1) // half
    served = []
    for w in range(nwaves):
        base = (w & 1) * half
        paged_experts_decide_wave(
            topk, E, half, w, src, dst, n_out, idx, slot_base=base
        )
        r_src, r_dst, r_idx = _ref_wave(experts, E, half, w, slot_base=base)
        n = int(n_out.item())
        assert idx.tolist() == r_idx, f"banked wave {w}: idx"
        assert src[:n].tolist() == r_src, f"banked wave {w}: src"
        assert dst[:n].tolist() == r_dst, f"banked wave {w}: dst"
        assert all(base <= s < base + half for s in dst[:n].tolist())
        served += src[:n].tolist()
    assert sorted(served) == sorted(experts)


@requires_cuda
def test_decide_is_cuda_graph_capturable():
    """Capture the decide kernel once, then replay with topk mutated in place. Because the step counter is
    on-device, the captured replays evolve the residency state *identically* to an eager run.
    """
    E, K = 16, 6
    steps = [[1, 5, 9], [2, 5, 11], [9, 3, 14], [5, 1, 2], [0, 7, 11]]

    # eager reference run (kernel, no capture)
    sc, se, es, lu, fq, src, dst, n_out, idx = _new_state(E, K)
    eager_idx = []
    for experts in steps:
        paged_experts_decide(
            _i32(experts), sc, se, es, lu, fq, False, src, dst, n_out, idx
        )
        eager_idx.append(idx.tolist())

    # captured run: fixed input/state buffers, topk mutated in place between replays
    sc, se, es, lu, fq, src, dst, n_out, idx = _new_state(E, K)
    topk_buf = _i32(steps[0])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):  # warmup (required before capture)
        paged_experts_decide(topk_buf, sc, se, es, lu, fq, False, src, dst, n_out, idx)
    torch.cuda.current_stream().wait_stream(s)
    # reset state after warmup so the captured replays reproduce the eager sequence from step 1
    sc.zero_()
    se.copy_(_i32(list(range(K))))
    es.copy_(_new_state(E, K)[2])
    lu.zero_()
    fq.zero_()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        paged_experts_decide(topk_buf, sc, se, es, lu, fq, False, src, dst, n_out, idx)

    captured_idx = []
    for experts in steps:
        topk_buf.copy_(_i32(experts))
        g.replay()
        torch.cuda.synchronize()
        captured_idx.append(idx.tolist())

    assert (
        captured_idx == eager_idx
    )  # exact match: on-device counter makes replay == eager


@requires_cuda
def test_gather_dynamic_count():
    """decide -> gather: the gather moves exactly the experts decide chose (count read on-device), placing
    the right host rows in the right slots."""
    E, K, W = 16, 6, 32  # 32 float32 = 128 B/expert (16-byte aligned)
    store = torch.empty((E, W), dtype=torch.float32, device="cpu", pin_memory=True)
    for e in range(E):
        store[e].fill_(float(e + 1))  # expert e's data == e+1
    devptr = paged_experts_host_devptr(store)
    slot = torch.zeros((K, W), dtype=torch.float32, device="cuda")
    for s in range(K):
        slot[s].fill_(
            float(s + 1)
        )  # slots 0..K-1 start holding experts 0..K-1 (value s+1)

    sc, se, es, lu, fq, src, dst, n_out, idx = _new_state(E, K)
    # route to expert 1 (resident hit) + 9, 12 (misses) -> decide pages 2 experts
    paged_experts_decide(
        _i32([1, 9, 12]), sc, se, es, lu, fq, False, src, dst, n_out, idx
    )
    n = int(n_out.item())
    assert n == 2, n
    paged_experts_gather(devptr, slot, src, dst, n_out, W * 4)
    torch.cuda.synchronize()
    # the two paged experts now sit in their assigned slots, with the right host values
    for i in range(n):
        e, s = int(src[i].item()), int(dst[i].item())
        assert (slot[s] == e + 1).all().item(), f"expert {e} -> slot {s}"
    # untouched slots keep their original contents (gather moved exactly n, not K)
    touched = set(int(dst[i].item()) for i in range(n))
    for s in range(K):
        if s not in touched:
            assert (slot[s] == s + 1).all().item(), f"slot {s} should be untouched"


@requires_cuda
def test_decide_gather_capturable():
    """The full per-step primitive (decide -> gather) captured once and replayed with topk mutated in
    place: each replay pages exactly the misses for that step (dynamic count survives capture).
    """
    E, K, W = 16, 6, 32
    store = torch.empty((E, W), dtype=torch.float32, device="cpu", pin_memory=True)
    for e in range(E):
        store[e].fill_(float(e + 1))
    devptr = paged_experts_host_devptr(store)
    slot = torch.zeros((K, W), dtype=torch.float32, device="cuda")
    for s in range(K):
        slot[s].fill_(float(s + 1))
    sc, se, es, lu, fq, src, dst, n_out, idx = _new_state(E, K)
    topk_buf = _i32([1, 9, 12])

    def step():
        paged_experts_decide(topk_buf, sc, se, es, lu, fq, False, src, dst, n_out, idx)
        paged_experts_gather(devptr, slot, src, dst, n_out, W * 4)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()

    # replay with a fresh routing step; the experts it pages must end up resident in their slots
    topk_buf.copy_(_i32([2, 10, 13]))
    g.replay()
    torch.cuda.synchronize()
    es_l = es.tolist()
    for e in (2, 10, 13):
        sidx = es_l[e]
        assert (
            sidx >= 0 and (slot[sidx] == e + 1).all().item()
        ), f"expert {e} not gathered to its slot"


# ---------------------------------------------------------------------------
# Bounded (pinned-window) decide
# ---------------------------------------------------------------------------
def _ref_decide_bounded(
    topk,
    slot_expert,
    expert_slot,
    slot_lastuse,
    freq,
    step,
    lfu,
    log2hot,
):
    """In-place bounded keep-warm decision; returns (src, dst, cold_log, needed).

    Victim choice is TIER-AWARE (mirrors the kernel): among non-needed slots, hot-tier residents and
    empty slots are strictly preferred over cold-tier residents (whose re-fetch costs a deferred round),
    then LFU/LRU as usual."""
    E, K = len(expert_slot), len(slot_expert)
    distinct = [int(e) for e in topk if 0 <= e < E]
    for e in distinct:
        freq[e] += 1
        s = expert_slot[e]
        if s >= 0:
            slot_lastuse[s] = step
    src, dst, cold_log = [], [], []
    for e in distinct:
        if expert_slot[e] >= 0:
            continue
        hi = log2hot[e]
        if hi < 0:  # cold miss: record logical id, no eviction, stays masked
            cold_log.append(e)
            continue
        victim, best_key = -1, None
        for s in range(K):
            se = slot_expert[s]
            if se in distinct:  # never evict a slot needed this step
                continue
            cold = 1 if (se >= 0 and log2hot[se] < 0) else 0
            f = freq[se] if (lfu and se >= 0) else 0
            lu = slot_lastuse[s]
            key = (cold, f, lu)
            if best_key is None or key < best_key:
                best_key, victim = key, s
        if victim < 0:
            continue
        old = slot_expert[victim]
        if old >= 0:
            expert_slot[old] = -1
        slot_expert[victim] = e
        expert_slot[e] = victim
        slot_lastuse[victim] = step
        src.append(hi)  # windowed -> on-device gather from host_hot
        dst.append(victim)
    needed = [
        1 if (slot_expert[s] >= 0 and slot_expert[s] in distinct) else 0
        for s in range(K)
    ]
    return src, dst, cold_log, needed


def _window_map(E, W):
    """Experts [0, W) are hot (pinned window), [W, E) are cold. log2hot as the kernel expects."""
    return _i32([e if e < W else -1 for e in range(E)])


def _new_bounded_buffers(K):
    cold_log = _i32([0] * K)
    cold_n = _i32([0])
    needed = _i32([0] * K)
    return cold_log, cold_n, needed


@requires_cuda
@pytest.mark.parametrize("lfu", [False, True])
def test_decide_bounded_matches_reference(lfu):
    """Bounded decide matches the pure-Python reference over a random multi-step run: window hits land in
    (src,dst), cold misses defer (logical id in cold_log, unresident — the host stages them out-of-graph),
    and needed[] marks slots in use this step."""
    E, K, W = 16, 6, 8  # experts 0..7 hot (window), 8..15 cold; K=6 resident slots
    log2hot = _window_map(E, W)
    sc, se, es, lu, fq, src, dst, n_out, idx = _new_state(E, K)
    cold_log, cold_n, needed = _new_bounded_buffers(K)
    r_se, r_es = list(range(K)), [-1] * E
    r_es[:K] = list(range(K))
    r_lu, r_fq = [0] * K, [0] * E
    rlog2hot = [e if e < W else -1 for e in range(E)]

    torch.manual_seed(1)
    for step in range(1, 25):
        ndist = int(
            torch.randint(1, K + 1, (1,)).item()
        )  # 1..K distinct (keep-warm regime)
        experts = torch.randperm(E)[:ndist].tolist()
        paged_experts_decide_bounded(
            _i32(experts),
            sc,
            se,
            es,
            lu,
            fq,
            lfu,
            log2hot,
            src,
            dst,
            n_out,
            cold_log,
            cold_n,
            idx,
            needed,
        )
        r_src, r_dst, r_cl, r_needed = _ref_decide_bounded(
            experts, r_se, r_es, r_lu, r_fq, step, lfu, rlog2hot
        )
        nw, nc = int(n_out.item()), int(cold_n.item())
        assert int(sc.item()) == step, f"step {step}: counter"
        assert se.tolist() == r_se, f"step {step}: slot_expert"
        assert es.tolist() == r_es, f"step {step}: expert_slot"
        assert idx.tolist() == r_es, f"step {step}: idx snapshot"
        assert src[:nw].tolist() == r_src, f"step {step}: windowed src"
        assert dst[:nw].tolist() == r_dst, f"step {step}: windowed dst"
        assert cold_log[:nc].tolist() == r_cl, f"step {step}: cold_log"
        assert needed.tolist() == r_needed, f"step {step}: needed"
        # deferred (window-miss) experts must stay unresident this step
        for e in cold_log[:nc].tolist():
            assert (
                es[e].item() == -1
            ), f"step {step}: deferred expert {e} should be unresident"


@requires_cuda
def test_decide_bounded_is_cuda_graph_capturable():
    """Capture decide_bounded once and replay with topk mutated in place; on-device counter makes the
    captured residency evolution identical to the eager run (the substrate for replay-twice).
    """
    E, K, W = 16, 6, 8
    log2hot = _window_map(E, W)
    steps = [[1, 6, 9], [2, 7, 11], [9, 3, 14], [6, 1, 2], [0, 7, 11]]

    def fresh():
        st = _new_state(E, K)
        return (*st, *_new_bounded_buffers(K))

    # eager reference (kernel, no capture)
    sc, se, es, lu, fq, src, dst, n_out, idx, cl, cn, nd = fresh()
    eager_idx = []
    for experts in steps:
        paged_experts_decide_bounded(
            _i32(experts),
            sc,
            se,
            es,
            lu,
            fq,
            False,
            log2hot,
            src,
            dst,
            n_out,
            cl,
            cn,
            idx,
            nd,
        )
        eager_idx.append(idx.tolist())

    # captured run
    sc, se, es, lu, fq, src, dst, n_out, idx, cl, cn, nd = fresh()
    topk_buf = _i32(steps[0])
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        paged_experts_decide_bounded(
            topk_buf,
            sc,
            se,
            es,
            lu,
            fq,
            False,
            log2hot,
            src,
            dst,
            n_out,
            cl,
            cn,
            idx,
            nd,
        )
    torch.cuda.current_stream().wait_stream(s)
    sc.zero_()
    se.copy_(_i32(list(range(K))))
    es.copy_(_new_state(E, K)[2])
    lu.zero_()
    fq.zero_()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        paged_experts_decide_bounded(
            topk_buf,
            sc,
            se,
            es,
            lu,
            fq,
            False,
            log2hot,
            src,
            dst,
            n_out,
            cl,
            cn,
            idx,
            nd,
        )

    captured_idx = []
    for experts in steps:
        topk_buf.copy_(_i32(experts))
        g.replay()
        torch.cuda.synchronize()
        captured_idx.append(idx.tolist())

    assert captured_idx == eager_idx


@requires_cuda
def test_decide_bounded_prefers_hot_tier_victims():
    """Tier-aware eviction: a cold-tier resident (whose re-fetch costs a deferred replay/break round) is
    kept over an older hot-tier resident (cheap in-graph re-gather). Simulates the post-staging state
    where cold experts occupy slots."""
    E, K, W = 16, 4, 8
    log2hot = _window_map(E, W)
    sc = _i32([10])
    se = _i32([0, 9, 2, 12])  # slots 1 and 3 hold COLD experts (>= W)
    es = _i32([-1] * E)
    for s, e in enumerate([0, 9, 2, 12]):
        es[e] = s
    lu = _i32([5, 1, 4, 2])  # plain LRU would evict slot 1 (cold expert 9)
    fq = _i32([0] * E)
    src, dst, n_out = _i32([0] * K), _i32([0] * K), _i32([0])
    idx = _i32([-1] * E)
    cold_log, cold_n, needed = _new_bounded_buffers(K)

    paged_experts_decide_bounded(
        _i32([3]),  # hot miss -> needs a victim
        sc,
        se,
        es,
        lu,
        fq,
        False,
        log2hot,
        src,
        dst,
        n_out,
        cold_log,
        cold_n,
        idx,
        needed,
    )
    assert int(n_out.item()) == 1
    assert se.tolist()[2] == 3, "oldest HOT slot (2) must be the victim"
    assert (
        int(es[9].item()) == 1 and int(es[12].item()) == 3
    ), "cold residents must survive"
    assert int(es[2].item()) == -1, "evicted hot expert unmapped"


@requires_cuda
def test_remap_mask_matches_reference():
    """One fused launch == gather + 2x where + 2x zeros_like chain."""
    E, T = 16, 12
    idx = _i32([2, -1, 5, -1, 0, 1, -1, 3, -1, -1, 4, -1, -1, -1, -1, -1])
    topk = _i32([0, 1, 2, 4, 7, 15, -1, 3, 10, 5, 0, 2])
    torch.manual_seed(3)
    tw = torch.rand(T, dtype=torch.float32, device="cuda")
    safe_ids = _i32([0] * T)
    masked_tw = torch.zeros(T, dtype=torch.float32, device="cuda")

    paged_experts_remap_mask(topk, idx, tw, safe_ids, masked_tw)
    torch.cuda.synchronize()

    idx_l, topk_l, tw_l = idx.tolist(), topk.tolist(), tw.tolist()
    for i in range(T):
        e = topk_l[i]
        s = idx_l[e] if 0 <= e < E else -1
        assert safe_ids[i].item() == (s if s >= 0 else 0), f"safe_ids[{i}]"
        expect_w = tw_l[i] if s >= 0 else 0.0
        assert abs(masked_tw[i].item() - expect_w) < 1e-7, f"masked_tw[{i}]"


@requires_cuda
def test_gather_multi_matches_per_tensor_gather():
    """One fused launch pages all tensors: results identical to per-tensor gather over the same plan."""
    E, K = 16, 4
    torch.manual_seed(4)
    hosts = [
        torch.rand(
            E, 64, dtype=torch.float32, pin_memory=True
        ),  # 256 B/expert -> e16=16
        torch.rand(E, 8, dtype=torch.float32, pin_memory=True),  # 32 B/expert  -> e16=2
    ]
    gpus = [
        torch.zeros(K, t.shape[1], dtype=torch.float32, device="cuda") for t in hosts
    ]
    dev = "cuda"
    stores = torch.tensor(
        [paged_experts_host_devptr(t) for t in hosts], dtype=torch.int64, device=dev
    )
    slots = torch.tensor([g.data_ptr() for g in gpus], dtype=torch.int64, device=dev)
    e16s = torch.tensor(
        [t.shape[1] * 4 // 16 for t in hosts], dtype=torch.int64, device=dev
    )
    src, dst, n_out = (
        _i32([3, 7, 1, 0]),
        _i32([0, 2, 1, 0]),
        _i32([3]),
    )  # 3 of 4 entries live

    paged_experts_gather_multi(stores, slots, e16s, src, dst, n_out)
    torch.cuda.synchronize()
    for h, g in zip(hosts, gpus):
        for s_e, d_s in [(3, 0), (7, 2), (1, 1)]:
            assert torch.equal(g[d_s].cpu(), h[s_e]), f"expert {s_e} -> slot {d_s}"
        assert g[3].abs().sum().item() == 0, "slot 3 untouched (n=3)"

    # n = 0 -> strict no-op
    for g in gpus:
        g.zero_()
    n_out.zero_()
    paged_experts_gather_multi(stores, slots, e16s, src, dst, n_out)
    torch.cuda.synchronize()
    assert all(g.abs().sum().item() == 0 for g in gpus)


@requires_cuda
def test_decide_bounded_doorbell():
    """The kernel writes the cold count to a mapped pinned host address (host-visible) — the BCG break
    spins on it instead of a per-layer stream sync."""
    E, K, W = 16, 4, 8
    log2hot = _window_map(E, W)
    sc, se, es, lu, fq, src, dst, n_out, idx = _new_state(E, K)
    cold_log, cold_n, needed = _new_bounded_buffers(K)
    bell = torch.full((1,), -1, dtype=torch.int32, pin_memory=True)
    bell_ptr = paged_experts_host_devptr(bell)

    # two cold misses (>= W) and one hot miss
    paged_experts_decide_bounded(
        _i32([9, 12, 5]),
        sc,
        se,
        es,
        lu,
        fq,
        False,
        log2hot,
        src,
        dst,
        n_out,
        cold_log,
        cold_n,
        idx,
        needed,
        doorbell=bell_ptr,
    )
    torch.cuda.synchronize()
    assert bell[0].item() == 2, bell[0].item()
    assert int(cold_n.item()) == 2  # device count matches the doorbell


@requires_cuda
def test_scatter_multi_matches_reference():
    """One fused launch scatters n staged rows into arbitrary slots for all tensors."""
    K, n = 6, 3
    torch.manual_seed(5)
    stages = [
        torch.rand(K, 64, dtype=torch.float32, device="cuda"),  # e16 = 16
        torch.rand(K, 8, dtype=torch.float32, device="cuda"),  # e16 = 2
    ]
    pools = [
        torch.zeros(K, t.shape[1], dtype=torch.float32, device="cuda") for t in stages
    ]
    dev = "cuda"
    stage_ptrs = torch.tensor(
        [t.data_ptr() for t in stages], dtype=torch.int64, device=dev
    )
    slot_ptrs = torch.tensor(
        [g.data_ptr() for g in pools], dtype=torch.int64, device=dev
    )
    e16s = torch.tensor(
        [t.shape[1] * 4 // 16 for t in stages], dtype=torch.int64, device=dev
    )
    dst = _i32([4, 0, 2, 0, 0, 0])  # first n entries live

    paged_experts_scatter_multi(stage_ptrs, slot_ptrs, e16s, dst, n)
    torch.cuda.synchronize()
    for st, g in zip(stages, pools):
        for i, slot in enumerate([4, 0, 2]):
            assert torch.equal(g[slot], st[i]), f"row {i} -> slot {slot}"
        assert g[1].abs().sum().item() == 0 and g[3].abs().sum().item() == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


@requires_cuda
def test_scratch_split_and_d2d_gather():
    """The streaming-prefill split: residents plan a device-to-device copy (pool slot -> scratch
    expert row), the complement a host-to-device copy — and gather_multi executes BOTH (its copy is
    address-agnostic: pool device pointers for D2D, UVA host pointers for H2D). End state: scratch
    holds every expert's store row exactly."""
    E, K = 16, 6
    l2g_host = [-1] * E
    for slot, e in enumerate([3, 9, 0, 14, 7, 11]):  # pool slot -> expert
        l2g_host[e] = slot
    l2g = _i32(l2g_host)
    mk = lambda n: torch.zeros(n, dtype=torch.int32, device="cuda")
    res_src, res_dst, res_n = mk(E), mk(E), mk(1)
    h2d_src, h2d_dst, h2d_n = mk(E), mk(E), mk(1)
    paged_experts_scratch_split(l2g, res_src, res_dst, res_n, h2d_src, h2d_dst, h2d_n)
    torch.cuda.synchronize()
    nr, nh = int(res_n.item()), int(h2d_n.item())
    assert nr == K and nh == E - K
    # reference: expert order, residents to their slots
    r_res = [(l2g_host[e], e) for e in range(E) if l2g_host[e] >= 0]
    r_h2d = [(e, e) for e in range(E) if l2g_host[e] < 0]
    assert list(zip(res_src[:nr].tolist(), res_dst[:nr].tolist())) == r_res
    assert list(zip(h2d_src[:nh].tolist(), h2d_dst[:nh].tolist())) == r_h2d

    # execute both plans with gather_multi; scratch must equal the store row-for-row
    torch.manual_seed(7)
    host = torch.rand(E, 64, dtype=torch.float32, pin_memory=True)  # e16 = 16
    pool = torch.zeros(K, 64, dtype=torch.float32, device="cuda")
    for e in range(E):  # pool holds exact store-row copies (as page_in guarantees)
        if l2g_host[e] >= 0:
            pool[l2g_host[e]].copy_(host[e])
    scratch = torch.zeros(E, 64, dtype=torch.float32, device="cuda")
    e16s = torch.tensor([64 * 4 // 16], dtype=torch.int64, device="cuda")
    store_base = torch.tensor(
        [paged_experts_host_devptr(host)], dtype=torch.int64, device="cuda"
    )
    pool_base = torch.tensor([pool.data_ptr()], dtype=torch.int64, device="cuda")
    scratch_base = torch.tensor([scratch.data_ptr()], dtype=torch.int64, device="cuda")
    paged_experts_gather_multi(store_base, scratch_base, e16s, h2d_src, h2d_dst, h2d_n)
    paged_experts_gather_multi(pool_base, scratch_base, e16s, res_src, res_dst, res_n)
    torch.cuda.synchronize()
    assert torch.equal(scratch.cpu(), host), "scratch != store after split fill"
