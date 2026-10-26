"""Muse Spark EP4 expert replicas (gfx950): the per-step replica assignment of ``_scatter_a4_replica``.

Every rank derives the same assignment from the global top-k ids: each route is computed by exactly one rank, that
rank's per-slot expert row is bitwise the row its home rank computes without replicas, and the assignment is the
reference greedy (distinct active experts per rank, entries in plan order, ties to the home rank, two sweeps).
"""

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

pytestmark = pytest.mark.skipif(
    not (is_hip() and is_gfx95_supported()), reason="Tuned for MI355X (gfx950)."
)

NE, NR, E0 = 256, 4, 64


def _plan(r, seed):
    """[(slot, holder)]: every holder takes r / 3 random experts of each other rank, shuffled plan order."""
    gen = torch.Generator().manual_seed(seed)
    out = []
    taken = set()
    for h in range(NR):
        for o in range(NR):
            if o == h:
                continue
            cand = [
                e
                for e in (o * E0 + torch.randperm(E0, generator=gen)).tolist()
                if e not in taken
            ]
            for e in cand[: r // 3]:
                taken.add(e)
                out.append((e, h))
    order = torch.randperm(len(out), generator=gen).tolist()
    return [out[i] for i in order]


def _ref_lids(ids, plan, rank, passes):
    from sglang.srt.rocm_muse_spark.expert_replicas import replica_slots

    flat = ids.flatten().tolist()
    active = set(flat)
    rep = {g for g, _ in plan}
    load = [0] * NR
    for g in active:
        if g not in rep:
            load[g // E0] += 1
    home = [True] * len(plan)
    for p in range(passes):
        for i, (g, o) in enumerate(plan):
            if g not in active:
                continue
            h = g // E0
            if p:
                load[h if home[i] else o] -= 1
            home[i] = load[h] <= load[o]
            load[h if home[i] else o] += 1
    slots = replica_slots(plan, NR, E0)
    owner = {g: (g // E0 if home[i] else o) for i, (g, o) in enumerate(plan)}
    out = []
    for g in flat:
        r = owner.get(g, g // E0)
        if r != rank:
            out.append(-1)
        elif r == g // E0:
            out.append(g - rank * E0)
        else:
            out.append(slots[(g, rank)])
    return torch.tensor(out, dtype=torch.int32).view(ids.shape)


def _ids(m, seed, hot=None):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.rand((m, NE), generator=gen, device="cuda")
    if hot is not None:
        logits[:, hot] += 0.5  # skew: some experts on most rows
    return logits.argsort(dim=1, descending=True)[:, :8].to(torch.int32).contiguous()


@pytest.mark.parametrize("r", [3, 6])
@pytest.mark.parametrize("m", [8, 64, 256, 512])
def test_replica_assignment_matches_reference(r, m):
    import triton

    from sglang.kernels.ops.moe.mxfp4_compact_a4_moe import (
        REPLICA_PASSES,
        _scatter_a4_replica,
    )
    from sglang.srt.rocm_muse_spark.expert_replicas import build_table

    plan = _plan(r, seed=m + r)
    tables = [build_table(plan, NR, E0, rank, NE).cuda() for rank in range(NR)]
    hot = torch.tensor(
        [g for g, _ in plan[: len(plan) // 2]] + [1, 70, 130], device="cuda"
    )
    for seed in range(4):
        ids = _ids(m, seed, hot if seed % 2 else None)
        e_local = E0 + r
        rows = m * 8
        cap = rows + e_local * 15
        nb = triton.cdiv(cap, 16)
        seen = torch.zeros(rows, dtype=torch.int32)
        for rank in range(NR):
            sid = torch.empty(cap, device="cuda", dtype=torch.int32)
            eid = torch.empty(nb, device="cuda", dtype=torch.int32)
            lids = torch.empty((m, 8), device="cuda", dtype=torch.int32)
            _scatter_a4_replica[(e_local,)](
                ids, sid, eid, lids, tables[rank], m, NE, E0, e_local, triton.next_power_of_2(e_local),
                len(plan), triton.next_power_of_2(len(plan)), REPLICA_PASSES, triton.next_power_of_2(rows),
                triton.next_power_of_2(triton.cdiv(rows, 16)), 16, nb, triton.next_power_of_2(nb), num_warps=4,
            )  # fmt: skip
            ref = _ref_lids(ids.cpu(), plan, rank, REPLICA_PASSES)
            assert torch.equal(lids.cpu(), ref), (rank, seed)
            seen += (lids.cpu().flatten() >= 0).int()
            # Block map: every local route sits in a block of its local expert, each exactly once.
            eid_c, sid_c, flat = eid.cpu(), sid.cpu(), lids.cpu().flatten()
            placed = []
            for b in range(nb):
                if eid_c[b] < 0:
                    continue
                for slot in sid_c[b * 16 : (b + 1) * 16].tolist():
                    if slot < rows:
                        assert flat[slot] == eid_c[b]
                        placed.append(slot)
            assert sorted(placed) == (flat >= 0).nonzero().flatten().tolist()
        assert torch.all(seen == 1)


@pytest.mark.parametrize("m", [8, 128, 256])
def test_replicated_rows_match_home_rows(m):
    """Each rank's expert rows (home or replica slot) are bitwise the home rank's rows without replicas."""
    from sglang.kernels.ops.moe.mxfp4_compact_a4_moe import (
        compact_a4_experts,
        compact_a4_experts_replicated,
    )
    from sglang.srt.rocm_muse_spark.expert_replicas import build_table, replica_slots

    r = 3
    torch.manual_seed(m)
    dev = "cuda"
    x = (torch.randn(m, 4096, device=dev) * 0.5).to(torch.bfloat16)
    w13 = torch.randint(0, 256, (NE, 8192, 2048), device=dev, dtype=torch.uint8)
    w2 = torch.randint(0, 256, (NE, 4096, 2048), device=dev, dtype=torch.uint8)
    s13 = torch.randint(118, 124, (NE, 8192, 128), device=dev, dtype=torch.uint8)
    s2 = torch.randint(118, 124, (NE, 4096, 128), device=dev, dtype=torch.uint8)
    plan = _plan(r, seed=7)
    slots = replica_slots(plan, NR, E0)
    ids = _ids(m, 3, torch.tensor([g for g, _ in plan], device=dev))
    flat = ids.flatten()
    seen = torch.zeros(m * 8, dtype=torch.int32, device=dev)
    for rank in range(NR):
        mine = sorted((s, g) for (g, h), s in slots.items() if h == rank)
        sel = list(range(rank * E0, rank * E0 + E0)) + [g for _, g in mine]
        local = [t[sel].contiguous() for t in (w13, w2, s13, s2)]
        table = build_table(plan, NR, E0, rank, NE).to(dev)
        y, lids = compact_a4_experts_replicated(
            x, ids, *local, table=table, num_entries=len(plan)
        )
        lf = lids.flatten()
        seen += (lf >= 0).int()
        for home in range(NR):
            lo = home * E0
            ref = compact_a4_experts(
                x,
                ids,
                *[t[lo : lo + E0].contiguous() for t in (w13, w2, s13, s2)],
                local_start=lo,
            )
            rows = ((lf >= 0) & (flat >= lo) & (flat < lo + E0)).nonzero().flatten()
            assert torch.equal(y[rows], ref[rows])
    assert torch.all(seen == 1)
