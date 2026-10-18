"""Parallel expert counting for Day0's existing sorted-slot ABI."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _parallel_align(
    IDS,
    SORTED,
    EXPERT,
    POST,
    START: tl.constexpr,
    N: tl.constexpr,
    E: tl.constexpr,
    BM: tl.constexpr,
    S: tl.constexpr,
    CAP: tl.constexpr,
    NB: tl.constexpr,
    CB: tl.constexpr,
    NBB: tl.constexpr,
    PER_E: tl.constexpr,
):
    s = tl.arange(0, S)
    e = tl.arange(0, E)
    ids = tl.load(IDS + s, s < N, other=-1)
    hits = (ids[None, :] == START + e[:, None]) & (s[None, :] < N)
    counts = tl.sum(hits.to(tl.int32), axis=1)
    blocks = tl.cdiv(counts, BM)
    first = tl.cumsum(blocks, axis=0) - blocks
    total = tl.sum(blocks, axis=0)
    ranks = tl.cumsum(hits.to(tl.int32), axis=1) - 1

    c = tl.arange(0, CB)
    b = tl.arange(0, NBB)
    tl.store(SORTED + c, N, c < CAP)
    tl.store(EXPERT + b, -1, b < NB)
    # The later scatter may be owned by a different lane than initialization.
    tl.debug_barrier()
    tl.store(SORTED + first[:, None] * BM + ranks, s[None, :], hits)
    eb = tl.arange(0, PER_E)
    tl.store(
        EXPERT + first[:, None] + eb[None, :], e[:, None], eb[None, :] < blocks[:, None]
    )
    tl.store(POST, total * BM)


def parallel_align(ids, num_local_experts, local_start, block_m):
    n = ids.numel()
    if num_local_experts != 32 or n > 512 or block_m != 16:
        raise ValueError(
            "experimental alignment requires E32, BM16, at most 512 routes"
        )
    cap = n + num_local_experts * (block_m - 1)
    nb = tr.cdiv(cap, block_m)
    sorted_ids = torch.empty(cap, device=ids.device, dtype=torch.int32)
    expert_ids = torch.empty(nb, device=ids.device, dtype=torch.int32)
    post = torch.empty(1, device=ids.device, dtype=torch.int32)
    _parallel_align[(1,)](
        ids,
        sorted_ids,
        expert_ids,
        post,
        local_start,
        n,
        num_local_experts,
        block_m,
        tr.next_power_of_2(n),
        cap,
        nb,
        tr.next_power_of_2(cap),
        tr.next_power_of_2(nb),
        tr.next_power_of_2(tr.cdiv(n, block_m)),
        num_warps=4,
    )
    return sorted_ids, expert_ids, post
