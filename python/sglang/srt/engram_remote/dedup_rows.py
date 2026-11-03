"""Per-lookup representatives; no rows or request state survive workspace reset."""

import torch
import triton
import triton.language as tl


@triton.jit
def _representatives(
    ids, out, table, query_list, lo, hi, S: tl.constexpr, COMPACT: tl.constexpr
):
    q = tl.program_id(0).to(tl.int64)
    row = tl.load(ids + q).to(tl.int64)
    representative = tl.full((), -1, tl.int64)
    if (row >= lo) & (row < hi):
        slot = (row * 2654435761) & (S - 1)
        key = ((row.to(tl.uint64) + 1) << 32) | q.to(tl.uint64)
        seen = tl.atomic_cas(
            table + slot, tl.full((), 0, tl.uint64), key, sem="relaxed"
        )
        while (seen != 0) & ((seen >> 32) != (row.to(tl.uint64) + 1)):
            slot = (slot + 1) & (S - 1)
            seen = tl.atomic_cas(
                table + slot, tl.full((), 0, tl.uint64), key, sem="relaxed"
            )
        representative = tl.where(seen == 0, q, (seen & 0xFFFFFFFF).to(tl.int64))
        if COMPACT:
            if seen == 0:
                slot_out = tl.atomic_add(
                    table + S, tl.full((), 1, tl.uint64), sem="relaxed"
                )
                tl.store(query_list + slot_out, q)
    tl.store(out + q, representative)


def workspace(n, device):
    slots = triton.next_power_of_2(max(2, 2 * n))
    return torch.empty(slots + 1, dtype=torch.uint64, device=device), torch.empty(
        n, dtype=torch.int64, device=device
    )


def prepare(ids, lo, hi, scratch, out, query_list=None):
    if not (
        ids.dtype == torch.int64
        and ids.is_cuda
        and ids.is_contiguous()
        and out.shape == ids.shape
    ):
        raise ValueError("invalid dedup IDs/output")
    if not 0 <= lo <= hi < 2**32 or ids.numel() >= 2**32:
        raise ValueError("dedup key/index range")
    if scratch.numel() < 2 * ids.numel():
        raise ValueError("dedup table capacity")
    scratch.zero_()
    if ids.numel():
        _representatives[(ids.numel(),)](
            ids,
            out,
            scratch,
            query_list if query_list is not None else out,
            lo,
            hi,
            S=scratch.numel() - 1,
            COMPACT=query_list is not None,
            num_warps=1,
        )
    return out
