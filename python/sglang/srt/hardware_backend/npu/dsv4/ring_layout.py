"""Per-request ring addressing for the NPU SWA request window and draft ring.

A request's logical position ``p`` lives at row ``slot * ring + p % ring`` of
its ring; the A3 sparse-attention operator reaches it through a per-request
page table, one row of which maps logical page ``q`` to ring page
``q % ring_pages``.
"""

from typing import Optional

import torch


def sink_dead_rows(loc: torch.Tensor, sink: int) -> torch.Tensor:
    """Map dead (-1) store rows to an unused sink row without a host sync."""
    return torch.where(loc >= 0, loc, loc.new_full((), sink))


def draft_swa_ring_size(pool) -> Optional[int]:
    """The draft worker's per-request SWA ring length (None: a paged pool)."""
    return getattr(pool, "swa_req_ring_size", None)


def _ring_page_pattern(ring_pages: int, width: int, device) -> torch.Tensor:
    return torch.arange(width, dtype=torch.int32, device=device) % ring_pages


def ring_row_loc(slots, positions, capacity: int) -> torch.Tensor:
    """Physical row of a request slot's ring for logical ``positions``."""
    slots = torch.as_tensor(slots, dtype=torch.int64)
    positions = torch.as_tensor(positions, dtype=torch.int64)
    return slots * capacity + positions % capacity


def ring_page_indices(capacity: int, page_size: int, slot, start: int, end: int):
    """Physical page ids of ``slot``'s ring rows in [start, end); capacity is
    whole pages, so they pair with the paged SWA pages one for one."""
    assert start % page_size == 0, (start, page_size)
    positions = torch.arange(start, end, page_size)
    return (
        (ring_row_loc(slot, positions, capacity) // page_size).to(torch.int32).numpy()
    )


def reset_tag_rows(tags: torch.Tensor, slots, capacity: int) -> None:
    """Clear every ring row's tag for ``slots`` without a host sync."""
    slots = torch.as_tensor(slots, dtype=torch.int64)
    loc = slots[:, None] * capacity + torch.arange(capacity, device=tags.device)
    # index_fill_ stays on the device; a scalar index_put waits for the stream.
    tags.index_fill_(1, loc.flatten(), -1)


class RingPageTableCache:
    """Reusable per-request ring page tables.

    Logical page ``p`` of request slot ``s`` maps to ring page ``p % ring_pages``
    of that slot's ring, i.e. row ``s * ring_pages + p % ring_pages``. The table
    is rewritten in place each step on the compute stream, so one buffer per
    (rows, ring_pages, device) removes the per-step allocation; the buffer grows
    by doubling when a batch needs more logical pages. Extra columns past the
    caller's width wrap onto valid ring pages and are never read below
    ``seqused_kv``.
    """

    def __init__(self) -> None:
        self._tables: dict[tuple, torch.Tensor] = {}
        self._patterns: dict[tuple, torch.Tensor] = {}

    def table(
        self,
        *,
        req_pool_indices: torch.Tensor,
        ring_pages: int,
        logical_pages: int,
    ) -> torch.Tensor:
        req = req_pool_indices.reshape(-1)
        device = req.device
        ring_pages = int(ring_pages)
        key = (int(req.numel()), ring_pages, device)
        table = self._tables.get(key)
        if table is None or table.shape[1] < logical_pages:
            width = (
                int(logical_pages)
                if table is None
                else max(int(logical_pages), 2 * table.shape[1])
            )
            table = torch.empty((key[0], width), dtype=torch.int32, device=device)
            self._tables[key] = table
        pattern = self._patterns.get((ring_pages, table.shape[1], device))
        if pattern is None:
            pattern = _ring_page_pattern(ring_pages, table.shape[1], device)
            self._patterns[(ring_pages, table.shape[1], device)] = pattern
        # table[i, q] = req_i * ring_pages + q % ring_pages, scaled in place: no
        # base vector allocation, and the contiguous rewrite beats a strided slice.
        if req.dtype != torch.int32:
            req = req.to(torch.int32)
        table.copy_(req[:, None])
        table.mul_(ring_pages)
        table.add_(pattern)
        # The cache doubles its width; callers copy this into a graph buffer
        # sized for the live table, so expose only the valid columns.
        return table[:, : int(logical_pages)]
