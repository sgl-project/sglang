"""A request's storage lifecycle, as far as the scheduler drives it today.

`acquire` binds the requests' rows in the request table and, when the model has
recurrent state, their state and tracking slots. If the state slots run out
after the rows were bound, it gives back what it took (the rows it bound and
the slots it handed out) before the failure surfaces. Giving a request's state
slots back on finish is `StateStorage.release`, which the radix tree's state
component calls before the row goes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from sglang.srt.mem_cache.kv.state_storage import StateSlotsExhausted

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.kv.state_storage import StateStorage
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


def acquire(
    table: ReqToTokenPool, state: Optional[StateStorage], reqs: List[Req]
) -> Optional[List[int]]:
    """The requests' rows, or None when the table has too few free rows. A
    request already holding a row (a later chunk) keeps it, and one already
    holding a state slot (a radix hit) keeps that."""
    fresh_rows = [req.kv.req_pool_idx is None for req in reqs]
    rows = table.alloc(reqs)
    if rows is None or state is None:
        return rows
    try:
        state.acquire(table, reqs, rows, fresh_rows)
    except StateSlotsExhausted:
        for req, fresh in zip(reqs, fresh_rows):
            if fresh:
                table.free(req)
        raise
    return rows
