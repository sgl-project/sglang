from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.test.scripted_runtime.context.api import ScriptedContext


def _get_all_reqs(ctx: "ScriptedContext") -> Iterator["Req"]:
    s = ctx._scheduler
    if s.chunked_req is not None:
        yield s.chunked_req
    yield from s.waiting_queue
    if s.running_batch is not None:
        yield from s.running_batch.reqs
    if s.last_batch is not None:
        yield from s.last_batch.reqs


def list_active_reqs(ctx: "ScriptedContext") -> List["Req"]:
    return list(set(_get_all_reqs(ctx)))


def batch_composition(ctx: "ScriptedContext") -> Dict[str, List[str]]:
    s = ctx._scheduler
    chunked_rid = s.chunked_req.rid if s.chunked_req is not None else None
    chunked = [chunked_rid] if chunked_rid is not None else []
    running = (
        [r.rid for r in s.running_batch.reqs] if s.running_batch is not None else []
    )

    prefill: List[str] = []
    decode: List[str] = []
    batch = s.last_batch
    if batch is not None and not batch.is_empty() and batch.forward_mode is not None:
        bucket = prefill if batch.forward_mode.is_extend() else decode
        bucket.extend(r.rid for r in batch.reqs if r.rid != chunked_rid)

    return {
        "prefill": prefill,
        "decode": decode,
        "chunked": chunked,
        "running": running,
    }


def find_req_by_rid(ctx: "ScriptedContext", rid: str) -> Optional["Req"]:
    req = next((r for r in _get_all_reqs(ctx) if r.rid == rid), None)
    if req is not None:
        ctx._seen_rids.add(rid)
    return req


def is_finished(ctx: "ScriptedContext", rid: str) -> bool:
    req = find_req_by_rid(ctx, rid)
    if req is None:
        # A started req that is absent from every scheduler structure is finished
        # only if it was once observed live; one that has not yet been pulled out
        # of the recv buffer into the waiting queue is still pending, not finished.
        return rid in ctx._seen_rids
    return req.finished()


def is_chunking(ctx: "ScriptedContext", rid: str) -> bool:
    s = ctx._scheduler
    return s.chunked_req is not None and s.chunked_req.rid == rid


def chunks_done(ctx: "ScriptedContext", rid: str) -> int:
    return sum(
        1
        for record in ctx._scheduler_hook._batch_log
        if record.chunked_rid == rid and rid in record.rids
    )
