from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, List, Optional

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


def _unique_rids(reqs: Iterator["Req"]) -> List[str]:
    seen: set[str] = set()
    rids: List[str] = []
    for req in reqs:
        if req.rid not in seen:
            seen.add(req.rid)
            rids.append(req.rid)
    return rids


def list_active_reqs(ctx: "ScriptedContext") -> List["Req"]:
    seen: set[str] = set()
    reqs: List["Req"] = []
    for req in _get_all_reqs(ctx):
        if req.rid not in seen:
            seen.add(req.rid)
            reqs.append(req)
    return reqs


def running_rids(ctx: "ScriptedContext") -> List[str]:
    batch = ctx._scheduler.running_batch
    return _unique_rids(iter(batch.reqs)) if batch is not None else []


def batch_rids(ctx: "ScriptedContext") -> List[str]:
    return running_rids(ctx)


def batch_size(ctx: "ScriptedContext") -> int:
    batch = ctx._scheduler.running_batch
    return len(running_rids(ctx)) if batch is not None else 0


def waiting_rids(ctx: "ScriptedContext") -> List[str]:
    return _unique_rids(iter(ctx._scheduler.waiting_queue))


def get_chunked_req_rid(ctx: "ScriptedContext") -> Optional[str]:
    chunked_req = ctx._scheduler.chunked_req
    return chunked_req.rid if chunked_req is not None else None


def chunked_in_flight_count(ctx: "ScriptedContext") -> int:
    return 1 if ctx._scheduler.chunked_req is not None else 0


def find_req_by_rid(ctx: "ScriptedContext", rid: str) -> Optional["Req"]:
    return next((r for r in _get_all_reqs(ctx) if r.rid == rid), None)


def is_finished(ctx: "ScriptedContext", rid: str) -> bool:
    req = find_req_by_rid(ctx, rid)
    if req is None:
        return rid in ctx._started_rids
    return req.finished()


def is_chunking(ctx: "ScriptedContext", rid: str) -> bool:
    s = ctx._scheduler
    return s.chunked_req is not None and s.chunked_req.rid == rid
