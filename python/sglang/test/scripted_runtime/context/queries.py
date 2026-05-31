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


def list_active_reqs(ctx: "ScriptedContext") -> List["Req"]:
    return list(set(_get_all_reqs(ctx)))


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
