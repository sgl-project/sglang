from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.test.scripted_runtime.context.api import ScriptedContext


def find_req_by_rid(ctx: "ScriptedContext", rid: str) -> Optional["Req"]:
    s = ctx._scheduler
    chunked = s.chunked_req
    if chunked is not None and chunked.rid == rid:
        return chunked
    for r in s.waiting_queue:
        if r.rid == rid:
            return r
    if s.running_batch is not None:
        for r in s.running_batch.reqs:
            if r.rid == rid:
                return r
    last_batch = getattr(s, "last_batch", None)
    if last_batch is not None:
        for r in last_batch.reqs:
            if r.rid == rid:
                return r
    return None


def is_finished(ctx: "ScriptedContext", rid: str) -> bool:
    req = find_req_by_rid(ctx, rid)
    if req is None:
        return rid in ctx._req_handles
    return req.finished()


def is_chunking(ctx: "ScriptedContext", rid: str) -> bool:
    s = ctx._scheduler
    return s.chunked_req is not None and s.chunked_req.rid == rid
