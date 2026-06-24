from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.test.scripted_runtime.context.api import ScriptedContext


def _get_all_reqs(ctx: ScriptedContext) -> Iterator[Req]:
    s = ctx.scheduler
    if s.chunked_req is not None:
        yield s.chunked_req
    yield from s.waiting_queue
    if s.ps.pp_size > 1:
        for mb in (*s.mbs, *s.last_mbs, *s.running_mbs):
            if mb is not None:
                yield from mb.reqs
    else:
        if s.running_batch is not None:
            yield from s.running_batch.reqs
        if s.last_iter is not None:
            yield from s.last_iter.reqs


def list_active_reqs(ctx: ScriptedContext) -> List[Req]:
    return list(set(_get_all_reqs(ctx)))


def batch_composition(ctx: ScriptedContext) -> Dict[str, List[str]]:
    s = ctx.scheduler
    chunked_rid = s.chunked_req.rid if s.chunked_req is not None else None
    chunked = [chunked_rid] if chunked_rid is not None else []
    running = (
        [r.rid for r in s.running_batch.reqs] if s.running_batch is not None else []
    )

    prefill: List[str] = []
    decode: List[str] = []
    last_iter = s.last_iter
    if (
        last_iter is not None
        and not last_iter.is_empty
        and last_iter.forward_mode is not None
    ):
        bucket = prefill if last_iter.forward_mode.is_extend() else decode
        bucket.extend(r.rid for r in last_iter.reqs if r.rid != chunked_rid)

    return {
        "prefill": prefill,
        "decode": decode,
        "chunked": chunked,
        "running": running,
    }


def is_idle(ctx: ScriptedContext) -> bool:
    s = ctx.scheduler
    return (
        s.chunked_req is None
        and len(s.waiting_queue) == 0
        and (s.running_batch is None or s.running_batch.is_empty())
    )


def is_fully_idle(ctx: ScriptedContext) -> bool:
    s = ctx.scheduler
    return is_idle(ctx) and (s.last_iter is None or s.last_iter.is_empty)


def last_batch_forward_mode(ctx: ScriptedContext) -> Optional[str]:
    s = ctx.scheduler
    if s.last_iter is not None and s.last_iter.forward_mode is not None:
        return s.last_iter.forward_mode.name
    return None


def find_req_by_rid(ctx: ScriptedContext, rid: str) -> Optional[Req]:
    req = next((r for r in _get_all_reqs(ctx) if r.rid == rid), None)
    if req is not None:
        ctx._seen_rids.add(rid)
    return req


def is_finished(ctx: ScriptedContext, rid: str) -> bool:
    req = find_req_by_rid(ctx, rid)
    if req is not None:
        return req.finished()
    if rid in ctx._seen_rids:
        return True
    # Fallback: if the req ran in a forward batch (recorded in _batch_log) but
    # is now absent from all active scheduler sets, it must have finished.
    # This catches requests that completed without ever being observed via
    # find_req_by_rid (e.g. when Python short-circuit evaluation prevents the
    # query while another request is still running).
    log = ctx._scheduler_hook._batch_log
    if any(rid in record.rids for record in log):
        ctx._seen_rids.add(rid)
        return True
    return False


def is_chunking(ctx: ScriptedContext, rid: str) -> bool:
    s = ctx.scheduler
    return s.chunked_req is not None and s.chunked_req.rid == rid


def status(ctx: ScriptedContext, rid: str) -> str:
    s = ctx.scheduler
    if rid in {r.rid for r in s.waiting_queue}:
        return "waiting"
    req = find_req_by_rid(ctx, rid)
    if req is not None:
        return "finished" if req.finished() else "running"
    if rid in ctx._seen_rids:
        return "finished"
    return "unknown"


def remaining_prompt_tokens(ctx: ScriptedContext, rid: str) -> int:
    req = find_req_by_rid(ctx, rid)
    if req is None:
        return 0
    return max(0, len(req.origin_input_ids) - req.kv_committed_len)


def chunks_done(ctx: ScriptedContext, rid: str) -> int:
    log = ctx._scheduler_hook._batch_log
    held = sum(1 for record in log if record.chunked_rid == rid and rid in record.rids)
    if held == 0:
        return 0
    completed = any(
        rid in record.extend_rids and record.chunked_rid != rid for record in log
    )
    return held + (1 if completed else 0)


def chunked_parks(ctx: ScriptedContext, rid: str) -> int:
    return sum(
        1
        for record in ctx._scheduler_hook._batch_log
        if record.chunked_rid == rid and rid not in record.rids
    )
