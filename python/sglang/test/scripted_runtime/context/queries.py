from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.test.scripted_runtime.context.api import ScriptedContext


def _last_step(ctx: ScriptedContext) -> Optional[ScheduleBatch]:
    """The step the last scheduler iteration launched; None when it launched
    nothing."""
    return ctx.scheduler.counters.cur_batch_for_debug


def _get_all_reqs(ctx: ScriptedContext) -> Iterator[Req]:
    s = ctx.scheduler
    if s.lists.chunked_req is not None:
        yield s.lists.chunked_req
    yield from s.lists.waiting_queue
    yield from s.lists.decoding
    # A request the last step finished has left the lists already.
    last_step = _last_step(ctx)
    if last_step is not None:
        yield from last_step.reqs


def list_active_reqs(ctx: ScriptedContext) -> List[Req]:
    return list(set(_get_all_reqs(ctx)))


def batch_composition(ctx: ScriptedContext) -> Dict[str, List[str]]:
    s = ctx.scheduler
    chunked_rid = s.lists.chunked_req.rid if s.lists.chunked_req is not None else None
    chunked = [chunked_rid] if chunked_rid is not None else []
    running = [r.rid for r in s.lists.decoding]

    prefill: List[str] = []
    decode: List[str] = []
    batch = _last_step(ctx)
    if batch is not None and not batch.is_empty() and batch.forward_mode is not None:
        bucket = prefill if batch.forward_mode.is_extend() else decode
        bucket.extend(r.rid for r in batch.reqs if r.rid != chunked_rid)

    return {
        "prefill": prefill,
        "decode": decode,
        "chunked": chunked,
        "running": running,
    }


def is_idle(ctx: ScriptedContext) -> bool:
    s = ctx.scheduler
    return (
        s.lists.chunked_req is None
        and len(s.lists.waiting_queue) == 0
        and not s.lists.decoding
    )


def is_fully_idle(ctx: ScriptedContext) -> bool:
    return is_idle(ctx) and len(ctx.scheduler.pending) == 0


def last_batch_forward_mode(ctx: ScriptedContext) -> Optional[str]:
    batch = _last_step(ctx)
    if batch is not None and batch.forward_mode is not None:
        return batch.forward_mode.name
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
    return s.lists.chunked_req is not None and s.lists.chunked_req.rid == rid


def status(ctx: ScriptedContext, rid: str) -> str:
    s = ctx.scheduler
    if rid in {r.rid for r in s.lists.waiting_queue}:
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
    return max(0, len(req.origin_input_ids) - req.kv.kv_committed_len)


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
