from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

import aiohttp
import orjson

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.test.scripted_runtime.context.api import ScriptedContext
    from sglang.test.scripted_runtime.req_handle import _RequestEpoch


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
        if s.last_batch is not None:
            yield from s.last_batch.reqs


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


def is_idle(ctx: ScriptedContext) -> bool:
    s = ctx.scheduler
    return (
        s.chunked_req is None
        and len(s.waiting_queue) == 0
        and (s.running_batch is None or s.running_batch.is_empty())
    )


def is_fully_idle(ctx: ScriptedContext) -> bool:
    s = ctx.scheduler
    return is_idle(ctx) and (s.last_batch is None or s.last_batch.is_empty())


def last_batch_forward_mode(ctx: ScriptedContext) -> Optional[str]:
    s = ctx.scheduler
    if s.last_batch is not None and s.last_batch.forward_mode is not None:
        return s.last_batch.forward_mode.name
    return None


def _resolve_epoch_req(ctx: ScriptedContext, *, epoch: _RequestEpoch) -> Optional[Req]:
    if epoch.req is not None or epoch.closed:
        return epoch.req
    candidates = list(_get_all_reqs(ctx))
    for record in ctx._scheduler_hook._batch_log[epoch.batch_start :]:
        candidates.extend(record.reqs)
        if record.chunked_req is not None:
            candidates.append(record.chunked_req)
    for req in candidates:
        if req.rid == epoch.rid and all(req is not old for old in epoch.excluded_reqs):
            epoch.req = req
            return req
    return None


def _resolve_req(
    ctx: ScriptedContext, *, rid: str, epoch: Optional[_RequestEpoch] = None
) -> Optional[Req]:
    if epoch is None:
        epoch = ctx._request_epochs.get(rid)
    if epoch is not None:
        return _resolve_epoch_req(ctx, epoch=epoch)
    return next((r for r in _get_all_reqs(ctx) if r.rid == rid), None)


def find_req_by_rid(
    ctx: ScriptedContext, rid: str, *, epoch: Optional[_RequestEpoch] = None
) -> Optional[Req]:
    req = _resolve_req(ctx, rid=rid, epoch=epoch)
    return next((r for r in _get_all_reqs(ctx) if r is req), None)


def _check_epoch_post_result(ctx: ScriptedContext, *, epoch: _RequestEpoch) -> bool:
    """Check a completed response, accepting only an intentional terminal abort."""
    if not epoch.post_future.done():
        return False
    try:
        epoch.post_future.result()
    except aiohttp.ClientResponseError as exc:
        if not epoch.abort_requested or exc.status != 400:
            raise
        req = _resolve_epoch_req(ctx, epoch=epoch)
        if (
            req is not None
            and not req.finished()
            and any(r is req for r in _get_all_reqs(ctx))
        ):
            raise
        try:
            payload = orjson.loads(exc.message)
        except orjson.JSONDecodeError:
            raise exc from None
        if not (
            isinstance(payload, dict)
            and isinstance(payload.get("error"), dict)
            and payload["error"].get("message") in ("Aborted", "Abort in waiting queue")
        ):
            raise
    return True


def is_finished(
    ctx: ScriptedContext, rid: str, *, epoch: Optional[_RequestEpoch] = None
) -> bool:
    if epoch is None:
        epoch = ctx._request_epochs.get(rid)
    post_finished = epoch is not None and _check_epoch_post_result(ctx, epoch=epoch)
    req = _resolve_req(ctx, rid=rid, epoch=epoch)
    return post_finished or (req is not None and req.finished())


def is_chunking(
    ctx: ScriptedContext, rid: str, *, epoch: Optional[_RequestEpoch] = None
) -> bool:
    req = _resolve_req(ctx, rid=rid, epoch=epoch)
    return req is not None and ctx.scheduler.chunked_req is req


def status(
    ctx: ScriptedContext, rid: str, *, epoch: Optional[_RequestEpoch] = None
) -> str:
    if is_finished(ctx, rid=rid, epoch=epoch):
        return "finished"
    req = find_req_by_rid(ctx, rid=rid, epoch=epoch)
    if req is None:
        return "unknown"
    if any(r is req for r in ctx.scheduler.waiting_queue):
        return "waiting"
    return "running"


def remaining_prompt_tokens(
    ctx: ScriptedContext, rid: str, *, epoch: Optional[_RequestEpoch] = None
) -> int:
    req = find_req_by_rid(ctx, rid=rid, epoch=epoch)
    if req is None:
        return 0
    return max(0, len(req.origin_input_ids) - req.kv.kv_committed_len)


def chunks_done(
    ctx: ScriptedContext, rid: str, *, epoch: Optional[_RequestEpoch] = None
) -> int:
    req = _resolve_req(ctx, rid=rid, epoch=epoch)
    if req is None:
        return 0
    log = ctx._scheduler_hook._batch_log
    held = sum(
        1
        for record in log
        if record.chunked_req is req and any(r is req for r in record.reqs)
    )
    if held == 0:
        return 0
    completed = any(
        req.rid in record.extend_rids
        and any(r is req for r in record.reqs)
        and record.chunked_req is not req
        for record in log
    )
    return held + (1 if completed else 0)


def chunked_parks(
    ctx: ScriptedContext, rid: str, *, epoch: Optional[_RequestEpoch] = None
) -> int:
    req = _resolve_req(ctx, rid=rid, epoch=epoch)
    if req is None:
        return 0
    return sum(
        1
        for record in ctx._scheduler_hook._batch_log
        if record.chunked_req is req and all(r is not req for r in record.reqs)
    )
