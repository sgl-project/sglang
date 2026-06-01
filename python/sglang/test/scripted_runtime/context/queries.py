from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.test.scripted_runtime.context.api import ScriptedContext


def _get_all_reqs(ctx: "ScriptedContext") -> Iterator["Req"]:
    s = ctx.scheduler
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


def is_idle(ctx: "ScriptedContext") -> bool:
    s = ctx.scheduler
    return (
        s.chunked_req is None
        and len(s.waiting_queue) == 0
        and (s.running_batch is None or s.running_batch.is_empty())
    )


def is_fully_idle(ctx: "ScriptedContext") -> bool:
    s = ctx.scheduler
    return is_idle(ctx) and (s.last_batch is None or s.last_batch.is_empty())


def last_batch_forward_mode(ctx: "ScriptedContext") -> Optional[str]:
    s = ctx.scheduler
    if s.last_batch is not None and s.last_batch.forward_mode is not None:
        return s.last_batch.forward_mode.name
    return None


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
    s = ctx.scheduler
    return s.chunked_req is not None and s.chunked_req.rid == rid


def status(ctx: "ScriptedContext", rid: str) -> str:
    s = ctx.scheduler
    if rid in {r.rid for r in s.waiting_queue}:
        return "waiting"
    req = find_req_by_rid(ctx, rid)
    if req is not None:
        return "finished" if req.finished() else "running"
    if rid in ctx._seen_rids:
        return "finished"
    return "unknown"


def remaining_prompt_tokens(ctx: "ScriptedContext", rid: str) -> int:
    req = find_req_by_rid(ctx, rid)
    if req is None:
        return 0
    return max(0, len(req.origin_input_ids) - req.kv_committed_len)


def chunks_done(ctx: "ScriptedContext", rid: str) -> int:
    return sum(
        1
        for record in ctx._scheduler_hook._batch_log
        if record.chunked_rid == rid and rid in record.rids
    )


def _send_idx_series(ctx: "ScriptedContext", rid: str) -> List[int]:
    # Ordered start_send_idx values observed for this rid across every step it
    # appeared in the batch. on_run_batch snapshots at step start, so the final
    # last_chunk send (done in the prefill step's result processing) is captured
    # by the subsequent decode step that still carries the req.
    series: List[int] = []
    for record in ctx._scheduler_hook._batch_log:
        if rid in record.send_idx_of_rid:
            series.append(record.send_idx_of_rid[rid])
    return series


def _prompt_len_for_rid(ctx: "ScriptedContext", rid: str) -> Optional[int]:
    for record in ctx._scheduler_hook._batch_log:
        origin_input_ids = record.origin_input_ids_of_rid.get(rid)
        if origin_input_ids is not None:
            return len(origin_input_ids)
    return None


def kv_send_events(ctx: "ScriptedContext", rid: str) -> int:
    # Number of kv-chunk sends = number of strict increases of start_send_idx
    # across the snapshotted series. Each successful send advances start_send_idx
    # to the just-sent end index (disagg prefill send_kv_chunk).
    series = _send_idx_series(ctx, rid)
    return sum(1 for prev, curr in zip(series, series[1:]) if curr > prev)


def kv_send_last_chunk_events(ctx: "ScriptedContext", rid: str) -> int:
    # The last_chunk send is the one whose end index reaches the full prompt
    # length; it advances start_send_idx to len(origin_input_ids). Count the
    # increases that land exactly on that value (expected to happen once).
    prompt_len = _prompt_len_for_rid(ctx, rid)
    if prompt_len is None:
        return 0
    series = _send_idx_series(ctx, rid)
    return sum(
        1
        for prev, curr in zip(series, series[1:])
        if curr > prev and curr == prompt_len
    )


def chunked_parks(ctx: "ScriptedContext", rid: str) -> int:
    # The dual of chunks_done: iterations where the scheduler still held this rid
    # as its chunked_req but did NOT run it in the batch -- i.e. add_chunked_req's
    # hybrid-SWA early-return parked it instead of scheduling its next chunk.
    # on_run_batch records chunked_rid right after get_next_batch_to_run in the
    # same loop iteration, so "chunked_rid set but absent from the batch" is an
    # exact, refactor-stable signal for a park, independent of any scheduler flag.
    return sum(
        1
        for record in ctx._scheduler_hook._batch_log
        if record.chunked_rid == rid and rid not in record.rids
    )
