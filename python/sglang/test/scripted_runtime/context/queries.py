from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional

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


def is_idle(ctx: "ScriptedContext") -> bool:
    s = ctx._scheduler
    return (
        s.chunked_req is None
        and len(s.waiting_queue) == 0
        and (s.running_batch is None or s.running_batch.is_empty())
    )


def is_fully_idle(ctx: "ScriptedContext") -> bool:
    s = ctx._scheduler
    return is_idle(ctx) and (s.last_batch is None or s.last_batch.is_empty())


def last_batch_forward_mode(ctx: "ScriptedContext") -> Optional[str]:
    s = ctx._scheduler
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
    s = ctx._scheduler
    return s.chunked_req is not None and s.chunked_req.rid == rid


def status(ctx: "ScriptedContext", rid: str) -> str:
    s = ctx._scheduler
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


def hisparse_dma_in_flight(ctx: "ScriptedContext", rid: str) -> bool:
    # Req.hisparse_staging is the real per-request bool the engine sets while this
    # request's KV is staged for an in-flight hisparse DMA. A finished/aborted req
    # that has released its staging is absent or has the flag cleared.
    req = find_req_by_rid(ctx, rid)
    return req is not None and req.hisparse_staging


def eagle_topk_p_captured(ctx: "ScriptedContext", rid: str) -> bool:
    # True if any step where this rid was in the batch carried a populated
    # EagleDraftInput.topk_p, i.e. the eagle draft input was captured during the
    # request's lifetime (read-only OR over on_run_batch snapshots).
    return any(
        record.eagle_topk_p_present
        for record in ctx._scheduler_hook._batch_log
        if rid in record.rids
    )


def eagle_topk_index_captured(ctx: "ScriptedContext", rid: str) -> bool:
    return any(
        record.eagle_topk_index_present
        for record in ctx._scheduler_hook._batch_log
        if rid in record.rids
    )


def eagle_hidden_states_captured(ctx: "ScriptedContext", rid: str) -> bool:
    return any(
        record.eagle_hidden_states_present
        for record in ctx._scheduler_hook._batch_log
        if rid in record.rids
    )


def _send_idx_series(ctx: "ScriptedContext", rid: str) -> List[int]:
    # Ordered start_send_idx values observed for this rid across every step it
    # appeared in the batch. on_run_batch snapshots at step start, so the final
    # last_chunk send (done in the prefill step's result processing) is captured
    # by the subsequent decode step that still carries the req.
    series: List[int] = []
    for record in ctx._scheduler_hook._batch_log:
        for record_rid, send_idx in record.send_idx_by_rid:
            if record_rid == rid:
                series.append(send_idx)
    return series


def _prompt_len_for_rid(ctx: "ScriptedContext", rid: str) -> Optional[int]:
    for record in ctx._scheduler_hook._batch_log:
        for record_rid, prompt_len in record.prompt_len_by_rid:
            if record_rid == rid:
                return prompt_len
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


def load_inquirer_num_pending_tokens(ctx: "ScriptedContext") -> int:
    # The scheduler's own SchedulerLoadInquirer lives on the driver process, so
    # its pending-token tally is directly readable here. _get_num_pending_tokens
    # is the exact value the scheduler reports for load balancing; reading it (not
    # recomputing) keeps the test honest about what the engine actually decides.
    return ctx._scheduler.load_inquirer._get_num_pending_tokens()


def _pending_tokens_count_for_rid(ctx: "ScriptedContext", rid: str) -> int:
    # Per-rid contribution to the load inquirer's pending-token tally, derived
    # from the same real state _get_num_pending_tokens sums over: a waiting-queue
    # req contributes its full seqlen, the single chunked req contributes only
    # the part not yet committed to its prefix.
    s = ctx._scheduler
    chunked = s.chunked_req
    if chunked is not None and chunked.rid == rid:
        return chunked.seqlen - len(chunked.prefix_indices)
    for req in s.waiting_queue:
        if req.rid == rid:
            return req.seqlen
    return 0


def load_inquirer_snapshot(ctx: "ScriptedContext") -> Dict[str, Any]:
    # Read-only derived view of the load inquirer at this instant. Not cached: the
    # callable re-derives from live scheduler state each time it is invoked.
    pending_tokens_count_for_rid: Callable[[str], int] = (
        lambda rid: _pending_tokens_count_for_rid(ctx, rid)
    )
    return {
        "num_pending_tokens": load_inquirer_num_pending_tokens(ctx),
        "pending_tokens_count_for_rid": pending_tokens_count_for_rid,
    }


def in_flight_other_mb_rids(ctx: "ScriptedContext") -> List[str]:
    # Under pipeline parallelism the scheduler keeps one ScheduleBatch per
    # micro-batch slot in running_mbs; the slot it is currently servicing is
    # running_batch. Reqs sitting in the *other* slots are in flight in another
    # micro-batch and must be excluded from the local running set by filter_batch.
    # running_mbs only exists once init_pp_loop_state has run (PP path); without
    # PP there are no other micro-batches.
    s = ctx._scheduler
    if not hasattr(s, "running_mbs"):
        return []
    current = s.running_batch
    rids: List[str] = []
    for mb in s.running_mbs:
        if mb is current or mb is None:
            continue
        rids.extend(r.rid for r in mb.reqs)
    return rids


def stream_events(ctx: "ScriptedContext", rid: str) -> List[Any]:
    # The streamed batch outputs the scheduler has emitted toward the tokenizer
    # for this rid so far. While a request is still mid-chunk (prefill not done)
    # the scheduler emits nothing, so this list stays empty; once the request
    # produces decode tokens each streamed step shows up as one entry.
    proxy = ctx._tokenizer_recv_proxy
    assert (
        proxy is not None
    ), "stream_events requires the tokenizer recv proxy (start the server with it)"
    return proxy.buffered_objects_for_rid(rid)


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
