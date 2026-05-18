from __future__ import annotations

import atexit
import csv
import json
import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


COMMON_FIELDNAMES = ["wall_time_ns", "op", "duration_ms"]


def _fields(*fieldnames: str) -> list[str]:
    return [*COMMON_FIELDNAMES, *fieldnames]


TRACE_EVENT_SCHEMAS: dict[tuple[str, str], list[str]] = {
    ("scheduler", "forward_batch"): _fields(
        "worker_role",
        "forward_mode",
        "model_forward_mode",
        "can_run_cuda_graph",
        "graph_path",
        "batch_size",
        "rids",
        "committed_lens_by_req",
        "output_lens_by_req",
    ),
    ("verifier", "build_sync_batch"): _fields(
        "forward_mode",
        "batch_size",
        "rids",
        "committed_lens_by_req",
        "output_lens_by_req",
        "dst_drafter_ranks",
    ),
    ("verifier", "snapshot_tail_batch"): _fields(
        "forward_mode",
        "batch_size",
        "rids",
        "valid_tail_lens_by_req",
        "raw_tail_lens_by_req",
        "committed_lens_by_req",
        "output_lens_by_req",
    ),
    ("verifier", "build_update_batch"): _fields(
        "forward_mode",
        "batch_size",
        "commit_rids",
        "close_rids",
        "num_commit",
        "num_close",
        "pre_committed_lens_by_req",
        "draft_buffer_lens_by_req",
        "accepted_tail_lens_by_req",
        "bonus_token_ids_by_req",
        "snapshot_candidate_token_ids_by_req",
        "committed_lens_by_req",
        "commit_output_lens_by_req",
        "commit_dst_drafter_ranks",
        "close_output_lens_by_req",
        "close_dst_drafter_ranks",
    ),
    ("drafter", "sync_control_messages"): _fields(
        "num_messages",
        "num_sync",
        "num_commit",
        "num_close",
        "num_created_reqs",
        "num_applied_commit",
        "num_pending_commit",
        "num_sleeping_reqs",
    ),
    ("drafter", "emit_draft_tokens"): _fields(
        "num_emit_candidates",
        "num_stream_outputs",
        "committed_lens_by_req",
        "output_lens_by_req",
    ),
    ("drafter", "sleep_requests"): _fields(
        "num_slept",
        "slept_rids",
    ),
    ("drafter", "wake_requests"): _fields(
        "num_woken",
        "woken_rids",
    ),
    ("draft_proxy", "send_control_batch"): _fields(
        "verifier_rank",
        "dst_drafter_rank",
        "batch_size",
        "rids",
        "num_sync",
        "num_commit",
        "num_close",
    ),
    ("draft_proxy", "recv_tail_stream_batch"): _fields(
        "verifier_rank",
        "batch_size",
        "rids",
        "num_stream_outputs",
        "draft_token_lens_by_req",
    ),
    ("draft_proxy", "append_tail_stream_batch"): _fields(
        "verifier_rank",
        "batch_size",
        "rids",
        "num_stream_outputs",
        "draft_token_lens_by_req",
        "num_appended_outputs",
        "num_duplicate_outputs",
        "num_stale_base_outputs",
        "num_already_committed_outputs",
        "num_stale_gap_outputs",
        "num_unknown_request_outputs",
        "appended_token_lens_by_req",
        "tail_lens_after_by_req",
        "consumable_tail_lens_after_by_req",
        "committed_lens_after_by_req",
    ),
    ("draft_proxy", "apply_control_batch"): _fields(
        "verifier_rank",
        "batch_size",
        "rids",
        "num_sync",
        "num_commit",
        "num_close",
        "commit_rids",
        "pre_committed_lens_by_req",
        "accepted_tail_lens_by_req",
        "raw_tail_lens_before_by_req",
        "bonus_token_ids_by_req",
        "buffer_candidate_token_ids_by_req",
        "bonus_match_by_req",
        "preserved_suffix_lens_by_req",
        "tail_lens_after_by_req",
        "committed_lens_after_by_req",
    ),
    ("token_sync_thread", "recv_control_batch"): _fields(
        "drafter_rank",
        "batch_size",
        "rids",
        "num_sync",
        "num_commit",
        "num_close",
    ),
    ("token_sync_thread", "drain_control_batch"): _fields(
        "drafter_rank",
        "batch_size",
        "rids",
        "num_sync",
        "num_commit",
        "num_close",
    ),
    ("token_sync_thread", "enqueue_draft_result_batch"): _fields(
        "drafter_rank",
        "batch_size",
        "rids",
        "num_stream_outputs",
        "emitted_token_lens_by_req",
    ),
    ("token_sync_thread", "send_result_batch"): _fields(
        "drafter_rank",
        "dst_verifier_rank",
        "batch_size",
        "rids",
        "num_stream_outputs",
        "emitted_token_lens_by_req",
    ),
    ("token_sync_thread", "drain_outgoing_results"): _fields(
        "drafter_rank",
        "queue_size_before",
        "queue_size_after",
        "num_result_batches",
        "num_stream_outputs",
    ),
    ("token_sync_thread", "drain_control_socket"): _fields(
        "drafter_rank",
        "pending_controls_before",
        "pending_controls_after",
        "num_control_batches",
        "num_control_messages",
    ),
    ("token_sync_thread", "idle_wait"): _fields(
        "drafter_rank",
        "wait_timeout_ms",
        "wakeup_set_before_wait",
        "wakeup_set_after_wait",
        "queue_size_before_wait",
        "queue_size_after_wait",
        "pending_controls_before_wait",
        "pending_controls_after_wait",
    ),
}


@dataclass
class _TraceEvent:
    component: str
    row: dict[str, Any]


TraceFieldsFn = Callable[
    [Any, tuple[Any, ...], dict[str, Any], Any], dict[str, Any] | None
]
TracePredicate = Callable[[Any, tuple[Any, ...], dict[str, Any], Any], bool]
TraceHook = Callable[[Any, tuple[Any, ...], dict[str, Any], Any], None]
TraceGetter = Callable[[Any], Any]


class DecoupledSpecTraceEvent(Enum):
    SCHEDULER_FORWARD_BATCH = auto()
    VERIFIER_BUILD_SYNC_BATCH = auto()
    VERIFIER_SNAPSHOT_TAIL_BATCH = auto()
    VERIFIER_BUILD_UPDATE_BATCH = auto()
    DRAFTER_SYNC_CONTROL_MESSAGES = auto()
    DRAFTER_EMIT_DRAFT_TOKENS = auto()
    DRAFTER_SLEEP_REQUESTS = auto()
    DRAFTER_WAKE_REQUESTS = auto()
    DRAFT_PROXY_SEND_CONTROL_BATCH = auto()
    DRAFT_PROXY_RECV_TAIL_STREAM_BATCH = auto()
    DRAFT_PROXY_APPEND_TAIL_STREAM_BATCH = auto()
    DRAFT_PROXY_APPLY_CONTROL_BATCH = auto()
    TOKEN_SYNC_RECV_CONTROL_BATCH = auto()
    TOKEN_SYNC_DRAIN_CONTROL_BATCH = auto()
    TOKEN_SYNC_ENQUEUE_DRAFT_RESULT_BATCH = auto()
    TOKEN_SYNC_SEND_RESULT_BATCH = auto()
    TOKEN_SYNC_DRAIN_OUTGOING_RESULTS = auto()
    TOKEN_SYNC_DRAIN_CONTROL_SOCKET = auto()
    TOKEN_SYNC_IDLE_WAIT = auto()


@dataclass(frozen=True)
class _TraceEventSpec:
    component: str
    op: str
    fields: TraceFieldsFn
    record_if: TracePredicate | None = None
    trace_if: TracePredicate | None = None
    before_start: TraceHook | None = None
    before_record: TraceHook | None = None


def _default_trace_getter(owner: Any) -> Any:
    tracer = getattr(owner, "tracer", None)
    if tracer is None:
        tracer = getattr(owner, "decoupled_spec_tracer", None)
    return tracer


def _payload_fields(
    owner: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any] | None:
    if isinstance(result, dict):
        return result
    result_payload = getattr(result, "_decoupled_spec_payload", None)
    if isinstance(result_payload, dict):
        return result_payload
    if args and isinstance(args[0], dict):
        return args[0]
    return None


def _non_empty_result(
    owner: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> bool:
    return bool(result)


def _control_batch_messages(batch: Any) -> list[Any]:
    return [
        *list(getattr(batch, "sync_messages", []) or []),
        *list(getattr(batch, "verify_commit_messages", []) or []),
        *list(getattr(batch, "close_messages", []) or []),
    ]


def _is_sync_message(message: Any) -> bool:
    return hasattr(message, "committed_output_ids")


def _is_commit_message(message: Any) -> bool:
    return hasattr(message, "bonus_token_pos")


def _is_close_message(message: Any) -> bool:
    return hasattr(message, "reason")


def _draft_proxy_send_control_fields(
    owner: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    batch = args[0]
    fields = _draft_proxy_control_fields(owner, batch)
    fields["dst_drafter_rank"] = int(batch.dst_drafter_rank)
    return fields


def _draft_proxy_apply_control_fields(
    owner: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    fields = _draft_proxy_control_fields(owner, args[0])
    fields.update(
        result
        or {
            "commit_rids": [],
            "pre_committed_lens_by_req": [],
            "accepted_tail_lens_by_req": [],
            "raw_tail_lens_before_by_req": [],
            "bonus_token_ids_by_req": [],
            "buffer_candidate_token_ids_by_req": [],
            "bonus_match_by_req": [],
            "preserved_suffix_lens_by_req": [],
            "tail_lens_after_by_req": [],
            "committed_lens_after_by_req": [],
        }
    )
    return fields


def _draft_proxy_control_fields(owner: Any, batch: Any) -> dict[str, Any]:
    messages = _control_batch_messages(batch)
    return {
        "verifier_rank": int(owner.verifier_rank),
        "batch_size": len(messages),
        "rids": [message.request_id for message in messages],
        "num_sync": len(getattr(batch, "sync_messages", []) or []),
        "num_commit": len(getattr(batch, "verify_commit_messages", []) or []),
        "num_close": len(getattr(batch, "close_messages", []) or []),
    }


def _draft_proxy_recv_tail_fields(
    owner: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    return _draft_proxy_tail_fields(owner, result)


def _draft_proxy_append_tail_fields(
    owner: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    output_batch = args[0]
    request_ids, draft_token_lens_by_req = _count_outputs_by_request(output_batch)
    fields: dict[str, Any] = {
        "verifier_rank": int(owner.verifier_rank),
        "batch_size": len(request_ids),
        "rids": request_ids,
        "num_stream_outputs": len(getattr(output_batch, "outputs", []) or []),
        "draft_token_lens_by_req": draft_token_lens_by_req,
    }
    fields.update(
        result
        or {
            "num_appended_outputs": 0,
            "num_duplicate_outputs": 0,
            "num_stale_base_outputs": 0,
            "num_already_committed_outputs": 0,
            "num_stale_gap_outputs": 0,
            "num_unknown_request_outputs": 0,
            "appended_token_lens_by_req": [0] * len(request_ids),
            "tail_lens_after_by_req": [0] * len(request_ids),
            "consumable_tail_lens_after_by_req": [0] * len(request_ids),
            "committed_lens_after_by_req": [
                owner.draft_tail_buffer.get_committed_len(request_id) or 0
                for request_id in request_ids
            ],
        }
    )
    return fields


def _draft_proxy_tail_fields(owner: Any, output_batch: Any) -> dict[str, Any]:
    request_ids, draft_token_lens_by_req = _count_outputs_by_request(output_batch)
    return {
        "verifier_rank": int(owner.verifier_rank),
        "batch_size": len(request_ids),
        "rids": request_ids,
        "num_stream_outputs": len(getattr(output_batch, "outputs", []) or []),
        "draft_token_lens_by_req": draft_token_lens_by_req,
    }


def _token_sync_control_message_fields(
    owner: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    messages = result
    if messages is None:
        return None
    return {
        "drafter_rank": int(owner.drafter_rank),
        "batch_size": len(messages),
        "rids": [message.request_id for message in messages],
        "num_sync": sum(_is_sync_message(message) for message in messages),
        "num_commit": sum(_is_commit_message(message) for message in messages),
        "num_close": sum(_is_close_message(message) for message in messages),
    }


def _token_sync_enqueue_result_fields(
    owner: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    return _token_sync_draft_result_fields(owner, args[0])


def _token_sync_send_result_fields(
    owner: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    return _token_sync_draft_result_fields(
        owner,
        args[1],
        dst_verifier_rank=args[0],
    )


def _token_sync_draft_result_fields(
    owner: Any,
    result_batch: Any,
    *,
    dst_verifier_rank: int | None = None,
) -> dict[str, Any]:
    request_ids, emitted_token_lens_by_req = _count_outputs_by_request(result_batch)
    fields: dict[str, Any] = {
        "drafter_rank": int(owner.drafter_rank),
        "batch_size": len(request_ids),
        "rids": request_ids,
        "num_stream_outputs": len(getattr(result_batch, "outputs", []) or []),
        "emitted_token_lens_by_req": emitted_token_lens_by_req,
    }
    if dst_verifier_rank is not None:
        fields["dst_verifier_rank"] = int(dst_verifier_rank)
    return fields


def _count_outputs_by_request(batch: Any) -> tuple[list[str], list[int]]:
    counts_by_request: dict[str, int] = {}
    for output in getattr(batch, "outputs", []) or []:
        counts_by_request[output.request_id] = (
            counts_by_request.get(output.request_id, 0) + 1
        )
    request_ids = list(counts_by_request.keys())
    return request_ids, [counts_by_request[request_id] for request_id in request_ids]


def _scheduler_forward_call_args(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, int | None, Any]:
    batch = args[0] if args else kwargs.get("batch")
    start_ns = args[1] if len(args) > 1 else kwargs.get("start_ns")
    result = args[2] if len(args) > 2 else kwargs.get("result")
    return batch, start_ns, result


def _scheduler_component(owner: Any) -> str | None:
    if owner.spec_algorithm.is_decoupled_verify():
        return "verifier"
    if owner.spec_algorithm.is_decoupled_draft():
        return "drafter"
    if owner.spec_algorithm.is_none():
        return "decode"
    return None


def _scheduler_forward_fields(
    owner: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    result: Any,
) -> dict[str, Any] | None:
    worker_role = _scheduler_component(owner)
    if worker_role is None:
        return None
    batch, start_ns, forward_result = _scheduler_forward_call_args(args, kwargs)
    if start_ns is None or batch is None:
        return None
    can_run_cuda_graph = getattr(forward_result, "can_run_cuda_graph", None)
    model_forward_mode = getattr(forward_result, "model_forward_mode", None)
    committed_lens_by_req = []
    if worker_role == "drafter":
        committed_lens_by_req = [
            int(owner._get_draft_state_by_req(req).verifier_committed_prefix_len)
            for req in batch.reqs
        ]
    elif worker_role == "verifier":
        committed_lens_by_req = [len(req.output_ids) for req in batch.reqs]
    event_fields = dict(
        duration_ms=owner.decoupled_spec_tracer.elapsed_ms(start_ns),
        worker_role=worker_role,
        forward_mode=str(batch.forward_mode),
        model_forward_mode=model_forward_mode or str(batch.forward_mode),
        can_run_cuda_graph=(
            bool(can_run_cuda_graph) if can_run_cuda_graph is not None else False
        ),
        graph_path=(
            owner._infer_forward_graph_path(batch, bool(can_run_cuda_graph))
            if can_run_cuda_graph is not None
            else ""
        ),
        batch_size=len(batch.reqs),
        rids=[
            (
                owner._get_draft_state_by_req(req).key.request_id
                if owner.spec_algorithm.is_decoupled_draft()
                else req.rid
            )
            for req in batch.reqs
        ],
        committed_lens_by_req=committed_lens_by_req,
        output_lens_by_req=[len(req.output_ids) for req in batch.reqs],
    )
    return event_fields


TRACE_EVENT_SPECS: dict[DecoupledSpecTraceEvent, _TraceEventSpec] = {
    DecoupledSpecTraceEvent.SCHEDULER_FORWARD_BATCH: _TraceEventSpec(
        "scheduler", "forward_batch", _scheduler_forward_fields
    ),
    DecoupledSpecTraceEvent.VERIFIER_BUILD_SYNC_BATCH: _TraceEventSpec(
        "verifier", "build_sync_batch", _payload_fields
    ),
    DecoupledSpecTraceEvent.VERIFIER_SNAPSHOT_TAIL_BATCH: _TraceEventSpec(
        "verifier", "snapshot_tail_batch", _payload_fields
    ),
    DecoupledSpecTraceEvent.VERIFIER_BUILD_UPDATE_BATCH: _TraceEventSpec(
        "verifier", "build_update_batch", _payload_fields
    ),
    DecoupledSpecTraceEvent.DRAFTER_SYNC_CONTROL_MESSAGES: _TraceEventSpec(
        "drafter", "sync_control_messages", _payload_fields
    ),
    DecoupledSpecTraceEvent.DRAFTER_EMIT_DRAFT_TOKENS: _TraceEventSpec(
        "drafter", "emit_draft_tokens", _payload_fields
    ),
    DecoupledSpecTraceEvent.DRAFTER_SLEEP_REQUESTS: _TraceEventSpec(
        "drafter", "sleep_requests", _payload_fields
    ),
    DecoupledSpecTraceEvent.DRAFTER_WAKE_REQUESTS: _TraceEventSpec(
        "drafter", "wake_requests", _payload_fields
    ),
    DecoupledSpecTraceEvent.DRAFT_PROXY_SEND_CONTROL_BATCH: _TraceEventSpec(
        "draft_proxy", "send_control_batch", _draft_proxy_send_control_fields
    ),
    DecoupledSpecTraceEvent.DRAFT_PROXY_RECV_TAIL_STREAM_BATCH: _TraceEventSpec(
        "draft_proxy", "recv_tail_stream_batch", _draft_proxy_recv_tail_fields
    ),
    DecoupledSpecTraceEvent.DRAFT_PROXY_APPEND_TAIL_STREAM_BATCH: _TraceEventSpec(
        "draft_proxy", "append_tail_stream_batch", _draft_proxy_append_tail_fields
    ),
    DecoupledSpecTraceEvent.DRAFT_PROXY_APPLY_CONTROL_BATCH: _TraceEventSpec(
        "draft_proxy", "apply_control_batch", _draft_proxy_apply_control_fields
    ),
    DecoupledSpecTraceEvent.TOKEN_SYNC_RECV_CONTROL_BATCH: _TraceEventSpec(
        "token_sync_thread", "recv_control_batch", _token_sync_control_message_fields
    ),
    DecoupledSpecTraceEvent.TOKEN_SYNC_DRAIN_CONTROL_BATCH: _TraceEventSpec(
        "token_sync_thread",
        "drain_control_batch",
        _token_sync_control_message_fields,
        _non_empty_result,
    ),
    DecoupledSpecTraceEvent.TOKEN_SYNC_ENQUEUE_DRAFT_RESULT_BATCH: _TraceEventSpec(
        "token_sync_thread",
        "enqueue_draft_result_batch",
        _token_sync_enqueue_result_fields,
        lambda owner, args, kwargs, result: bool(getattr(args[0], "outputs", [])),
    ),
    DecoupledSpecTraceEvent.TOKEN_SYNC_SEND_RESULT_BATCH: _TraceEventSpec(
        "token_sync_thread", "send_result_batch", _token_sync_send_result_fields
    ),
    DecoupledSpecTraceEvent.TOKEN_SYNC_DRAIN_OUTGOING_RESULTS: _TraceEventSpec(
        "token_sync_thread", "drain_outgoing_results", _payload_fields
    ),
    DecoupledSpecTraceEvent.TOKEN_SYNC_DRAIN_CONTROL_SOCKET: _TraceEventSpec(
        "token_sync_thread", "drain_control_socket", _payload_fields
    ),
    DecoupledSpecTraceEvent.TOKEN_SYNC_IDLE_WAIT: _TraceEventSpec(
        "token_sync_thread", "idle_wait", _payload_fields
    ),
}


def trace_decoupled_spec(
    event: DecoupledSpecTraceEvent,
    tracer_getter: TraceGetter | None = None,
    inject_trace_enabled: str | None = None,
):
    """Trace one decoupled-spec event selected by enum."""
    event_spec = TRACE_EVENT_SPECS[event]

    def decorator(fn):
        @wraps(fn)
        def wrapper(owner, *args, **kwargs):
            tracer = (
                tracer_getter(owner)
                if tracer_getter is not None
                else _default_trace_getter(owner)
            )
            if not getattr(tracer, "enabled", False):
                if inject_trace_enabled is not None:
                    kwargs.setdefault(inject_trace_enabled, False)
                return fn(owner, *args, **kwargs)

            if inject_trace_enabled is not None:
                kwargs.setdefault(inject_trace_enabled, True)
            start_ns = tracer.start_timer()
            result = fn(owner, *args, **kwargs)
            if event_spec.record_if is not None and not event_spec.record_if(
                owner, args, kwargs, result
            ):
                return result

            event_fields = event_spec.fields(owner, args, kwargs, result)
            if event_fields is None:
                return result
            tracer.record_duration(
                event_spec.component,
                event_spec.op,
                start_ns,
                **event_fields,
            )
            return result

        return wrapper

    return decorator


class NullDecoupledSpecTracer:
    enabled = False

    def start_timer(self) -> None:
        return None

    def elapsed_ms(self, start_ns: int | None) -> float:
        return 0.0

    def record(self, component: str, op: str, **fields: Any) -> None:
        return

    def record_duration(
        self,
        component: str,
        op: str,
        start_ns: int | None,
        **fields: Any,
    ) -> None:
        return

    def close(self) -> None:
        return


class DecoupledSpecCsvTracer:
    enabled = True

    def __init__(self, *, output_dir: str | Path, file_names: dict[str, str]) -> None:
        self.output_dir = Path(output_dir).expanduser()
        self.file_names = dict(file_names)
        self._queue: queue.Queue[_TraceEvent | None] = queue.Queue(maxsize=0)
        self._closed = threading.Event()
        self._writers: dict[tuple[str, str], csv.DictWriter] = {}
        self._files: dict[tuple[str, str], Any] = {}
        self._thread = threading.Thread(
            target=self._run,
            name="sglang-decoupled-spec-trace-writer",
            daemon=True,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        atexit.register(self.close)

    def start_timer(self) -> int:
        return time.perf_counter_ns()

    def elapsed_ms(self, start_ns: int | None) -> float:
        if start_ns is None:
            return 0.0
        return (time.perf_counter_ns() - start_ns) / 1_000_000

    def record(self, component: str, op: str, **fields: Any) -> None:
        if self._closed.is_set():
            return
        row = {"wall_time_ns": time.time_ns(), "op": op}
        row.update(fields)
        self._validate_event(component, op, row)
        self._queue.put(_TraceEvent(component=component, row=row))

    def record_duration(
        self,
        component: str,
        op: str,
        start_ns: int | None,
        **fields: Any,
    ) -> None:
        fields.setdefault("duration_ms", self.elapsed_ms(start_ns))
        self.record(component, op, **fields)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join()

    def _run(self) -> None:
        try:
            while True:
                event = self._queue.get()
                if event is None:
                    break
                self._write_event(event)
        except Exception:
            logger.exception("Decoupled spec trace writer failed")
        finally:
            self._flush_files()
            for file in self._files.values():
                try:
                    file.close()
                except Exception:
                    logger.exception("Failed to close decoupled spec trace file")

    def _write_event(self, event: _TraceEvent) -> None:
        op = str(event.row["op"])
        writer = self._get_writer(event.component, op)
        fieldnames = TRACE_EVENT_SCHEMAS[(event.component, op)]
        row = {}
        for key, value in event.row.items():
            row[key] = self._serialize_value(value)
        writer.writerow(row)
        # Scheduler processes may be killed during engine shutdown; flush each
        # event so trace CSVs are usable even without a graceful close.
        self._files[(event.component, op)].flush()

    def _flush_files(self) -> None:
        for file in self._files.values():
            try:
                file.flush()
            except Exception:
                logger.exception("Failed to flush decoupled spec trace file")

    def _get_writer(self, component: str, op: str) -> csv.DictWriter:
        event_key = (component, op)
        fieldnames = TRACE_EVENT_SCHEMAS[event_key]
        writer = self._writers.get(event_key)
        if writer is not None:
            return writer

        file_name = self._event_file_name(component, op)
        path = self.output_dir / file_name
        file = path.open("w", newline="", buffering=1024 * 1024)
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        self._files[event_key] = file
        self._writers[event_key] = writer
        return writer

    def _event_file_name(self, component: str, op: str) -> str:
        event_file_name = self.file_names.get(f"{component}.{op}")
        if event_file_name is not None:
            return event_file_name

        component_file_name = self.file_names.get(component)
        if component_file_name is not None and self._component_has_single_op(component):
            return component_file_name

        component_file_name = component_file_name or f"{component}.csv"
        path = Path(component_file_name)
        suffix = path.suffix or ".csv"
        stem = path.name[: -len(path.suffix)] if path.suffix else path.name
        return str(path.with_name(f"{stem}__{op}{suffix}"))

    def _component_has_single_op(self, component: str) -> bool:
        return sum(
            1
            for event_component, _ in TRACE_EVENT_SCHEMAS
            if event_component == component
        ) == 1

    def _validate_event(self, component: str, op: str, row: dict[str, Any]) -> None:
        event_key = (component, op)
        fieldnames = TRACE_EVENT_SCHEMAS.get(event_key)
        if fieldnames is None:
            raise ValueError(
                f"Unknown decoupled spec trace event: component={component} op={op}"
            )

        expected = set(fieldnames)
        actual = set(row)
        missing = expected - actual
        if missing:
            raise ValueError(
                "Missing decoupled spec trace fields for "
                f"component={component} op={op}: {sorted(missing)}"
            )
        extra = actual - expected
        if extra:
            raise ValueError(
                "Unexpected decoupled spec trace fields for "
                f"component={component} op={op}: {sorted(extra)}"
            )
        null_fields = [key for key, value in row.items() if value is None]
        if null_fields:
            raise ValueError(
                "Null decoupled spec trace fields for "
                f"component={component} op={op}: {sorted(null_fields)}"
            )

    def _serialize_value(self, value: Any) -> str:
        if value is None:
            raise ValueError("Decoupled spec trace fields cannot be None")
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def build_decoupled_spec_tracer(
    *,
    enabled: bool,
    output_dir: str | None,
    file_names: dict[str, str],
) -> NullDecoupledSpecTracer | DecoupledSpecCsvTracer:
    if not enabled:
        return NullDecoupledSpecTracer()
    return DecoupledSpecCsvTracer(
        output_dir=output_dir or "decoupled_spec_trace",
        file_names=file_names,
    )
