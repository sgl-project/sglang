from __future__ import annotations

import atexit
import csv
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


COMMON_FIELDNAMES = ["wall_time_ns", "op", "duration_ms"]


def _fields(*fieldnames: str) -> list[str]:
    return [*COMMON_FIELDNAMES, *fieldnames]


TRACE_EVENT_SCHEMAS: dict[tuple[str, str], list[str]] = {
    ("verifier", "forward_batch"): _fields(
        "forward_mode",
        "model_forward_mode",
        "can_run_cuda_graph",
        "graph_path",
        "batch_size",
        "rids",
        "committed_lens_by_req",
        "output_lens_by_req",
    ),
    ("verifier", "verify_worker_timing"): _fields(
        "model_forward_mode",
        "batch_size",
        "draft_token_num",
        "num_input_tokens",
        "seq_lens_sum",
        "draft_tree_mask_numel",
        "target_can_run_cuda_graph",
        "reported_can_run_cuda_graph",
        "valid_tail_sum",
        "valid_tail_min",
        "valid_tail_max",
        "num_accepted_drafts",
        "accepted_tokens_num",
        "draft_ms",
        "draft_preamble_ms",
        "draft_build_tokens_ms",
        "draft_get_verify_buffers_ms",
        "draft_build_tree_ms",
        "draft_terminal_mask_ms",
        "prepare_verify_ms",
        "target_forward_ms",
        "cuda_graph_replay_prepare_ms",
        "cuda_graph_replay_ms",
        "verify_impl",
        "eagle_verify_ms",
        "assert_ms",
        "total_worker_ms",
    ),
    ("decode", "forward_batch"): _fields(
        "forward_mode",
        "model_forward_mode",
        "can_run_cuda_graph",
        "graph_path",
        "batch_size",
        "rids",
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
    ("verifier", "build_commit_batch"): _fields(
        "forward_mode",
        "batch_size",
        "rids",
        "pre_committed_lens_by_req",
        "draft_buffer_lens_by_req",
        "accepted_tail_lens_by_req",
        "bonus_token_ids_by_req",
        "snapshot_candidate_token_ids_by_req",
        "committed_lens_by_req",
        "output_lens_by_req",
        "dst_drafter_ranks",
    ),
    ("verifier", "build_close_batch"): _fields(
        "forward_mode",
        "batch_size",
        "rids",
        "num_close",
        "output_lens_by_req",
        "dst_drafter_ranks",
    ),
    ("drafter", "forward_batch"): _fields(
        "forward_mode",
        "model_forward_mode",
        "can_run_cuda_graph",
        "graph_path",
        "batch_size",
        "rids",
        "committed_lens_by_req",
        "output_lens_by_req",
    ),
    ("drafter", "recv_sync_batch"): _fields(
        "batch_size",
        "rids",
        "committed_lens_by_req",
        "output_lens_by_req",
    ),
    ("drafter", "create_draft_req_batch"): _fields(
        "batch_size",
        "rids",
        "committed_lens_by_req",
        "output_lens_by_req",
    ),
    ("drafter", "apply_commit_batch"): _fields(
        "forward_mode",
        "batch_size",
        "rids",
        "committed_lens_by_req",
        "num_applied_commit",
        "num_deferred_commit",
    ),
    ("drafter", "post_decode_control_batch"): _fields(
        "forward_mode",
        "batch_size",
        "rids",
        "num_commit",
        "num_close",
        "num_applied_commit",
        "num_deferred_commit",
    ),
    ("drafter", "emit_tail_batch"): _fields(
        "forward_mode",
        "batch_size",
        "rids",
        "num_stream_outputs",
        "committed_lens_by_req",
        "output_lens_by_req",
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
    ("token_sync_thread", "drain_sync_batch"): _fields(
        "drafter_rank",
        "batch_size",
        "rids",
        "num_sync",
    ),
    ("token_sync_thread", "drain_post_result_batch"): _fields(
        "drafter_rank",
        "batch_size",
        "rids",
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


class NullDecoupledSpecTracer:
    enabled = False

    def record(self, component: str, op: str, **fields: Any) -> None:
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

    def record(self, component: str, op: str, **fields: Any) -> None:
        if self._closed.is_set():
            return
        row = {"wall_time_ns": time.time_ns(), "op": op}
        row.update(fields)
        self._validate_event(component, op, row)
        self._queue.put(_TraceEvent(component=component, row=row))

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

        component_file_name = self.file_names.get(component, f"{component}.csv")
        path = Path(component_file_name)
        suffix = path.suffix or ".csv"
        stem = path.name[: -len(path.suffix)] if path.suffix else path.name
        return str(path.with_name(f"{stem}__{op}{suffix}"))

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
