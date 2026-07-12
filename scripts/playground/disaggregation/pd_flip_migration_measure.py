#!/usr/bin/env python3
"""Sample and summarize PD flip migration link events.

This helper is intentionally standard-library only and Python 3.6 compatible so
it can run directly on older controller hosts.
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


JsonDict = Dict[str, Any]


class HttpClient:
    def __init__(self, timeout_seconds: float = 5.0, api_key: Optional[str] = None):
        self.timeout_seconds = timeout_seconds
        self.api_key = api_key

    def get_json(self, url: str) -> Any:
        headers = {}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
        return json.loads(raw)


def parse_node_specs(specs: Sequence[str]) -> List[JsonDict]:
    nodes = []
    for spec in specs:
        item = {}
        for part in spec.split(","):
            if not part:
                continue
            if "=" not in part:
                raise ValueError("node spec part must be key=value: %s" % part)
            key, value = part.split("=", 1)
            item[key.strip()] = value.strip()
        if "name" not in item or "worker_url" not in item:
            raise ValueError("node spec requires name and worker_url: %s" % spec)
        nodes.append(item)
    return nodes


def sample_loop(
    *,
    router_url: str,
    nodes: Sequence[JsonDict],
    output_events: Path,
    interval_seconds: float,
    duration_seconds: float,
    http_timeout_seconds: float,
    api_key: Optional[str] = None,
    router_api_key: Optional[str] = None,
) -> None:
    output_events.parent.mkdir(parents=True, exist_ok=True)
    client = HttpClient(timeout_seconds=http_timeout_seconds, api_key=api_key)
    router_client = HttpClient(
        timeout_seconds=http_timeout_seconds, api_key=router_api_key or api_key
    )
    deadline = time.monotonic() + duration_seconds
    url_to_name = {node["worker_url"].rstrip("/"): node["name"] for node in nodes}

    with output_events.open("w", encoding="utf-8") as f:
        while time.monotonic() < deadline:
            sample_started = time.monotonic()
            write_event(f, collect_router_event(router_client, router_url, url_to_name))
            for node in nodes:
                for event in collect_worker_events(client, node):
                    write_event(f, event)
            sleep_s = interval_seconds - (time.monotonic() - sample_started)
            if sleep_s > 0:
                time.sleep(sleep_s)


def now_fields() -> JsonDict:
    return {"ts_wall": time.time(), "ts_mono": time.monotonic()}


def write_event(handle: Any, event: JsonDict) -> None:
    handle.write(json.dumps(event, sort_keys=True) + "\n")
    handle.flush()


def collect_router_event(
    client: HttpClient, router_url: str, url_to_name: Dict[str, str]
) -> JsonDict:
    event = now_fields()
    event.update({"event_type": "router_workers", "component": "router"})
    try:
        payload = client.get_json(router_url.rstrip("/") + "/pd_flip/router/workers")
        workers = payload.get("workers") if isinstance(payload, dict) else []
        normalized = []
        for worker in workers or []:
            if not isinstance(worker, dict):
                continue
            url = (worker.get("url") or worker.get("worker_id") or "").rstrip("/")
            row = {
                "name": url_to_name.get(url),
                "worker_id": worker.get("worker_id"),
                "url": worker.get("url"),
                "role": worker.get("role"),
                "draining": bool(worker.get("draining")),
                "active_load": worker.get("active_load"),
                "bootstrap_port": worker.get("bootstrap_port"),
            }
            normalized.append(row)
        event["workers"] = normalized
        event["ok"] = True
    except Exception as exc:
        event["ok"] = False
        event["error"] = repr(exc)
    return event


def collect_worker_events(client: HttpClient, node: JsonDict) -> List[JsonDict]:
    name = node["name"]
    base = node["worker_url"].rstrip("/")
    events = []

    server_event = now_fields()
    server_event.update(
        {"event_type": "worker_status", "component": "worker", "node": name, "url": base}
    )
    try:
        server_info = client.get_json(base + "/server_info")
        server_event["ok"] = True
        server_event["pd_flip"] = extract_pd_flip(server_info)
    except Exception as exc:
        server_event["ok"] = False
        server_event["error"] = repr(exc)
    events.append(server_event)

    migration_event = now_fields()
    migration_event.update(
        {
            "event_type": "migration_status",
            "component": "worker",
            "node": name,
            "url": base,
        }
    )
    try:
        migration_payload = client.get_json(base + "/pd_flip/migration/status")
        migration_event["ok"] = True
        migration_event["status"] = unwrap_migration_status(migration_payload)
    except Exception as exc:
        migration_event["ok"] = False
        migration_event["error"] = repr(exc)
    events.append(migration_event)

    load_event = now_fields()
    load_event.update(
        {"event_type": "worker_load", "component": "worker", "node": name, "url": base}
    )
    try:
        load_event["ok"] = True
        load_event["load"] = client.get_json(base + "/v1/loads?include=all")
    except Exception as exc:
        load_event["ok"] = False
        load_event["error"] = repr(exc)
    events.append(load_event)
    return events


def extract_pd_flip(server_info: Any) -> JsonDict:
    if isinstance(server_info, dict):
        for state in server_info.get("internal_states", []) or []:
            if isinstance(state, dict) and isinstance(state.get("pd_flip"), dict):
                return compact_dict(state["pd_flip"])
        if isinstance(server_info.get("pd_flip"), dict):
            return compact_dict(server_info["pd_flip"])
    return {}


def unwrap_migration_status(payload: Any) -> JsonDict:
    candidates = payload if isinstance(payload, list) else [payload]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        status = candidate.get("status")
        if isinstance(status, dict):
            return compact_dict(status)
        if "state" in candidate and "pending_reqs" in candidate:
            return compact_dict(candidate)
    return {}


def compact_dict(data: JsonDict) -> JsonDict:
    result = {}
    for key, value in data.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[key] = value
        elif isinstance(value, list):
            compact_items = []
            for item in value:
                if isinstance(item, dict):
                    compact_items.append(compact_dict(item))
                elif isinstance(item, (str, int, float, bool)) or item is None:
                    compact_items.append(item)
            result[key] = compact_items
        elif isinstance(value, dict):
            result[key] = compact_dict(value)
    return result


def load_jsonl(path: Path) -> List[JsonDict]:
    rows = []
    if not path or not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def build_timeline(events: Sequence[JsonDict]) -> List[JsonDict]:
    timeline = []
    seen = set()
    source_node = None
    source_drained = False

    def add(stage: str, event: JsonDict, node: Optional[str] = None, details: Optional[JsonDict] = None) -> None:
        if stage in seen:
            return
        seen.add(stage)
        row = {
            "stage": stage,
            "ts_wall": event.get("ts_wall"),
            "ts_mono": event.get("ts_mono"),
            "node": node or event.get("node"),
            "session_id": None,
        }
        if details:
            row.update(details)
        timeline.append(row)

    for event in sorted(events, key=lambda item: float(item.get("ts_mono") or 0.0)):
        event_type = event.get("event_type")
        if event_type == "router_workers":
            for worker in event.get("workers") or []:
                node = worker.get("name")
                role = worker.get("role")
                draining = bool(worker.get("draining"))
                if role == "decode" and draining and "router_source_drained" not in seen:
                    source_node = node
                    source_drained = True
                    add(
                        "router_source_drained",
                        event,
                        node=node,
                        details={
                            "worker_url": worker.get("url"),
                            "active_load": worker.get("active_load"),
                        },
                    )
                if source_node and node == source_node and role == "prefill":
                    add(
                        "source_role_committed",
                        event,
                        node=node,
                        details={"worker_url": worker.get("url")},
                    )
                if source_drained and source_node and node == source_node and not draining:
                    add(
                        "cleanup_router_undrain",
                        event,
                        node=node,
                        details={"worker_url": worker.get("url"), "role": role},
                    )

        elif event_type == "worker_status":
            pd_flip = event.get("pd_flip") or {}
            if pd_flip.get("admission_paused") or pd_flip.get("pd_flip_admission_paused"):
                add("source_admission_paused", event, node=event.get("node"))

        elif event_type == "migration_status":
            status = event.get("status") or {}
            state = str(status.get("state") or "")
            role = str(status.get("role") or "")
            session_id = status.get("session_id")
            details = {
                "session_id": session_id,
                "migration_role": role,
                "migration_state": state,
                "pending_reqs": status.get("pending_reqs"),
                "transferred_reqs": status.get("transferred_reqs"),
                "failed_reqs": status.get("failed_reqs"),
                "held_reqs": status.get("held_reqs"),
                "last_error": status.get("last_error"),
            }
            if role == "source" and state == "source_started":
                add("source_migration_started", event, node=event.get("node"), details=details)
            if role == "target" and state == "target_prepared":
                add("target_migration_prepared", event, node=event.get("node"), details=details)

            transferred = int_or_zero(status.get("transferred_reqs"))
            pending = int_or_zero(status.get("pending_reqs"))
            failed = int_or_zero(status.get("failed_reqs"))
            if transferred > 0:
                add("kv_transfer_first_progress", event, node=event.get("node"), details=details)
            if transferred > 0 and pending == 0 and failed == 0:
                add("kv_transfer_complete", event, node=event.get("node"), details=details)
            if failed > 0 or "failed" in state or "aborted" in state:
                add("migration_abort_or_failed", event, node=event.get("node"), details=details)

    return timeline


def int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def build_stage_durations(timeline: Sequence[JsonDict]) -> List[JsonDict]:
    rows = []
    ordered = [row for row in timeline if row.get("ts_mono") is not None]
    for index, row in enumerate(ordered):
        next_row = ordered[index + 1] if index + 1 < len(ordered) else None
        rows.append(
            {
                "stage": row.get("stage"),
                "node": row.get("node"),
                "ts_mono": row.get("ts_mono"),
                "next_stage": next_row.get("stage") if next_row else None,
                "duration_to_next_s": (
                    float(next_row.get("ts_mono")) - float(row.get("ts_mono"))
                    if next_row
                    else None
                ),
            }
        )
    return rows


def label_request_impact(metrics: Sequence[JsonDict], timeline: Sequence[JsonDict]) -> List[JsonDict]:
    migration_start = first_stage_time(timeline, ["source_migration_started"])
    migration_end = first_stage_time(
        timeline,
        ["migration_abort_or_failed", "cleanup_router_undrain", "source_role_committed", "kv_transfer_complete"],
        after=migration_start,
    )
    rows = []
    for metric in metrics:
        start = metric.get("start_monotonic")
        end = metric.get("end_monotonic")
        phase = "unknown"
        active_during = False
        start_delta = None
        end_delta = None
        if start is not None and migration_start is not None:
            start_f = float(start)
            end_f = float(end) if end is not None else start_f
            start_delta = start_f - migration_start
            if migration_end is not None:
                end_delta = end_f - migration_end

            active_during = (
                (migration_end is None and end_f >= migration_start)
                or (
                    migration_end is not None
                    and start_f <= migration_end
                    and end_f >= migration_start
                )
            )
            if end_f < migration_start:
                phase = "before_migration"
            elif migration_end is not None and start_f > migration_end:
                phase = "after_migration"
            elif start_f < migration_start:
                phase = "overlaps_migration"
            elif migration_end is None or start_f <= migration_end:
                phase = "during_migration"
            else:
                phase = "after_migration"
        row = dict(metric)
        row["migration_phase"] = phase
        row["active_during_migration"] = active_during
        row["start_delta_from_migration_start_s"] = start_delta
        row["end_delta_from_migration_end_s"] = end_delta
        row["migration_start_mono"] = migration_start
        row["migration_end_mono"] = migration_end
        rows.append(row)
    return rows


def join_request_migration(
    request_rows: Sequence[JsonDict], migration_rows: Sequence[JsonDict]
) -> List[JsonDict]:
    """Join only unambiguous target-owner proof by ``(session_id, rid)``."""
    candidates = {}
    for migration in migration_rows:
        rid = migration.get("rid")
        session_id = migration.get("session_id")
        if rid and session_id and migration.get("final_owner") == "target":
            candidates.setdefault((session_id, rid), []).append(migration)
    joined = []
    for request in request_rows:
        row = dict(request)
        rid = request.get("request_id")
        session_id = request.get("migration_session_id")
        if session_id is None:
            sessions = {
                candidate_session
                for candidate_session, candidate_rid in candidates
                if candidate_rid == rid
            }
            if len(sessions) > 1:
                raise RuntimeError("ambiguous migration session for RID %s" % rid)
            session_id = next(iter(sessions), None)
        matches = candidates.get((session_id, rid), []) if session_id else []
        proof_fields = (
            "p_tokens",
            "h_tokens",
            "c0_tokens",
            "c1_tokens",
            "stitch_mode",
            "mooncake_bytes",
            "mooncake_bytes_available",
            "source_bytes",
            "delta_bytes",
            "source_queue",
            "final_owner",
            "output_boundary",
            "rollback_reason",
        )
        signatures = {
            tuple(measurement.get(field) for field in proof_fields)
            for measurement in matches
        }
        if len(signatures) > 1:
            raise RuntimeError(
                "conflicting target proofs for session/RID %s/%s"
                % (session_id, rid)
            )
        measurement = (
            max(matches, key=lambda item: float(item.get("ts_mono") or 0))
            if matches
            else None
        )
        row["migration_measurement_found"] = measurement is not None
        if measurement:
            for key, value in measurement.items():
                row["worker_" + key] = value
        joined.append(row)
    return joined


def first_stage_time(
    timeline: Sequence[JsonDict], stages: Sequence[str], after: Optional[float] = None
) -> Optional[float]:
    for row in timeline:
        if row.get("stage") not in stages or row.get("ts_mono") is None:
            continue
        ts = float(row["ts_mono"])
        if after is None or ts >= after:
            return ts
    return None


def parse_controller_json(log_path: Optional[Path]) -> JsonDict:
    if log_path is None or not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        return obj if isinstance(obj, dict) else {"parsed": obj}
    return {}


def flatten_controller_actions(controller: JsonDict) -> List[JsonDict]:
    actions = []

    def walk(value: Any, parent_step: Optional[str] = None) -> None:
        if isinstance(value, dict):
            if "step" in value:
                row = {}
                for key in [
                    "step",
                    "target",
                    "method",
                    "url",
                    "success",
                    "elapsed_seconds",
                    "message",
                ]:
                    row[key] = value.get(key)
                row["parent_step"] = parent_step
                actions.append(row)
                parent_step = str(value.get("step"))
            for child in value.values():
                walk(child, parent_step)
        elif isinstance(value, list):
            for child in value:
                walk(child, parent_step)

    walk(controller)
    for index, action in enumerate(actions):
        action["action_index"] = index
    return actions


def flatten_migration_samples(events: Sequence[JsonDict]) -> List[JsonDict]:
    rows = []
    for event in events:
        if event.get("event_type") != "migration_status":
            continue
        status = event.get("status") or {}
        row = {
            "ts_wall": event.get("ts_wall"),
            "ts_mono": event.get("ts_mono"),
            "node": event.get("node"),
            "ok": event.get("ok"),
            "error": event.get("error"),
            "enabled": status.get("enabled"),
            "role": status.get("role"),
            "state": status.get("state"),
            "session_id": status.get("session_id"),
            "pending_reqs": status.get("pending_reqs"),
            "transferred_reqs": status.get("transferred_reqs"),
            "released_reqs": status.get("released_reqs"),
            "failed_reqs": status.get("failed_reqs"),
            "held_reqs": status.get("held_reqs"),
            "last_error": status.get("last_error"),
            "waiting_reqs": status.get("waiting_reqs"),
            "waiting_manifest_count": status.get("waiting_manifest_count"),
            "waiting_skipped_count": status.get("waiting_skipped_count"),
            "waiting_skipped": json.dumps(
                status.get("waiting_skipped") or [], sort_keys=True
            ),
            "index_debug": json.dumps(status.get("index_debug") or [], sort_keys=True),
            "timing_debug": json.dumps(
                status.get("timing_debug") or {}, sort_keys=True
            ),
            "request_measurements": json.dumps(
                status.get("request_measurements") or [], sort_keys=True
            ),
        }
        rows.append(row)
    return rows


def flatten_migration_request_samples(events: Sequence[JsonDict]) -> List[JsonDict]:
    rows = []
    request_fields = migration_request_fields()[4:]
    for event in events:
        if event.get("event_type") != "migration_status":
            continue
        status = event.get("status") or {}
        sessions = [status]
        while sessions:
            session = sessions.pop(0)
            sessions.extend(
                archived
                for archived in (session.get("session_archive") or [])
                if isinstance(archived, dict)
            )
            for measurement in session.get("request_measurements") or []:
                row = {
                    "ts_wall": event.get("ts_wall"),
                    "ts_mono": event.get("ts_mono"),
                    "node": event.get("node"),
                    "session_id": session.get("session_id"),
                }
                for field in request_fields:
                    row[field] = measurement.get(field)
                rows.append(row)
    return rows


def flatten_router_samples(events: Sequence[JsonDict]) -> List[JsonDict]:
    rows = []
    for event in events:
        if event.get("event_type") != "router_workers":
            continue
        for worker in event.get("workers") or []:
            rows.append(
                {
                    "ts_wall": event.get("ts_wall"),
                    "ts_mono": event.get("ts_mono"),
                    "name": worker.get("name"),
                    "worker_id": worker.get("worker_id"),
                    "url": worker.get("url"),
                    "role": worker.get("role"),
                    "draining": worker.get("draining"),
                    "active_load": worker.get("active_load"),
                    "bootstrap_port": worker.get("bootstrap_port"),
                }
            )
    return rows


def flatten_pd_flip_samples(events: Sequence[JsonDict]) -> List[JsonDict]:
    rows = []
    for event in events:
        if event.get("event_type") != "worker_status":
            continue
        pd_flip = event.get("pd_flip") or {}
        rows.append(
            {
                "ts_wall": event.get("ts_wall"),
                "ts_mono": event.get("ts_mono"),
                "node": event.get("node"),
                "ok": event.get("ok"),
                "error": event.get("error"),
                "state": pd_flip.get("state"),
                "direction": pd_flip.get("direction"),
                "current_role": pd_flip.get("current_role"),
                "target_role": pd_flip.get("target_role") or pd_flip.get("requested_role"),
                "admission_paused": pd_flip.get("admission_paused")
                or pd_flip.get("pd_flip_admission_paused"),
                "is_idle_for_flip": pd_flip.get("is_idle_for_flip"),
                "migration_state": pd_flip.get("migration_state"),
                "migration_pending_reqs": pd_flip.get("migration_pending_reqs"),
                "migration_transferred_reqs": pd_flip.get("migration_transferred_reqs"),
                "migration_failed_reqs": pd_flip.get("migration_failed_reqs"),
                "migration_last_error": pd_flip.get("migration_last_error"),
            }
        )
    return rows


def flatten_worker_load_samples(events: Sequence[JsonDict]) -> List[JsonDict]:
    rows = []
    for event in events:
        if event.get("event_type") != "worker_load":
            continue
        if not event.get("ok"):
            rows.append(
                {
                    "ts_wall": event.get("ts_wall"),
                    "ts_mono": event.get("ts_mono"),
                    "node": event.get("node"),
                    "ok": event.get("ok"),
                    "error": event.get("error"),
                }
            )
            continue
        for index, load in enumerate(iter_load_items(event.get("load"))):
            running = int_or_zero(load.get("num_running_reqs"))
            waiting = int_or_zero(load.get("num_waiting_reqs"))
            decode_prealloc = load_metric(load, "decode_prealloc_queue_reqs")
            decode_transfer = load_metric(load, "decode_transfer_queue_reqs")
            decode_retracted = load_metric(load, "decode_retracted_queue_reqs")
            prefill_bootstrap = load_metric(load, "prefill_bootstrap_queue_reqs")
            prefill_inflight = load_metric(load, "prefill_inflight_queue_reqs")
            rows.append(
                {
                    "ts_wall": event.get("ts_wall"),
                    "ts_mono": event.get("ts_mono"),
                    "node": event.get("node"),
                    "url": event.get("url"),
                    "load_index": index,
                    "ok": event.get("ok"),
                    "error": event.get("error"),
                    "num_running_reqs": running,
                    "num_waiting_reqs": waiting,
                    "num_total_tokens": load.get("num_total_tokens"),
                    "token_usage": load.get("token_usage"),
                    "decode_prealloc_queue_reqs": decode_prealloc,
                    "decode_transfer_queue_reqs": decode_transfer,
                    "decode_retracted_queue_reqs": decode_retracted,
                    "prefill_bootstrap_queue_reqs": prefill_bootstrap,
                    "prefill_inflight_queue_reqs": prefill_inflight,
                    "source_total_residual_reqs": running
                    + waiting
                    + decode_prealloc
                    + decode_transfer
                    + decode_retracted
                    + prefill_bootstrap
                    + prefill_inflight,
                    "raw_load": json.dumps(load, sort_keys=True),
                }
            )
    return rows


def iter_load_items(payload: Any) -> Iterable[JsonDict]:
    if isinstance(payload, dict):
        loads = payload.get("loads", [])
    elif isinstance(payload, list):
        loads = payload
    else:
        loads = []
    for item in loads or []:
        if isinstance(item, dict):
            yield item


def load_metric(load: JsonDict, field: str) -> int:
    value = load.get(field)
    if value is None and isinstance(load.get("disaggregation"), dict):
        value = load["disaggregation"].get(field)
    return int_or_zero(value)


def write_outputs(
    *,
    events_path: Path,
    output_dir: Path,
    controller_log: Optional[Path],
    request_metrics_path: Optional[Path],
    errors_path: Optional[Path],
) -> JsonDict:
    output_dir.mkdir(parents=True, exist_ok=True)
    events = load_jsonl(events_path)
    timeline = build_timeline(events)
    durations = build_stage_durations(timeline)
    migration_samples = flatten_migration_samples(events)
    migration_request_samples = flatten_migration_request_samples(events)
    router_samples = flatten_router_samples(events)
    pd_flip_samples = flatten_pd_flip_samples(events)
    worker_load_samples = flatten_worker_load_samples(events)
    controller = parse_controller_json(controller_log)
    controller_actions = flatten_controller_actions(controller)
    state_trace = controller.get("state_trace") if isinstance(controller.get("state_trace"), list) else []
    request_metrics = load_jsonl(request_metrics_path) if request_metrics_path else []
    request_impact = label_request_impact(request_metrics, timeline)
    request_migration_join = join_request_migration(
        request_impact, migration_request_samples
    )
    error_rows = load_jsonl(errors_path) if errors_path else []

    write_jsonl(output_dir / "migration_timeline.jsonl", timeline)
    write_csv(output_dir / "migration_timeline.csv", timeline, timeline_fields())
    write_csv(output_dir / "migration_stage_durations.csv", durations, duration_fields())
    write_csv(output_dir / "migration_status_samples.csv", migration_samples, migration_status_fields())
    write_jsonl(
        output_dir / "migration_request_samples.jsonl", migration_request_samples
    )
    write_csv(
        output_dir / "migration_request_samples.csv",
        migration_request_samples,
        migration_request_fields(),
    )
    write_csv(output_dir / "router_worker_samples.csv", router_samples, router_fields())
    write_csv(output_dir / "worker_pd_flip_samples.csv", pd_flip_samples, pd_flip_fields())
    write_csv(output_dir / "worker_load_samples.csv", worker_load_samples, worker_load_fields())
    write_csv(output_dir / "controller_actions.csv", controller_actions, controller_action_fields())
    write_csv(output_dir / "controller_state_trace.csv", state_trace, controller_state_fields())
    write_csv(output_dir / "request_impact_by_stage.csv", request_impact, request_impact_fields())
    write_jsonl(output_dir / "request_migration_join.jsonl", request_migration_join)

    summary = {
        "events_path": str(events_path),
        "event_count": len(events),
        "timeline_stage_count": len(timeline),
        "timeline_stages": [row.get("stage") for row in timeline],
        "worker_load_sample_count": len(worker_load_samples),
        "migration_request_sample_count": len(migration_request_samples),
        "migration_outcome": migration_outcome(timeline),
        "controller_message": controller.get("message"),
        "controller_success": controller.get("success"),
        "request_count": len(request_metrics),
        "request_migration_join_count": sum(
            1 for row in request_migration_join if row["migration_measurement_found"]
        ),
        "request_error_count": len(error_rows),
    }
    with (output_dir / "migration_link_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    write_measurement_plan(output_dir)
    return summary


def migration_outcome(timeline: Sequence[JsonDict]) -> str:
    stages = [row.get("stage") for row in timeline]
    if "source_role_committed" in stages:
        return "committed"
    if "migration_abort_or_failed" in stages:
        return "aborted_or_failed"
    if "kv_transfer_complete" in stages:
        return "kv_transfer_complete_without_commit_observed"
    if "source_migration_started" in stages:
        return "migration_started_without_terminal_observed"
    return "no_migration_observed"


def write_measurement_plan(output_dir: Path) -> None:
    text = """# PD Flip Migration Link Measurement Plan

Goal: measure every observable step in one PD flip migration attempt, from SLO risk through router drain, source admission pause, KV source/target migration, commit or abort, cleanup, and request impact.

Collected raw files:
- `migration_events.jsonl`: sidecar polling raw events.
- `migration_timeline.csv` / `migration_timeline.jsonl`: first-observed chain stages.
- `migration_stage_durations.csv`: time from each observed stage to the next stage.
- `migration_status_samples.csv`: per-node `/pd_flip/migration/status` samples, including pending/transferred/failed/released/held counts and index debug.
- `router_worker_samples.csv`: router role/drain/load view.
- `worker_pd_flip_samples.csv`: worker state-machine view from `/server_info`.
- `worker_load_samples.csv`: per-node `/v1/loads?include=all` queue counts, including running, waiting queue, and decode/prefill residual queues.
- `controller_actions.csv`: controller HTTP actions parsed from the monitor log.
- `controller_state_trace.csv`: controller high-level state trace.
- `request_impact_by_stage.csv`: request metrics labeled as before, overlaps, during, or after migration.
- `request_migration_join.jsonl`: request/SLO rows joined to worker measurements by request ID/RID.

Interpretation:
- A successful chain should show `router_source_drained`, `source_admission_paused`, `source_migration_started`, `target_migration_prepared`, `kv_transfer_first_progress`, `kv_transfer_complete`, `source_role_committed`, and `cleanup_router_undrain`.
- If `source_migration_started` appears without `kv_transfer_first_progress`, KV transfer did not visibly progress.
- If `migration_abort_or_failed` appears, inspect `migration_status_samples.csv` and `controller_actions.csv` around the same timestamps for the failing node, `last_error`, pending request count, and index debug.
"""
    (output_dir / "migration_measurement_plan.md").write_text(text, encoding="utf-8")


def timeline_fields() -> List[str]:
    return [
        "stage",
        "ts_wall",
        "ts_mono",
        "node",
        "session_id",
        "worker_url",
        "active_load",
        "migration_role",
        "migration_state",
        "pending_reqs",
        "transferred_reqs",
        "failed_reqs",
        "held_reqs",
        "last_error",
    ]


def duration_fields() -> List[str]:
    return ["stage", "node", "ts_mono", "next_stage", "duration_to_next_s"]


def migration_status_fields() -> List[str]:
    return [
        "ts_wall",
        "ts_mono",
        "node",
        "ok",
        "error",
        "enabled",
        "role",
        "state",
        "session_id",
        "pending_reqs",
        "transferred_reqs",
        "released_reqs",
        "failed_reqs",
        "held_reqs",
        "last_error",
        "waiting_reqs",
        "waiting_manifest_count",
        "waiting_skipped_count",
        "waiting_skipped",
        "index_debug",
        "timing_debug",
        "request_measurements",
    ]


def migration_request_fields() -> List[str]:
    return [
        "ts_wall",
        "ts_mono",
        "node",
        "session_id",
        "rid",
        "p_tokens",
        "h_tokens",
        "c0_tokens",
        "c1_tokens",
        "stitch_mode",
        "mooncake_bytes",
        "mooncake_bytes_available",
        "mooncake_restore_tokens",
        "source_bytes",
        "delta_bytes",
        "mooncake_duration_seconds",
        "source_duration_seconds",
        "delta_duration_seconds",
        "held_at_mono",
        "freeze_at_mono",
        "commit_at_mono",
        "activate_at_mono",
        "source_queue",
        "final_owner",
        "output_boundary",
        "rollback_reason",
    ]


def router_fields() -> List[str]:
    return [
        "ts_wall",
        "ts_mono",
        "name",
        "worker_id",
        "url",
        "role",
        "draining",
        "active_load",
        "bootstrap_port",
    ]


def pd_flip_fields() -> List[str]:
    return [
        "ts_wall",
        "ts_mono",
        "node",
        "ok",
        "error",
        "state",
        "direction",
        "current_role",
        "target_role",
        "admission_paused",
        "is_idle_for_flip",
        "migration_state",
        "migration_pending_reqs",
        "migration_transferred_reqs",
        "migration_failed_reqs",
        "migration_last_error",
    ]


def worker_load_fields() -> List[str]:
    return [
        "ts_wall",
        "ts_mono",
        "node",
        "url",
        "load_index",
        "ok",
        "error",
        "num_running_reqs",
        "num_waiting_reqs",
        "num_total_tokens",
        "token_usage",
        "decode_prealloc_queue_reqs",
        "decode_transfer_queue_reqs",
        "decode_retracted_queue_reqs",
        "prefill_bootstrap_queue_reqs",
        "prefill_inflight_queue_reqs",
        "source_total_residual_reqs",
        "raw_load",
    ]


def controller_action_fields() -> List[str]:
    return [
        "action_index",
        "parent_step",
        "step",
        "target",
        "method",
        "url",
        "success",
        "elapsed_seconds",
        "message",
    ]


def controller_state_fields() -> List[str]:
    return [
        "snapshot_index",
        "action_index",
        "state",
        "reason",
        "source",
        "migration_target",
        "direction",
        "role_before",
        "role_after",
        "configured_ratio",
        "effective_ratio",
        "capacity_fallback_count",
        "prefill_slo_good",
        "prefill_slo_total",
        "decode_slo_good",
        "decode_slo_total",
    ]


def request_impact_fields() -> List[str]:
    return [
        "request_id",
        "prompt_kind",
        "arrival_offset_s",
        "migration_phase",
        "active_during_migration",
        "start_delta_from_migration_start_s",
        "end_delta_from_migration_end_s",
        "status",
        "ttft_slo_s",
        "ttft_s",
        "ttft_met",
        "tpot_slo_s",
        "avg_tpot_s",
        "p95_tpot_s",
        "tpot_avg_met",
        "all_met",
        "error",
    ]


def write_jsonl(path: Path, rows: Sequence[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[JsonDict], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    sample = subparsers.add_parser("sample")
    sample.add_argument("--router-url", required=True)
    sample.add_argument("--node", action="append", required=True)
    sample.add_argument("--output-events", required=True)
    sample.add_argument("--interval-seconds", type=float, default=0.2)
    sample.add_argument("--duration-seconds", type=float, default=420.0)
    sample.add_argument("--http-timeout-seconds", type=float, default=3.0)
    sample.add_argument("--api-key-env", default="ADMIN_API_KEY")
    sample.add_argument(
        "--router-api-key-env", default="PD_FLIP_ROUTER_ADMIN_API_KEY"
    )

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--events-jsonl", required=True)
    summarize.add_argument("--output-dir", required=True)
    summarize.add_argument("--controller-log", default=None)
    summarize.add_argument("--request-metrics-jsonl", default=None)
    summarize.add_argument("--errors-jsonl", default=None)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    if args.command == "sample":
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise ValueError("measurement API key environment variable is empty")
        sample_loop(
            router_url=args.router_url,
            nodes=parse_node_specs(args.node),
            output_events=Path(args.output_events),
            interval_seconds=args.interval_seconds,
            duration_seconds=args.duration_seconds,
            http_timeout_seconds=args.http_timeout_seconds,
            api_key=api_key,
            router_api_key=os.environ.get(args.router_api_key_env)
            or api_key,
        )
        return 0
    if args.command == "summarize":
        summary = write_outputs(
            events_path=Path(args.events_jsonl),
            output_dir=Path(args.output_dir),
            controller_log=Path(args.controller_log) if args.controller_log else None,
            request_metrics_path=Path(args.request_metrics_jsonl)
            if args.request_metrics_jsonl
            else None,
            errors_path=Path(args.errors_jsonl) if args.errors_jsonl else None,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    raise AssertionError("unhandled command: %s" % args.command)


if __name__ == "__main__":
    sys.exit(main())
