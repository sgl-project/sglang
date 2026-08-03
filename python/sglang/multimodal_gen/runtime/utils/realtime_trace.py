# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any, Callable

TRACE_LOG_PREFIX = "realtime_trace"
CLIENT_TRACE_EVENT_KIND = "client_trace"
MAX_TRACE_ID_LENGTH = 128
MAX_CLIENT_TRACE_EVENTS = 32

_TRACE_ID_RE = re.compile(r"[^A-Za-z0-9_.:-]")
_HOSTNAME = socket.gethostname()
_TRACE_SINKS: dict[str, list[Callable[[dict[str, Any]], None]]] = {}
_TRACE_SINKS_LOCK = threading.RLock()
_RESERVED_PAYLOAD_KEYS = {
    "event",
    "trace_id",
    "session_id",
    "server_epoch_ms",
    "server_elapsed_ms",
    "host",
}


def normalize_trace_id(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    trace_id = _TRACE_ID_RE.sub("_", str(value).strip())
    if not trace_id:
        return fallback
    return trace_id[:MAX_TRACE_ID_LENGTH]


def compact_client_trace(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None

    compacted: dict[str, Any] = {}
    for key in (
        "trace_id",
        "time_origin_ms",
        "created_epoch_ms",
        "created_perf_ms",
        "user_agent",
        "location",
    ):
        if key in value:
            compacted[key] = _json_safe(value[key])

    events = value.get("events")
    if isinstance(events, list):
        compacted["events"] = [
            _compact_client_trace_event(event)
            for event in events[:MAX_CLIENT_TRACE_EVENTS]
            if isinstance(event, Mapping)
        ]

    return compacted


def compact_client_trace_event(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"value": _json_safe(value)}
    return _compact_client_trace_event(value)


def log_realtime_trace(
    logger,
    session,
    event: str,
    **fields: Any,
) -> None:
    payload = realtime_trace_payload(session, event, **fields)
    logger.info(
        "%s %s",
        TRACE_LOG_PREFIX,
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
    )
    _notify_realtime_trace_sinks(payload)


def register_realtime_trace_sink(
    trace_id: str,
    sink: Callable[[dict[str, Any]], None],
) -> None:
    if not trace_id:
        return
    with _TRACE_SINKS_LOCK:
        sinks = _TRACE_SINKS.setdefault(trace_id, [])
        if sink not in sinks:
            sinks.append(sink)


def unregister_realtime_trace_sink(
    trace_id: str,
    sink: Callable[[dict[str, Any]], None],
) -> None:
    if not trace_id:
        return
    with _TRACE_SINKS_LOCK:
        sinks = _TRACE_SINKS.get(trace_id)
        if not sinks:
            return
        if sink in sinks:
            sinks.remove(sink)
        if not sinks:
            _TRACE_SINKS.pop(trace_id, None)


def log_realtime_trace_for_batch(
    logger,
    batch,
    event: str,
    **fields: Any,
) -> None:
    trace_id = getattr(batch, "realtime_trace_id", None)
    if not trace_id:
        return
    session = SimpleNamespace(
        id=getattr(batch, "realtime_session_id", None),
        trace_id=trace_id,
        trace_started_at=getattr(batch, "realtime_trace_started_at", None),
    )
    log_realtime_trace(logger, session, event, **fields)


def realtime_trace_span(
    logger,
    batch,
    event: str,
    *,
    component: str,
    measure_cuda: bool | None = None,
    input_tensor: Any = None,
    **fields: Any,
):
    return RealtimeTraceSpan(
        logger,
        batch,
        event,
        component=component,
        measure_cuda=measure_cuda,
        input_tensor=input_tensor,
        **fields,
    )


class RealtimeTraceSpan:
    def __init__(
        self,
        logger,
        batch,
        event: str,
        *,
        component: str,
        measure_cuda: bool | None = None,
        input_tensor: Any = None,
        **fields: Any,
    ) -> None:
        self.logger = logger
        self.batch = batch
        self.event = event
        self.measure_cuda = _should_measure_cuda(measure_cuda)
        self.fields = {
            "component": component,
            **tensor_trace_metadata(input_tensor, prefix="input"),
            **fields,
        }
        self.start_time = 0.0
        self.start_elapsed_ms = 0.0
        self.cuda_start = None
        self.cuda_end = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.start_elapsed_ms = _batch_elapsed_ms(self.batch, self.start_time)
        self.cuda_start, self.cuda_end = _new_cuda_events(self.measure_cuda)
        if self.cuda_start is not None:
            self.cuda_start.record()
        return self

    def add_fields(self, **fields: Any) -> None:
        self.fields.update(fields)

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.perf_counter()
        duration_ms = (end_time - self.start_time) * 1000.0
        fields = {
            **self.fields,
            "component_start_elapsed_ms": round(self.start_elapsed_ms, 3),
            "component_end_elapsed_ms": round(
                _batch_elapsed_ms(self.batch, end_time), 3
            ),
            "duration_ms": round(duration_ms, 3),
        }
        cuda_ms = _finish_cuda_timing(self.cuda_start, self.cuda_end)
        if cuda_ms is not None:
            fields["cuda_ms"] = round(cuda_ms, 3)
        if exc_val is not None:
            fields["error"] = str(exc_val).splitlines()[0]
        log_realtime_trace_for_batch(self.logger, self.batch, self.event, **fields)
        return False


def tensor_trace_metadata(value: Any, *, prefix: str) -> dict[str, Any]:
    if value is None:
        return {}
    shape = getattr(value, "shape", None)
    if shape is None:
        return {}
    metadata = {
        f"{prefix}_shape": [int(dim) for dim in shape],
        f"{prefix}_dtype": str(getattr(value, "dtype", "")),
        f"{prefix}_device": str(getattr(value, "device", "")),
    }
    try:
        metadata[f"{prefix}_bytes"] = int(value.numel() * value.element_size())
    except Exception:
        pass
    return metadata


def realtime_trace_payload(session, event: str, **fields: Any) -> dict[str, Any]:
    now = time.perf_counter()
    started_at = getattr(session, "trace_started_at", None) or now
    payload: dict[str, Any] = {
        "event": event,
        "trace_id": getattr(session, "trace_id", getattr(session, "id", "")),
        "session_id": getattr(session, "id", None),
        "server_epoch_ms": int(time.time() * 1000),
        "server_elapsed_ms": round((now - started_at) * 1000.0, 3),
        "host": _HOSTNAME,
    }
    for key, value in fields.items():
        if value is not None and key not in _RESERVED_PAYLOAD_KEYS:
            payload[key] = _json_safe(value)
    return payload


def _notify_realtime_trace_sinks(payload: dict[str, Any]) -> None:
    trace_id = str(payload.get("trace_id") or "")
    if not trace_id:
        return
    with _TRACE_SINKS_LOCK:
        sinks = list(_TRACE_SINKS.get(trace_id, ()))
    for sink in sinks:
        try:
            sink(dict(payload))
        except Exception:
            pass


def _batch_elapsed_ms(batch, now: float) -> float:
    started_at = getattr(batch, "realtime_trace_started_at", None) or now
    return (now - started_at) * 1000.0


def _should_measure_cuda(measure_cuda: bool | None) -> bool:
    if measure_cuda is not None:
        return measure_cuda
    return os.environ.get("SGLANG_REALTIME_TRACE_SYNC_CUDA", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _new_cuda_events(enabled: bool):
    if not enabled:
        return None, None
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        return (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
    except Exception:
        return None, None


def _finish_cuda_timing(cuda_start, cuda_end) -> float | None:
    if cuda_start is None or cuda_end is None:
        return None
    try:
        cuda_end.record()
        cuda_end.synchronize()
        return float(cuda_start.elapsed_time(cuda_end))
    except Exception:
        return None


def _compact_client_trace_event(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _json_safe(event_value)
        for key, event_value in value.items()
        if key
        in {
            "name",
            "seq",
            "trace_id",
            "client_perf_ms",
            "client_epoch_ms",
            "chunk_index",
            "event_id",
            "kind",
            "content_type",
            "num_frames",
            "payload_bytes",
            "transport",
            "fps",
            "code",
            "reason",
            "decode_ms",
            "display_lag_ms",
            "queue_frames",
            "buffer_ms",
            "ws_buffered_amount",
        }
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)
