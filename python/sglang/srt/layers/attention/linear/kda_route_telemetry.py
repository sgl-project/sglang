"""Terminal route telemetry for the Phase-A CAKE KDA serving paths.

Each KDA layer call produces exactly one immutable terminal event.  The three
terminal outcomes are mutually exclusive and exhaustive::

    considered == cake_success + triton_fallback + fatal

CUDA graph capture needs special handling because Python executes during graph
construction but does not execute during replay.  Graph runners therefore wrap
``backend.capture_one`` in :func:`capture_kda_route_plan`.  Backend warmups are
ignored; only records made while ``torch.cuda.is_current_stream_capturing()`` is
true enter the immutable per-``ShapeKey`` plan.  A successful capture emits the
capture events once.  :func:`replay_kda_route_plan` emits that exact plan after
each successful ``backend.replay`` without pretending that the Python KDA
wrapper ran again.

``copy_count`` is a host-side, statically audited materialization count, *not* a
CUPTI activity count.  It is zero only for the packed row-strided CAKE ABI,
whose admitted inputs are metadata-only views.  Adapter and fallback paths use
JSON ``null`` with ``copy_count_source=\"unknown_requires_cupti\"``.  The raw
events are deliberately sufficient for an external CUPTI kernel/memcpy name and
count cross-check; this module never fabricates activity counts.

Only synchronous Python/capture/replay exceptions can be attributed here.
``cake_success`` therefore means that the selected host route returned without
a synchronous exception; it is not a claim that an asynchronously launched GPU
graph has completed.
Asynchronous GPU faults need a synchronization/CUPTI receipt at the owning
launch boundary and are intentionally not assigned to a KDA layer by this
telemetry.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Hashable, Iterator, Optional, TypeVar

logger = logging.getLogger(__name__)

KDA_ROUTE_EVENT_PREFIX = "SGLANG_KDA_ROUTE_EVENT "
KDA_ROUTE_SCHEMA_VERSION = 1

_VALID_MODES = frozenset(("decode", "prefill"))
_VALID_PHASES = frozenset(("direct", "capture", "replay"))
_VALID_COPY_COUNT_SOURCES = frozenset(
    ("static_zero_copy_row_view", "unknown_requires_cupti")
)


class KDATelemetryError(RuntimeError):
    """Base error for fail-closed KDA route telemetry."""


class KDACudaGraphPlanError(KDATelemetryError):
    """Raised when a capture/replay plan is missing or inconsistent."""


@dataclass(frozen=True)
class KDATerminalRouteEvent:
    """One immutable, terminal KDA route decision."""

    schema_version: int
    mode: str
    layer_id: int
    considered: int
    eligible: int
    attempted_cake: int
    cake_success: int
    triton_fallback: int
    fatal: int
    reason: str
    detail: str
    graph_phase: str
    copy_count: Optional[int]
    copy_count_source: str

    def __post_init__(self) -> None:
        if self.schema_version != KDA_ROUTE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported KDA route schema version: {self.schema_version}"
            )
        if self.mode not in _VALID_MODES:
            raise ValueError(f"invalid KDA route mode: {self.mode!r}")
        if isinstance(self.layer_id, bool) or not isinstance(self.layer_id, int):
            raise TypeError("KDA route layer_id must be an int")
        if self.layer_id < 0:
            raise ValueError("KDA route layer_id must be non-negative")
        for name in (
            "considered",
            "eligible",
            "attempted_cake",
            "cake_success",
            "triton_fallback",
            "fatal",
        ):
            value = getattr(self, name)
            if value not in (0, 1):
                raise ValueError(f"KDA route {name} must be 0 or 1, got {value!r}")
        if self.considered != 1:
            raise ValueError("every KDA terminal route event must have considered=1")
        if self.considered != (self.cake_success + self.triton_fallback + self.fatal):
            raise ValueError(
                "KDA route terminal invariant failed: considered must equal "
                "cake_success + triton_fallback + fatal"
            )
        if self.cake_success and not (self.eligible and self.attempted_cake):
            raise ValueError("cake_success requires eligible=1 and attempted_cake=1")
        if self.triton_fallback and self.attempted_cake:
            raise ValueError("successful Triton fallback cannot attempt CAKE")
        if not self.reason or not isinstance(self.reason, str):
            raise ValueError("KDA route reason must be a stable non-empty string")
        if not isinstance(self.detail, str):
            raise TypeError("KDA route detail must be a string")
        if self.graph_phase not in _VALID_PHASES:
            raise ValueError(f"invalid KDA graph phase: {self.graph_phase!r}")
        if self.copy_count is not None and (
            isinstance(self.copy_count, bool)
            or not isinstance(self.copy_count, int)
            or self.copy_count < 0
        ):
            raise ValueError("copy_count must be a non-negative int or None")
        if self.copy_count_source not in _VALID_COPY_COUNT_SOURCES:
            raise ValueError(f"invalid copy_count_source: {self.copy_count_source!r}")
        if self.copy_count is None and self.copy_count_source != (
            "unknown_requires_cupti"
        ):
            raise ValueError(
                "unknown copy_count must use copy_count_source='unknown_requires_cupti'"
            )
        if self.copy_count is not None and self.copy_count_source == (
            "unknown_requires_cupti"
        ):
            raise ValueError("known copy_count needs a statically audited source")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable machine-readable event mapping."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize deterministically for JSONL receipts."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def for_phase(self, graph_phase: str) -> KDATerminalRouteEvent:
        return replace(self, graph_phase=graph_phase)

    def as_fatal(self, *, reason: str, detail: str) -> KDATerminalRouteEvent:
        return replace(
            self,
            cake_success=0,
            triton_fallback=0,
            fatal=1,
            reason=reason,
            detail=detail,
            copy_count=None,
            copy_count_source="unknown_requires_cupti",
        )


@dataclass
class _CounterTotals:
    considered: int = 0
    eligible: int = 0
    attempted_cake: int = 0
    cake_success: int = 0
    triton_fallback: int = 0
    fatal: int = 0
    known_copy_calls: int = 0
    known_copy_count: int = 0
    unknown_copy_calls: int = 0

    def add(self, event: KDATerminalRouteEvent) -> None:
        self.considered += event.considered
        self.eligible += event.eligible
        self.attempted_cake += event.attempted_cake
        self.cake_success += event.cake_success
        self.triton_fallback += event.triton_fallback
        self.fatal += event.fatal
        if event.copy_count is None:
            self.unknown_copy_calls += 1
        else:
            self.known_copy_calls += 1
            self.known_copy_count += event.copy_count


class KDATerminalRouteTelemetry:
    """Thread-safe raw event buffer, JSONL emitter, and closed counters."""

    def __init__(
        self,
        *,
        raw_event_capacity: int = 4096,
        emit_log: bool = True,
        event_logger: logging.Logger = logger,
    ) -> None:
        if raw_event_capacity <= 0:
            raise ValueError("raw_event_capacity must be positive")
        self._lock = threading.RLock()
        self._events: deque[KDATerminalRouteEvent] = deque(maxlen=raw_event_capacity)
        self._event_count = 0
        self._counters: dict[tuple[str, int, str, str], _CounterTotals] = defaultdict(
            _CounterTotals
        )
        self._emit_log = emit_log
        self._logger = event_logger

    def emit(self, event: KDATerminalRouteEvent) -> None:
        """Atomically retain and count one already-validated event."""
        key = (event.mode, event.layer_id, event.graph_phase, event.reason)
        with self._lock:
            self._events.append(event)
            self._event_count += 1
            self._counters[key].add(event)
        if self._emit_log:
            self._logger.info("%s%s", KDA_ROUTE_EVENT_PREFIX, event.to_json())

    def raw_events_snapshot(self) -> tuple[KDATerminalRouteEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def raw_events_jsonl(self) -> str:
        return "\n".join(event.to_json() for event in self.raw_events_snapshot())

    def counter_snapshot(self) -> dict[str, Any]:
        """Return deterministic counters keyed by mode/layer/phase/reason."""
        with self._lock:
            rows = []
            for (mode, layer_id, phase, reason), totals in sorted(
                self._counters.items()
            ):
                row = {
                    "mode": mode,
                    "layer_id": layer_id,
                    "graph_phase": phase,
                    "reason": reason,
                    **asdict(totals),
                }
                if row["considered"] != (
                    row["cake_success"] + row["triton_fallback"] + row["fatal"]
                ):
                    raise KDATelemetryError(
                        "KDA route counter closure failed for "
                        f"{(mode, layer_id, phase, reason)!r}"
                    )
                rows.append(row)
            retained = len(self._events)
            total = self._event_count
        return {
            "schema_version": KDA_ROUTE_SCHEMA_VERSION,
            "key_fields": ["mode", "layer_id", "graph_phase", "reason"],
            "events_total": total,
            "events_retained": retained,
            "events_dropped_from_memory": total - retained,
            "counters": rows,
        }

    def counter_snapshot_json(self) -> str:
        return json.dumps(
            self.counter_snapshot(), sort_keys=True, separators=(",", ":")
        )


@dataclass
class _CaptureSession:
    mode: str
    shape_key: Hashable
    capture_probe: Callable[[], bool]
    records: list[KDATerminalRouteEvent]


_ACTIVE_CAPTURE: ContextVar[Optional[_CaptureSession]] = ContextVar(
    "sglang_kda_active_capture", default=None
)


class KDACudaGraphRoutePlans:
    """Immutable KDA route plans, partitioned by mode and keyed by ShapeKey."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._plans: dict[str, dict[Hashable, tuple[KDATerminalRouteEvent, ...]]] = {
            mode: {} for mode in _VALID_MODES
        }

    def bind(
        self,
        mode: str,
        shape_key: Hashable,
        plan: tuple[KDATerminalRouteEvent, ...],
    ) -> bool:
        """Bind once. Return true only for the first identical binding."""
        _validate_mode(mode)
        with self._lock:
            existing = self._plans[mode].get(shape_key)
            if existing is None:
                self._plans[mode][shape_key] = plan
                return True
            if existing != plan:
                raise KDACudaGraphPlanError(
                    "KDA CUDA graph route plan changed for immutable key "
                    f"mode={mode!r}, shape_key={shape_key!r}"
                )
            return False

    def require(
        self, mode: str, shape_key: Hashable
    ) -> tuple[KDATerminalRouteEvent, ...]:
        _validate_mode(mode)
        with self._lock:
            try:
                return self._plans[mode][shape_key]
            except KeyError as exc:
                raise KDACudaGraphPlanError(
                    "missing KDA CUDA graph route plan for "
                    f"mode={mode!r}, shape_key={shape_key!r}"
                ) from exc

    def snapshot(
        self, mode: str, shape_key: Hashable
    ) -> tuple[KDATerminalRouteEvent, ...]:
        return self.require(mode, shape_key)


def _raw_event_capacity_from_env() -> int:
    raw_value = os.getenv("SGLANG_KDA_ROUTE_EVENT_BUFFER_SIZE", "4096")
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            "SGLANG_KDA_ROUTE_EVENT_BUFFER_SIZE must be an integer"
        ) from exc
    if value <= 0:
        raise ValueError("SGLANG_KDA_ROUTE_EVENT_BUFFER_SIZE must be positive")
    return value


KDA_ROUTE_TELEMETRY = KDATerminalRouteTelemetry(
    raw_event_capacity=_raw_event_capacity_from_env()
)
KDA_CUDA_GRAPH_ROUTE_PLANS = KDACudaGraphRoutePlans()


def _validate_mode(mode: str) -> None:
    if mode not in _VALID_MODES:
        raise ValueError(f"invalid KDA route mode: {mode!r}")


def stable_kda_exception_detail(exc: BaseException) -> str:
    """Return a stable detail without embedding request data or error text."""
    cls = type(exc)
    return f"{cls.__module__}.{cls.__qualname__}"


def _torch_capture_probe() -> bool:
    # Lazy import keeps the telemetry model independently CPU-testable.
    import torch

    return bool(torch.cuda.is_current_stream_capturing())


def record_kda_terminal_route(
    *,
    mode: str,
    layer_id: int,
    eligible: bool,
    attempted_cake: bool,
    cake_success: bool,
    triton_fallback: bool,
    fatal: bool,
    reason: str,
    detail: str = "",
    copy_count: Optional[int] = None,
    copy_count_source: str = "unknown_requires_cupti",
    telemetry: KDATerminalRouteTelemetry = KDA_ROUTE_TELEMETRY,
) -> Optional[KDATerminalRouteEvent]:
    """Record or capture exactly one terminal decision.

    ``None`` means the call was a backend warmup inside an explicit capture
    context and was intentionally suppressed.
    """
    session = _ACTIVE_CAPTURE.get()
    graph_phase = "capture" if session is not None else "direct"
    event = KDATerminalRouteEvent(
        schema_version=KDA_ROUTE_SCHEMA_VERSION,
        mode=mode,
        layer_id=layer_id,
        considered=1,
        eligible=int(eligible),
        attempted_cake=int(attempted_cake),
        cake_success=int(cake_success),
        triton_fallback=int(triton_fallback),
        fatal=int(fatal),
        reason=reason,
        detail=detail,
        graph_phase=graph_phase,
        copy_count=copy_count,
        copy_count_source=copy_count_source,
    )
    if session is None:
        telemetry.emit(event)
        return event
    if session.mode != mode:
        raise KDATelemetryError(
            "KDA route mode does not match active CUDA graph capture: "
            f"record={mode!r}, capture={session.mode!r}"
        )
    if not session.capture_probe():
        return None
    session.records.append(event)
    return event


@contextmanager
def capture_kda_route_plan(
    shape_key: Hashable,
    mode: str,
    *,
    capture_probe: Callable[[], bool] = _torch_capture_probe,
    telemetry: KDATerminalRouteTelemetry = KDA_ROUTE_TELEMETRY,
    plans: KDACudaGraphRoutePlans = KDA_CUDA_GRAPH_ROUTE_PLANS,
) -> Iterator[None]:
    """Collect only the actual capture pass and bind an immutable route plan."""
    _validate_mode(mode)
    if _ACTIVE_CAPTURE.get() is not None:
        raise KDACudaGraphPlanError("nested KDA CUDA graph capture contexts")
    session = _CaptureSession(mode, shape_key, capture_probe, [])
    token = _ACTIVE_CAPTURE.set(session)
    try:
        try:
            yield
        except BaseException as exc:
            detail = stable_kda_exception_detail(exc)
            for event in session.records:
                telemetry.emit(
                    event
                    if event.fatal
                    else event.as_fatal(
                        reason="cuda_graph_capture_exception", detail=detail
                    )
                )
            raise
        else:
            plan = tuple(session.records)
            try:
                first_binding = plans.bind(mode, shape_key, plan)
            except BaseException as exc:
                detail = stable_kda_exception_detail(exc)
                for event in session.records:
                    telemetry.emit(
                        event
                        if event.fatal
                        else event.as_fatal(
                            reason="cuda_graph_plan_bind_exception", detail=detail
                        )
                    )
                raise
            if first_binding:
                for event in plan:
                    telemetry.emit(event)
    finally:
        _ACTIVE_CAPTURE.reset(token)


_ReplayResult = TypeVar("_ReplayResult")


def replay_kda_route_plan(
    shape_key: Hashable,
    mode: str,
    replay: Callable[[], _ReplayResult],
    *,
    telemetry: KDATerminalRouteTelemetry = KDA_ROUTE_TELEMETRY,
    plans: KDACudaGraphRoutePlans = KDA_CUDA_GRAPH_ROUTE_PLANS,
) -> _ReplayResult:
    """Replay the backend, then emit its exact captured KDA route plan."""
    plan = plans.require(mode, shape_key)
    try:
        result = replay()
    except BaseException as exc:
        detail = stable_kda_exception_detail(exc)
        for event in plan:
            telemetry.emit(
                event.for_phase("replay").as_fatal(
                    reason="cuda_graph_replay_exception", detail=detail
                )
            )
        raise
    for event in plan:
        telemetry.emit(event.for_phase("replay"))
    return result


def get_kda_route_raw_events() -> tuple[KDATerminalRouteEvent, ...]:
    return KDA_ROUTE_TELEMETRY.raw_events_snapshot()


def get_kda_route_raw_events_jsonl() -> str:
    return KDA_ROUTE_TELEMETRY.raw_events_jsonl()


def get_kda_route_counter_snapshot() -> dict[str, Any]:
    return KDA_ROUTE_TELEMETRY.counter_snapshot()


def get_kda_route_counter_snapshot_json() -> str:
    return KDA_ROUTE_TELEMETRY.counter_snapshot_json()
