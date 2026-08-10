"""Terminal route telemetry for the Phase-A CAKE KDA serving paths.

Each KDA layer call produces exactly one immutable terminal event.  The three
terminal outcomes are mutually exclusive and exhaustive::

    considered == cake_success + triton_fallback + fatal

CUDA graph capture needs special handling because a route can belong either to
a physically captured graph segment or to live Python in a breakable/piecewise
replay seam.  Graph runners therefore wrap ``backend.capture_one`` in
:func:`capture_kda_route_plan`.  Backend warmups are ignored; only the backend's
actual capture pass enters an immutable per-``ShapeKey`` plan, with replay
ownership recorded for every route.  A successful capture emits the capture
events once.  :func:`replay_kda_route_plan` stages live-Python events, merges
them with synthesized captured-segment events, and commits only after the whole
backend replay returns.  A later segment or bridge exception therefore
fatalizes the transaction instead of leaving a false success receipt.

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


class CakePackedDecodeReason:
    """Stable CAKE packed-decode admission reason codes for schema v1."""

    ELIGIBLE = "eligible"
    KERNEL_UNAVAILABLE = "kernel_unavailable"
    REPLAYSSM_REQUESTED = "replayssm_requested"
    UNSUPPORTED_CONTRACT = "unsupported_contract"
    INNER_STRIDE = "inner_stride"
    ZERO_ROW_STRIDE = "zero_row_stride"
    NEGATIVE_ROW_STRIDE = "negative_row_stride"
    OVERLAPPING_ROW_STRIDE = "overlapping_row_stride"
    STORAGE_ALIAS = "storage_alias"
    CACHE_INDEX_UNVERIFIED = "cache_index_unverified"
    CACHE_INDEX_OOB = "cache_index_oob"
    CACHE_INDEX_DUPLICATE = "cache_index_duplicate"


class CakePrefillReason:
    """Stable CAKE ordinary-prefill admission reason codes for schema v1."""

    ELIGIBLE = "eligible"
    SPEC_DECODE = "spec_decode"
    INTERIOR_CHECKPOINT = "interior_checkpoint"
    INVALID_LOWER_BOUND = "invalid_lower_bound"
    MISSING_GATE_PARAMS = "missing_gate_params"
    UNSUPPORTED_Q_CONTRACT = "unsupported_q_contract"
    CUDA_GRAPH_ALLOCATION = "cuda_graph_allocation"
    T1_DECODE_SHAPE = "t1_decode_shape"
    UNSUPPORTED_HEAD_DIM = "unsupported_head_dim"
    SHAPE_MISMATCH = "shape_mismatch"
    UNSUPPORTED_ARCH = "unsupported_arch"
    UNSUPPORTED_CONTRACT = "unsupported_contract"


PREFILL_SELECTOR_EXCEPTION = "prefill_selector_exception"
PACKED_SELECTOR_EXCEPTION = "packed_selector_exception"
TRITON_FALLBACK_EXCEPTION = "triton_fallback_exception"
CAKE_PREFILL_EXCEPTION = "cake_prefill_exception"
CAKE_PACKED_EXCEPTION = "cake_packed_exception"
CUDA_GRAPH_CAPTURE_EXCEPTION = "cuda_graph_capture_exception"
CUDA_GRAPH_PLAN_BIND_EXCEPTION = "cuda_graph_plan_bind_exception"
CUDA_GRAPH_REPLAY_EXCEPTION = "cuda_graph_replay_exception"
CUDA_GRAPH_REPLAY_PLAN_MISMATCH = "cuda_graph_replay_plan_mismatch"

_DECODE_FALLBACK_REASONS = frozenset(
    (
        CakePackedDecodeReason.KERNEL_UNAVAILABLE,
        CakePackedDecodeReason.REPLAYSSM_REQUESTED,
        CakePackedDecodeReason.UNSUPPORTED_CONTRACT,
        CakePackedDecodeReason.INNER_STRIDE,
        CakePackedDecodeReason.ZERO_ROW_STRIDE,
        CakePackedDecodeReason.NEGATIVE_ROW_STRIDE,
        CakePackedDecodeReason.OVERLAPPING_ROW_STRIDE,
        CakePackedDecodeReason.STORAGE_ALIAS,
        CakePackedDecodeReason.CACHE_INDEX_UNVERIFIED,
        CakePackedDecodeReason.CACHE_INDEX_OOB,
        CakePackedDecodeReason.CACHE_INDEX_DUPLICATE,
    )
)
_PREFILL_FALLBACK_REASONS = frozenset(
    (
        CakePrefillReason.SPEC_DECODE,
        CakePrefillReason.INTERIOR_CHECKPOINT,
        CakePrefillReason.INVALID_LOWER_BOUND,
        CakePrefillReason.MISSING_GATE_PARAMS,
        CakePrefillReason.UNSUPPORTED_Q_CONTRACT,
        CakePrefillReason.CUDA_GRAPH_ALLOCATION,
        CakePrefillReason.T1_DECODE_SHAPE,
        CakePrefillReason.UNSUPPORTED_HEAD_DIM,
        CakePrefillReason.SHAPE_MISMATCH,
        CakePrefillReason.UNSUPPORTED_ARCH,
        CakePrefillReason.UNSUPPORTED_CONTRACT,
    )
)
_DIRECT_FATAL_REASONS = {
    ("decode", 0): frozenset((PACKED_SELECTOR_EXCEPTION, TRITON_FALLBACK_EXCEPTION)),
    ("decode", 1): frozenset((CAKE_PACKED_EXCEPTION,)),
    ("prefill", 0): frozenset((PREFILL_SELECTOR_EXCEPTION, TRITON_FALLBACK_EXCEPTION)),
    ("prefill", 1): frozenset((CAKE_PREFILL_EXCEPTION,)),
}
_GRAPH_FATAL_PHASES = {
    CUDA_GRAPH_CAPTURE_EXCEPTION: "capture",
    CUDA_GRAPH_PLAN_BIND_EXCEPTION: "capture",
    CUDA_GRAPH_REPLAY_EXCEPTION: "replay",
    CUDA_GRAPH_REPLAY_PLAN_MISMATCH: "replay",
}


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
        if self.eligible != self.attempted_cake:
            raise ValueError(
                "KDA route funnel invariant failed: eligible must equal attempted_cake"
            )
        route_bits = (self.eligible, self.attempted_cake)
        if self.cake_success and route_bits != (1, 1):
            raise ValueError("cake_success requires eligible=1 and attempted_cake=1")
        if self.triton_fallback and route_bits != (0, 0):
            raise ValueError(
                "successful Triton fallback requires eligible=0 and attempted_cake=0"
            )
        if self.fatal and route_bits not in ((0, 0), (1, 1)):
            raise ValueError("fatal KDA route has invalid eligibility/attempt bits")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("KDA route reason must be a stable non-empty string")
        if not isinstance(self.detail, str):
            raise TypeError("KDA route detail must be a string")
        if self.graph_phase not in _VALID_PHASES:
            raise ValueError(f"invalid KDA graph phase: {self.graph_phase!r}")

        if self.cake_success:
            allowed_reasons = frozenset((CakePackedDecodeReason.ELIGIBLE,))
        elif self.triton_fallback:
            allowed_reasons = (
                _DECODE_FALLBACK_REASONS
                if self.mode == "decode"
                else _PREFILL_FALLBACK_REASONS
            )
        else:
            allowed_reasons = _DIRECT_FATAL_REASONS[
                (self.mode, self.eligible)
            ] | frozenset(_GRAPH_FATAL_PHASES)
        if self.reason not in allowed_reasons:
            outcome = (
                "cake_success"
                if self.cake_success
                else "triton_fallback" if self.triton_fallback else "fatal"
            )
            raise ValueError(
                "invalid KDA route reason for mode/outcome/funnel: "
                f"mode={self.mode!r}, outcome={outcome!r}, "
                f"eligible={self.eligible}, reason={self.reason!r}"
            )
        required_phase = _GRAPH_FATAL_PHASES.get(self.reason)
        if required_phase is not None and self.graph_phase != required_phase:
            raise ValueError(
                f"KDA graph reason {self.reason!r} requires "
                f"graph_phase={required_phase!r}"
            )

        if self.copy_count is not None and (
            isinstance(self.copy_count, bool)
            or not isinstance(self.copy_count, int)
            or self.copy_count < 0
        ):
            raise ValueError("copy_count must be a non-negative int or None")
        if self.copy_count_source not in _VALID_COPY_COUNT_SOURCES:
            raise ValueError(f"invalid copy_count_source: {self.copy_count_source!r}")
        packed_zero_copy_success = bool(
            self.mode == "decode"
            and self.cake_success
            and self.reason == CakePackedDecodeReason.ELIGIBLE
        )
        expected_copy_state = (
            (0, "static_zero_copy_row_view")
            if packed_zero_copy_success
            else (None, "unknown_requires_cupti")
        )
        if (self.copy_count, self.copy_count_source) != expected_copy_state:
            raise ValueError(
                "invalid KDA copy-count state for route: expected "
                f"{expected_copy_state!r}, got "
                f"{(self.copy_count, self.copy_count_source)!r}"
            )

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
        self.emit_many((event,))

    def emit_many(self, events: tuple[KDATerminalRouteEvent, ...]) -> None:
        """Atomically retain and count one committed event transaction."""
        with self._lock:
            for event in events:
                key = (event.mode, event.layer_id, event.graph_phase, event.reason)
                self._events.append(event)
                self._event_count += 1
                self._counters[key].add(event)
        if self._emit_log:
            for event in events:
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
                if row["eligible"] != row["attempted_cake"]:
                    raise KDATelemetryError(
                        "KDA route counter funnel failed for "
                        f"{(mode, layer_id, phase, reason)!r}"
                    )
                if row["cake_success"] > row["attempted_cake"]:
                    raise KDATelemetryError(
                        "KDA route counter success funnel failed for "
                        f"{(mode, layer_id, phase, reason)!r}"
                    )
                if row["triton_fallback"] > (row["considered"] - row["eligible"]):
                    raise KDATelemetryError(
                        "KDA route counter fallback funnel failed for "
                        f"{(mode, layer_id, phase, reason)!r}"
                    )
                if (
                    row["known_copy_calls"] + row["unknown_copy_calls"]
                    != row["considered"]
                ):
                    raise KDATelemetryError(
                        "KDA route counter copy-call closure failed for "
                        f"{(mode, layer_id, phase, reason)!r}"
                    )
                if row["known_copy_count"] != 0:
                    raise KDATelemetryError(
                        "KDA route schema v1 only permits a statically known zero "
                        "copy count for "
                        f"{(mode, layer_id, phase, reason)!r}"
                    )
                rows.append(row)
            retained = len(self._events)
            total = self._event_count
            if sum(row["considered"] for row in rows) != total:
                raise KDATelemetryError(
                    "KDA route counter total does not match emitted event total"
                )
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


_REPLAY_OWNER_CAPTURED_GRAPH = "captured_graph"
_REPLAY_OWNER_LIVE_PYTHON = "live_python"
_VALID_REPLAY_OWNERS = frozenset(
    (_REPLAY_OWNER_CAPTURED_GRAPH, _REPLAY_OWNER_LIVE_PYTHON)
)


@dataclass(frozen=True)
class _KDAPlannedRoute:
    event: KDATerminalRouteEvent
    replay_owner: str

    def __post_init__(self) -> None:
        if self.replay_owner not in _VALID_REPLAY_OWNERS:
            raise ValueError(f"invalid KDA replay owner: {self.replay_owner!r}")


@dataclass
class _CaptureSession:
    mode: str
    shape_key: Hashable
    actual_capture_probe: Callable[[], bool]
    physical_capture_probe: Callable[[], bool]
    telemetry: KDATerminalRouteTelemetry
    records: list[_KDAPlannedRoute]


@dataclass
class _ReplaySession:
    mode: str
    shape_key: Hashable
    telemetry: KDATerminalRouteTelemetry
    records: list[KDATerminalRouteEvent]


_ACTIVE_CAPTURE: ContextVar[Optional[_CaptureSession]] = ContextVar(
    "sglang_kda_active_capture", default=None
)
_ACTIVE_REPLAY: ContextVar[Optional[_ReplaySession]] = ContextVar(
    "sglang_kda_active_replay", default=None
)
_SUPPRESS_RECORDING: ContextVar[bool] = ContextVar(
    "sglang_kda_suppress_recording", default=False
)


class KDACudaGraphRoutePlans:
    """Immutable KDA route plans, partitioned by mode and keyed by ShapeKey."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._plans: dict[str, dict[Hashable, tuple[_KDAPlannedRoute, ...]]] = {
            mode: {} for mode in _VALID_MODES
        }

    def bind(
        self,
        mode: str,
        shape_key: Hashable,
        plan: tuple[_KDAPlannedRoute, ...],
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

    def require(self, mode: str, shape_key: Hashable) -> tuple[_KDAPlannedRoute, ...]:
        _validate_mode(mode)
        with self._lock:
            try:
                return self._plans[mode][shape_key]
            except KeyError as exc:
                raise KDACudaGraphPlanError(
                    "missing KDA CUDA graph route plan for "
                    f"mode={mode!r}, shape_key={shape_key!r}"
                ) from exc

    def snapshot(self, mode: str, shape_key: Hashable) -> tuple[_KDAPlannedRoute, ...]:
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


@contextmanager
def suppress_kda_route_recording() -> Iterator[None]:
    """Suppress non-serving warmup/compile calls without touching GPU state."""
    token = _SUPPRESS_RECORDING.set(True)
    try:
        yield
    finally:
        _SUPPRESS_RECORDING.reset(token)


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
    if _SUPPRESS_RECORDING.get():
        return None
    capture_session = _ACTIVE_CAPTURE.get()
    replay_session = _ACTIVE_REPLAY.get()
    if capture_session is not None and replay_session is not None:
        raise KDACudaGraphPlanError(
            "overlapping KDA CUDA graph capture and replay contexts"
        )
    graph_phase = (
        "capture"
        if capture_session is not None
        else "replay" if replay_session is not None else "direct"
    )
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
    session = capture_session or replay_session
    if session is None:
        telemetry.emit(event)
        return event
    if session.mode != mode:
        raise KDATelemetryError(
            "KDA route mode does not match active CUDA graph transaction: "
            f"record={mode!r}, transaction={session.mode!r}"
        )
    if session.telemetry is not telemetry:
        raise KDATelemetryError(
            "KDA route telemetry sink does not match active graph transaction"
        )
    if capture_session is not None:
        if not capture_session.actual_capture_probe():
            return None
        replay_owner = (
            _REPLAY_OWNER_CAPTURED_GRAPH
            if capture_session.physical_capture_probe()
            else _REPLAY_OWNER_LIVE_PYTHON
        )
        capture_session.records.append(_KDAPlannedRoute(event, replay_owner))
    else:
        replay_session.records.append(event)
    return event


@contextmanager
def capture_kda_route_plan(
    shape_key: Hashable,
    mode: str,
    *,
    capture_probe: Callable[[], bool] = _torch_capture_probe,
    physical_capture_probe: Callable[[], bool] = _torch_capture_probe,
    telemetry: KDATerminalRouteTelemetry = KDA_ROUTE_TELEMETRY,
    plans: KDACudaGraphRoutePlans = KDA_CUDA_GRAPH_ROUTE_PLANS,
) -> Iterator[None]:
    """Collect only the actual capture pass and bind an immutable route plan."""
    _validate_mode(mode)
    if _ACTIVE_CAPTURE.get() is not None or _ACTIVE_REPLAY.get() is not None:
        raise KDACudaGraphPlanError("nested KDA CUDA graph telemetry transaction")
    session = _CaptureSession(
        mode,
        shape_key,
        capture_probe,
        physical_capture_probe,
        telemetry,
        [],
    )
    token = _ACTIVE_CAPTURE.set(session)
    try:
        try:
            yield
        except BaseException as exc:
            detail = stable_kda_exception_detail(exc)
            telemetry.emit_many(
                tuple(
                    (
                        entry.event
                        if entry.event.fatal
                        else entry.event.as_fatal(
                            reason=CUDA_GRAPH_CAPTURE_EXCEPTION, detail=detail
                        )
                    )
                    for entry in session.records
                )
            )
            raise
        else:
            plan = tuple(session.records)
            try:
                first_binding = plans.bind(mode, shape_key, plan)
            except BaseException as exc:
                detail = stable_kda_exception_detail(exc)
                telemetry.emit_many(
                    tuple(
                        (
                            entry.event
                            if entry.event.fatal
                            else entry.event.as_fatal(
                                reason=CUDA_GRAPH_PLAN_BIND_EXCEPTION, detail=detail
                            )
                        )
                        for entry in session.records
                    )
                )
                raise
            if first_binding:
                telemetry.emit_many(tuple(entry.event for entry in plan))
    finally:
        _ACTIVE_CAPTURE.reset(token)


_ReplayResult = TypeVar("_ReplayResult")


def _fatal_replay_events(
    plan: tuple[_KDAPlannedRoute, ...],
    live_records: tuple[KDATerminalRouteEvent, ...],
    *,
    reason: str,
    detail: str,
) -> tuple[KDATerminalRouteEvent, ...]:
    """Fatalize a replay transaction without retaining staged successes."""
    output: list[KDATerminalRouteEvent] = []
    live_index = 0
    for entry in plan:
        basis = entry.event.for_phase("replay")
        if entry.replay_owner == _REPLAY_OWNER_LIVE_PYTHON:
            if live_index < len(live_records):
                basis = live_records[live_index]
            live_index += 1
        output.append(
            basis if basis.fatal else basis.as_fatal(reason=reason, detail=detail)
        )
    for extra in live_records[live_index:]:
        output.append(
            extra if extra.fatal else extra.as_fatal(reason=reason, detail=detail)
        )
    return tuple(output)


def _merge_successful_replay(
    plan: tuple[_KDAPlannedRoute, ...],
    live_records: tuple[KDATerminalRouteEvent, ...],
) -> tuple[KDATerminalRouteEvent, ...]:
    expected_live = sum(
        entry.replay_owner == _REPLAY_OWNER_LIVE_PYTHON for entry in plan
    )
    if len(live_records) != expected_live:
        raise KDACudaGraphPlanError(
            "KDA live replay event count changed for immutable graph plan: "
            f"expected={expected_live}, actual={len(live_records)}"
        )

    output: list[KDATerminalRouteEvent] = []
    live_index = 0
    for entry in plan:
        if entry.replay_owner == _REPLAY_OWNER_CAPTURED_GRAPH:
            output.append(entry.event.for_phase("replay"))
            continue
        actual = live_records[live_index]
        live_index += 1
        if actual.for_phase("capture") != entry.event:
            raise KDACudaGraphPlanError(
                "KDA live replay route changed for immutable graph plan: "
                f"expected={entry.event.to_json()}, actual={actual.to_json()}"
            )
        output.append(actual)
    return tuple(output)


def replay_kda_route_plan(
    shape_key: Hashable,
    mode: str,
    replay: Callable[[], _ReplayResult],
    *,
    telemetry: KDATerminalRouteTelemetry = KDA_ROUTE_TELEMETRY,
    plans: KDACudaGraphRoutePlans = KDA_CUDA_GRAPH_ROUTE_PLANS,
) -> _ReplayResult:
    """Replay atomically, then commit one terminal event per planned layer."""
    plan = plans.require(mode, shape_key)
    if _ACTIVE_CAPTURE.get() is not None or _ACTIVE_REPLAY.get() is not None:
        raise KDACudaGraphPlanError("nested KDA CUDA graph telemetry transaction")
    session = _ReplaySession(mode, shape_key, telemetry, [])
    try:
        token = _ACTIVE_REPLAY.set(session)
        try:
            result = replay()
        finally:
            _ACTIVE_REPLAY.reset(token)
    except BaseException as exc:
        detail = stable_kda_exception_detail(exc)
        telemetry.emit_many(
            _fatal_replay_events(
                plan,
                tuple(session.records),
                reason=CUDA_GRAPH_REPLAY_EXCEPTION,
                detail=detail,
            )
        )
        raise

    try:
        committed = _merge_successful_replay(plan, tuple(session.records))
    except KDACudaGraphPlanError as exc:
        telemetry.emit_many(
            _fatal_replay_events(
                plan,
                tuple(session.records),
                reason=CUDA_GRAPH_REPLAY_PLAN_MISMATCH,
                detail=stable_kda_exception_detail(exc),
            )
        )
        raise
    telemetry.emit_many(committed)
    return result


def get_kda_route_raw_events() -> tuple[KDATerminalRouteEvent, ...]:
    return KDA_ROUTE_TELEMETRY.raw_events_snapshot()


def get_kda_route_raw_events_jsonl() -> str:
    return KDA_ROUTE_TELEMETRY.raw_events_jsonl()


def get_kda_route_counter_snapshot() -> dict[str, Any]:
    return KDA_ROUTE_TELEMETRY.counter_snapshot()


def get_kda_route_counter_snapshot_json() -> str:
    return KDA_ROUTE_TELEMETRY.counter_snapshot_json()
