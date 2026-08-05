"""Process-global registry of engine startup phase durations.

Startup work that explicit plumbing cannot reach (JIT compiles inside
kernel wrappers, distributed init, autotune, ...) accumulates named
durations here. The registry is frozen at scheduler-ready
and merged into the ``startup_time`` dict (see
sglang.srt.observability.startup_time); post-freeze work (e.g. lazy JIT
compiles) is drained into a separate counter instead.

Phases are independent measurements that may nest or overlap, so their sum
can exceed time-to-ready. This module must stay dependency-free so any
startup code path can import it without cycles.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_lock = threading.Lock()
_phases: dict[str, float] = {}
_frozen: dict[str, float] | None = None
_drained: dict[str, float] = {}
_prefix: ContextVar[str] = ContextVar("startup_phase_prefix", default="")


@contextmanager
def startup_phase_prefix(prefix: str) -> Iterator[None]:
    """Prefix every phase recorded in the enclosed scope, however deeply
    nested (e.g. ``draft_`` to attribute draft-model work). Scopes replace
    rather than compose: an inner scope overrides an outer one."""
    token = _prefix.set(prefix)
    try:
        yield
    finally:
        _prefix.reset(token)


def record_startup_phase(phase: str, seconds: float) -> None:
    """Add ``seconds`` to the running total for ``phase``."""
    name = _prefix.get() + phase
    with _lock:
        _phases[name] = _phases.get(name, 0.0) + seconds


@contextmanager
def startup_phase(phase: str) -> Iterator[None]:
    """Time the enclosed block and record it under ``phase``."""
    tic = time.perf_counter()
    try:
        yield
    finally:
        record_startup_phase(phase, time.perf_counter() - tic)


def get_startup_phases() -> dict[str, float]:
    """Snapshot of all recorded phase totals (live, ignores the freeze)."""
    with _lock:
        return dict(_phases)


def freeze_startup_phases() -> dict[str, float]:
    """Freeze the cold-start snapshot and return it. Called at
    scheduler-ready; idempotent, so late callers cannot fold post-ready
    work into the snapshot."""
    global _frozen
    with _lock:
        if _frozen is None:
            _frozen = dict(_phases)
        return dict(_frozen)


def drain_post_startup_deltas() -> dict[str, float]:
    """Per-phase seconds accumulated past the freeze, each returned exactly
    once (counter-friendly). Empty before the freeze."""
    with _lock:
        if _frozen is None:
            return {}
        deltas: dict[str, float] = {}
        for phase, total in _phases.items():
            watermark = _drained.get(phase, _frozen.get(phase, 0.0))
            delta = total - watermark
            if delta > 0.0:
                deltas[phase] = delta
                _drained[phase] = total
        return deltas


def reset_startup_phases() -> None:
    """Clear all registry state. Intended for tests."""
    global _frozen
    with _lock:
        _phases.clear()
        _drained.clear()
        _frozen = None
