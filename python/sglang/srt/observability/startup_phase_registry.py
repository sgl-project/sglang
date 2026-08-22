"""Process-global registry of engine startup phase durations.

Phases may nest or overlap, so their sum can exceed time-to-ready.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_lock = threading.Lock()
# Unreported work: freezing empties this, so afterwards it holds only what
# accumulated past scheduler-ready.
_phases: dict[str, float] = {}
_frozen: dict[str, float] | None = None
_prefix: ContextVar[str] = ContextVar("startup_phase_prefix", default="")


def _record(phase: str, seconds: float) -> None:
    name = _prefix.get() + phase
    with _lock:
        _phases[name] = _phases.get(name, 0.0) + seconds


@contextmanager
def startup_phase(
    phase: str | None = None, *, draft: bool | None = None
) -> Iterator[None]:
    """Measure a startup phase, attribute the work inside one, or both::

        with startup_phase("kv_cache_allocation"):        # measure
        with startup_phase(draft=runner.is_draft_worker): # attribute
        @startup_phase("load_weight")                     # measure a function

    ``draft`` names the model the enclosed work belongs to, covering ``phase``
    itself and every phase recorded inside it at any depth. A target scope inside 
    a draft one attributes to the target.
    """
    token = None if draft is None else _prefix.set("draft_" if draft else "")
    tic = time.perf_counter()
    try:
        yield
    finally:
        if phase is not None:
            _record(phase, time.perf_counter() - tic)
        if token is not None:
            _prefix.reset(token)


def freeze_startup_phases() -> dict[str, float]:
    """Close the cold-start snapshot and return it. Idempotent."""
    global _frozen
    with _lock:
        if _frozen is None:
            _frozen = dict(_phases)
            _phases.clear()
        return dict(_frozen)


def drain_post_startup_deltas() -> dict[str, float]:
    """Work recorded since the snapshot closed, returned once. Empty before."""
    with _lock:
        if _frozen is None:
            return {}
        deltas = dict(_phases)
        _phases.clear()
        return deltas


def reset_startup_phases() -> None:
    global _frozen
    with _lock:
        _phases.clear()
        _frozen = None
