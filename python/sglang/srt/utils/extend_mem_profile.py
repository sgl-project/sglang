# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Hook-local caching-allocator allocation deltas plus the whole-extend
allocated peak, for eager extend (prefill) forwards.

Enabled by ``SGLANG_EXTEND_MEM_PROFILE=1``, read once at import into
:data:`ENABLED`. When it is unset, :func:`phase` and :func:`record` are
module-level functions that return one shared inert context object and the
module touches nothing else; the model runner and the per-layer hooks check
:data:`ENABLED` before doing any work of their own.

What is measured: all numbers are ``torch.cuda`` caching-allocator counters
(``memory_allocated`` / ``max_memory_allocated``), i.e. bytes the allocator
has handed out to tensors on the current device. They do not include
allocator-cached-but-free segments, memory taken directly from the driver by
libraries (cuBLAS / NCCL / FlashInfer workspaces, kernel scratch outside the
allocator), or anything on other devices. Reading the counters is host-side
bookkeeping, so no device synchronization is added.

Usage: ``record(num_tokens)`` as a context manager around one extend forward
(``begin``/``end`` are the explicit form), ``phase(tag)`` as a context manager
around any sub-step (an attention backend's kernels, the multimodal embedding
step). Only extends with at least ``min_tokens`` real tokens are recorded.

* A phase's value is a hook-local allocation delta: ``max_memory_allocated``
  during its body minus ``memory_allocated`` at its entry. Each phase is
  measured against its own entry live set, so the values of different phases
  are not additive and are not disjoint parts of the extend peak: tensors a
  phase leaves alive (a conv output consumed by the next kernel) are inside
  the later phases' baselines, not their deltas. Repeated tags (one per
  decoder layer) keep their maximum; a nested phase's delta is folded into
  its enclosing phase.
* The whole-extend allocated peak is ``max_memory_allocated`` over the whole
  extend minus ``memory_allocated`` at ``begin``. The allocator keeps one
  device-wide peak counter, which every phase entry resets, so the extend
  peak is folded together from the counter's value at every reset point and
  at ``end``; allocations made outside any phase still count towards it.

``end`` logs one line with the extend peak and the largest phase deltas.
Phases whose delta or retained delta reaches ``LARGE_PHASE_BYTES`` are logged
immediately, so an OOM later in the same extend does not lose them. A phase
that still retains that much after it ends also requests an allocator-history
snapshot through :mod:`sglang.srt.utils.mem_forensics`, which writes one when
``SGLANG_MEM_FORENSICS_DIR`` is set and is a no-op otherwise.

The profiler is fail-open: an exception raised by the profiled code always
propagates unchanged (the context manager calls ``end`` on all exits), and an
allocator query that fails inside the profiler disables profiling for the
rest of that extend with one warning instead of raising.

State is module-global and the peak counter is per device. The profiler
assumes serialized forward calls on the current device (speculative decoding
builds a draft runner next to the target runner in the same process; their
forwards do not overlap); it is not thread-safe, and allocations made by
another thread or stream on the same device during a phase are attributed to
that phase.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

LARGE_PHASE_BYTES = 256 * 2**20
DEFAULT_MIN_TOKENS = 1024

# Bound once at import: the hooks read this constant, never the environment.
ENABLED: bool = envs.SGLANG_EXTEND_MEM_PROFILE.get()

_active: bool = False
_num_tokens: int = 0
_base: int = 0
_peak: int = 0  # highest allocator value observed so far in this extend
_phase_peaks: Dict[str, int] = {}
_open_phases: List[_Phase] = []
_extend_count: int = 0


def enabled() -> bool:
    return ENABLED


def _disable(where: str, exc: BaseException) -> None:
    """Fail open: stop profiling this extend rather than raise from the profiler."""
    global _active
    _active = False
    _open_phases.clear()
    logger.warning(
        "extend-mem-profile disabled for the rest of this extend: %s failed: %r",
        where,
        exc,
    )


def _observe_peak() -> int:
    """Fold the device-wide peak counter into the extend peak and return it."""
    global _peak
    peak = torch.cuda.max_memory_allocated()
    if peak > _peak:
        _peak = peak
    return peak


def begin(num_tokens: int, min_tokens: int = DEFAULT_MIN_TOKENS) -> None:
    """Start recording one extend forward of ``num_tokens`` real tokens."""
    global _active, _num_tokens, _base, _peak, _phase_peaks
    _active = False
    _open_phases.clear()
    if not ENABLED or num_tokens < min_tokens:
        return
    try:
        if not torch.cuda.is_available():
            return
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    except Exception as exc:
        _disable("begin", exc)
        return
    _num_tokens = num_tokens
    _base = base
    _peak = base
    _phase_peaks = {}
    _active = True


def end() -> None:
    """Finish the current extend and log its summary line. Never raises."""
    global _active, _extend_count
    if not _active:
        return
    try:
        _extend_count += 1
        _observe_peak()  # allocations since the last phase reset
        peak = _peak - _base
        live_now = torch.cuda.memory_allocated()
        free_b, _ = torch.cuda.mem_get_info()
        top = sorted(_phase_peaks.items(), key=lambda kv: kv[1], reverse=True)[:8]
        phases = ", ".join(f"{k}={v / 2**20:.0f}MiB" for k, v in top)
        logger.info(
            "extend-mem-profile #%d tokens=%d extend_alloc_peak=%.0fMiB "
            "live_at_entry=%.2fGiB live_after=%.2fGiB device_free_now=%.2fGiB "
            "top_phase_alloc_deltas[%s]",
            _extend_count,
            _num_tokens,
            peak / 2**20,
            _base / 2**30,
            live_now / 2**30,
            free_b / 2**30,
            phases,
        )
    except Exception as exc:
        _disable("end", exc)
    finally:
        _active = False
        _open_phases.clear()


class _NoopScope:
    """Shared inert context object handed out whenever nothing is recorded."""

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


_NOOP_SCOPE = _NoopScope()


class _ExtendScope:
    __slots__ = ("num_tokens", "min_tokens")

    def __init__(self, num_tokens: int, min_tokens: int):
        self.num_tokens = num_tokens
        self.min_tokens = min_tokens

    def __enter__(self) -> None:
        begin(self.num_tokens, self.min_tokens)
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        end()
        return False


class _Phase:
    __slots__ = ("tag", "before", "peak")

    def __init__(self, tag: str):
        self.tag = tag
        self.before = -1  # entry not completed: exit records nothing
        self.peak = 0  # highest allocator value attributable to this phase

    def __enter__(self) -> None:
        if not _active:
            return None
        try:
            # What accumulated since the last reset belongs to the enclosing
            # phases (and the whole extend); fold it in before resetting.
            outer_peak = _observe_peak()
            for outer in _open_phases:
                if outer_peak > outer.peak:
                    outer.peak = outer_peak
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        except Exception as exc:
            _disable(f"phase {self.tag} entry", exc)
            return None
        self.before = before
        self.peak = before
        _open_phases.append(self)
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self.before < 0 or not _active:
            return False
        try:
            self._finish()
        except Exception as e:
            _disable(f"phase {self.tag} exit", e)
        return False

    def _finish(self) -> None:
        peak_abs = max(self.peak, _observe_peak())
        if _open_phases and _open_phases[-1] is self:
            _open_phases.pop()
        for outer in _open_phases:
            if peak_abs > outer.peak:
                outer.peak = peak_abs
        peak = peak_abs - self.before
        prev = _phase_peaks.get(self.tag)
        if prev is None or peak > prev:
            _phase_peaks[self.tag] = peak
        # Report large phases immediately: an OOM later in the same extend
        # would otherwise take the numbers with it.
        live_after = torch.cuda.memory_allocated()
        retained = live_after - self.before
        if peak >= LARGE_PHASE_BYTES or retained >= LARGE_PHASE_BYTES:
            free_b, _ = torch.cuda.mem_get_info()
            logger.info(
                "extend-mem-profile phase %s alloc_delta=%.0fMiB retained=%.0fMiB "
                "live_after=%.2fGiB device_free=%.2fGiB",
                self.tag,
                peak / 2**20,
                retained / 2**20,
                live_after / 2**30,
                free_b / 2**30,
            )
        if retained >= LARGE_PHASE_BYTES:
            # A phase that leaves this much behind is the question the
            # profiler exists to answer; when allocator history is being
            # recorded (SGLANG_MEM_FORENSICS_DIR), snapshot it with stacks.
            from sglang.srt.utils.mem_forensics import maybe_dump_memory_forensics

            maybe_dump_memory_forensics(f"retained-{self.tag}")


def _record_enabled(num_tokens: int, min_tokens: int = DEFAULT_MIN_TOKENS):
    """``record`` when enabled: context manager for one extend forward,
    ``begin`` on entry and ``end`` on every exit (normal or exception).
    Returns the shared no-op object below ``min_tokens``."""
    if num_tokens < min_tokens:
        return _NOOP_SCOPE
    return _ExtendScope(num_tokens, min_tokens)


def _record_disabled(num_tokens: int, min_tokens: int = DEFAULT_MIN_TOKENS):
    """``record`` when disabled: the shared no-op object, nothing else."""
    return _NOOP_SCOPE


def _phase_enabled(tag: str):
    """``phase`` when enabled: context manager recording the hook-local
    allocation delta of the enclosed step under ``tag``; the shared no-op
    object outside an active extend."""
    return _Phase(tag) if _active else _NOOP_SCOPE


def _phase_disabled(tag: str):
    """``phase`` when disabled: the shared no-op object, nothing else."""
    return _NOOP_SCOPE


def _bind(enabled_flag: bool) -> None:
    """Select the enabled or disabled ``record`` / ``phase`` entry points.
    Called once at import from the env; tests call it to flip the profiler
    without re-importing. Hooks look the functions up on the module at call
    time, so rebinding here is enough."""
    global ENABLED, record, phase, _active
    ENABLED = bool(enabled_flag)
    _active = False
    _open_phases.clear()
    record = _record_enabled if ENABLED else _record_disabled
    phase = _phase_enabled if ENABLED else _phase_disabled


record = _record_disabled
phase = _phase_disabled
_bind(ENABLED)
