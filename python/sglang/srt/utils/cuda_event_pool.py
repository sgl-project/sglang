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
"""Reusable CUDA events for :meth:`torch.cuda.Stream.wait_stream`.

PyTorch implements ``destination.wait_stream(source)`` as::

    destination.wait_event(source.record_event())

When no event is supplied, ``record_event`` allocates a new lazy CUDA event.
On drivers where event creation contends on the CUDA context lock, that can
turn a lightweight stream dependency into a long host-side stall.

This module moves those allocations to worker initialization and reuses a
fixed FIFO of no-timing events for the narrow record-then-wait operation.  A
CUDA stream wait captures the event state when the wait is issued, so a later
record of the same event does not alter an already-enqueued wait.

The pre-create/borrow/recycle strategy follows TensorRT-LLM KVCM2's
``CachedCudaEvent`` pool.  This specialization has a shorter lease: KVCM2
keeps general events until query/synchronize/close, while a stream dependency
can return its event immediately after ``wait_event`` has been enqueued.  The
pool size therefore bounds the number of concurrent ``wait_stream`` callers,
not the number of pending GPU waits.

Explicit ``torch.cuda.Event`` instances are intentionally unaffected: their
lifetime can extend beyond one record/wait pair and must remain owned by their
caller.  PyTorch's original implementation is also used when:

- the current stream is capturing, because reusing external event objects
  across graph captures has different lifetime and graph-identity
  requirements.  Only the current stream is checked; a call made while another
  stream is capturing takes the pooled path, which issues the same record and
  wait calls on a reused event;
- the source is not a ``torch.cuda.Stream`` (e.g. a generic ``torch.Stream``),
  whose ``record_event`` accepts only ``torch.Event``;
- the source device has no prewarmed pool (see
  :func:`prewarm_cuda_event_pool`), so events are never created on the hot
  path;
- every event of the pool is leased.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Serializes install, uninstall, prewarm and stats. wait_stream never takes it.
_INSTALL_LOCK = threading.Lock()
_WARNING_LOCK = threading.Lock()

_INSTALLATION: _Installation | None = None
_EXHAUSTION_WARNED = False
_STATS: dict[str, int] = {
    "pooled_calls": 0,
    # Calls forwarded to PyTorch's wait_stream, for any of the reasons below.
    "original_calls": 0,
    # The current stream was capturing; other capturing streams are not checked.
    "capture_bypasses": 0,
    # The source was not a torch.cuda.Stream.
    "non_cuda_source_fallbacks": 0,
    # The source device had no prewarmed pool.
    "unprewarmed_fallbacks": 0,
    "pool_exhaustions": 0,
    "pool_init_failures": 0,
    "pooled_call_failures": 0,
    "events_materialized": 0,
}


def _increment(name: str, amount: int = 1) -> None:
    # These counters are diagnostic. Avoid adding another lock acquisition to
    # every stream dependency; a concurrent snapshot may be approximate.
    _STATS[name] += amount


class _DeviceEventPool:
    """Fixed-size, materialized event pool for one CUDA device."""

    def __init__(self, torch_module: Any, device_index: int, size: int) -> None:
        self.device_index = device_index
        self.size = size
        self._lock = threading.Lock()
        self._available: deque[Any] = deque()
        self._quarantined: list[Any] = []
        self._in_use = 0
        self._max_in_use = 0

        # torch.cuda.Event is lazy. Recording every event on a private stream
        # moves cudaEventCreateWithFlags into this controlled initialization.
        with torch_module.cuda.device(device_index):
            materialize_stream = torch_module.cuda.Stream(device=device_index)
            events = [torch_module.cuda.Event(enable_timing=False) for _ in range(size)]
            for event in events:
                event.record(materialize_stream)
            materialize_stream.synchronize()

        self._available.extend(events)
        _increment("events_materialized", len(events))

    def acquire(self) -> Any | None:
        with self._lock:
            if not self._available:
                return None
            event = self._available.popleft()
            self._in_use += 1
            self._max_in_use = max(self._max_in_use, self._in_use)
            return event

    def release(self, event: Any) -> None:
        with self._lock:
            self._in_use -= 1
            self._available.append(event)

    def quarantine(self, event: Any) -> None:
        """Keep a failed event alive without issuing it again."""
        with self._lock:
            self._in_use -= 1
            self._quarantined.append(event)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": self.size,
                "available": len(self._available),
                "in_use": self._in_use,
                "max_in_use": self._max_in_use,
                "quarantined": len(self._quarantined),
            }


class _Installation:
    """State of one install.

    The patched method closes over it, so a call still running after uninstall
    keeps a consistent view and falls back to PyTorch's method.
    """

    def __init__(
        self,
        torch_module: Any,
        stream_class: type,
        original_wait_stream: Callable[..., None],
        pool_size: int,
    ) -> None:
        self.torch = torch_module
        self.stream_class = stream_class
        self.original_wait_stream = original_wait_stream
        self.pool_size = pool_size
        self.pools: dict[int, _DeviceEventPool] = {}
        self.disabled_devices: dict[int, str] = {}


def _call_original(
    original_wait_stream: Callable[..., None],
    destination_stream: Any,
    source_stream: Any,
) -> None:
    _increment("original_calls")
    original_wait_stream(destination_stream, source_stream)


def _warn_pool_exhausted(device_index: int, pool_size: int) -> None:
    global _EXHAUSTION_WARNED

    if _EXHAUSTION_WARNED:
        return
    with _WARNING_LOCK:
        if _EXHAUSTION_WARNED:
            return
        _EXHAUSTION_WARNED = True
    logger.warning(
        "CUDA event pool exhausted on device %d: all %d events are held by "
        "concurrent wait_stream callers or were quarantined after failures. "
        "Falling back to PyTorch's per-call event allocation; increase "
        "SGLANG_CUDA_EVENT_POOL_SIZE if this recurs.",
        device_index,
        pool_size,
    )


def _make_pooled_wait_stream(installation: _Installation) -> Callable[..., None]:
    torch_module = installation.torch
    cuda_stream_class = installation.stream_class
    original_wait_stream = installation.original_wait_stream
    pools = installation.pools

    # Same parameter name as PyTorch, so wait_stream(stream=...) keeps working.
    def wait_stream(self: Any, stream: Any) -> None:
        # Preserve PyTorch's event ownership while the current stream is
        # capturing. SGLang invokes wait_stream with a participating stream
        # current in its graph paths.
        try:
            is_capturing = bool(torch_module.cuda.is_current_stream_capturing())
        except Exception:
            # Failure to prove that reuse is safe must retain the original
            # behavior.
            is_capturing = True
        if is_capturing:
            _increment("capture_bypasses")
            return _call_original(original_wait_stream, self, stream)

        # torch.Stream.record_event accepts only torch.Event, and before
        # PyTorch 2.11 torch.cuda.Event.record assumes a torch.cuda.Stream.
        if not isinstance(stream, cuda_stream_class):
            _increment("non_cuda_source_fallbacks")
            return _call_original(original_wait_stream, self, stream)

        # Events are bound to one device, so the pool is keyed by the source.
        # device_index is a plain int; stream.device builds a torch.device.
        pool = pools.get(stream.device_index)
        if pool is None:
            # Never create events in the hot path; that is the stall this
            # module removes.
            _increment("unprewarmed_fallbacks")
            return _call_original(original_wait_stream, self, stream)

        event = pool.acquire()
        if event is None:
            # Never block waiting for a host-side lease or grow the pool in the
            # hot path. Preserve PyTorch semantics and make exhaustion
            # observable.
            _increment("pool_exhaustions")
            _warn_pool_exhausted(pool.device_index, pool.size)
            return _call_original(original_wait_stream, self, stream)

        try:
            stream.record_event(event)
            self.wait_event(event)
        except BaseException:
            _increment("pooled_call_failures")
            pool.quarantine(event)
            raise
        else:
            pool.release(event)
            _increment("pooled_calls")

    # Marks the patch, so a second copy of this module does not wrap it again.
    setattr(wait_stream, "_sglang_cuda_event_pool", installation)
    return wait_stream


def install_cuda_event_pool(*, torch_module: Any = None, pool_size: int) -> bool:
    """Patch ``torch.cuda.Stream.wait_stream`` for the current process.

    ``pool_size`` is the number of events per device.  Calls only use events
    on devices materialized by :func:`prewarm_cuda_event_pool`.

    Returns ``True`` when this call installs the patch and ``False`` when the
    same patch is already installed or CUDA is unavailable.
    """

    global _INSTALLATION

    if pool_size <= 0:
        raise ValueError("CUDA event pool size must be positive")
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    with _INSTALL_LOCK:
        if _INSTALLATION is not None:
            if torch_module is not _INSTALLATION.torch:
                raise RuntimeError(
                    "CUDA event pool is already installed for another torch module"
                )
            if pool_size != _INSTALLATION.pool_size:
                raise ValueError(
                    "CUDA event pool is already installed with "
                    f"pool_size={_INSTALLATION.pool_size}, got {pool_size}"
                )
            return False
        if not torch_module.cuda.is_available():
            return False

        stream_class = torch_module.cuda.Stream
        original_wait_stream = stream_class.wait_stream
        if hasattr(original_wait_stream, "_sglang_cuda_event_pool"):
            raise RuntimeError(
                "torch.cuda.Stream.wait_stream is already patched by another "
                "copy of the CUDA event pool"
            )
        installation = _Installation(
            torch_module, stream_class, original_wait_stream, pool_size
        )
        stream_class.wait_stream = _make_pooled_wait_stream(installation)
        _INSTALLATION = installation
        return True


def prewarm_cuda_event_pool(device: int | None = None) -> dict[str, int] | None:
    """Materialize one device's pool after the worker selects its device.

    Returns the pool snapshot, or ``None`` if materialization failed; that
    device then keeps PyTorch's implementation.
    """

    with _INSTALL_LOCK:
        installation = _INSTALLATION
        if installation is None:
            raise RuntimeError("CUDA event pool is not installed")
        if device is None:
            device = int(installation.torch.cuda.current_device())
        pool = installation.pools.get(device)
        if pool is None and device not in installation.disabled_devices:
            try:
                pool = _DeviceEventPool(
                    installation.torch, device, installation.pool_size
                )
            except Exception as exc:
                _increment("pool_init_failures")
                reason = f"{type(exc).__name__}: {exc}"
                installation.disabled_devices[device] = reason
                logger.warning(
                    "Disabling the CUDA event pool on device %d after "
                    "initialization failed: %s",
                    device,
                    reason,
                )
            else:
                installation.pools[device] = pool
        return None if pool is None else pool.snapshot()


def cuda_event_pool_stats() -> dict[str, Any]:
    """Return process-local counters for diagnostics and tests."""

    with _INSTALL_LOCK:
        installation = _INSTALLATION
        pools = {} if installation is None else dict(installation.pools)
        disabled_devices = (
            {} if installation is None else dict(installation.disabled_devices)
        )
        counters: dict[str, Any] = dict(_STATS)
    # Derived rather than counted, to keep one increment off every call.
    counters["wait_stream_calls"] = (
        counters["pooled_calls"]
        + counters["original_calls"]
        + counters["pooled_call_failures"]
    )
    return counters | {
        "installed": installation is not None,
        "pool_size": None if installation is None else installation.pool_size,
        "devices": {
            str(device): pool.snapshot() for device, pool in sorted(pools.items())
        },
        "disabled_devices": disabled_devices,
    }


def uninstall_cuda_event_pool() -> bool:
    """Restore PyTorch's method when no later hook has wrapped this one.

    The pools are dropped, so a later install can use another size.
    """

    global _INSTALLATION
    with _INSTALL_LOCK:
        installation = _INSTALLATION
        if installation is None:
            return False
        patched = installation.stream_class.wait_stream
        if getattr(patched, "_sglang_cuda_event_pool", None) is not installation:
            return False
        installation.stream_class.wait_stream = installation.original_wait_stream
        # A call still in flight finds no pool and uses PyTorch's method.
        installation.pools.clear()
        _INSTALLATION = None
        return True


def _reset_cuda_event_pool_for_testing() -> None:
    global _EXHAUSTION_WARNED

    uninstall_cuda_event_pool()
    for key in _STATS:
        _STATS[key] = 0
    _EXHAUSTION_WARNED = False
