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
can return its event immediately after ``wait_event`` has been enqueued.

Explicit ``torch.cuda.Event`` instances are intentionally unaffected: their
lifetime can extend beyond one record/wait pair and must remain owned by their
caller.  CUDA graph capture also uses PyTorch's original implementation because
reusing external event objects across graph captures has different lifetime
and graph-identity requirements.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Callable

logger = logging.getLogger(__name__)

_INSTALL_LOCK = threading.RLock()
_POOLS_LOCK = threading.Lock()

_TORCH: Any = None
_STREAM_CLASS: type | None = None
_ORIGINAL_WAIT_STREAM: Callable[..., None] | None = None
_POOL_SIZE = 1024
_EXHAUSTION_WARNED = False
_POOLS: dict[int, _DeviceEventPool] = {}
_DISABLED_DEVICES: dict[int, str] = {}
_STATS: dict[str, int] = {
    "wait_stream_calls": 0,
    "pooled_calls": 0,
    "original_calls": 0,
    "capture_bypasses": 0,
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


def _source_device_index(source_stream: Any) -> int:
    device = getattr(source_stream, "device", None)
    index = getattr(device, "index", None)
    if index is not None:
        return int(index)
    if isinstance(device, int):
        return device
    assert _TORCH is not None
    return int(_TORCH.cuda.current_device())


def _pool_for_device(device_index: int) -> _DeviceEventPool | None:
    assert _TORCH is not None
    pool = _POOLS.get(device_index)
    if pool is not None:
        return pool
    with _POOLS_LOCK:
        if device_index in _DISABLED_DEVICES:
            return None
        pool = _POOLS.get(device_index)
        if pool is not None:
            return pool
        try:
            pool = _DeviceEventPool(_TORCH, device_index, _POOL_SIZE)
        except Exception as exc:
            _increment("pool_init_failures")
            reason = f"{type(exc).__name__}: {exc}"
            _DISABLED_DEVICES[device_index] = reason
            logger.warning(
                "Disabling the CUDA event pool on device %d after initialization "
                "failed: %s",
                device_index,
                reason,
            )
            return None
        _POOLS[device_index] = pool
        return pool


def _call_original(destination_stream: Any, source_stream: Any) -> None:
    assert _ORIGINAL_WAIT_STREAM is not None
    _increment("original_calls")
    _ORIGINAL_WAIT_STREAM(destination_stream, source_stream)


def _pooled_wait_stream(destination_stream: Any, source_stream: Any) -> None:
    global _EXHAUSTION_WARNED

    _increment("wait_stream_calls")
    assert _TORCH is not None

    # Preserve PyTorch's event ownership during graph capture. SGLang invokes
    # wait_stream with a participating stream current in its graph paths.
    try:
        is_capturing = bool(_TORCH.cuda.is_current_stream_capturing())
    except Exception:
        # Failure to prove that reuse is safe must retain the original behavior.
        is_capturing = True
    if is_capturing:
        _increment("capture_bypasses")
        return _call_original(destination_stream, source_stream)

    pool = _pool_for_device(_source_device_index(source_stream))
    if pool is None:
        return _call_original(destination_stream, source_stream)

    event = pool.acquire()
    if event is None:
        # Never block waiting for a host-side lease or grow the pool in the hot
        # path. Preserve PyTorch semantics and make exhaustion observable.
        _increment("pool_exhaustions")
        if not _EXHAUSTION_WARNED:
            with _INSTALL_LOCK:
                if not _EXHAUSTION_WARNED:
                    logger.warning(
                        "CUDA event pool exhausted; falling back to PyTorch's "
                        "Stream.wait_stream event allocation. Increase "
                        "SGLANG_CUDA_EVENT_POOL_SIZE above %d if this recurs.",
                        _POOL_SIZE,
                    )
                    _EXHAUSTION_WARNED = True
        return _call_original(destination_stream, source_stream)

    try:
        source_stream.record_event(event)
        destination_stream.wait_event(event)
    except BaseException:
        _increment("pooled_call_failures")
        pool.quarantine(event)
        raise
    else:
        pool.release(event)
        _increment("pooled_calls")


setattr(_pooled_wait_stream, "_sglang_cuda_event_pool", True)


def install_cuda_event_pool(*, torch_module: Any = None, pool_size: int = 1024) -> bool:
    """Patch ``torch.cuda.Stream.wait_stream`` for the current process.

    Returns ``True`` when this call installs the patch and ``False`` when the
    same patch is already installed or CUDA is unavailable.
    """

    global _TORCH, _STREAM_CLASS, _ORIGINAL_WAIT_STREAM, _POOL_SIZE

    if pool_size <= 0:
        raise ValueError("CUDA event pool size must be positive")
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    with _INSTALL_LOCK:
        if _ORIGINAL_WAIT_STREAM is not None:
            if torch_module is not _TORCH:
                raise RuntimeError(
                    "CUDA event pool is already installed for another torch module"
                )
            if pool_size != _POOL_SIZE:
                raise ValueError(
                    "CUDA event pool is already installed with "
                    f"pool_size={_POOL_SIZE}, got {pool_size}"
                )
            return False
        if not torch_module.cuda.is_available():
            return False

        stream_class = torch_module.cuda.Stream
        _TORCH = torch_module
        _STREAM_CLASS = stream_class
        _ORIGINAL_WAIT_STREAM = stream_class.wait_stream
        _POOL_SIZE = pool_size
        stream_class.wait_stream = _pooled_wait_stream
        return True


def prewarm_cuda_event_pool(device: int | None = None) -> dict[str, int] | None:
    """Materialize one device's pool after the worker selects its device."""

    if _TORCH is None:
        raise RuntimeError("CUDA event pool is not installed")
    if device is None:
        device = int(_TORCH.cuda.current_device())
    pool = _pool_for_device(device)
    return None if pool is None else pool.snapshot()


def cuda_event_pool_stats() -> dict[str, Any]:
    """Return process-local counters for diagnostics and tests."""

    with _POOLS_LOCK:
        devices = {
            str(device): pool.snapshot() for device, pool in sorted(_POOLS.items())
        }
        disabled_devices = dict(_DISABLED_DEVICES)
        counters: dict[str, Any] = dict(_STATS)
    return counters | {
        "installed": _ORIGINAL_WAIT_STREAM is not None,
        "pool_size": _POOL_SIZE,
        "devices": devices,
        "disabled_devices": disabled_devices,
    }


def uninstall_cuda_event_pool() -> bool:
    """Restore PyTorch's method when no later hook has wrapped this one."""

    global _TORCH, _STREAM_CLASS, _ORIGINAL_WAIT_STREAM
    with _INSTALL_LOCK:
        if _ORIGINAL_WAIT_STREAM is None or _STREAM_CLASS is None:
            return False
        if _STREAM_CLASS.wait_stream is not _pooled_wait_stream:
            return False
        _STREAM_CLASS.wait_stream = _ORIGINAL_WAIT_STREAM
        _TORCH = None
        _STREAM_CLASS = None
        _ORIGINAL_WAIT_STREAM = None
        return True


def _reset_cuda_event_pool_for_testing() -> None:
    global _EXHAUSTION_WARNED, _POOL_SIZE

    uninstall_cuda_event_pool()
    with _POOLS_LOCK:
        _POOLS.clear()
        _DISABLED_DEVICES.clear()
    for key in _STATS:
        _STATS[key] = 0
    _POOL_SIZE = 1024
    _EXHAUSTION_WARNED = False
