"""Stub implementations for APIs missing from ``torch.mps``.

``torch.mps`` lacks several APIs that ``torch.cuda`` provides (``Stream``,
``set_device``, ``get_device_properties``, …).  Rather than scattering
``hasattr`` / ``getattr`` guards throughout the codebase, we monkey-patch
``torch.mps`` once at startup so that generic device-agnostic code paths
just work.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Last-resort fallback when neither torch.mps.recommended_max_memory() nor
# MLX is available.  Uses the conservative end of Apple's 65-75% range.
_METAL_MEMORY_FRACTION_FALLBACK = 0.65


_cached_metal_max_memory: int | None = None


def _get_metal_max_memory() -> int:
    """Return the maximum memory Metal can safely use, in bytes.

    Priority chain:
    1. ``torch.mps.recommended_max_memory()`` — available in PyTorch 2.9+,
       maps directly to Metal's ``MTLDevice.recommendedMaxWorkingSetSize``.
    2. ``mx.device_info()["max_recommended_working_set_size"]`` — MLX's
       equivalent API.  When this path is used, also sets
       ``mx.set_wired_limit()`` to prevent Metal from paging GPU memory.
    3. 65% of system RAM — last-resort heuristic fallback.

    Allocating beyond Metal's recommended limit causes GPU paging,
    instability, and potential machine reboots on Apple Silicon.

    Reference: https://github.com/sgl-project/sglang/issues/21443
    """
    global _cached_metal_max_memory
    if _cached_metal_max_memory is not None:
        return _cached_metal_max_memory

    # 1. Prefer torch.mps (available in PyTorch 2.9+, no MLX needed).
    try:
        import torch

        if hasattr(torch, "mps") and hasattr(torch.mps, "recommended_max_memory"):
            max_mem = int(torch.mps.recommended_max_memory())
            if max_mem > 0:
                logger.info(
                    "Metal max working set: %.1f GB (system RAM: %.1f GB). "
                    "Source: torch.mps.recommended_max_memory().",
                    max_mem / (1 << 30),
                    _get_system_total_memory() / (1 << 30),
                )
                _cached_metal_max_memory = max_mem
                # Also set wired limit via MLX if available.
                _try_set_wired_limit(max_mem)
                return max_mem
    except (ImportError, AttributeError, Exception) as e:
        logger.debug("torch.mps.recommended_max_memory() unavailable: %s", e)

    # 2. Fall back to MLX.
    try:
        import mlx.core as mx

        if hasattr(mx, "device_info"):
            device_info = mx.device_info()
        else:
            device_info = mx.metal.device_info()
        max_mem = int(device_info.get("max_recommended_working_set_size", 0))
        if max_mem > 0:
            _try_set_wired_limit(max_mem)
            logger.info(
                "Metal max working set: %.1f GB (system RAM: %.1f GB). "
                "Source: MLX device_info().",
                max_mem / (1 << 30),
                _get_system_total_memory() / (1 << 30),
            )
            _cached_metal_max_memory = max_mem
            return max_mem
    except (ImportError, AttributeError, Exception) as e:
        logger.debug("MLX device_info() unavailable: %s", e)

    # 3. Last-resort heuristic.
    import psutil

    total = psutil.virtual_memory().total
    _cached_metal_max_memory = int(total * _METAL_MEMORY_FRACTION_FALLBACK)
    logger.warning(
        "Could not query Metal memory limit (torch.mps and MLX both unavailable). "
        "Using %.0f%% of system RAM (%.1f GB) as a conservative estimate.",
        _METAL_MEMORY_FRACTION_FALLBACK * 100,
        _cached_metal_max_memory / (1 << 30),
    )
    return _cached_metal_max_memory


def _try_set_wired_limit(limit_bytes: int) -> None:
    """Set MLX wired memory limit if MLX is available. No-op otherwise."""
    try:
        import mlx.core as mx

        if hasattr(mx, "set_wired_limit"):
            mx.set_wired_limit(limit_bytes)
            logger.debug("Wired memory limit set to %.1f GB.", limit_bytes / (1 << 30))
    except (ImportError, AttributeError, Exception):
        pass


def _get_system_total_memory() -> int:
    """Return total system RAM in bytes."""
    import psutil

    return psutil.virtual_memory().total


class Stream:
    """Minimal stand-in for ``torch.cuda.Stream``.

    MPS does not expose user-visible streams.  Every method is a no-op so
    that code written for CUDA's multi-stream model still runs.
    """

    def __init__(self, device: Any = None, priority: int = 0) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def wait_stream(self, stream: Any) -> None:
        pass

    def wait_event(self, event: Any) -> None:
        pass

    def record_event(self, event: Any = None) -> Any:
        return None

    def query(self) -> bool:
        return True

    # context-manager protocol (``with stream:``)
    def __enter__(self) -> "Stream":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class Event:
    """Minimal stand-in for ``torch.cuda.Event``."""

    def __init__(self, enable_timing: bool = False) -> None:
        pass

    def record(self, stream: Any = None) -> None:
        pass

    def wait(self, stream: Any = None) -> None:
        pass

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, end_event: Any) -> float:
        return 0.0


class StreamContext:
    """Minimal stand-in for ``torch.cuda.StreamContext``."""

    def __init__(self, stream: Any = None) -> None:
        pass

    def __enter__(self) -> "StreamContext":
        return self

    def __exit__(self, *args: Any) -> None:
        pass


_default_stream = Stream()


def current_stream(device: Any = None) -> Stream:
    """Return the default (and only) MPS stream."""
    return _default_stream


def stream(s: Any) -> Stream:
    """Return a context manager that is a no-op on MPS."""
    return s if s is not None else _default_stream


def set_device(device: Any) -> None:  # noqa: ARG001
    """Set the current device. This is a no-op for MPS as it has exactly one device."""
    pass


def current_device() -> int:
    """Return the index of the current MPS device (always 0)."""
    return 0


def device_count() -> int:
    """Return the number of available MPS devices (always 1)."""
    return 1


@dataclass
class _MPSDeviceProperties:
    """Mimics the object returned by ``torch.cuda.get_device_properties``."""

    name: str = "Apple MPS"
    total_memory: int = 0  # populated at install time
    multi_processor_count: int = 0
    warp_size: int = 32
    is_integrated: bool = True
    major: int = 0
    minor: int = 0
    # Extra attrs some callers inspect
    _extra: dict = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        # Return a safe default for any attribute we didn't anticipate
        try:
            return self._extra[name]
        except KeyError:
            return None


_cached_props: _MPSDeviceProperties | None = None


def get_device_properties(device: Any = 0) -> _MPSDeviceProperties:  # noqa: ARG001
    """Return the properties of the MPS device. Results are cached after first call."""
    global _cached_props
    if _cached_props is None:
        _cached_props = _MPSDeviceProperties(
            total_memory=_get_metal_max_memory(),
        )
    return _cached_props


class _MPSMemoryTracker:
    """Tracks peak memory values on top of ``torch.mps`` current-value APIs.

    * ``memory_allocated`` → ``torch.mps.current_allocated_memory()``
    * ``memory_reserved``  → ``torch.mps.driver_allocated_memory()``
    * ``max_memory_*``     → high-water marks of the above
    """

    def __init__(self) -> None:
        self._peak_allocated: int = 0
        self._peak_reserved: int = 0

    def memory_allocated(self, device: Any = None) -> int:  # noqa: ARG002
        import torch

        val = torch.mps.current_allocated_memory()
        if val > self._peak_allocated:
            self._peak_allocated = val
        return val

    def memory_reserved(self, device: Any = None) -> int:  # noqa: ARG002
        import torch

        val = torch.mps.driver_allocated_memory()
        if val > self._peak_reserved:
            self._peak_reserved = val
        return val

    def max_memory_allocated(self, device: Any = None) -> int:  # noqa: ARG002
        self.memory_allocated()
        return self._peak_allocated

    def max_memory_reserved(self, device: Any = None) -> int:  # noqa: ARG002
        self.memory_reserved()
        return self._peak_reserved

    def reset_peak_memory_stats(self, device: Any = None) -> None:  # noqa: ARG002
        import torch

        self._peak_allocated = torch.mps.current_allocated_memory()
        self._peak_reserved = torch.mps.driver_allocated_memory()


_memory_tracker = _MPSMemoryTracker()


def _patch_non_blocking() -> None:
    """Force ``non_blocking=False`` for copies targeting the MPS device.

    Unlike CUDA, MPS does not guarantee that a subsequent kernel on the same
    "stream" will wait for an async host-to-device transfer to finish.  Reading
    the tensor before the transfer completes yields uninitialised (garbage)
    data.  Patching ``Tensor.to`` and ``Tensor.copy_`` centrally avoids having
    to sprinkle ``non_blocking=not is_mps()`` at every call-site.
    """
    import torch

    _original_to = torch.Tensor.to

    @functools.wraps(_original_to)
    def _patched_to(self, *args, **kwargs):
        if kwargs.get("non_blocking"):
            # Detect target device from positional or keyword args
            device = None
            if args and isinstance(args[0], (str, torch.device)):
                device = torch.device(args[0]) if isinstance(args[0], str) else args[0]
            elif "device" in kwargs:
                d = kwargs["device"]
                device = torch.device(d) if isinstance(d, str) else d
            if device is not None and device.type == "mps":
                kwargs = {**kwargs, "non_blocking": False}
        return _original_to(self, *args, **kwargs)

    torch.Tensor.to = _patched_to

    _original_copy_ = torch.Tensor.copy_

    @functools.wraps(_original_copy_)
    def _patched_copy_(self, src, non_blocking=False):
        if non_blocking and self.device.type == "mps":
            non_blocking = False
        return _original_copy_(self, src, non_blocking=non_blocking)

    torch.Tensor.copy_ = _patched_copy_


_installed = False


def install() -> None:
    """Patch ``torch.mps`` with the stubs above.  Safe to call multiple times."""
    global _installed
    if _installed:
        return

    import torch

    mps = torch.mps
    # Only patch attributes that are actually missing
    for name, obj in [
        ("Stream", Stream),
        ("StreamContext", StreamContext),
        ("Event", Event),
        ("current_stream", current_stream),
        ("stream", stream),
        ("set_device", set_device),
        ("current_device", current_device),
        ("device_count", device_count),
        ("get_device_properties", get_device_properties),
        ("reset_peak_memory_stats", _memory_tracker.reset_peak_memory_stats),
        ("memory_allocated", _memory_tracker.memory_allocated),
        ("memory_reserved", _memory_tracker.memory_reserved),
        ("max_memory_allocated", _memory_tracker.max_memory_allocated),
        ("max_memory_reserved", _memory_tracker.max_memory_reserved),
    ]:
        if not hasattr(mps, name):
            setattr(mps, name, obj)

    _patch_non_blocking()

    _installed = True
