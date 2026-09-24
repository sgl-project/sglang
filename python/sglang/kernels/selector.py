"""Device-aware fixed-path kernel resolution over the :data:`registry`.

There is no priority *ranking* or preference heuristic. Resolution of an op's
call path is deterministic:

- an op with a single registered backend resolves to it directly;
- an op with several registered backends is filtered by the detected platform
  (a hard :class:`~sglang.kernels.spec.CapabilityRequirement` check, not a
  preference). If exactly one backend is usable on this device, it is the fixed
  call path; if several remain usable, the caller must name the backend
  explicitly (``backend=...``).

Because ``KernelBackend`` is now device-agnostic provenance, the *same* backend
(e.g. ``AOT``) may be registered for an op on more than one device; the
availability filter is what makes ``select_kernel`` pick the right one per
platform. Filtering by device is a hard eligibility gate, not the ranked
auto-selection that ``BaseFusedOp`` performs.

:func:`get_kernel` is the fast path used by the public ``ops.*`` wrappers: it
resolves the spec to its callable and caches the result so repeated calls do
not re-run resolution or re-import (the platform is constant per process).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

from sglang.kernels.registry import registry
from sglang.kernels.spec import KernelBackend, KernelSpec, PlatformInfo


@lru_cache(maxsize=1)
def _platform() -> PlatformInfo:
    return PlatformInfo.detect()


def select_kernel(op: str, backend: Optional[KernelBackend] = None) -> KernelSpec:
    """Return the :class:`KernelSpec` for ``op`` (its fixed call path).

    Parameters
    ----------
    op:
        Operator id, ``"<group>.<name>"``.
    backend:
        Required only when ``op`` has more than one backend *usable on the
        current device*; selects which one. Otherwise optional.

    Raises
    ------
    KeyError
        If ``op`` is unknown, or if ``backend`` is requested but not registered.
    ValueError
        If ``op`` has multiple device-eligible backends and ``backend`` is not
        given, or if none are eligible on this platform.
    """
    specs = registry.get(op)
    if not specs:
        raise KeyError(f"No kernels registered for op {op!r}")

    if backend is not None:
        for spec in specs:
            if spec.backend == backend:
                return spec
        raise KeyError(f"No '{backend.value}' backend registered for op {op!r}")

    if len(specs) == 1:
        return specs[0]

    # Multiple backends: hard-filter by device eligibility.
    platform = _platform()
    eligible = [s for s in specs if s.is_available(platform)]
    if len(eligible) == 1:
        return eligible[0]
    if not eligible:
        raise ValueError(
            f"op {op!r} has no backend usable on device {platform.device.value!r} "
            f"(registered: {[s.backend.value for s in specs]})"
        )
    raise ValueError(
        f"op {op!r} has multiple backends usable on device "
        f"{platform.device.value!r} ({[s.backend.value for s in eligible]}); "
        f"pass backend=... to choose one"
    )


@lru_cache(maxsize=None)
def _resolve(op: str, backend: Optional[KernelBackend]) -> Callable:
    return select_kernel(op, backend=backend).load()


def get_kernel(op: str, backend: Optional[KernelBackend] = None) -> Callable:
    """Resolve ``op`` to a callable kernel and cache it.

    This is what the public ``sglang.kernels.ops.*`` wrappers call. The first
    call resolves and imports the backend; later calls hit the cache.
    """
    return _resolve(op, backend)


def clear_cache() -> None:
    """Drop the resolved-callable cache (used by tests)."""
    _resolve.cache_clear()
