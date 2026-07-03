"""Heuristic kernel selection over the :data:`registry`.

RFC #29630 asks the first version to "start with a heuristic policy only" and
leave autotuning for later. :func:`select_kernel` ranks the registered backends
for an operator by:

1. explicit backend request (if given, only that backend is considered);
2. runnability on the target platform (capability check);
3. spec ``priority`` (higher first);
4. a stable backend preference order as the final tie-breaker.

:func:`get_kernel` is the fast path used by the public ``ops.*`` wrappers: it
resolves the selected spec to its callable and caches the result so repeated
calls do not re-run selection or re-import.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

from sglang.kernels.registry import registry
from sglang.kernels.spec import KernelBackend, KernelSpec, PlatformInfo

# Final tie-breaker when priority is equal. AOT (the stable wheel) is preferred
# over JIT, then Triton, then the torch fallback last.
_BACKEND_PREFERENCE = {
    KernelBackend.CUDA_AOT: 6,
    KernelBackend.FLASHINFER: 5,
    KernelBackend.CUDA_JIT: 4,
    KernelBackend.CUTE_DSL: 3,
    KernelBackend.DEEPGEMM: 2,
    KernelBackend.TRITON: 1,
    KernelBackend.TORCH: 0,
}


def select_kernel(
    op: str,
    backend: Optional[KernelBackend] = None,
    platform: Optional[PlatformInfo] = None,
) -> KernelSpec:
    """Return the best :class:`KernelSpec` for ``op``.

    Parameters
    ----------
    op:
        Operator id, ``"<group>.<name>"``.
    backend:
        If given, restrict the choice to this backend (no heuristic ranking).
    platform:
        Platform to check capabilities against. Defaults to
        :meth:`PlatformInfo.detect`.

    Raises
    ------
    KeyError
        If ``op`` is unknown, or if ``backend`` is requested but not registered.
    RuntimeError
        If no registered backend can run on ``platform``.
    """
    specs = registry.get(op)
    if not specs:
        raise KeyError(f"No kernels registered for op {op!r}")

    if backend is not None:
        for spec in specs:
            if spec.backend == backend:
                return spec
        raise KeyError(f"No '{backend.value}' backend registered for op {op!r}")

    plat = platform if platform is not None else PlatformInfo.detect()
    runnable = [s for s in specs if s.is_available(plat)]
    if not runnable:
        raise RuntimeError(
            f"No registered backend for op {op!r} can run on platform {plat}. "
            f"Registered backends: {[s.backend.value for s in specs]}"
        )

    runnable.sort(
        key=lambda s: (s.priority, _BACKEND_PREFERENCE.get(s.backend, -1)),
        reverse=True,
    )
    return runnable[0]


@lru_cache(maxsize=None)
def _resolve(op: str, backend: Optional[KernelBackend]) -> Callable:
    return select_kernel(op, backend=backend).load()


def get_kernel(op: str, backend: Optional[KernelBackend] = None) -> Callable:
    """Resolve ``op`` to a callable kernel and cache it.

    This is what the public ``sglang.kernels.ops.*`` wrappers call. The first
    call selects and imports the backend; later calls hit the cache. Note the
    cache keys only on ``(op, backend)`` and therefore does not react to the
    live platform — callers that need platform-sensitive re-selection should
    use :func:`select_kernel` directly.
    """
    return _resolve(op, backend)


def clear_cache() -> None:
    """Drop the resolved-callable cache (used by tests)."""
    _resolve.cache_clear()
