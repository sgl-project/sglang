"""Fixed-path kernel resolution over the :data:`registry`.

There is no priority ranking or heuristic backend selection. Each operator has
a fixed call path — its :attr:`KernelSpec.target`:

- an op with a single registered backend resolves to it directly;
- an op with several registered backends must be resolved by naming the backend
  explicitly (``backend=...``). The extra backends exist only as inventory, and
  are never silently auto-picked.

:func:`get_kernel` is the fast path used by the public ``ops.*`` wrappers: it
resolves the spec to its callable and caches the result so repeated calls do
not re-run resolution or re-import.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Optional

from sglang.kernels.registry import registry
from sglang.kernels.spec import KernelBackend, KernelSpec


def select_kernel(op: str, backend: Optional[KernelBackend] = None) -> KernelSpec:
    """Return the :class:`KernelSpec` for ``op`` (its fixed call path).

    Parameters
    ----------
    op:
        Operator id, ``"<group>.<name>"``.
    backend:
        Required only when ``op`` has more than one registered backend; selects
        which one. For single-backend ops it is optional.

    Raises
    ------
    KeyError
        If ``op`` is unknown, or if ``backend`` is requested but not registered.
    ValueError
        If ``op`` has multiple backends and ``backend`` is not given.
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

    raise ValueError(
        f"op {op!r} has multiple registered backends "
        f"({[s.backend.value for s in specs]}); pass backend=... to choose one"
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
