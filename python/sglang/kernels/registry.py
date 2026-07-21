"""In-memory registry of :class:`KernelSpec` entries.

The registry is the single inventory of "which operators have which backend
implementations". It is populated at import time by the ``sglang.kernels.ops.*``
group packages, using only metadata (import path strings) — registering a spec
never imports ``torch`` or a kernel backend.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from sglang.kernels.spec import KernelBackend, KernelSpec


class KernelRegistry:
    """Maps ``"<group>.<name>"`` operator ids to their :class:`KernelSpec` list."""

    def __init__(self) -> None:
        self._by_op: Dict[str, List[KernelSpec]] = defaultdict(list)

    def register(self, spec: KernelSpec) -> KernelSpec:
        """Register ``spec``.

        Re-registering the same ``(op, backend)`` pair replaces the previous
        entry so that module reloads during tests stay idempotent.
        """
        existing = self._by_op[spec.op]
        for i, other in enumerate(existing):
            if other.backend == spec.backend:
                existing[i] = spec
                return spec
        existing.append(spec)
        return spec

    def get(self, op: str) -> List[KernelSpec]:
        """All registered specs for ``op`` (empty list if none)."""
        return list(self._by_op.get(op, ()))

    def get_backend(self, op: str, backend: KernelBackend) -> KernelSpec:
        """The spec for ``op`` provided by ``backend``.

        Raises ``KeyError`` if no such implementation is registered.
        """
        for spec in self._by_op.get(op, ()):
            if spec.backend == backend:
                return spec
        raise KeyError(f"No '{backend.value}' backend registered for op {op!r}")

    def has(self, op: str) -> bool:
        return bool(self._by_op.get(op))

    def ops(self) -> List[str]:
        """Sorted list of all registered operator ids."""
        return sorted(self._by_op.keys())

    def all_specs(self) -> List[KernelSpec]:
        specs: List[KernelSpec] = []
        for op in self.ops():
            specs.extend(self._by_op[op])
        return specs


# Process-wide registry. Group packages register into this instance on import.
registry = KernelRegistry()


def register_kernel(spec: KernelSpec) -> KernelSpec:
    """Register ``spec`` in the process-wide :data:`registry`."""
    return registry.register(spec)
