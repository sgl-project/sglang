"""Load-snapshot source protocol and the manager/Router adapters."""

from __future__ import annotations

from typing import Any, Collection, Optional, Protocol, runtime_checkable


@runtime_checkable
class LoadSnapshotSource(Protocol):
    """Protocol for a load-snapshot data source."""

    async def get_loads(self) -> list:
        """Return the source's latest scheduler load snapshots."""
        raise NotImplementedError

    def expected_dp_ranks(self) -> frozenset:
        """Return the authoritative DP ranks required for a full snapshot."""
        raise NotImplementedError


class ManagerLoadSnapshotSource:
    """Adapt a manager to the load-snapshot protocol."""

    def __init__(
        self,
        manager: Any,
        expected_dp_ranks: Collection[int],
        *,
        snapshot_reader: Optional[Any] = None,
    ) -> None:
        """Wrap a manager with an authoritative rank fallback."""
        self._manager = manager
        self._snapshot_reader = snapshot_reader
        self._expected: frozenset[int] = frozenset(expected_dp_ranks)

    async def get_loads(self) -> list:
        """Fetch core load snapshots from the wrapped manager."""
        if self._snapshot_reader is not None:
            return self._snapshot_reader.read_all()
        return await self._manager.get_loads(include=["core"])

    def expected_dp_ranks(self) -> frozenset[int]:
        """Return the manager's current authoritative DP rank set."""
        worker_count = getattr(self._manager, "elastic_worker_count", None)
        if (
            isinstance(worker_count, int)
            and not isinstance(worker_count, bool)
            and worker_count > 0
        ):
            return frozenset(range(worker_count))
        return self._expected


class RouterLoadSnapshotSource:
    """Adapt a load-snapshot reader to the load-snapshot protocol."""

    def __init__(self, reader: Any, expected_dp_ranks: Collection[int]) -> None:
        """Wrap a shared-memory reader with an authoritative rank set."""
        self._reader = reader
        self._expected: frozenset[int] = frozenset(expected_dp_ranks)

    async def get_loads(self) -> list:
        """Read all load snapshots currently published in shared memory."""
        return self._reader.read_all()

    def expected_dp_ranks(self) -> frozenset[int]:
        """Return the current authoritative DP rank set."""
        return self._expected

    def update_expected_dp_ranks(self, ranks: Collection[int]) -> bool:
        """Update the authoritative rank set.  Returns True if it changed."""
        updated = frozenset(ranks)
        if updated == self._expected:
            return False
        self._expected = updated
        return True
