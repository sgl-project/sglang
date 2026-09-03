"""Authoritative control state for an experimental shared same-GPU KV pool.

The CUDA allocation is intentionally outside this module. A CUDA-IPC pool owner
keeps the slab alive while this control plane leases the slab's page IDs to
schedulers. A page may be written by exactly one lease holder and becomes
cross-replica readable only after immutable prefix publication.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class SharedKVPageState(str, Enum):
    FREE = "free"
    RESERVED = "reserved"
    PUBLISHED = "published"


@dataclass(frozen=True)
class SharedKVWriteLease:
    lease_id: int
    owner: str
    page_ids: tuple[int, ...]
    generations: tuple[int, ...]


@dataclass(frozen=True)
class SharedKVReadLease:
    lease_id: int
    key: tuple[int, ...]
    page_ids: tuple[int, ...]
    generations: tuple[int, ...]


@dataclass
class _Page:
    generation: int = 0
    state: SharedKVPageState = SharedKVPageState.FREE
    write_lease_id: Optional[int] = None
    published_refs: int = 0
    read_refs: int = 0


@dataclass
class _PublishedPrefix:
    page_ids: tuple[int, ...]
    generations: tuple[int, ...]
    read_refs: int = 0


class SharedKVPoolControlPlane:
    """Thread-safe global page allocator and immutable-prefix directory.

    This is deliberately a control-plane-only primitive. Scheduler processes
    must access it through one service process; they must not deserialize a
    private copy of this object. The service will pair it with one CUDA-IPC
    exported slab in the Phase 2 runtime integration.
    """

    def __init__(self, num_pages: int):
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")
        self._pages = [_Page() for _ in range(num_pages)]
        self._prefixes: dict[tuple[int, ...], _PublishedPrefix] = {}
        self._write_leases: dict[int, SharedKVWriteLease] = {}
        self._read_leases: dict[int, SharedKVReadLease] = {}
        self._next_lease_id = 1
        self._lock = threading.RLock()

    @property
    def num_pages(self) -> int:
        return len(self._pages)

    def available_pages(self) -> int:
        with self._lock:
            return sum(page.state is SharedKVPageState.FREE for page in self._pages)

    def reserve(self, owner: str, num_pages: int) -> Optional[SharedKVWriteLease]:
        """Lease globally free pages for one scheduler's exclusive write."""
        if not owner:
            raise ValueError("owner must be non-empty")
        if num_pages <= 0:
            raise ValueError("num_pages must be positive")
        with self._lock:
            page_ids = tuple(
                index
                for index, page in enumerate(self._pages)
                if page.state is SharedKVPageState.FREE
            )[:num_pages]
            if len(page_ids) != num_pages:
                return None

            lease_id = self._new_lease_id()
            generations = []
            for page_id in page_ids:
                page = self._pages[page_id]
                page.generation += 1
                page.state = SharedKVPageState.RESERVED
                page.write_lease_id = lease_id
                generations.append(page.generation)
            lease = SharedKVWriteLease(
                lease_id=lease_id,
                owner=owner,
                page_ids=page_ids,
                generations=tuple(generations),
            )
            self._write_leases[lease_id] = lease
            return lease

    def publish_prefix(self, key: tuple[int, ...], write_lease_id: int) -> None:
        """Publish a completed, immutable prefix owned by a write lease.

        Publishing transfers the pages from exclusive write ownership to the
        directory. The caller must ensure its CUDA writes are visible before
        this method is invoked; the first runtime implementation will use a
        conservative producer synchronization barrier for that transition.
        """
        if not key:
            raise ValueError("cannot publish an empty prefix")
        with self._lock:
            if key in self._prefixes:
                raise ValueError("prefix is already published")
            lease = self._write_leases.pop(write_lease_id, None)
            if lease is None:
                raise ValueError("unknown or already released write lease")
            self._validate_write_lease(lease)
            for page_id in lease.page_ids:
                page = self._pages[page_id]
                page.state = SharedKVPageState.PUBLISHED
                page.write_lease_id = None
                page.published_refs += 1
            self._prefixes[key] = _PublishedPrefix(
                page_ids=lease.page_ids,
                generations=lease.generations,
            )

    def acquire_prefix(self, key: tuple[int, ...]) -> Optional[SharedKVReadLease]:
        """Return the longest published prefix of ``key`` with a read lease."""
        with self._lock:
            prefix_key = next(
                (key[:length] for length in range(len(key), 0, -1) if key[:length] in self._prefixes),
                None,
            )
            if prefix_key is None:
                return None
            prefix = self._prefixes[prefix_key]
            for page_id, generation in zip(prefix.page_ids, prefix.generations):
                page = self._pages[page_id]
                if (
                    page.state is not SharedKVPageState.PUBLISHED
                    or page.generation != generation
                ):
                    raise RuntimeError("published prefix points to a stale page")
                page.read_refs += 1
            prefix.read_refs += 1
            lease = SharedKVReadLease(
                lease_id=self._new_lease_id(),
                key=prefix_key,
                page_ids=prefix.page_ids,
                generations=prefix.generations,
            )
            self._read_leases[lease.lease_id] = lease
            return lease

    def release_read(self, lease_id: int) -> None:
        with self._lock:
            lease = self._read_leases.pop(lease_id, None)
            if lease is None:
                raise ValueError("unknown or already released read lease")
            prefix = self._prefixes.get(lease.key)
            if prefix is None:
                raise RuntimeError("read lease outlived its published prefix")
            for page_id, generation in zip(lease.page_ids, lease.generations):
                page = self._pages[page_id]
                if page.generation != generation or page.read_refs <= 0:
                    raise RuntimeError("invalid read lease release")
                page.read_refs -= 1
            prefix.read_refs -= 1

    def release_write(self, lease_id: int) -> None:
        """Release an unpublished write lease and make its pages reusable."""
        with self._lock:
            lease = self._write_leases.pop(lease_id, None)
            if lease is None:
                raise ValueError("unknown or already released write lease")
            self._validate_write_lease(lease)
            for page_id in lease.page_ids:
                page = self._pages[page_id]
                page.state = SharedKVPageState.FREE
                page.write_lease_id = None

    def evict_prefix(self, key: tuple[int, ...]) -> bool:
        """Evict one unreferenced immutable prefix and free its pages."""
        with self._lock:
            prefix = self._prefixes.get(key)
            if prefix is None or prefix.read_refs:
                return False
            if any(self._pages[page_id].read_refs for page_id in prefix.page_ids):
                return False
            del self._prefixes[key]
            for page_id, generation in zip(prefix.page_ids, prefix.generations):
                page = self._pages[page_id]
                if page.generation != generation:
                    raise RuntimeError("cannot evict a stale prefix page")
                page.published_refs -= 1
                if page.published_refs == 0:
                    page.state = SharedKVPageState.FREE
            return True

    def _validate_write_lease(self, lease: SharedKVWriteLease) -> None:
        for page_id, generation in zip(lease.page_ids, lease.generations):
            page = self._pages[page_id]
            if (
                page.state is not SharedKVPageState.RESERVED
                or page.write_lease_id != lease.lease_id
                or page.generation != generation
            ):
                raise RuntimeError("write lease no longer owns its pages")

    def _new_lease_id(self) -> int:
        lease_id = self._next_lease_id
        self._next_lease_id += 1
        return lease_id
