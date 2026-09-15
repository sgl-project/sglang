"""Bounded CPU storage and single-flight coordination for MM preprocessing.

Model processors store prompt-independent ``MediaArtifact`` values here. This
module knows nothing about a model or media format: it provides byte-accounted
LRU storage and ensures concurrent misses for one key share one computation.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import (
    Any,
    Awaitable,
    Callable,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import torch
from PIL import Image

K = TypeVar("K")
V = TypeVar("V")
_USE_RESULT = object()


@dataclass(frozen=True)
class CacheLookup(Generic[V]):
    """A resolved value returned immediately or after shared computation."""

    value: V
    hit: bool
    joined: bool = False


@dataclass(frozen=True)
class CacheMiss(Generic[K, V]):
    """Handle for one in-flight cache miss.

    Exactly one handle has ``should_compute=True`` and must publish the result.
    Other handles for the same key wait on the shared ``future``.
    """

    key: K
    future: concurrent.futures.Future[V]
    generation: int
    should_compute: bool


@dataclass
class _Entry(Generic[V]):
    value: V
    size_bytes: int


@runtime_checkable
class CacheSizeProvider(Protocol):
    """Explicitly expose the owned values that count against a cache budget."""

    def cache_size_items(self) -> Sequence[Any]: ...


def estimate_cache_size_bytes(value: Any) -> Optional[int]:
    """Estimate owned CPU bytes, returning None for GPU-backed artifacts."""
    seen: set[int] = set()

    def visit(item: Any) -> Optional[int]:
        if item is None or isinstance(item, (bool, int, float)):
            return sys.getsizeof(item)
        item_id = id(item)
        if item_id in seen:
            return 0
        seen.add(item_id)

        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu":
                return None
            return item.untyped_storage().nbytes()
        if isinstance(item, np.ndarray):
            return int(item.nbytes)
        if isinstance(item, Image.Image):
            return len(item.tobytes())
        if isinstance(item, (bytes, bytearray, memoryview)):
            return len(item)
        if isinstance(item, str):
            return len(item.encode())
        if isinstance(item, CacheSizeProvider):
            return visit(item.cache_size_items())
        if isinstance(item, Mapping):
            total = 0
            for key, child in item.items():
                key_size = visit(key)
                child_size = visit(child)
                if key_size is None or child_size is None:
                    return None
                total += key_size + child_size
            return total
        if isinstance(item, (list, tuple, set)):
            total = 0
            for child in item:
                child_size = visit(child)
                if child_size is None:
                    return None
                total += child_size
            return total
        return sys.getsizeof(item)

    return visit(value)


class MultimodalPreprocessCache(Generic[K, V]):
    """Thread-safe CPU LRU with per-key async single-flight.

    ``max_size_bytes`` and ``max_entries`` bound retained values. In-flight
    computations are tracked separately and are not part of the LRU budget.
    ``clear()`` invalidates cache writes from old computations so a flush cannot
    be undone by work that started before it.
    """

    def __init__(self, max_size_bytes: int, max_entries: int = 8192):
        if max_size_bytes < 0:
            raise ValueError("max_size_bytes must be non-negative")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self._entries: OrderedDict[K, _Entry[V]] = OrderedDict()
        self._inflight: dict[K, tuple[concurrent.futures.Future[V], int]] = {}
        self._background_tasks: set[asyncio.Task] = set()
        self._lock = threading.Lock()
        self._generation = 0
        self.current_size_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.singleflight_joins = 0

    @property
    def enabled(self) -> bool:
        """Whether values can be retained; zero bytes is the cache kill switch."""
        return self.max_size_bytes > 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: K) -> bool:
        with self._lock:
            return key in self._entries

    def get(self, key: K) -> Optional[V]:
        """Read and touch an LRU entry, recording a hit or miss."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry.value

    def get_if_present(
        self,
        key: K,
        predicate: Callable[[V], bool],
        *,
        evict_on_reject: bool = False,
    ) -> Optional[V]:
        """Use a compatible entry without recording an absent speculative miss.

        The predicate runs while holding the cache lock, so a caller cannot use
        an entry that another thread replaces between validation and lookup.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if not predicate(entry.value):
                if evict_on_reject:
                    self._entries.pop(key)
                    self.current_size_bytes -= entry.size_bytes
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry.value

    def put(
        self,
        key: K,
        value: V,
        size_bytes: Optional[int] = None,
        *,
        _generation: Optional[int] = None,
    ) -> bool:
        """Insert a value if it is CPU-sizeable and fits the configured budget.

        With automatic sizing, returns ``False`` when caching is disabled, the
        value contains a GPU tensor, the value is too large, or its generation
        predates ``clear()``.
        """
        if not self.enabled:
            return False
        if size_bytes is None:
            size_bytes = estimate_cache_size_bytes(value)
        if size_bytes is None or size_bytes < 0 or size_bytes > self.max_size_bytes:
            return False

        with self._lock:
            # A pre-flush computation may finish, but it must not repopulate the
            # new cache generation.
            if _generation is not None and _generation != self._generation:
                return False
            old = self._entries.pop(key, None)
            if old is not None:
                self.current_size_bytes -= old.size_bytes
            while self._entries and (
                self.current_size_bytes + size_bytes > self.max_size_bytes
                or len(self._entries) >= self.max_entries
            ):
                _, evicted = self._entries.popitem(last=False)
                self.current_size_bytes -= evicted.size_bytes
                self.evictions += 1
            self._entries[key] = _Entry(value=value, size_bytes=size_bytes)
            self.current_size_bytes += size_bytes
            return True

    def pop(self, key: K) -> Optional[V]:
        """Remove and return one entry without changing hit/miss counters."""
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                return None
            self.current_size_bytes -= entry.size_bytes
            return entry.value

    def clear(self) -> None:
        """Drop values and prevent older in-flight work from repopulating them."""
        with self._lock:
            self._entries.clear()
            self.current_size_bytes = 0
            # Let active requests finish, but prevent work started before this
            # flush from repopulating the cache afterwards.
            self._generation += 1

    async def get_or_compute(
        self,
        key: K,
        compute: Callable[[], Awaitable[V]],
        *,
        size_bytes: Optional[Callable[[V], Optional[int]]] = None,
    ) -> CacheLookup[V]:
        """Return a cached value or share one async computation for ``key``.

        Cancellation affects only the caller that is awaiting the result. The
        shared computation remains alive for other callers.
        """
        if not self.enabled:
            return CacheLookup(await compute(), hit=False)

        cached = self.get(key)
        if cached is not None:
            return CacheLookup(cached, hit=True)

        with self._lock:
            inflight = self._inflight.get(key)
            if inflight is None or inflight[1] != self._generation:
                future = concurrent.futures.Future()
                generation = self._generation
                self._inflight[key] = (future, generation)
                should_compute = True
            else:
                future, generation = inflight
                self.singleflight_joins += 1
                should_compute = False

        if should_compute:
            self.create_background_task(
                self._compute_shared_value(
                    key, future, generation, compute, size_bytes=size_bytes
                )
            )

        # The cache owns the shared computation. Cancelling either its first
        # computing caller or a later waiter ends only that caller's local await.
        value = await asyncio.shield(asyncio.wrap_future(future))
        return CacheLookup(value, hit=False, joined=not should_compute)

    async def _compute_shared_value(
        self,
        key: K,
        future: concurrent.futures.Future[V],
        generation: int,
        compute: Callable[[], Awaitable[V]],
        *,
        size_bytes: Optional[Callable[[V], Optional[int]]],
    ) -> None:
        """Compute once, cache the result, and wake every caller for this key."""
        try:
            value = await compute()
            measured = size_bytes(value) if size_bytes is not None else None
            self.put(key, value, measured, _generation=generation)
            future.set_result(value)
        except BaseException as exc:
            future.set_exception(exc)
            # Retrieve the exception locally when no waiter joined, avoiding a
            # noisy "Future exception was never retrieved" warning.
            future.exception()
        finally:
            with self._lock:
                if self._inflight.get(key) == (future, generation):
                    self._inflight.pop(key, None)

    def _background_task_done(self, task: asyncio.Task) -> None:
        with self._lock:
            self._background_tasks.discard(task)
        try:
            task.exception()
        except asyncio.CancelledError:
            pass

    def create_background_task(self, awaitable: Awaitable[Any]) -> asyncio.Task:
        """Keep shared cache work alive independently of one caller task."""
        task = asyncio.create_task(awaitable)
        with self._lock:
            self._background_tasks.add(task)
        task.add_done_callback(self._background_task_done)
        return task

    def lookup_or_claim_many(
        self,
        keys: list[K],
        *,
        predicate: Optional[Callable[[K, V], bool]] = None,
    ) -> list[CacheLookup[V] | CacheMiss[K, V]]:
        """Return hits and single-flight miss handles in input-key order.

        For each missing key, one result has ``should_compute=True``. Repeated
        keys or concurrent callers receive handles with ``should_compute=False``
        and should call ``wait_for_miss`` instead of recomputing the value.
        """
        results: list[CacheLookup[V] | CacheMiss[K, V]] = []
        with self._lock:
            for key in keys:
                if not self.enabled:
                    future: concurrent.futures.Future[V] = concurrent.futures.Future()
                    results.append(
                        CacheMiss(key, future, self._generation, should_compute=True)
                    )
                    continue

                entry = self._entries.get(key)
                if entry is not None and (
                    predicate is None or predicate(key, entry.value)
                ):
                    self._entries.move_to_end(key)
                    self.hits += 1
                    results.append(CacheLookup(entry.value, hit=True))
                    continue
                if entry is not None:
                    self._entries.pop(key)
                    self.current_size_bytes -= entry.size_bytes

                self.misses += 1
                inflight = self._inflight.get(key)
                if inflight is None or inflight[1] != self._generation:
                    future: concurrent.futures.Future[V] = concurrent.futures.Future()
                    generation = self._generation
                    self._inflight[key] = (future, generation)
                    results.append(
                        CacheMiss(key, future, generation, should_compute=True)
                    )
                else:
                    future, generation = inflight
                    self.singleflight_joins += 1
                    results.append(
                        CacheMiss(key, future, generation, should_compute=False)
                    )
        return results

    def complete_miss(
        self,
        miss: CacheMiss[K, V],
        value: V,
        *,
        cache_value: V | object = _USE_RESULT,
        size_bytes: Optional[int] = None,
    ) -> None:
        """Publish a computed miss to waiters and optionally retain a copy.

        ``value`` is returned to current waiters. ``cache_value`` may be a
        smaller representation retained for future requests.
        """
        if not miss.should_compute:
            raise ValueError("Only the caller computing a cache miss can complete it")
        self.put(
            miss.key,
            value if cache_value is _USE_RESULT else cache_value,
            size_bytes,
            _generation=miss.generation,
        )
        miss.future.set_result(value)
        with self._lock:
            if self._inflight.get(miss.key) == (
                miss.future,
                miss.generation,
            ):
                self._inflight.pop(miss.key, None)

    def fail_miss(self, miss: CacheMiss[K, V], error: BaseException) -> None:
        """Publish a computation failure to every waiter for this miss."""
        if not miss.should_compute:
            raise ValueError("Only the caller computing a cache miss can fail it")
        miss.future.set_exception(error)
        miss.future.exception()
        with self._lock:
            if self._inflight.get(miss.key) == (
                miss.future,
                miss.generation,
            ):
                self._inflight.pop(miss.key, None)

    async def wait_for_miss(self, miss: CacheMiss[K, V]) -> V:
        """Wait for another caller's computation without cancelling it."""
        return await asyncio.shield(asyncio.wrap_future(miss.future))

    def stats(self) -> dict[str, int]:
        """Return a lock-consistent snapshot of cache and single-flight state."""
        with self._lock:
            return {
                "entries": len(self._entries),
                "size_bytes": self.current_size_bytes,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "singleflight_joins": self.singleflight_joins,
                "inflight": len(self._inflight),
            }
