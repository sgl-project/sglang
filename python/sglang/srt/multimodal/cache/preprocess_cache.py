"""Bounded CPU cache and single-flight coordination for MM preprocessing."""

from __future__ import annotations

import asyncio
import concurrent.futures
import dataclasses
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Generic, Optional, TypeVar

import numpy as np
import torch
from PIL import Image

K = TypeVar("K")
V = TypeVar("V")
_USE_RESULT = object()


@dataclass(frozen=True)
class CacheLookup(Generic[V]):
    value: V
    hit: bool
    joined: bool = False


@dataclass(frozen=True)
class CacheReservation(Generic[K, V]):
    key: K
    future: concurrent.futures.Future[V]
    generation: int
    owner: bool


@dataclass
class _Entry(Generic[V]):
    value: V
    size_bytes: int


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
        if dataclasses.is_dataclass(item):
            total = 0
            for field in dataclasses.fields(item):
                child_size = visit(getattr(item, field.name))
                if child_size is None:
                    return None
                total += child_size
            return total
        if isinstance(item, dict):
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
    """Thread-safe byte-accounted LRU with per-key async single-flight."""

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
        return self.max_size_bytes > 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: K) -> bool:
        with self._lock:
            return key in self._entries

    def get(self, key: K) -> Optional[V]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry.value

    def peek(self, key: K) -> Optional[V]:
        """Return a value without changing LRU order or counters."""
        with self._lock:
            entry = self._entries.get(key)
            return None if entry is None else entry.value

    def put(
        self,
        key: K,
        value: V,
        size_bytes: Optional[int] = None,
        *,
        _generation: Optional[int] = None,
    ) -> bool:
        if not self.enabled:
            return False
        if size_bytes is None:
            size_bytes = estimate_cache_size_bytes(value)
        if size_bytes is None or size_bytes < 0 or size_bytes > self.max_size_bytes:
            return False

        with self._lock:
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
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                return None
            self.current_size_bytes -= entry.size_bytes
            return entry.value

    def clear(self) -> None:
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
                owner = True
            else:
                future, generation = inflight
                self.singleflight_joins += 1
                owner = False

        if owner:
            self.create_background_task(
                self._compute_owned_value(
                    key, future, generation, compute, size_bytes=size_bytes
                )
            )

        # The cache owns the shared computation. Cancelling either its first
        # caller or a later joiner ends only that caller's local await.
        value = await asyncio.shield(asyncio.wrap_future(future))
        return CacheLookup(value, hit=False, joined=not owner)

    async def _compute_owned_value(
        self,
        key: K,
        future: concurrent.futures.Future[V],
        generation: int,
        compute: Callable[[], Awaitable[V]],
        *,
        size_bytes: Optional[Callable[[V], Optional[int]]],
    ) -> None:
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

    def reserve_many(
        self, keys: list[K]
    ) -> list[CacheLookup[V] | CacheReservation[K, V]]:
        """Reserve cache misses so owners can compute them in one batch."""
        results: list[CacheLookup[V] | CacheReservation[K, V]] = []
        with self._lock:
            for key in keys:
                if not self.enabled:
                    future: concurrent.futures.Future[V] = concurrent.futures.Future()
                    results.append(
                        CacheReservation(key, future, self._generation, owner=True)
                    )
                    continue

                entry = self._entries.get(key)
                if entry is not None:
                    self._entries.move_to_end(key)
                    self.hits += 1
                    results.append(CacheLookup(entry.value, hit=True))
                    continue

                self.misses += 1
                inflight = self._inflight.get(key)
                if inflight is None or inflight[1] != self._generation:
                    future: concurrent.futures.Future[V] = concurrent.futures.Future()
                    generation = self._generation
                    self._inflight[key] = (future, generation)
                    results.append(
                        CacheReservation(key, future, generation, owner=True)
                    )
                else:
                    future, generation = inflight
                    self.singleflight_joins += 1
                    results.append(
                        CacheReservation(key, future, generation, owner=False)
                    )
        return results

    def fulfill(
        self,
        reservation: CacheReservation[K, V],
        value: V,
        *,
        cache_value: V | object = _USE_RESULT,
        size_bytes: Optional[int] = None,
    ) -> None:
        if not reservation.owner:
            raise ValueError("Only an owning reservation can be fulfilled")
        self.put(
            reservation.key,
            value if cache_value is _USE_RESULT else cache_value,
            size_bytes,
            _generation=reservation.generation,
        )
        reservation.future.set_result(value)
        with self._lock:
            if self._inflight.get(reservation.key) == (
                reservation.future,
                reservation.generation,
            ):
                self._inflight.pop(reservation.key, None)

    def fail(self, reservation: CacheReservation[K, V], error: BaseException) -> None:
        if not reservation.owner:
            raise ValueError("Only an owning reservation can fail")
        reservation.future.set_exception(error)
        reservation.future.exception()
        with self._lock:
            if self._inflight.get(reservation.key) == (
                reservation.future,
                reservation.generation,
            ):
                self._inflight.pop(reservation.key, None)

    async def wait(self, reservation: CacheReservation[K, V]) -> V:
        return await asyncio.shield(asyncio.wrap_future(reservation.future))

    def record_singleflight_joins(self, count: int) -> None:
        """Include external work coalesced around this cache in its metrics."""
        if count < 0:
            raise ValueError("single-flight join count must be non-negative")
        with self._lock:
            self.singleflight_joins += count

    def stats(self) -> dict[str, int]:
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
