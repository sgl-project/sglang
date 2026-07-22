from __future__ import annotations

import hashlib
import math
import threading
from collections import deque
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Iterator, NamedTuple, Sequence

import torch

_l3_staging_allocation: ContextVar[bool] = ContextVar(
    "hicache_l3_staging_allocation", default=False
)


@dataclass(slots=True)
class HostEvictionScanBudget:
    remaining_nodes: int


_host_eviction_scan_budget: ContextVar[HostEvictionScanBudget | None] = ContextVar(
    "hicache_host_eviction_scan_budget", default=None
)


@contextmanager
def l3_staging_allocation() -> Iterator[None]:
    """Route host allocations through the isolated L3 staging reserve."""

    token = _l3_staging_allocation.set(True)
    try:
        yield
    finally:
        _l3_staging_allocation.reset(token)


@contextmanager
def bounded_host_eviction_scan(
    max_nodes: int,
) -> Iterator[HostEvictionScanBudget]:
    """Bound host-LRU work performed by one staging-reclaim pass."""

    budget = HostEvictionScanBudget(remaining_nodes=max_nodes)
    token = _host_eviction_scan_budget.set(budget)
    try:
        yield budget
    finally:
        _host_eviction_scan_budget.reset(token)


def get_host_eviction_scan_budget() -> HostEvictionScanBudget | None:
    return _host_eviction_scan_budget.get()


def _index_values(indices: Any) -> list[int]:
    values = indices.tolist() if hasattr(indices, "tolist") else list(indices)
    return [int(value) for value in values]


def _take_indices(indices: Any, positions: list[int]) -> Any:
    if len(positions) == len(indices):
        return indices
    if isinstance(indices, list):
        return [indices[position] for position in positions]
    position_tensor = torch.tensor(positions, dtype=torch.int64, device=indices.device)
    return indices[position_tensor]


def _string_sequence_digest(values: Sequence[str]) -> tuple[int, ...]:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return tuple(
        int.from_bytes(digest.digest()[offset : offset + 4], "big") & 0x7FFFFFFF
        for offset in range(0, digest.digest_size, 4)
    )


class StoragePrefetchState(Enum):
    NOT_ATTEMPTED = auto()
    DEFERRED = auto()
    QUERYING = auto()
    READING = auto()
    READY = auto()
    MISS = auto()
    FAILED = auto()

    @property
    def is_terminal(self) -> bool:
        return self in {self.READY, self.MISS, self.FAILED}

    @property
    def is_in_progress(self) -> bool:
        return self in {self.QUERYING, self.READING}


class StoragePrefetchTracker:
    """Track storage-prefetch state until scheduler admission consumes it."""

    def __init__(self) -> None:
        self._states: dict[str, StoragePrefetchState] = {}

    def get(self, request_id: str) -> StoragePrefetchState:
        return self._states.get(request_id, StoragePrefetchState.NOT_ATTEMPTED)

    def set(self, request_id: str, state: StoragePrefetchState) -> None:
        self._states[request_id] = state

    def forget(self, request_id: str) -> None:
        self._states.pop(request_id, None)

    def clear(self) -> None:
        self._states.clear()


class PrefetchOwnershipTransition(NamedTuple):
    """Immutable query-worker result consumed by the scheduler thread."""

    request_id: str
    generation: int
    state: StoragePrefetchState
    retained_tokens: int
    hash_values: tuple[str, ...]


@dataclass(slots=True, kw_only=True)
class StoragePrefetchOwnership:
    """Host memory owned by one scheduler-managed storage prefetch."""

    operation_id: int
    generation: int
    host_indices: Any
    owned_tokens: int


@dataclass(slots=True, kw_only=True)
class L3StagingQuota:
    """A physically isolated allocation reserve inside one host pool."""

    name: str
    pool: Any
    capacity_tokens: int
    page_size: int
    original_alloc: Callable[[int], Any | None]
    original_free: Callable[[Any], int]
    original_available_size: Callable[[], int]
    state_changed: Callable[[], None]
    free_chunks: list[Any] = field(default_factory=list)
    staging_indices: set[int] = field(default_factory=set)
    in_use_indices: set[int] = field(default_factory=set)

    @classmethod
    def install(
        cls,
        *,
        name: str,
        pool: Any,
        capacity_tokens: int,
        state_changed: Callable[[], None],
    ) -> L3StagingQuota:
        quota = cls(
            name=name,
            pool=pool,
            capacity_tokens=capacity_tokens,
            page_size=pool.page_size,
            original_alloc=pool.alloc,
            original_free=pool.free,
            original_available_size=pool.available_size,
            state_changed=state_changed,
        )
        quota._reserve_after_clear()
        pool.alloc = quota.alloc
        pool.free = quota.free
        pool.available_size = quota.available_size
        return quota

    @property
    def available_tokens(self) -> int:
        return self.capacity_tokens - len(self.in_use_indices)

    @property
    def used_tokens(self) -> int:
        return len(self.in_use_indices)

    def alloc(self, need_size: int) -> Any | None:
        if not _l3_staging_allocation.get():
            return self.original_alloc(need_size)
        if need_size % self.page_size != 0:
            raise ValueError(
                f"L3 staging allocation for {self.name} must be page-aligned: "
                f"need={need_size} page_size={self.page_size}"
            )
        if need_size == 0:
            return self.original_alloc(0)
        if need_size > self.available_tokens:
            return None

        remaining = need_size
        allocated_chunks = []
        while remaining > 0:
            chunk = self.free_chunks.pop()
            if len(chunk) <= remaining:
                allocated_chunks.append(chunk)
                remaining -= len(chunk)
                continue
            allocated_chunks.append(chunk[:remaining])
            self.free_chunks.append(chunk[remaining:])
            remaining = 0
        indices = (
            allocated_chunks[0]
            if len(allocated_chunks) == 1
            else torch.cat(allocated_chunks, dim=0)
        )
        values = _index_values(indices)
        if any(value not in self.staging_indices for value in values):
            raise RuntimeError(f"L3 staging free-list corruption in pool {self.name}")
        if any(value in self.in_use_indices for value in values):
            raise RuntimeError(f"L3 staging double allocation in pool {self.name}")
        self.in_use_indices.update(values)
        return indices

    def available_size(self) -> int:
        if _l3_staging_allocation.get():
            return self.available_tokens
        return self.original_available_size()

    def free(self, indices: Any) -> int:
        values = _index_values(indices)
        staging_positions = [
            position
            for position, value in enumerate(values)
            if value in self.staging_indices
        ]
        ordinary_positions = [
            position
            for position, value in enumerate(values)
            if value not in self.staging_indices
        ]
        if staging_positions:
            staging_values = {values[position] for position in staging_positions}
            if not staging_values.issubset(self.in_use_indices):
                raise RuntimeError(f"L3 staging double free in pool {self.name}")
            self.in_use_indices.difference_update(staging_values)
            self.free_chunks.append(_take_indices(indices, staging_positions))
            self.state_changed()
        if ordinary_positions:
            self.original_free(_take_indices(indices, ordinary_positions))
        return len(values)

    def count_in_use(self, indices: Any) -> int:
        return sum(value in self.in_use_indices for value in _index_values(indices))

    def promote(self, indices: Any) -> int:
        """Reclassify live staging indices as L2 and reserve free L2 slots."""

        values = _index_values(indices)
        staging_positions = [
            position
            for position, value in enumerate(values)
            if value in self.in_use_indices
        ]
        if not staging_positions:
            return 0
        staging_indices = _take_indices(indices, staging_positions)
        replacement = self.original_alloc(len(staging_indices))
        if replacement is None:
            return 0

        staging_values = set(_index_values(staging_indices))
        replacement_values = set(_index_values(replacement))
        if replacement_values.intersection(self.staging_indices):
            raise RuntimeError(f"L3 staging replacement collision in pool {self.name}")
        self.in_use_indices.difference_update(staging_values)
        self.staging_indices.difference_update(staging_values)
        self.staging_indices.update(replacement_values)
        self.free_chunks.append(replacement)
        self.state_changed()
        return len(staging_values)

    def reset_after_clear(self) -> None:
        self.free_chunks.clear()
        self.staging_indices.clear()
        self.in_use_indices.clear()
        self._reserve_after_clear()

    def uninstall(self) -> int:
        self.pool.alloc = self.original_alloc
        self.pool.free = self.original_free
        self.pool.available_size = self.original_available_size
        if self.free_chunks:
            free_indices = (
                self.free_chunks[0]
                if len(self.free_chunks) == 1
                else torch.cat(self.free_chunks, dim=0)
            )
            self.original_free(free_indices)
        reclassified_tokens = len(self.in_use_indices)
        self.free_chunks.clear()
        self.staging_indices.clear()
        self.in_use_indices.clear()
        return reclassified_tokens

    def _reserve_after_clear(self) -> None:
        if self.capacity_tokens == 0:
            return
        reserved = self.original_alloc(self.capacity_tokens)
        if reserved is None or len(reserved) != self.capacity_tokens:
            if reserved is not None:
                self.original_free(reserved)
            raise RuntimeError(
                f"Cannot atomically reserve {self.capacity_tokens} L3 staging "
                f"tokens from host pool {self.name}"
            )
        values = set(_index_values(reserved))
        if len(values) != self.capacity_tokens:
            self.original_free(reserved)
            raise RuntimeError(f"Duplicate indices in L3 staging pool {self.name}")
        self.free_chunks.append(reserved)
        self.staging_indices.update(values)


@dataclass(frozen=True, slots=True, kw_only=True)
class _L3StagingPoolPlan:
    name: str
    pool: Any
    reserve_tokens: int


@dataclass(slots=True)
class L3StagingManager:
    """Own the staging partitions associated with one HiCache controller."""

    reserve_ratio: float = 0.0
    reserve_tokens: int = 0
    shortfall_tokens: int = 0
    generation: int = 0
    quotas: dict[int, L3StagingQuota] = field(default_factory=dict)

    def install(
        self,
        mem_pool_host: Any,
        reserve_ratio: float,
        synchronize_values: Callable[[list[int]], list[int]],
    ) -> None:
        ratio_valid = math.isfinite(reserve_ratio) and 0 <= reserve_ratio < 1
        plans = self._build_plans(mem_pool_host, reserve_ratio if ratio_valid else 0.0)
        plan_digest = hashlib.sha256()
        plan_digest.update(str(reserve_ratio).encode())
        plan_digest.update(b"\0")
        for plan in plans:
            plan_digest.update(plan.name.encode())
            plan_digest.update(b"\0")
            for value in (
                plan.pool.size,
                plan.pool.page_size,
                plan.reserve_tokens,
                plan.pool.available_size(),
            ):
                plan_digest.update(str(int(value)).encode())
                plan_digest.update(b"\0")
        local_plan = [
            int(ratio_valid),
            len(plans),
            *[
                int.from_bytes(plan_digest.digest()[offset : offset + 4], "big")
                & 0x7FFFFFFF
                for offset in range(0, plan_digest.digest_size, 4)
            ],
            *[
                value
                for plan in plans
                for value in (
                    plan.pool.size,
                    plan.pool.page_size,
                    plan.reserve_tokens,
                    plan.pool.available_size(),
                )
            ],
        ]
        extrema = synchronize_values(local_plan + [-value for value in local_plan])
        split = len(local_plan)
        if extrema[:split] != [-value for value in extrema[split:]]:
            raise RuntimeError(
                "HiCache L3 staging install plan diverged across prefetch ranks"
            )
        if not ratio_valid:
            raise ValueError(
                "SGLANG_HICACHE_L3_STAGING_RESERVE_RATIO must be a number in "
                f"[0, 1), got {reserve_ratio!r}"
            )

        self.uninstall()
        self.reserve_ratio = reserve_ratio
        if reserve_ratio == 0:
            return

        def state_changed() -> None:
            self.generation += 1

        installed: dict[int, L3StagingQuota] = {}
        install_error: Exception | None = None
        try:
            for plan in plans:
                installed[id(plan.pool)] = L3StagingQuota.install(
                    name=plan.name,
                    pool=plan.pool,
                    capacity_tokens=plan.reserve_tokens,
                    state_changed=state_changed,
                )
        except Exception as error:
            install_error = error

        all_installed = synchronize_values([int(install_error is None)])[0]
        if all_installed == 0:
            for quota in installed.values():
                quota.uninstall()
            if install_error is not None:
                raise install_error
            raise RuntimeError("A peer rank failed to install the HiCache L3 reserve")

        self.quotas = installed
        anchor_pool = getattr(
            getattr(mem_pool_host, "anchor_entry", None),
            "host_pool",
            mem_pool_host,
        )
        anchor_quota = installed.get(id(anchor_pool))
        self.reserve_tokens = (
            anchor_quota.capacity_tokens if anchor_quota is not None else 0
        )

    def reset_after_clear(self) -> None:
        for quota in self.quotas.values():
            quota.reset_after_clear()

    def uninstall(self) -> int:
        reclassified = sum(quota.uninstall() for quota in self.quotas.values())
        self.quotas.clear()
        self.reserve_tokens = 0
        self.shortfall_tokens = 0
        return reclassified

    def usage(self) -> dict[str, str]:
        return {
            quota.name: f"{quota.used_tokens}/{quota.capacity_tokens}"
            for quota in self.quotas.values()
        }

    def quota_for_pool(self, pool: Any) -> L3StagingQuota | None:
        return self.quotas.get(id(pool))

    @staticmethod
    def _build_plans(
        mem_pool_host: Any, reserve_ratio: float
    ) -> tuple[_L3StagingPoolPlan, ...]:
        entries = getattr(mem_pool_host, "entries", None)
        if entries:
            anchor_entry = getattr(mem_pool_host, "anchor_entry", None)
            named_pools = [
                (str(entry.name), entry.host_pool)
                for entry in entries
                if entry is anchor_entry
                or entry.is_primary_index_anchor
                or entry.host_evict_fn is not None
            ]
        else:
            named_pools = [("KV", mem_pool_host)]

        plans_by_pool = {}
        for name, pool in named_pools:
            reserve_tokens = (
                int(pool.size * reserve_ratio) // pool.page_size * pool.page_size
            )
            plans_by_pool[id(pool)] = _L3StagingPoolPlan(
                name=name,
                pool=pool,
                reserve_tokens=reserve_tokens,
            )
        return tuple(sorted(plans_by_pool.values(), key=lambda plan: plan.name))


@dataclass(slots=True, kw_only=True)
class L3StagingLease:
    """Tree ownership metadata for live indices from a staging partition."""

    quota: L3StagingQuota
    indices: Any
    owner_node: Any | None = None
    component_type: Any | None = None
    reclaim_index_cursor: int = 0
    discard_scan_node: Any | None = None
    discard_scan_remaining: frozenset[int] | None = None


@dataclass(slots=True, kw_only=True)
class StoragePrefetchAdmissionPin:
    node: Any
    lock_params: Any
    occupied_tokens: int
    staging_leases: tuple[L3StagingLease, ...] = ()


@dataclass(slots=True, kw_only=True)
class StoragePrefetchDeviceAdmissionPin:
    node: Any
    lock_params: Any


class CheckpointRetryResource(Enum):
    HOST = auto()
    AUXILIARY = auto()
    STORAGE = auto()
    SCHEDULER = auto()


@dataclass(slots=True, kw_only=True)
class PendingStorageCheckpoint:
    """Pinned path and bounded retry cursor for one durability checkpoint."""

    handle: str
    generation: int
    radix_key: Any
    path: tuple[Any, ...]
    path_hashes: tuple[str, ...]
    path_hash_digest: tuple[int, ...] = field(init=False)
    cursor_pages: int = 0
    device_pin: tuple[Any, Any] | None = None
    host_pin: tuple[Any, Any] | None = None
    staging_leases: list[L3StagingLease] = field(default_factory=list)
    staging_lease_keys: set[tuple[Any, ...]] = field(init=False)
    active_operation_ids: set[int] = field(default_factory=set)
    blocked_resource: CheckpointRetryResource | None = None
    blocked_generation: int = -1
    retry_queued: bool = False
    retry_ticket: int = 0
    requires_device_pin: bool = False
    path_scan_node: Any | None = None
    path_scan_reverse_nodes: deque[Any] = field(default_factory=deque)
    path_scan_drive_reverse_nodes: deque[Any] = field(default_factory=deque)
    path_scan_hash_cursor: int = 0
    path_scan_key_tokens: int = 0
    path_scan_valid: bool = True
    path_scan_tree_generation: int = -1

    def __post_init__(self) -> None:
        self.path_hash_digest = _string_sequence_digest(self.path_hashes)
        self.staging_lease_keys = set()


@dataclass(slots=True)
class StorageCheckpointRetryQueues:
    """Resource-indexed, generation-gated retry queues."""

    queues: dict[CheckpointRetryResource, deque[tuple[str, int, int]]] = field(
        default_factory=lambda: {
            resource: deque() for resource in CheckpointRetryResource
        }
    )
    generations: dict[CheckpointRetryResource, int] = field(
        default_factory=lambda: {resource: 0 for resource in CheckpointRetryResource}
    )
    cursor: int = 0
    continuation_resources: set[CheckpointRetryResource] = field(default_factory=set)

    def clear(self) -> None:
        for queue in self.queues.values():
            queue.clear()
        self.generations = {resource: 0 for resource in CheckpointRetryResource}
        self.cursor = 0
        self.continuation_resources.clear()


@dataclass(frozen=True)
class StorageWriteCompletion:
    operation_id: int
    page_hashes: tuple[str, ...]
    durable_pages: int
    pending_hashes: frozenset[str] = frozenset()


class StorageWriteTracker:
    """Correlate storage lookups with writes that have not acknowledged yet."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._operation_hashes: dict[int, tuple[str, ...]] = {}
        self._pending_operations_by_hash: dict[str, set[int]] = {}
        self._durable_hashes: set[str] = set()

    def register(self, operation_id: int, page_hashes: list[str]) -> None:
        with self._condition:
            hashes = tuple(page_hashes)
            self._operation_hashes[operation_id] = hashes
            for page_hash in hashes:
                self._pending_operations_by_hash.setdefault(page_hash, set()).add(
                    operation_id
                )
            self._condition.notify_all()

    def has_pending(self, page_hashes: Sequence[str]) -> bool:
        with self._condition:
            return self._has_pending(page_hashes)

    def get_durable_hashes(self, page_hashes: Sequence[str]) -> frozenset[str]:
        with self._condition:
            return frozenset(self._durable_hashes.intersection(page_hashes))

    def mark_durable(self, page_hashes: Sequence[str]) -> None:
        with self._condition:
            self._durable_hashes.update(page_hashes)

    def wait_until_clear(
        self,
        page_hashes: Sequence[str],
        cancelled: Callable[[], bool],
    ) -> bool:
        with self._condition:
            while self._has_pending(page_hashes):
                if cancelled():
                    return False
                self._condition.wait(timeout=0.05)
            return not cancelled()

    def complete(
        self, operation_id: int, durable_pages: int
    ) -> StorageWriteCompletion | None:
        with self._condition:
            page_hashes = self._operation_hashes.pop(operation_id, None)
            if page_hashes is None:
                return None
            for page_hash in page_hashes:
                pending_operations = self._pending_operations_by_hash[page_hash]
                pending_operations.discard(operation_id)
                if not pending_operations:
                    del self._pending_operations_by_hash[page_hash]
            durable_pages = max(0, min(durable_pages, len(page_hashes)))
            self._durable_hashes.update(page_hashes[:durable_pages])
            pending_hashes = frozenset(
                page_hash
                for page_hash in page_hashes
                if page_hash in self._pending_operations_by_hash
            )
            self._condition.notify_all()
            return StorageWriteCompletion(
                operation_id=operation_id,
                page_hashes=page_hashes,
                durable_pages=durable_pages,
                pending_hashes=pending_hashes,
            )

    def clear(self) -> None:
        with self._condition:
            self._operation_hashes.clear()
            self._pending_operations_by_hash.clear()
            self._durable_hashes.clear()
            self._condition.notify_all()

    def _has_pending(self, page_hashes: Sequence[str]) -> bool:
        return any(
            page_hash in self._pending_operations_by_hash for page_hash in page_hashes
        )


class StorageCheckpointState(Enum):
    PENDING = auto()
    READY = auto()
    FAILED = auto()


@dataclass
class StorageCheckpoint:
    handle: str
    expected_hashes: frozenset[str] | None = None
    durable_hashes: set[str] = field(default_factory=set)
    state: StorageCheckpointState = StorageCheckpointState.PENDING


class StorageCheckpointRegistry:
    """Track the L3 durability of explicitly requested rollout checkpoints."""

    def __init__(self) -> None:
        self._checkpoints: dict[str, StorageCheckpoint] = {}
        self._checkpoint_handles_by_hash: dict[str, set[str]] = {}

    def reserve(self, handle: str) -> StorageCheckpoint:
        if handle in self._checkpoints:
            self.forget(handle)
        checkpoint = StorageCheckpoint(handle=handle)
        self._checkpoints[handle] = checkpoint
        return checkpoint

    def create(
        self,
        handle: str,
        page_hashes: Sequence[str],
        durable_hashes: Sequence[str] = (),
        has_pending: Callable[[Sequence[str]], bool] | None = None,
    ) -> StorageCheckpoint:
        if handle in self._checkpoints:
            self.forget(handle)
        checkpoint = StorageCheckpoint(
            handle=handle,
            expected_hashes=frozenset(page_hashes),
        )
        if not checkpoint.expected_hashes:
            checkpoint.state = StorageCheckpointState.FAILED
            checkpoint.expected_hashes = None
        else:
            checkpoint.durable_hashes.update(
                checkpoint.expected_hashes.intersection(durable_hashes)
            )
            if checkpoint.durable_hashes == checkpoint.expected_hashes and not (
                has_pending is not None and has_pending(checkpoint.expected_hashes)
            ):
                self._set_terminal(checkpoint, StorageCheckpointState.READY)
            else:
                for page_hash in checkpoint.expected_hashes:
                    self._checkpoint_handles_by_hash.setdefault(page_hash, set()).add(
                        handle
                    )
        self._checkpoints[handle] = checkpoint
        return checkpoint

    def get_state(self, handle: str) -> StorageCheckpointState:
        checkpoint = self._checkpoints.get(handle)
        if checkpoint is None:
            return StorageCheckpointState.FAILED
        return checkpoint.state

    def record_write_completion(
        self,
        completion: StorageWriteCompletion,
        has_pending: Callable[[Sequence[str]], bool] | None = None,
    ) -> None:
        durable_hashes = completion.page_hashes[: completion.durable_pages]
        failed_hashes = completion.page_hashes[completion.durable_pages :]
        affected_handles = {
            handle
            for page_hash in completion.page_hashes
            for handle in self._checkpoint_handles_by_hash.get(page_hash, ())
        }
        for handle in affected_handles:
            checkpoint = self._checkpoints[handle]
            if checkpoint.expected_hashes is None:
                continue
            checkpoint.durable_hashes.update(
                checkpoint.expected_hashes.intersection(durable_hashes)
            )
            unresolved_failed_hashes = (
                checkpoint.expected_hashes.intersection(failed_hashes)
                - checkpoint.durable_hashes
            )
            definitive_failed_hashes = (
                unresolved_failed_hashes - completion.pending_hashes
            )
            if definitive_failed_hashes:
                self._set_terminal(checkpoint, StorageCheckpointState.FAILED)
            elif checkpoint.durable_hashes == checkpoint.expected_hashes:
                pending = (
                    has_pending(checkpoint.expected_hashes)
                    if has_pending is not None
                    else bool(
                        checkpoint.expected_hashes.intersection(
                            completion.pending_hashes
                        )
                    )
                )
                if not pending:
                    self._set_terminal(checkpoint, StorageCheckpointState.READY)

    def fail(self, handle: str) -> None:
        checkpoint = self._checkpoints.get(handle)
        if checkpoint is None:
            checkpoint = self.create(handle, ())
        self._set_terminal(checkpoint, StorageCheckpointState.FAILED)

    def fail_if_present(self, handle: str) -> None:
        if handle in self._checkpoints:
            self.fail(handle)

    def forget(self, handle: str) -> None:
        checkpoint = self._checkpoints.pop(handle, None)
        if checkpoint is not None:
            self._remove_reverse_entries(checkpoint)

    def clear(self) -> None:
        self._checkpoints.clear()
        self._checkpoint_handles_by_hash.clear()

    def _remove_reverse_entries(self, checkpoint: StorageCheckpoint) -> None:
        for page_hash in checkpoint.expected_hashes or ():
            handles = self._checkpoint_handles_by_hash.get(page_hash)
            if handles is None:
                continue
            handles.discard(checkpoint.handle)
            if not handles:
                del self._checkpoint_handles_by_hash[page_hash]

    def _set_terminal(
        self,
        checkpoint: StorageCheckpoint,
        state: StorageCheckpointState,
    ) -> None:
        assert state is not StorageCheckpointState.PENDING
        self._remove_reverse_entries(checkpoint)
        checkpoint.state = state
        checkpoint.expected_hashes = None
        checkpoint.durable_hashes.clear()


def make_storage_checkpoint_handle(request_id: str) -> str:
    return f"hicache:{request_id}"
