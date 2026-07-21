from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Sequence


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
