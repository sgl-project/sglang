"""CPU page-lifetime ledger, independent of trees, CUDA, and transport workers.

The allocator owner calls allocation/free/epoch methods. Background threads may
publish, pin, and release, but never physically free a page. A Pin protects its
pages immediately, including prepared/partial exports. The adapter supplies a
common TP epoch and maps its conservative protection union into rank-local
PageRefs; this module never runs collectives. The allocator must apply returned
reclaim batches before reuse.
"""

import threading
from dataclasses import dataclass, field
from enum import Enum


class SlotGuardError(RuntimeError):
    """A safe rejection; no requested partial acquisition or reclaim occurred."""


class PageUnavailable(SlotGuardError):
    pass


class PinsFrozen(SlotGuardError):
    pass


class QuotaExceeded(SlotGuardError):
    pass


class ReplayConflict(SlotGuardError):
    pass


class OwnerThreadViolation(SlotGuardError):
    pass


class EpochError(SlotGuardError):
    pass


class UnsafeReset(SlotGuardError):
    pass


def _text(name, value):
    if type(value) is not str or not value.strip() or len(value) > 4096:
        raise ValueError(f"{name} must be nonempty bounded text")


def _epoch_id(epoch_id):
    # None would equal an idle ledger's absent frozen and installed epochs.
    if type(epoch_id) is not int:
        raise EpochError("allocation epoch id must be an integer")


def _distinct(refs):
    # A repeated page would be counted twice by the owned and held counters.
    if len(set(refs)) != len(refs):
        raise ValueError("pages must be distinct")


@dataclass(frozen=True, order=True)
class PageRef:
    pool_epoch: int
    page_id: int
    generation: int


@dataclass(frozen=True)
class LeaseIdentity:
    source_incarnation: str
    target_incarnation: str
    # The wire protocol's TransferId: a frozen value of target_incarnation, rid
    # and attempt. Pin requests decode it from the control wire, so its fields
    # are bounded here.
    transfer_id: object
    nonce: str

    def __post_init__(self):
        for name in ("source_incarnation", "target_incarnation", "nonce"):
            _text(name, getattr(self, name))
        if self.transfer_id.target_incarnation != self.target_incarnation:
            raise ValueError("transfer target incarnation mismatch")
        _text("transfer_id.rid", self.transfer_id.rid)
        attempt = self.transfer_id.attempt
        if type(attempt) is not int or not 0 <= attempt < 1 << 63:
            raise ValueError("transfer_id.attempt must be a bounded integer")


class PageState(str, Enum):
    ALLOCATED = "ALLOCATED"
    PUBLISHED = "PUBLISHED"
    DEFERRED = "DEFERRED"
    RECLAIM_PENDING = "RECLAIM_PENDING"
    FREE = "FREE"


@dataclass(frozen=True)
class EpochSnapshot:
    """Held and reclaimable pages at epoch begin; constructing it never scans
    the pool. Candidates still need the common epoch protection check."""

    held_pages: tuple[PageRef, ...]
    reclaim_candidate_pages: tuple[PageRef, ...]


@dataclass(frozen=True)
class OwnerCounts:
    """O(1) owner proof inputs; local cached pages are not remote owners."""

    active_leases: int
    held_pages: int
    local_owned_pages: int
    retired_pages: int
    pending_reclaim_pages: int
    reclaim_overflow_pages: int
    frozen_epoch: int | None
    credit_remaining: int
    revision: int


@dataclass
class _Page:
    ref: PageRef
    state: PageState = PageState.ALLOCATED
    local_owned: bool = True
    readers: set[LeaseIdentity] = field(default_factory=set)


class SlotGuard:
    """Bounded page metadata for remote Pins and owner reclaim.

    max_pages bounds the physical registration domain (including FREE records);
    the allocator checks its pool against it. A full pending queue retains
    overflow in the existing page records as DEFERRED and backpressures all new
    Pins. Callers own Pin replay: each lease identity is pinned once and
    released once. No callbacks, hashing of payloads, I/O or accelerator work
    run under _lock.
    """

    def __init__(
        self,
        pool_epoch: int,
        source_incarnation: str,
        *,
        max_remote_pinned_pages: int,
        max_active_leases: int,
        max_pending_reclaim_pages: int,
        max_pages: int = 1 << 20,
    ):
        self.pool_epoch = pool_epoch
        self._allocation_faulted = False
        self.source_incarnation = source_incarnation
        self.max_remote_pinned_pages = max_remote_pinned_pages
        self.max_active_leases = max_active_leases
        self.max_pending_reclaim_pages = max_pending_reclaim_pages
        self.max_pages = max_pages
        # New Pins draw on headroom that a finished common epoch verified.
        self._credit_remaining = 0
        self._owner_thread = threading.get_ident()
        self._lock = threading.RLock()
        self._pages: dict[int, _Page] = {}
        self._active: dict[LeaseIdentity, tuple[PageRef, ...]] = {}
        self._held_count = self._pending_count = self._overflow_count = 0
        self._local_owned_count = 0
        self._owner_revision = 0
        self._held_pages: set[PageRef] = set()
        self._retired_pages: set[PageRef] = set()
        self._reclaim_candidates: set[PageRef] = set()
        self._frozen_epoch = None
        self._installed_epoch = None
        self._last_epoch = -1
        self._protected = frozenset()
        self._epoch_held = frozenset()

    def _owner(self):
        if threading.get_ident() != self._owner_thread:
            raise OwnerThreadViolation(
                "operation belongs to the allocator owner thread"
            )

    def _current(self, ref):
        if ref.pool_epoch != self.pool_epoch:
            raise PageUnavailable("pool epoch mismatch")
        page = self._pages.get(ref.page_id)
        if page is None:
            raise PageUnavailable("page is not registered")
        if page.ref != ref:
            raise PageUnavailable("page generation mismatch")
        return page

    def _identity(self, identity):
        if identity.source_incarnation != self.source_incarnation:
            raise ValueError("source incarnation mismatch")

    def _backpressured(self):
        return self._overflow_count > 0

    def _allocation_refs_locked(self, page_ids):
        self._allocation_epoch_preflight_locked()
        _distinct(page_ids)
        refs = []
        for page_id in page_ids:
            old = self._pages.get(page_id)
            if old is not None and old.state is not PageState.FREE:
                raise PageUnavailable("page has not been reclaimed by the owner")
            generation = 1 if old is None else old.ref.generation + 1
            refs.append(PageRef(self.pool_epoch, page_id, generation))
        return tuple(refs)

    def _allocation_epoch_preflight_locked(self):
        # A faulted owner keeps its frozen epoch uninstalled, so this fences it.
        if self._frozen_epoch is not None and self._installed_epoch is None:
            raise EpochError("allocation requires the frozen epoch to be committed")

    def allocation_preflight(self, page_ids=None):
        """Read-only check before the allocator mutates queues or launches kernels;
        ``page_ids`` also checks those candidates are FREE in this epoch."""
        self._owner()
        with self._lock:
            if page_ids is None:
                self._allocation_epoch_preflight_locked()
            else:
                self._allocation_refs_locked(page_ids)

    def register_allocation(self, page_ids: tuple[int, ...]) -> tuple[PageRef, ...]:
        """``page_ids`` come from the owner's own free queue: physical page ids."""
        self._owner()
        with self._lock:
            refs = self._allocation_refs_locked(page_ids)
            for ref in refs:
                self._pages[ref.page_id] = _Page(ref)
            self._local_owned_count += len(refs)
            self._owner_revision += 1
            return refs

    def publish(self, refs: tuple[PageRef, ...]):
        """Publish only after the caller has independently verified producer ready."""
        with self._lock:
            pages = [self._current(ref) for ref in refs]
            if any(not page.local_owned for page in pages):
                raise PageUnavailable("retired pages cannot be published")
            for page in pages:
                page.state = PageState.PUBLISHED
            self._owner_revision += 1

    def published_refs_current(self, refs: tuple[PageRef, ...]) -> bool:
        """Bounded metadata preflight, repeated atomically by try_pin later."""
        with self._lock:
            try:
                return all(
                    self._current(ref).local_owned
                    and self._current(ref).state is PageState.PUBLISHED
                    for ref in refs
                )
            except PageUnavailable:
                return False

    def try_pin(self, identity: LeaseIdentity, refs: tuple[PageRef, ...]):
        """Pin ``refs`` for one lease. The caller pins each identity at most once
        and bounds the batch size."""
        self._identity(identity)
        _distinct(refs)
        with self._lock:
            if identity in self._active:
                raise ReplayConflict("lease is already active")
            pages = [self._current(ref) for ref in refs]
            if any(
                not page.local_owned or page.state is not PageState.PUBLISHED
                for page in pages
            ):
                raise PageUnavailable(
                    "Pin requires current published locally owned pages"
                )
            if self._frozen_epoch is not None:
                raise PinsFrozen("new Pins are frozen through allocation commit")
            if self._backpressured():
                raise QuotaExceeded("retired page overflow requires owner reclaim")
            if len(self._active) >= self.max_active_leases:
                raise QuotaExceeded("active lease quota exhausted")
            newly_held = sum(not page.readers for page in pages)
            if self._held_count + newly_held > self.max_remote_pinned_pages:
                raise QuotaExceeded("unique remotely protected page quota exhausted")
            if newly_held > self._credit_remaining:
                raise QuotaExceeded("verified cumulative headroom credit exhausted")
            for page in pages:
                if not page.readers:
                    self._held_pages.add(page.ref)
                page.readers.add(identity)
            self._held_count += newly_held
            self._credit_remaining -= newly_held
            self._active[identity] = refs
            self._owner_revision += 1

    def _queue(self, page):
        self._retired_pages.add(page.ref)
        self._reclaim_candidates.add(page.ref)
        if self._pending_count < self.max_pending_reclaim_pages:
            page.state = PageState.RECLAIM_PENDING
            self._pending_count += 1
        else:
            page.state = PageState.DEFERRED
            self._overflow_count += 1

    def release(self, identity: LeaseIdentity, refs: tuple[PageRef, ...]):
        """Release a Pin once its reader has drained and unmapped every export.
        The caller releases each pinned identity once, with its pinned pages."""
        with self._lock:
            if self._active.get(identity) != refs:
                raise SlotGuardError("release does not match an active lease")
            del self._active[identity]
            for ref in refs:
                page = self._pages[ref.page_id]
                page.readers.remove(identity)
                if not page.readers:
                    self._held_count -= 1
                    self._held_pages.remove(page.ref)
                    if not page.local_owned:
                        self._queue(page)
            self._owner_revision += 1

    def on_free(self, refs: tuple[PageRef, ...]) -> tuple[PageRef, ...]:
        """Retire the owner's pages, which also closes them to new Pins, and
        return those reusable now. The allocator's stream ordering already
        covers local accesses."""
        self._owner()
        _distinct(refs)
        with self._lock:
            pages = [self._current(ref) for ref in refs]
            if any(not page.local_owned for page in pages):
                raise PageUnavailable("page was already retired")
            reusable = []
            for page in pages:
                # Retirement and closing Pin admission share the same lock.
                page.local_owned = False
                self._local_owned_count -= 1
                if page.readers:
                    page.state = PageState.DEFERRED
                    self._retired_pages.add(page.ref)
                elif (
                    self._installed_epoch is not None
                    and page.ref not in self._protected
                ):
                    page.state = PageState.FREE
                    reusable.append(page.ref)
                else:
                    self._queue(page)
            self._owner_revision += 1
            return tuple(reusable)

    def begin_epoch(self, epoch_id: int) -> EpochSnapshot:
        self._owner()
        _epoch_id(epoch_id)
        with self._lock:
            if self._frozen_epoch is not None or epoch_id <= self._last_epoch:
                raise EpochError("an unfinished or stale allocation epoch cannot begin")
            self._frozen_epoch = epoch_id
            self._owner_revision += 1
            self._epoch_held = frozenset(self._held_pages)
            return EpochSnapshot(
                held_pages=tuple(sorted(self._epoch_held)),
                reclaim_candidate_pages=tuple(sorted(self._reclaim_candidates)),
            )

    def install_epoch(
        self,
        epoch_id: int,
        protected_pages: tuple[PageRef, ...],
        reclaim: tuple[PageRef, ...] | None = None,
    ) -> tuple[PageRef, ...]:
        """Install an externally committed conservative union, never shrink it,
        and reclaim ``reclaim`` (default: every unprotected candidate). Returns
        the reclaimed pages; a rejection changes neither."""
        self._owner()
        if reclaim is not None:
            _distinct(reclaim)
        with self._lock:
            protected = self._install_preflight_locked(epoch_id, protected_pages)
            pages = self._reclaim_batch_locked(reclaim, protected)
            self._protected = protected
            self._installed_epoch = epoch_id
            return self._drain_locked(pages)

    def _install_preflight_locked(self, epoch_id, protected_pages):
        _epoch_id(epoch_id)
        if self._allocation_faulted:
            raise EpochError("allocation owner is permanently faulted")
        if self._frozen_epoch != epoch_id:
            raise EpochError("allocation epoch is absent or mismatched")
        if self._installed_epoch is not None:
            raise EpochError("allocation epoch protection is immutable")
        protected = frozenset(protected_pages)
        if not self._epoch_held <= protected:
            raise EpochError("committed protection drops a frozen epoch reference")
        for ref in protected:
            if self._current(ref).state is PageState.FREE:
                raise EpochError("protected page has already been returned")
        if not self._held_pages <= protected:
            raise EpochError(
                "committed protection drops a local prepared/active reference"
            )
        return protected

    def _reclaim_batch_locked(self, refs, protected):
        if refs is None:
            return [
                self._current(ref)
                for ref in sorted(self._reclaim_candidates - protected)
            ]
        pages = [self._current(ref) for ref in refs]
        if any(
            p.local_owned
            or p.readers
            or p.ref in protected
            or p.state is PageState.FREE
            or p.ref not in self._reclaim_candidates
            for p in pages
        ):
            raise PageUnavailable("reclaim batch is not wholly safe in this epoch")
        return pages

    def _drain_locked(self, pages):
        for page in pages:
            if page.state is PageState.RECLAIM_PENDING:
                self._pending_count -= 1
            else:
                self._overflow_count -= 1
            self._retired_pages.remove(page.ref)
            self._reclaim_candidates.remove(page.ref)
            page.state = PageState.FREE
        self._owner_revision += 1
        return tuple(page.ref for page in pages)

    def owner_counts(self) -> OwnerCounts:
        with self._lock:
            return OwnerCounts(
                len(self._active),
                self._held_count,
                self._local_owned_count,
                len(self._retired_pages),
                self._pending_count,
                self._overflow_count,
                self._frozen_epoch,
                self._credit_remaining,
                self._owner_revision,
            )

    def finish_epoch(self, epoch_id: int, *, verified_headroom_pages: int):
        """Close the epoch; new Pins may then hold up to the headroom, in pages,
        that every rank verified for this epoch."""
        self._owner()
        with self._lock:
            self._finish_preflight_locked(epoch_id)
            self._credit_remaining = verified_headroom_pages
            self._last_epoch = epoch_id
            self._frozen_epoch = self._installed_epoch = None
            self._protected = frozenset()
            self._epoch_held = frozenset()
            self._owner_revision += 1

    def _finish_preflight_locked(self, epoch_id):
        _epoch_id(epoch_id)
        # A faulted epoch stays uninstalled, so it can never finish.
        if self._frozen_epoch != epoch_id or self._installed_epoch != epoch_id:
            raise EpochError("allocation epoch is absent or not committed")

    def prevalidate_finish_epoch(self, epoch_id):
        """Owner-only read-only input validation before the common finish vote."""
        self._owner()
        with self._lock:
            self._finish_preflight_locked(epoch_id)

    def fault_allocation_epoch(self, epoch_id):
        """Permanent fence, retaining all physical bookkeeping and cleanup paths."""
        self._owner()
        with self._lock:
            self._allocation_faulted = True
            if self._frozen_epoch is None:
                self._frozen_epoch = epoch_id
            self._installed_epoch = None
            self._owner_revision += 1

    def _assert_reset_safe_locked(self):
        if (
            self._active
            or self._frozen_epoch is not None
            or self._local_owned_count
            or self._retired_pages
        ):
            raise UnsafeReset(
                "reset requires all pages reclaimed and no active epoch/leases"
            )

    def assert_reset_safe(self):
        """O(1), read-only lifetime preflight for tree/row metadata destruction."""
        self._owner()
        with self._lock:
            self._assert_reset_safe_locked()

    def reset(self, new_pool_epoch: int):
        """Clear only an already reclaimed pool."""
        self._owner()
        with self._lock:
            self._assert_reset_safe_locked()
            if new_pool_epoch <= self.pool_epoch:
                raise ValueError("pool epoch must increase")
            self._pages.clear()
            self._held_pages.clear()
            self._retired_pages.clear()
            self._reclaim_candidates.clear()
            self._local_owned_count = 0
            self.pool_epoch = new_pool_epoch
            self._owner_revision += 1
            self._credit_remaining = 0
