# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Eager speculative KV lifetime, independent of CUDA and indexer allocation.

The copy backend owns host allocation and submission. Its ``copy`` must return
a CompletionFence, or, on submission failure, synchronize any partial work
before raising. ``publish`` is atomic; ``release`` frees only unpublished host
slots. Physical allocator methods never allocate/free logical or indexer pages.
All calls are serialized by the coordinator. Readers register CompletionFence
before retirement and arm it with record() after their final stream operation.
A publish exception must leave host metadata unchanged; poll can retry or cancel.
"""

from dataclasses import dataclass, field
from typing import Protocol

from sglang.srt.mem_cache.hisparse_spec_state import ProvisionalArena, SpecTxnKey


class CompletionFence:
    """Unrecorded GPU events are never mistaken for completed operations."""

    def __init__(self, event):
        self._event = event
        self._armed = False

    def record(self, *args, **kwargs):
        if self._armed:
            raise ValueError("completion fence already recorded")
        self._event.record(*args, **kwargs)
        self._armed = True

    def query(self):
        return self._armed and self._event.query()


@dataclass(frozen=True)
class CommitPlan:
    key: SpecTxnKey
    positions: tuple[int, ...]
    logical_ids: tuple[int, ...]
    device_ids: tuple[int, ...]
    host_ids: tuple[int, ...]


@dataclass(frozen=True)
class HostReservation:
    """Unpublished host page allocation; reused tail pages are not owned here.

    new_page_rows retains *all* rows of newly allocated host pages, including
    padding. publish transfers these pages to request ownership and installs
    their full mapping; only positions/host_ids become KV-valid. release frees
    new_page_rows only, never host_ids (which can share a committed tail page).
    Allocate against a private mapping snapshot so failure cannot change caller
    req_to_host_pool or allocated_len. A backend may retain additional request
    metadata keyed by key, but must release it with this reservation.
    """

    key: SpecTxnKey
    positions: tuple[int, ...]
    host_ids: tuple[int, ...]
    new_page_rows: tuple[int, ...]


class CommitBackend(Protocol):
    def allocate(
        self, key: SpecTxnKey, positions: tuple[int, ...]
    ) -> HostReservation | None: ...
    def copy(self, plan: CommitPlan) -> CompletionFence: ...
    def publish(self, plan: CommitPlan, reservation: HostReservation) -> None: ...
    def release(self, reservation: HostReservation) -> None: ...


@dataclass
class _Transaction:
    arena: ProvisionalArena
    logical_ids: tuple[int, ...]
    old_len: int
    fences: list[CompletionFence] = field(default_factory=list)
    plan: CommitPlan | None = None
    backup: CompletionFence | None = None
    reservation: HostReservation | None = None
    published: bool = False
    cancelled: bool = False
    retiring: bool = False
    verified: bool = False
    quarantined: bool = False


class SpeculativeKVLifecycle:
    """Coordinator-owned transactions; stale callbacks never touch new owners.

    begin() must run even when scheduler logical growth is zero. Pass only the
    real verifier input IDs, in root/chain order, and the independent maximum R.
    commit's accept_len includes the root (accepted drafts + one), not fresh
    bonus output KV. release/cancel poll fences and return False while busy.
    """

    def __init__(self, allocator, backend: CommitBackend):
        self.allocator = allocator
        self.backend = backend
        self._active = {}
        self._latest = {}

    def _get(self, key):
        txn = self._active.get(key.request_slot)
        if txn is None or txn.arena.key != key:
            raise ValueError("stale or retired speculative transaction")
        return txn

    def begin(self, key, old_kv_len, logical_write_ids, reserved_rows):
        if type(key) is not SpecTxnKey or type(old_kv_len) is not int or old_kv_len < 0:
            raise ValueError("invalid transaction identity or prefix")
        if key.request_slot in self._active:
            raise ValueError("request still owns a live transaction")
        previous = self._latest.get(key.request_slot)
        stamp = (key.request_generation, key.iteration_id)
        if previous is not None and stamp <= previous:
            raise ValueError("stale transaction generation/iteration")
        logical = tuple(logical_write_ids)
        arena = self.allocator.ensure_provisional_mapping(
            key, old_kv_len, logical, reserved_rows
        )
        self._active[key.request_slot] = _Transaction(arena, logical, old_kv_len)
        self._latest[key.request_slot] = stamp
        return arena

    def add_reader(self, key, fence: CompletionFence):
        txn = self._get(key)
        if txn.retiring:
            raise ValueError("cannot register reader after retirement starts")
        if not isinstance(fence, CompletionFence):
            raise ValueError("reader requires explicitly recorded CompletionFence")
        txn.fences.append(fence)

    def verified(self, key):
        txn = self._get(key)
        if txn.retiring:
            raise ValueError("transaction retiring")
        txn.verified = True

    def commit(self, key, accept_len):
        txn = self._get(key)
        if txn.cancelled or txn.retiring or not txn.verified:
            raise ValueError("transaction is not ready for commit")
        if type(accept_len) is not int or not 1 <= accept_len <= len(txn.logical_ids):
            raise ValueError("accept_len must include root and only verified inputs")
        if txn.plan is not None:
            if len(txn.plan.positions) != accept_len:
                raise ValueError("acceptance changed after copy submission")
            return txn.plan
        self.allocator.validate_provisional_mapping(txn.arena, txn.logical_ids)
        positions = tuple(range(txn.old_len, txn.old_len + accept_len))
        try:
            reservation = self.backend.allocate(key, positions)
        except Exception:
            self.cancel(key)
            raise
        if reservation is None:
            self.cancel(key)
            raise MemoryError("accepted KV host allocation failed")
        if not isinstance(reservation, HostReservation) or reservation.key != key:
            # A mismatched reservation may belong to another live request.
            # Never call its release callback on this transaction's behalf.
            self.cancel(key)
            raise ValueError("host reservation belongs to another transaction")
        host = reservation.host_ids
        try:
            if reservation.positions != positions:
                raise ValueError("host reservation does not cover accepted positions")
            if (
                len(host) != accept_len
                or len(set(host)) != accept_len
                or any(type(i) is not int or i < 0 for i in host)
            ):
                raise ValueError("invalid or aliased accepted host destinations")
            plan = CommitPlan(
                key,
                tuple(range(txn.old_len, txn.old_len + accept_len)),
                txn.logical_ids[:accept_len],
                tuple(slot for _, slot in txn.arena.position_slots[:accept_len]),
                host,
            )
            backup = self.backend.copy(plan)
        except Exception:
            self.backend.release(reservation)
            self.cancel(key)
            raise
        txn.plan, txn.reservation = plan, reservation
        if not isinstance(backup, CompletionFence):
            # Submission returned without a usable completion proof. Retain both
            # host and device ownership even if the caller immediately cancels.
            txn.quarantined = txn.cancelled = txn.retiring = True
            raise TypeError(
                "copy backend returned no CompletionFence; ownership quarantined"
            )
        txn.backup = backup
        return plan

    def resolve_quarantine(self, key, drain_fence: CompletionFence):
        """Attach a fence recorded after ALL possibly submitted copy operations.

        This only permits cancellation cleanup; a malformed submission is never
        published. An unarmed replacement remains incomplete until recorded.
        """
        txn = self._get(key)
        if not txn.quarantined or not isinstance(drain_fence, CompletionFence):
            raise ValueError("quarantined copy requires an explicit drain fence")
        txn.backup = drain_fence
        txn.quarantined = False
        return self.poll(key)

    def poll(self, key):
        txn = self._get(key)
        if txn.quarantined:
            return False
        if txn.backup is not None and not txn.backup.query():
            return False
        if txn.plan is not None and not txn.published and not txn.cancelled:
            self.backend.publish(txn.plan, txn.reservation)
            txn.published = True
        if not txn.retiring or not all(event.query() for event in txn.fences):
            return False
        if txn.plan is not None and not txn.published:
            self.backend.release(txn.reservation)
            txn.plan = None
        self.allocator.retire_provisional_mapping(txn.arena, txn.logical_ids)
        del self._active[key.request_slot]
        return True

    def release(self, key):
        txn = self._get(key)
        if txn.plan is None and not txn.cancelled:
            raise ValueError("cannot release uncommitted transaction; cancel instead")
        txn.retiring = True
        return self.poll(key)

    def cancel(self, key):
        txn = self._get(key)
        txn.cancelled = True
        txn.retiring = True
        return self.poll(key)
