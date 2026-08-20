from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Iterable, Protocol

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool


@dataclass(frozen=True, slots=True)
class ReqSlotLease:
    """Identity of one request-pool row across reuse of the same index."""

    index: int
    generation: int

    @classmethod
    def capture(cls, req_to_token_pool: ReqToTokenPool, req: Req) -> ReqSlotLease:
        index = req.req_pool_idx
        assert index is not None, "cannot lease an unallocated request"
        return cls(
            index=int(index),
            generation=int(req_to_token_pool.req_generation[index].item()),
        )

    def validate(self, req_to_token_pool: ReqToTokenPool, req: Req) -> None:
        assert req.req_pool_idx == self.index, (
            "request-pool row changed while its retirement was deferred: "
            f"expected={self.index}, actual={req.req_pool_idx}"
        )
        actual_generation = int(req_to_token_pool.req_generation[self.index].item())
        assert actual_generation == self.generation, (
            "request-pool row was reused while its retirement was deferred: "
            f"index={self.index}, expected_generation={self.generation}, "
            f"actual_generation={actual_generation}"
        )


@dataclass(frozen=True, slots=True)
class DeferredReqRetirement:
    req: Req
    lease: ReqSlotLease
    is_insert: bool


class ForwardDoneEvent(Protocol):
    """Host-visible completion marker for one submitted forward."""

    def synchronize(self) -> None: ...


@dataclass(slots=True)
class ForwardResourceEpoch:
    """Resources protected by one submitted forward."""

    full_done_event: ForwardDoneEvent
    protected_slots: frozenset[ReqSlotLease]
    retirements: list[DeferredReqRetirement] = field(default_factory=list)
    completed: bool = False


class ForwardResourceLease:
    """Fence scheduler mutations at the resource, rather than loop, boundary.

    A launched overlap forward leases the request-pool rows it reads. Result
    handling may retire a request from the preceding result while that request
    is still present in the launched forward. Mapping writes are ordered after
    the forward's early read-done marker, while whole request/KV retirements are
    quarantined until the forward-completion marker. This distinction matters
    because host-driven transfer engines do not inherit CUDA stream ordering.
    """

    def __init__(
        self,
        *,
        req_to_token_pool: ReqToTokenPool,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator,
        wait_for_read_done: Callable[[], None],
        release_finished_req: Callable[..., None],
    ) -> None:
        self._req_to_token_pool = req_to_token_pool
        self._token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self._wait_for_read_done = wait_for_read_done
        self._release_finished_req = release_finished_req
        self._read_pending = False
        self._active_epoch: ForwardResourceEpoch | None = None
        self._retirement_epochs: list[ForwardResourceEpoch] = []

    @property
    def read_pending(self) -> bool:
        return self._read_pending

    @property
    def num_pending_retirements(self) -> int:
        return sum(
            len(epoch.retirements)
            for epoch in self._retirement_epochs
            + ([self._active_epoch] if self._active_epoch is not None else [])
        )

    def arm_after_launch(
        self,
        reqs: Iterable[Req],
        *,
        full_done_event: ForwardDoneEvent,
    ) -> ForwardResourceEpoch:
        assert not self._read_pending, "previous forward read fence is still pending"
        if self._active_epoch is not None and self._active_epoch.retirements:
            self._retirement_epochs.append(self._active_epoch)
        self._active_epoch = ForwardResourceEpoch(
            full_done_event=full_done_event,
            protected_slots=frozenset(
                ReqSlotLease.capture(self._req_to_token_pool, req)
                for req in reqs
                if req.req_pool_idx is not None
            ),
        )
        self._read_pending = True
        return self._active_epoch

    def wait_read_done(self) -> None:
        if not self._read_pending:
            return
        self._wait_for_read_done()
        self._read_pending = False

    def try_defer_finished_req(self, req: Req, is_insert: bool) -> bool:
        """Defer a release only when the in-flight forward leased its row.

        A result request outside the active lease is conservatively fenced and
        returned to the caller for immediate retirement.
        """
        epoch = self._active_epoch
        if epoch is None or epoch.completed:
            return False
        if req.req_pool_idx is None:
            self.wait_read_done()
            return False

        lease = ReqSlotLease.capture(self._req_to_token_pool, req)
        if lease not in epoch.protected_slots:
            self.wait_read_done()
            return False

        assert all(
            item.lease != lease for item in epoch.retirements
        ), f"request-pool row {lease.index} was deferred more than once"
        epoch.retirements.append(
            DeferredReqRetirement(req=req, lease=lease, is_insert=is_insert)
        )
        return True

    def _epochs_with_retirements(self) -> list[ForwardResourceEpoch]:
        epochs = list(self._retirement_epochs)
        if self._active_epoch is not None and self._active_epoch.retirements:
            epochs.append(self._active_epoch)
        return epochs

    def _release_epochs(self, epochs: list[ForwardResourceEpoch]) -> None:
        if not epochs:
            return

        # Validate the whole transaction before returning any row/page to a
        # free list. If an invariant is broken, fail without a partial release
        # that could make retry or crash diagnostics ambiguous.
        for epoch in epochs:
            for item in epoch.retirements:
                item.lease.validate(self._req_to_token_pool, item.req)

        self._token_to_kv_pool_allocator.free_group_begin()
        try:
            for epoch in epochs:
                for item in epoch.retirements:
                    self._release_finished_req(
                        item.req,
                        is_insert=item.is_insert,
                    )
        finally:
            self._token_to_kv_pool_allocator.free_group_end()
        for epoch in epochs:
            epoch.retirements.clear()
        self._retirement_epochs = [
            epoch for epoch in self._retirement_epochs if epoch.retirements
        ]

    def mark_forward_completed(self, epoch: ForwardResourceEpoch) -> None:
        """Complete one logical epoch after its result copy has synchronized.

        Completion is driven by result-queue order instead of rank-local event
        queries. All TP ranks therefore mutate allocator state in the same
        logical iteration even if their GPU completion times differ slightly.
        """
        epoch.completed = True
        self._release_epochs(
            [item for item in self._epochs_with_retirements() if item.completed]
        )

    def wait_mapping_read_done(self) -> None:
        self.wait_read_done()

    def synchronize_all_and_drain(self) -> None:
        """Quiesce in-flight forwards before a control-plane mutation."""
        self.wait_read_done()
        epochs = self._epochs_with_retirements()

        # A control handler can mutate the active epoch even when it has no
        # deferred retirement yet, so synchronize it as well.
        events = [
            epoch.full_done_event
            for epoch in self._retirement_epochs
            if not epoch.completed
        ]
        if self._active_epoch is not None and not self._active_epoch.completed:
            events.append(self._active_epoch.full_done_event)
        for event in events:
            event.synchronize()
        for epoch in self._retirement_epochs:
            epoch.completed = True
        if self._active_epoch is not None:
            self._active_epoch.completed = True
        self._release_epochs(epochs)
