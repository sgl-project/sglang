"""CPU ownership proofs for deferred draft-prefill staging.

Only a request's final transfer chunk reads the hidden staging and source
metadata. Each sender and that chunk share one fence object; a completion for
an earlier sender therefore cannot settle a newer use of its bootstrap room.
"""

from __future__ import annotations

import enum
import itertools
import logging
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

logger = logging.getLogger(__name__)


class DeferredDraftPrefillRetirementState(str, enum.Enum):
    FREE = "free"
    HOLD = "hold"
    QUARANTINE = "quarantine"


class DeferredDraftPrefillSourceFence:
    """Track one sender's final chunk, including timeouts with unknown drain.

    ``submit`` must run before queue publication. ``seal`` closes admission of
    later chunks, making never-submitted a positive safety proof. The worker
    calls ``settled`` only after skipping that chunk or completing all submitted
    calls and its cleanup. A timeout or exception that may leave a transfer
    alive must call ``poison``; a later worker finally block cannot undo it.
    """

    _generations = itertools.count(1)

    def __init__(self):
        self.generation = next(self._generations)
        self._lock = threading.Lock()
        self._submitted = False
        self._sealed = False
        self._settled = False
        self._poison_reason: str | None = None

    def _state_locked(self) -> DeferredDraftPrefillRetirementState:
        if self._poison_reason is not None:
            return DeferredDraftPrefillRetirementState.QUARANTINE
        if not self._sealed:
            return DeferredDraftPrefillRetirementState.HOLD
        if not self._submitted or self._settled:
            return DeferredDraftPrefillRetirementState.FREE
        return DeferredDraftPrefillRetirementState.HOLD

    @property
    def retirement_state(self) -> DeferredDraftPrefillRetirementState:
        with self._lock:
            return self._state_locked()

    @property
    def submitted(self) -> bool:
        with self._lock:
            return self._submitted

    @property
    def poison_reason(self) -> str | None:
        with self._lock:
            return self._poison_reason

    def submit(self) -> bool:
        with self._lock:
            if self._sealed or self._submitted or self._poison_reason is not None:
                return False
            self._submitted = True
            self._settled = False
            return True

    def seal(self) -> DeferredDraftPrefillRetirementState:
        with self._lock:
            self._sealed = True
            return self._state_locked()

    def settled(self) -> DeferredDraftPrefillRetirementState:
        with self._lock:
            self._settled = True
            return self._state_locked()

    def poison(
        self, reason: str = "Transfer drain could not be proven"
    ) -> DeferredDraftPrefillRetirementState:
        with self._lock:
            if self._poison_reason is None:
                self._poison_reason = reason
            return self._state_locked()


RetirementKey = tuple[str, int, str]


def deferred_draft_prefill_retirement_key(req) -> RetirementKey:
    room = req.bootstrap_room
    if not isinstance(room, int) or isinstance(room, bool):
        raise ValueError("Deferred draft source retirement needs a bootstrap room")
    if not isinstance(req.rid, str) or not req.rid:
        raise ValueError("Deferred draft source retirement needs a request RID")
    generation = getattr(req, "pd_deferred_draft_source_generation", None)
    if not isinstance(generation, str) or not generation:
        raise ValueError("Deferred draft source retirement needs an intake generation")
    return req.rid, room, generation


def combine_deferred_draft_prefill_retirement_states(
    states: Iterable[DeferredDraftPrefillRetirementState],
) -> DeferredDraftPrefillRetirementState:
    states = [DeferredDraftPrefillRetirementState(state) for state in states]
    if not states:
        raise ValueError("At least one retirement state is required")
    if DeferredDraftPrefillRetirementState.QUARANTINE in states:
        return DeferredDraftPrefillRetirementState.QUARANTINE
    if DeferredDraftPrefillRetirementState.HOLD in states:
        return DeferredDraftPrefillRetirementState.HOLD
    return DeferredDraftPrefillRetirementState.FREE


def merge_deferred_draft_prefill_retirements(
    left: Mapping[RetirementKey, DeferredDraftPrefillRetirementState],
    right: Mapping[RetirementKey, DeferredDraftPrefillRetirementState],
) -> dict[RetirementKey, DeferredDraftPrefillRetirementState]:
    """Intersect registered generations and preserve a peer's unsafe verdict."""
    return {
        key: combine_deferred_draft_prefill_retirement_states((state, right[key]))
        for key, state in left.items()
        if key in right
    }


class DeferredDraftPrefillReleaseRids(list):
    """Keep the existing RID list interface while carrying a PP release vote."""

    def __init__(self, rids=(), retirement_states=None):
        super().__init__(rids)
        self.retirement_states = dict(retirement_states or {})


@dataclass(frozen=True)
class _DeferredDraftPrefillReleaseEnvelope:
    payload: DeferredDraftPrefillReleaseRids

    def __len__(self) -> int:
        # Generic PP/TP pyobj transport encodes len(data)==0 as bare []. A
        # nonempty wire envelope preserves votes after the RID list drains.
        return 1


def pack_deferred_draft_prefill_release(data):
    if isinstance(data, DeferredDraftPrefillReleaseRids):
        return _DeferredDraftPrefillReleaseEnvelope(data)
    return data


def unpack_deferred_draft_prefill_release(data):
    if isinstance(data, _DeferredDraftPrefillReleaseEnvelope):
        return data.payload
    return data


@dataclass
class _SourceRetirement:
    req: Any
    pool: Any
    slot: int
    metadata_index: int
    fence: DeferredDraftPrefillSourceFence


class DeferredDraftPrefillSourceRetirements:
    """Hold metadata and hidden ownership until a local or PP release decision.

    PP callers pass the fully merged generation map to ``resolve``. They must
    not resolve their local snapshot independently. PP1 may use ``resolve()``.
    A quarantine removes request ownership but never returns either resource to
    its allocator. Counts therefore remain aligned for source admission.

    The intake generation is shared across PP ranks, so a delayed vote cannot
    retire a newer request even when both its RID and bootstrap room are reused.
    """

    def __init__(self, allocator):
        self.allocator = allocator
        self._entries: dict[RetirementKey, _SourceRetirement] = {}
        self._quarantined_metadata: dict[int, RetirementKey] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def quarantined_metadata_slots(self) -> int:
        return len(self._quarantined_metadata)

    def register(self, req) -> bool:
        pool = getattr(req, "deferred_draft_prefill_pool", None)
        if pool is None or not pool.owns(req):
            return False
        key = deferred_draft_prefill_retirement_key(req)
        if key in self._entries:
            if self._entries[key].req is not req:
                raise ValueError(f"Another request already owns retirement {key}")
            return True
        if req.metadata_buffer_index < 0:
            raise RuntimeError("Hidden source retirement lost its metadata ownership")
        sender = getattr(req, "disagg_kv_sender", None)
        fence = getattr(sender, "deferred_draft_prefill_fence", None)
        if fence is None:
            # Missing provenance is not evidence that no transfer was submitted.
            fence = DeferredDraftPrefillSourceFence()
            fence.poison("Source sender has no deferred draft transfer fence")
        fence.seal()
        self._entries[key] = _SourceRetirement(
            req, pool, pool.get_slot(req), req.metadata_buffer_index, fence
        )
        return True

    def snapshot(self) -> dict[RetirementKey, DeferredDraftPrefillRetirementState]:
        return {
            key: entry.fence.retirement_state for key, entry in self._entries.items()
        }

    def resolve(
        self,
        states: Mapping[RetirementKey, DeferredDraftPrefillRetirementState]
        | None = None,
    ) -> list[RetirementKey]:
        states = self.snapshot() if states is None else states
        retired = []
        for key, state in states.items():
            entry = self._entries.get(key)
            state = DeferredDraftPrefillRetirementState(state)
            if entry is None or state == DeferredDraftPrefillRetirementState.HOLD:
                continue
            req, pool = entry.req, entry.pool
            if (
                deferred_draft_prefill_retirement_key(req) != key
                or req.metadata_buffer_index != entry.metadata_index
                or not pool.owns(req, entry.slot)
            ):
                raise RuntimeError(f"Deferred draft source ownership changed for {key}")
            if state == DeferredDraftPrefillRetirementState.FREE:
                if entry.fence.retirement_state != state:
                    raise RuntimeError(
                        f"Unsafe deferred draft source release for {key}"
                    )
                pool.free_source(req)
                self.allocator.free(entry.metadata_index)
            else:
                reason = (
                    entry.fence.poison_reason
                    or "A pipeline peer could not prove its source transfer drained"
                )
                pool.quarantine(req, reason=reason)
                self._quarantined_metadata[entry.metadata_index] = key
                logger.warning(
                    "Quarantined deferred draft-prefill metadata slot %d for %s",
                    entry.metadata_index,
                    key,
                )
            req.metadata_buffer_index = -1
            del self._entries[key]
            retired.append(key)
        return retired
