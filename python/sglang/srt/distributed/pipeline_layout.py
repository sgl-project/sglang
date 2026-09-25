from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class PipelineStage:
    stage_id: int
    physical_rank: int
    layer_ids: Tuple[int, ...]


@dataclass(frozen=True)
class PipelineWavefrontAction:
    tick: int
    batch_seq: int
    slot_id: int
    stage_id: int
    physical_rank: int


class PipelineControlKind(str, Enum):
    REQUEST = "request"
    BOOTSTRAP_STATUS = "bootstrap_status"
    TRANSFER_STATUS = "transfer_status"
    ADMIT = "admit"
    CANCEL = "cancel"
    FIRST_PASS_DONE = "first_pass_done"
    COMPLETION = "completion"
    RESOURCE = "resource"
    PREFIX_MATERIALIZED = "prefix_materialized"
    PREFIX_COMMIT = "prefix_commit"
    CACHE_UPDATE = "cache_update"
    EVICT_ACK = "evict_ack"


@dataclass(frozen=True)
class PipelineControlEnvelope:
    protocol_version: int
    runtime_epoch: int
    layout_digest: str
    kind: PipelineControlKind
    source_rank: int
    batch_seq: int = -1
    generation: int = -1
    slot_id: int = -1
    hops: int = 0
    payload: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "__msg_type__": "vpp_control",
            "protocol_version": self.protocol_version,
            "runtime_epoch": self.runtime_epoch,
            "layout_digest": self.layout_digest,
            "kind": self.kind.value,
            "source_rank": self.source_rank,
            "batch_seq": self.batch_seq,
            "generation": self.generation,
            "slot_id": self.slot_id,
            "hops": self.hops,
            "payload": self.payload or {},
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> PipelineControlEnvelope:
        if value.get("__msg_type__") != "vpp_control":
            raise ValueError("not a VPP control envelope")
        try:
            return cls(
                protocol_version=int(value["protocol_version"]),
                runtime_epoch=int(value["runtime_epoch"]),
                layout_digest=str(value["layout_digest"]),
                kind=PipelineControlKind(value["kind"]),
                source_rank=int(value["source_rank"]),
                batch_seq=int(value.get("batch_seq", -1)),
                generation=int(value.get("generation", -1)),
                slot_id=int(value.get("slot_id", -1)),
                hops=int(value.get("hops", 0)),
                payload=dict(value.get("payload") or {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid VPP control envelope: {exc}") from exc

    def forwarded(self) -> PipelineControlEnvelope:
        return replace(self, hops=self.hops + 1)


@dataclass(frozen=True)
class PipelineRankTransition:
    batch_seq: int
    stage_id: int
    successor_stage_id: Optional[int]
    first_pass_done: bool
    batch_complete: bool
    cancelled: bool


class PipelineRankSchedule:
    def __init__(
        self,
        physical_rank: int,
        physical_size: int,
        virtual_stages: int,
        max_inflight: int,
        prefill_burst_size: int = 1,
    ):
        if not 0 <= physical_rank < physical_size:
            raise ValueError("physical rank is outside pipeline")
        if virtual_stages != 2:
            raise ValueError("rank-local scheduler supports VPP2 only")
        if max_inflight < 1:
            raise ValueError("max inflight batches must be positive")
        if prefill_burst_size < 1:
            raise ValueError("prefill burst size must be positive")
        self.physical_rank = physical_rank
        self.physical_size = physical_size
        self.virtual_stages = virtual_stages
        self.max_inflight = max_inflight
        self.prefill_burst_size = prefill_burst_size
        self.logical_size = physical_size * virtual_stages
        self._slots: list[Optional[int]] = [None] * max_inflight
        self._ready: list[tuple[int, int]] = []
        self._running: Optional[PipelineWavefrontAction] = None
        self._completed: set[int] = set()
        self._cancelled: set[int] = set()
        self._burst_first_pass = True
        self._burst_progress = {True: 0, False: 0}

    @property
    def slot_batch_seqs(self) -> Tuple[Optional[int], ...]:
        return tuple(self._slots)

    @property
    def inflight_count(self) -> int:
        return sum(item is not None for item in self._slots)

    @property
    def running(self) -> Optional[PipelineWavefrontAction]:
        return self._running

    @property
    def ready_tasks(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(self._ready)

    def owns_stage(self, stage_id: int) -> bool:
        return (
            0 <= stage_id < self.logical_size
            and stage_id % self.physical_size == self.physical_rank
        )

    def can_admit(self, batch_seq: int) -> bool:
        if batch_seq < 0 or self.inflight_count >= self.max_inflight:
            return False
        return self._slots[batch_seq % self.max_inflight] is None

    def admit(self, batch_seq: int) -> None:
        if not self.can_admit(batch_seq):
            raise RuntimeError(f"cannot admit VPP batch {batch_seq}")
        self._slots[batch_seq % self.max_inflight] = batch_seq
        if self.physical_rank == 0:
            self.mark_ready(batch_seq, 0)

    def mark_ready(self, batch_seq: int, stage_id: int) -> None:
        if not self.owns_stage(stage_id):
            raise RuntimeError(
                f"PP rank {self.physical_rank} does not own stage {stage_id}"
            )
        slot_id = batch_seq % self.max_inflight
        if self._slots[slot_id] != batch_seq:
            raise RuntimeError(
                f"batch {batch_seq} does not own rank-local slot {slot_id}"
            )
        item = (batch_seq, stage_id)
        if item in self._ready or (
            self._running is not None
            and (self._running.batch_seq, self._running.stage_id) == item
        ):
            raise RuntimeError(f"duplicate rank-local ready task {item}")
        self._ready.append(item)

    def next_action(
        self,
        tick: int,
        is_ready: Optional[Callable[[int, int], bool]] = None,
    ) -> Optional[PipelineWavefrontAction]:
        if self._running is not None:
            return None
        available = [
            item
            for item in self._ready
            if is_ready is None or is_ready(item[0], item[1])
        ]
        if not available:
            return None
        if self.prefill_burst_size == 1:
            decode = [item for item in available if item[1] >= self.physical_size]
            selected = decode if decode else available
        else:
            preferred_stage = self.physical_rank + (
                0 if self._burst_first_pass else self.physical_size
            )
            preferred = [item for item in available if item[1] == preferred_stage]
            selected = preferred if preferred else available
        batch_seq, stage_id = min(selected)
        return self.dispatch(tick, batch_seq, stage_id)

    def dispatch(
        self,
        tick: int,
        batch_seq: int,
        stage_id: int,
    ) -> PipelineWavefrontAction:
        if self._running is not None:
            raise RuntimeError("rank-local scheduler already has a running task")
        if (batch_seq, stage_id) not in self._ready:
            raise RuntimeError(f"rank-local task {(batch_seq, stage_id)} is not ready")
        self._ready.remove((batch_seq, stage_id))
        self._running = PipelineWavefrontAction(
            tick=tick,
            batch_seq=batch_seq,
            slot_id=batch_seq % self.max_inflight,
            stage_id=stage_id,
            physical_rank=self.physical_rank,
        )
        return self._running

    def complete(self, action: PipelineWavefrontAction) -> PipelineRankTransition:
        if action != self._running:
            raise RuntimeError("rank-local completion does not match running task")
        self._running = None
        if self.prefill_burst_size > 1:
            is_first_pass = action.stage_id < self.physical_size
            self._burst_progress[is_first_pass] += 1
            while (
                self._burst_progress[self._burst_first_pass] >= self.prefill_burst_size
            ):
                self._burst_progress[self._burst_first_pass] -= self.prefill_burst_size
                self._burst_first_pass = not self._burst_first_pass
        successor = action.stage_id + 1
        batch_complete = successor == self.logical_size
        if batch_complete:
            successor = None
            self._completed.add(action.batch_seq)
        return PipelineRankTransition(
            batch_seq=action.batch_seq,
            stage_id=action.stage_id,
            successor_stage_id=successor,
            first_pass_done=action.stage_id == self.physical_rank,
            batch_complete=batch_complete,
            cancelled=action.batch_seq in self._cancelled,
        )

    def cancel(self, batch_seq: int) -> None:
        if batch_seq not in self._slots:
            return
        self._cancelled.add(batch_seq)

    def mark_completed(self, batch_seq: int) -> None:
        if batch_seq not in self._slots:
            raise RuntimeError(f"VPP batch {batch_seq} is not admitted")
        self._completed.add(batch_seq)

    def retire(self, batch_seq: int) -> None:
        if batch_seq not in self._completed:
            raise RuntimeError(f"VPP batch {batch_seq} is not complete")
        slot_id = batch_seq % self.max_inflight
        if self._slots[slot_id] != batch_seq:
            raise RuntimeError(f"VPP batch {batch_seq} does not own slot {slot_id}")
        self._completed.remove(batch_seq)
        self._cancelled.discard(batch_seq)
        self._slots[slot_id] = None


@dataclass(frozen=True)
class PipelineResourceSnapshot:
    request_slots: int
    kv_tokens: int
    activation_bytes: int
    pending_sends: int
    metadata_slots: int = 0


class PipelineResourceGate:
    def __init__(
        self,
        ranks: Iterable[int],
        activation_low_watermark: int,
        activation_high_watermark: int,
        max_pending_sends: int,
    ):
        if not 0 <= activation_low_watermark <= activation_high_watermark:
            raise ValueError("invalid activation watermarks")
        if max_pending_sends < 1:
            raise ValueError("max pending sends must be positive")
        self._ranks = tuple(sorted(set(ranks)))
        self._snapshots: Dict[int, PipelineResourceSnapshot] = {}
        self._blocked: set[int] = set()
        self.activation_low_watermark = activation_low_watermark
        self.activation_high_watermark = activation_high_watermark
        self.max_pending_sends = max_pending_sends

    def update(self, rank: int, snapshot: PipelineResourceSnapshot) -> None:
        if rank not in self._ranks:
            raise ValueError(f"unexpected resource rank {rank}")
        self._snapshots[rank] = snapshot
        if (
            snapshot.activation_bytes >= self.activation_high_watermark
            or snapshot.pending_sends >= self.max_pending_sends
        ):
            self._blocked.add(rank)
        elif (
            snapshot.activation_bytes <= self.activation_low_watermark
            and snapshot.pending_sends < self.max_pending_sends
        ):
            self._blocked.discard(rank)

    def can_admit(
        self,
        required_request_slots: int,
        required_kv_tokens: int,
        required_activation_bytes: int,
    ) -> bool:
        if len(self._snapshots) != len(self._ranks) or self._blocked:
            return False
        return all(
            snapshot.request_slots >= required_request_slots
            and snapshot.kv_tokens >= required_kv_tokens
            and snapshot.activation_bytes + required_activation_bytes
            <= self.activation_high_watermark
            for snapshot in self._snapshots.values()
        )

    def can_send(self, rank: int, activation_bytes: int) -> bool:
        snapshot = self._snapshots.get(rank)
        return (
            snapshot is not None
            and rank not in self._blocked
            and snapshot.pending_sends < self.max_pending_sends
            and snapshot.activation_bytes + activation_bytes
            <= self.activation_high_watermark
        )

    def can_bootstrap(self) -> bool:
        return len(self._snapshots) == len(self._ranks) and all(
            snapshot.metadata_slots > 0 for snapshot in self._snapshots.values()
        )


@dataclass
class PipelinePrefixFrontier:
    request_generation: int
    residency_generation: int
    required_stages: Tuple[int, ...]
    planned_end: int = 0
    committed_end: int = 0
    materialized_by_stage: Optional[Dict[int, int]] = None
    locked: bool = False

    def __post_init__(self):
        if self.materialized_by_stage is None:
            self.materialized_by_stage = {
                stage_id: self.committed_end for stage_id in self.required_stages
            }

    @property
    def materialized_end(self) -> int:
        return min(self.materialized_by_stage.values(), default=self.committed_end)


class PipelinePrefixRegistry:
    def __init__(self):
        self._entries: Dict[tuple[str, int], PipelinePrefixFrontier] = {}

    def get(
        self,
        rid: str,
        request_generation: int,
    ) -> Optional[PipelinePrefixFrontier]:
        return self._entries.get((rid, request_generation))

    def plan(
        self,
        rid: str,
        request_generation: int,
        residency_generation: int,
        start: int,
        end: int,
        required_stages: Iterable[int],
    ) -> PipelinePrefixFrontier:
        key = (rid, request_generation)
        entry = self._entries.get(key)
        stages = tuple(sorted(set(required_stages)))
        if entry is None:
            entry = PipelinePrefixFrontier(
                request_generation=request_generation,
                residency_generation=residency_generation,
                required_stages=stages,
                planned_end=start,
                committed_end=start,
            )
            self._entries[key] = entry
        if entry.residency_generation != residency_generation:
            raise RuntimeError("prefix residency generation changed")
        if entry.required_stages != stages:
            raise RuntimeError("prefix dependency stages changed")
        if start != entry.planned_end:
            raise RuntimeError(
                f"non-contiguous prefix plan: expected {entry.planned_end}, got {start}"
            )
        if end < start:
            raise ValueError("prefix end precedes start")
        entry.planned_end = end
        entry.locked = True
        return entry

    def mark_materialized(
        self,
        rid: str,
        request_generation: int,
        stage_id: int,
        end: int,
    ) -> int:
        entry = self._entries[(rid, request_generation)]
        if stage_id not in entry.materialized_by_stage:
            raise RuntimeError(f"stage {stage_id} is not a prefix dependency")
        current = entry.materialized_by_stage[stage_id]
        if not current <= end <= entry.planned_end:
            raise RuntimeError("invalid materialized prefix frontier")
        entry.materialized_by_stage[stage_id] = end
        return entry.materialized_end

    def commit(
        self,
        rid: str,
        request_generation: int,
        end: int,
    ) -> None:
        entry = self._entries[(rid, request_generation)]
        if end < entry.committed_end or end > entry.materialized_end:
            raise RuntimeError("prefix commit exceeds materialized frontier")
        entry.committed_end = end
        if end == entry.planned_end:
            entry.locked = False

    def invalidate_residency(
        self,
        rid: str,
        request_generation: int,
        residency_generation: int,
    ) -> None:
        entry = self._entries[(rid, request_generation)]
        if residency_generation <= entry.residency_generation:
            raise RuntimeError("residency generation must increase")
        entry.residency_generation = residency_generation
        entry.planned_end = entry.committed_end
        entry.materialized_by_stage = {
            stage_id: entry.committed_end for stage_id in entry.required_stages
        }
        entry.locked = False

    def release(self, rid: str, request_generation: int) -> None:
        self._entries.pop((rid, request_generation), None)


@dataclass(frozen=True)
class PipelineReplicaIdentity:
    content_id: str
    source_id: int
    format_version: int
    consumer_rank: int


@dataclass
class PipelineReplicaState:
    residency_generation: int
    valid_end: int = 0
    locked_until: int = 0
    lock_owners: Optional[Dict[int, int]] = None

    def __post_init__(self):
        if self.lock_owners is None:
            self.lock_owners = {}


class PipelineReplicaRegistry:
    def __init__(self):
        self._entries: Dict[PipelineReplicaIdentity, PipelineReplicaState] = {}

    def get(
        self,
        identity: PipelineReplicaIdentity,
    ) -> Optional[PipelineReplicaState]:
        return self._entries.get(identity)

    def register(
        self,
        identity: PipelineReplicaIdentity,
        residency_generation: int,
        valid_end: int = 0,
    ) -> PipelineReplicaState:
        current = self._entries.get(identity)
        if current is not None:
            if current.residency_generation != residency_generation:
                raise RuntimeError("replica residency generation mismatch")
            if valid_end < current.valid_end:
                raise RuntimeError("replica valid frontier regressed")
            current.valid_end = valid_end
            return current
        state = PipelineReplicaState(
            residency_generation=residency_generation,
            valid_end=valid_end,
        )
        self._entries[identity] = state
        return state

    def missing_range(
        self,
        identity: PipelineReplicaIdentity,
        residency_generation: int,
        required_end: int,
    ) -> Optional[Tuple[int, int]]:
        state = self._entries.get(identity)
        if state is None or state.residency_generation != residency_generation:
            return (0, required_end)
        if state.valid_end >= required_end:
            return None
        return (state.valid_end, required_end)

    def install(
        self,
        identity: PipelineReplicaIdentity,
        residency_generation: int,
        start: int,
        end: int,
    ) -> None:
        state = self._entries.get(identity)
        if state is None:
            if start != 0:
                raise RuntimeError("replica restore must start at zero")
            state = self.register(identity, residency_generation)
        if state.residency_generation != residency_generation:
            raise RuntimeError("stale replica installation")
        if start != state.valid_end or end < start:
            raise RuntimeError("replica installation is not contiguous")
        state.valid_end = end

    def lock(
        self,
        identity: PipelineReplicaIdentity,
        required_end: int,
        owner_id: int = -1,
    ) -> None:
        state = self._entries[identity]
        if state.valid_end < required_end:
            raise RuntimeError("cannot lock an incomplete replica")
        state.lock_owners[owner_id] = max(
            state.lock_owners.get(owner_id, 0),
            required_end,
        )
        state.locked_until = max(state.lock_owners.values(), default=0)

    def unlock(
        self,
        identity: PipelineReplicaIdentity,
        released_end: int,
        owner_id: int = -1,
    ) -> None:
        state = self._entries[identity]
        locked_end = state.lock_owners.get(owner_id)
        if locked_end is None:
            raise RuntimeError("replica owner does not hold a lock")
        if released_end < locked_end:
            raise RuntimeError("replica unlock does not cover active lock")
        state.lock_owners.pop(owner_id)
        state.locked_until = max(state.lock_owners.values(), default=0)

    def evict(
        self,
        identity: PipelineReplicaIdentity,
        residency_generation: int,
    ) -> int:
        state = self._entries[identity]
        if state.residency_generation != residency_generation:
            raise RuntimeError("stale replica eviction")
        if state.locked_until:
            raise RuntimeError("cannot evict a locked replica")
        next_generation = residency_generation + 1
        self._entries[identity] = PipelineReplicaState(
            residency_generation=next_generation,
        )
        return next_generation

    def common_boundary(
        self,
        identities: Iterable[PipelineReplicaIdentity],
        residency_generation: int,
    ) -> int:
        states = []
        for identity in identities:
            state = self._entries.get(identity)
            if state is None or state.residency_generation != residency_generation:
                return 0
            states.append(state.valid_end)
        return min(states, default=0)


@dataclass(frozen=True)
class PipelineLayout:
    num_hidden_layers: int
    physical_size: int
    virtual_stages: int
    stages: Tuple[PipelineStage, ...]

    @classmethod
    def build(
        cls,
        num_hidden_layers: int,
        physical_size: int,
        virtual_stages: int = 1,
        partition: Optional[Tuple[int, ...]] = None,
    ) -> PipelineLayout:
        if physical_size < 1 or virtual_stages < 1:
            raise ValueError("pipeline sizes must be positive")
        logical_size = physical_size * virtual_stages
        if num_hidden_layers < logical_size:
            raise ValueError(
                f"{num_hidden_layers=} must be >= logical pipeline size {logical_size}"
            )

        if partition is None:
            base, remainder = divmod(num_hidden_layers, logical_size)
            partition = tuple(
                base + (stage_id >= logical_size - remainder)
                for stage_id in range(logical_size)
            )
        if len(partition) != logical_size:
            raise ValueError(
                f"{len(partition)=} does not match logical pipeline size {logical_size}"
            )
        if sum(partition) != num_hidden_layers:
            raise ValueError(f"{sum(partition)=} does not match {num_hidden_layers=}")

        stages = []
        start = 0
        for stage_id, layer_count in enumerate(partition):
            end = start + layer_count
            stages.append(
                PipelineStage(
                    stage_id=stage_id,
                    physical_rank=stage_id % physical_size,
                    layer_ids=tuple(range(start, end)),
                )
            )
            start = end

        layout = cls(
            num_hidden_layers=num_hidden_layers,
            physical_size=physical_size,
            virtual_stages=virtual_stages,
            stages=tuple(stages),
        )
        layout.validate()
        return layout

    @property
    def logical_size(self) -> int:
        return len(self.stages)

    @property
    def is_interleaved(self) -> bool:
        return self.virtual_stages > 1

    @property
    def digest(self) -> str:
        manifest = [
            (stage.stage_id, stage.physical_rank, stage.layer_ids)
            for stage in self.stages
        ]
        return hashlib.sha256(
            json.dumps(manifest, separators=(",", ":")).encode()
        ).hexdigest()

    def stage(self, stage_id: int) -> PipelineStage:
        if not 0 <= stage_id < self.logical_size:
            raise ValueError(f"invalid logical pipeline stage {stage_id}")
        return self.stages[stage_id]

    def stages_for_rank(self, physical_rank: int) -> Tuple[PipelineStage, ...]:
        return tuple(
            stage for stage in self.stages if stage.physical_rank == physical_rank
        )

    def layer_ids_for_rank(self, physical_rank: int) -> Tuple[int, ...]:
        return tuple(
            layer_id
            for stage in self.stages_for_rank(physical_rank)
            for layer_id in stage.layer_ids
        )

    def validate(self) -> None:
        if len(self.stages) != self.physical_size * self.virtual_stages:
            raise ValueError("logical stage count does not match pipeline sizes")
        layer_ids = tuple(
            layer_id for stage in self.stages for layer_id in stage.layer_ids
        )
        if layer_ids != tuple(range(self.num_hidden_layers)):
            raise ValueError("logical stages must cover every layer exactly once")
        for stage in self.stages:
            if stage.physical_rank != stage.stage_id % self.physical_size:
                raise ValueError("interleaved stage owner does not match stage order")
