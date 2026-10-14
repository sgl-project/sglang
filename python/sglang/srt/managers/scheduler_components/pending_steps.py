"""Launched steps whose results are not committed yet.

Each event loop decides when it commits a pending step; this container only
keeps the records in launch order.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Deque,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Set,
    Union,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler_components.mode_stages import ScheduleSlot
    from sglang.srt.managers.utils import EmbeddingBatchResult, GenerationBatchResult


@dataclass(frozen=True, slots=True)
class PendingStep:
    step: ScheduleBatch
    result: Union[GenerationBatchResult, EmbeddingBatchResult]
    # The slot it was scheduled for; its commit reopens that slot's admission.
    slot: ScheduleSlot
    # Launch stamps: when the step launched, and whether the scheduler
    # launched nothing in the iteration before.
    launch_ts: float
    after_idle_gap: bool


def _output_reqs(step: ScheduleBatch) -> List[Req]:
    """The requests a step's commit appends an output token to."""
    if step.forward_mode.is_idle():
        return []
    return [req for req in step.reqs if req is not step.chunked_req]


def _kv_commit_reqs(step: ScheduleBatch) -> List[Req]:
    """The requests a step's commit adds one committed KV token to."""
    if not step.spec_algorithm.is_none():
        return []
    return step.decode_reqs()


class PendingSteps:
    def __init__(self):
        self._records: Deque[PendingStep] = deque()
        # Per record, the requests its commit appends an output to, and those
        # it commits a KV token to; built on the first query, so a push or
        # commit costs nothing per request.
        self._output_sets: Deque[Optional[FrozenSet[Req]]] = deque()
        self._kv_sets: Deque[Optional[FrozenSet[Req]]] = deque()

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[PendingStep]:
        return iter(self._records)

    def push(self, record: PendingStep) -> None:
        self._records.append(record)
        self._output_sets.append(None)
        self._kv_sets.append(None)

    def popleft(self) -> PendingStep:
        self._output_sets.popleft()
        self._kv_sets.popleft()
        return self._records.popleft()

    def newest(self) -> Optional[PendingStep]:
        return self._records[-1] if self._records else None

    def num_inflight_outputs(self, req: Req) -> int:
        """The output tokens the pending steps will commit to req."""
        num = 0
        for i, record in enumerate(self._records):
            outputs = self._output_sets[i]
            if outputs is None:
                outputs = self._output_sets[i] = frozenset(_output_reqs(record.step))
            num += req in outputs
        return num

    def num_inflight_kv(self, req: Req) -> int:
        """The KV tokens the pending steps will add to req's kv_committed_len
        at their commits (commit_step_counters): one per decode row of a
        non-speculative step. A speculative verify adds its accepted run when
        it resolves, so it counts none here."""
        if req.finished() or req.is_retracted:
            return 0
        num = 0
        for i, record in enumerate(self._records):
            kv_reqs = self._kv_sets[i]
            if kv_reqs is None:
                kv_reqs = self._kv_sets[i] = frozenset(_kv_commit_reqs(record.step))
            num += req in kv_reqs
        return num

    def plans_mamba_track_write(self, req: Req, position: int) -> bool:
        """Whether a pending step's forward may write req's mamba ping-pong
        slot at position; the tree does not take such a slot from a finishing
        request."""
        for record in self._records:
            step = record.step
            for i, row_req in enumerate(step.reqs):
                if row_req is req and step.mamba_track_write_position(i) == position:
                    return True
        return False

    def lora_ids(self) -> Set[Optional[str]]:
        """The LoRA adapters the pending steps run; they stay loaded."""
        return {req.lora_id for record in self._records for req in record.step.reqs}
