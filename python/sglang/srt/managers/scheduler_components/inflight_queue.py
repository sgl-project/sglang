"""Launched batches whose results the scheduler has not committed yet.

An event loop pushes one record per launched batch and commits records oldest
first. A loop of depth ``d`` commits a record once ``d`` loop steps have
completed after the step that launched it: depth 0 commits in the launching
step, depth 1 (overlap) in the next one. So at most ``d`` records are in flight
when a step starts planning, and ``d + 1`` right after its launch.

A ring loop pushes one record per step, an empty record (batch None) for a
step that launches nothing, so the record of step ``s`` sits in slot
``s % (depth + 1)``. Each slot shows its newest record until the slot is
pushed again, committed or not.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Deque, Iterator, List, Optional

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.utils import EmbeddingBatchResult, GenerationBatchResult


@dataclass(eq=False)
class StepRecord:
    # The batch the commit processes: the launched batch at depth 0, a copy
    # taken after the launch otherwise. None in an empty ring record.
    batch: Optional[ScheduleBatch]
    # None in a ring record: its result arrives over the pipeline relay.
    result: Optional[GenerationBatchResult | EmbeddingBatchResult]
    # The queue's completed step count when the batch was launched.
    step: int


class InflightQueue:
    def __init__(self, depth: int, blocking: bool = True, ring: bool = False):
        self.depth = depth
        # Read by the PD prefill loop: when False, it defers committing a due
        # record until its result is ready, testing readiness itself.
        self.blocking = blocking
        # Completed loop steps; only end_step advances it.
        self.step = 0
        self._records: Deque[StepRecord] = deque()
        # A ring keeps its last committed record, which still shows in its
        # slot until the next push to that slot.
        self.ring = ring
        self._last_committed: Optional[StepRecord] = None

    def can_launch(self) -> bool:
        return len(self._records) <= self.depth

    def push(
        self,
        batch: Optional[ScheduleBatch],
        result: Optional[GenerationBatchResult | EmbeddingBatchResult],
    ) -> None:
        self._records.append(StepRecord(batch=batch, result=result, step=self.step))

    def has_due(self) -> bool:
        return (
            len(self._records) > 0 and self._records[0].step <= self.step - self.depth
        )

    def pop_due(self) -> StepRecord:
        assert self.has_due()
        record = self._records.popleft()
        if self.ring:
            self._last_committed = record
        return record

    def drain(self) -> List[StepRecord]:
        """Remove every record, oldest first, regardless of age."""
        records = list(self._records)
        self._records.clear()
        self._last_committed = None
        return records

    def head(self) -> StepRecord:
        return self._records[0]

    def newest(self) -> Optional[StepRecord]:
        return self._records[-1] if self._records else None

    def slot_batches(self) -> List[Optional[ScheduleBatch]]:
        """The batch of each ring slot's newest record; None for an empty
        record or a slot not pushed yet."""
        assert self.ring
        slots: List[Optional[ScheduleBatch]] = [None] * (self.depth + 1)
        records = list(self._records)
        if self._last_committed is not None:
            records.insert(0, self._last_committed)
        for record in records:
            slots[record.step % len(slots)] = record.batch
        return slots

    def end_step(self) -> None:
        self.step += 1

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[StepRecord]:
        return iter(self._records)
