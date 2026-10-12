"""Launched batches whose results the scheduler has not committed yet.

An event loop pushes one record per launched batch and commits records oldest
first. A loop of depth ``d`` commits a record once ``d`` loop steps have
completed after the step that launched it: depth 0 commits in the launching
step, depth 1 (overlap) in the next one. So at most ``d`` records are in flight
when a step starts planning, and ``d + 1`` right after its launch.
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
    # taken after the launch otherwise.
    batch: ScheduleBatch
    result: GenerationBatchResult | EmbeddingBatchResult
    # The queue's completed step count when the batch was launched.
    step: int


class InflightQueue:
    def __init__(self, depth: int, blocking: bool = True):
        self.depth = depth
        # A non-blocking loop may defer committing a due record until its
        # result is ready; the loop tests readiness itself.
        self.blocking = blocking
        # Completed loop steps; only end_step advances it.
        self.step = 0
        # The live batch the last completed step launched, or None.
        self.last_launched: Optional[ScheduleBatch] = None
        self._records: Deque[StepRecord] = deque()

    def can_launch(self) -> bool:
        return len(self._records) <= self.depth

    def push(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult | EmbeddingBatchResult,
    ) -> None:
        self._records.append(StepRecord(batch=batch, result=result, step=self.step))

    def has_due(self) -> bool:
        return (
            len(self._records) > 0 and self._records[0].step <= self.step - self.depth
        )

    def pop_due(self) -> StepRecord:
        assert self.has_due()
        return self._records.popleft()

    def drain(self) -> List[StepRecord]:
        """Remove every record, oldest first, regardless of age."""
        records = list(self._records)
        self._records.clear()
        self.last_launched = None
        return records

    def head(self) -> StepRecord:
        return self._records[0]

    def newest(self) -> Optional[StepRecord]:
        return self._records[-1] if self._records else None

    def end_step(self, launched: Optional[ScheduleBatch]) -> None:
        self.step += 1
        self.last_launched = launched

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self) -> Iterator[StepRecord]:
        return iter(self._records)
