from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import msgspec

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch


class SchedulerBatches(msgspec.Struct):
    # The current forward batch
    cur_mbs: List[Optional[ScheduleBatch]]
    # The last forward batch
    last_mbs: List[Optional[ScheduleBatch]]
    # The running decoding batch for continuous batching
    running_mbs: List[ScheduleBatch]
    active_mb_id: int = 0

    @property
    def cur(self) -> Optional[ScheduleBatch]:
        return self.cur_mbs[self.active_mb_id]

    @cur.setter
    def cur(self, value: Optional[ScheduleBatch]) -> None:
        self.cur_mbs[self.active_mb_id] = value

    @property
    def last(self) -> Optional[ScheduleBatch]:
        return self.last_mbs[self.active_mb_id]

    @last.setter
    def last(self, value: Optional[ScheduleBatch]) -> None:
        self.last_mbs[self.active_mb_id] = value

    @property
    def running(self) -> ScheduleBatch:
        return self.running_mbs[self.active_mb_id]

    @running.setter
    def running(self, value: ScheduleBatch) -> None:
        self.running_mbs[self.active_mb_id] = value
