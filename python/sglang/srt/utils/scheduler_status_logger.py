from __future__ import annotations

import time
from typing import TYPE_CHECKING, List, Optional

import torch.distributed as dist

from sglang.srt.environ import envs
from sglang.srt.utils.log_utils import create_log_targets, log_json

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch


class SchedulerStatusLogger:
    def __init__(self, targets: List[str], dump_interval: float):
        self.loggers = create_log_targets(targets=targets, name_prefix=__name__)
        self.dump_interval = dump_interval
        self.last_dump_time = 0.0
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    @staticmethod
    def maybe_create() -> Optional["SchedulerStatusLogger"]:
        target = envs.SGLANG_LOG_SCHEDULER_STATUS_TARGET.get()
        if not target:
            return None

        return SchedulerStatusLogger(
            targets=[t.strip() for t in target.split(",") if t.strip()],
            dump_interval=envs.SGLANG_LOG_SCHEDULER_STATUS_INTERVAL.get(),
        )

    def maybe_dump(
        self, running_batch: "ScheduleBatch", waiting_queue: List["Req"]
    ) -> None:
        now = time.time()
        if now - self.last_dump_time < self.dump_interval:
            return

        self.last_dump_time = now
        log_json(
            self.loggers,
            "scheduler.status",
            {
                "rank": self.rank,
                "running_rids": [r.rid for r in running_batch.reqs],
                "queued_rids": [r.rid for r in waiting_queue],
            },
        )
