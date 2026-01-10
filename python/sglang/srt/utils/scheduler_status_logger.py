from __future__ import annotations

import time
from typing import List, Optional

from sglang.srt.utils.log_utils import create_log_target, log_json


class SchedulerStatusLogger:
    DUMP_INTERVAL_S = 60.0

    def __init__(self, targets: List[str]):
        self.loggers = [create_log_target(t) for t in targets]
        self.last_dump_time = 0.0

    def maybe_dump(self, running_rids: List[str], queued_rids: List[str]) -> None:
        now = time.time()
        if now - self.last_dump_time < self.DUMP_INTERVAL_S:
            return
        self.last_dump_time = now
        log_json(
            self.loggers,
            "scheduler.status",
            {
                "running_rids": running_rids,
                "queued_rids": queued_rids,
            },
        )

    @staticmethod
    def maybe_create() -> Optional["SchedulerStatusLogger"]:
        from sglang.srt.environ import envs

        raw = envs.SGLANG_LOG_SCHEDULER_STATUS_TARGET.get()
        if not raw:
            return None
        targets = [t.strip() for t in raw.split(",") if t.strip()]
        if not targets:
            return None
        return SchedulerStatusLogger(targets)
