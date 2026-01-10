from __future__ import annotations

import time
from typing import List, Optional

from sglang.srt.utils.log_utils import create_log_target, log_json


class SchedulerStatusLogger:
    DUMP_INTERVAL_S = 60.0

    def __init__(self, target: str):
        self.logger = create_log_target(target)
        self.last_dump_time = 0.0

    def maybe_dump(self, running_rids: List[str], queued_rids: List[str]) -> None:
        now = time.time()
        if now - self.last_dump_time < self.DUMP_INTERVAL_S:
            return
        self.last_dump_time = now
        log_json(
            self.logger,
            "scheduler.status",
            {
                "running_rids": running_rids,
                "queued_rids": queued_rids,
            },
        )

    @staticmethod
    def create_if_enabled() -> Optional["SchedulerStatusLogger"]:
        from sglang.srt.environ import envs

        target = envs.SGLANG_LOG_SCHEDULER_STATUS_TARGET.get()
        if not target:
            return None
        return SchedulerStatusLogger(target)
