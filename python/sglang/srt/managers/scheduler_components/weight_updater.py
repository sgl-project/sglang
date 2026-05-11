from __future__ import annotations  # noqa: F401

from typing import Callable  # noqa: F401


class SchedulerWeightUpdaterManager:
    """Hot weight-update / memory-occupation / model-save / weight-inspection
    control surface. Composition target on Scheduler
    (``self.weight_updater``)."""

    def __init__(
        self,
        *,
        tp_worker,
        draft_worker,
        tp_cpu_group,
        memory_saver_adapter,
        flush_cache: Callable[..., bool],
        is_fully_idle: Callable[..., bool],
    ) -> None:
        self.tp_worker = tp_worker
        self.draft_worker = draft_worker
        self.tp_cpu_group = tp_cpu_group
        self.memory_saver_adapter = memory_saver_adapter
        self.flush_cache = flush_cache
        self.is_fully_idle = is_fully_idle
        self.offload_tags: set = set()
