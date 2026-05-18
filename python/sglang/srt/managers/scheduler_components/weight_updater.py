from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(kw_only=True, slots=True)
class SchedulerWeightUpdaterManager:
    tp_worker: Any
    draft_worker: Any
    tp_cpu_group: Any
    memory_saver_adapter: Any
    flush_cache: Callable[..., bool]
    is_fully_idle: Callable[..., bool]
    offload_tags: set = field(default_factory=set)
    stashed_model_static_state: Any = None
