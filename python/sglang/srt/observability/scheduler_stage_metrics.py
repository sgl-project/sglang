# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import TypeVar, cast

from sglang.srt.utils.nvtx_utils import (
    NVTX_SCHEDULER_ENABLED,
    profile_range,
    scheduler_nvtx_method,
)

SCHEDULER_STAGE_OTHER = "other"
SCHEDULER_STAGE_RECV_REQUESTS = "recv_requests"
SCHEDULER_STAGE_PROCESS_REQUESTS = "process_input_requests"
SCHEDULER_STAGE_GET_NEXT_BATCH = "get_next_batch_to_run"
SCHEDULER_STAGE_PROCESS_QUEUE = "process_queue"
SCHEDULER_STAGE_RUN_BATCH = "run_batch"
SCHEDULER_STAGE_PROCESS_BATCH_RESULT = "process_batch_result"
SCHEDULER_STAGE_SANITY_CHECK_CACHE = "sanity_check_cache"
SCHEDULER_STAGE_IDLE = "idle"

SCHEDULER_STAGE_CATEGORIES = (
    SCHEDULER_STAGE_OTHER,
    SCHEDULER_STAGE_RECV_REQUESTS,
    SCHEDULER_STAGE_PROCESS_REQUESTS,
    SCHEDULER_STAGE_GET_NEXT_BATCH,
    SCHEDULER_STAGE_PROCESS_QUEUE,
    SCHEDULER_STAGE_RUN_BATCH,
    SCHEDULER_STAGE_PROCESS_BATCH_RESULT,
    SCHEDULER_STAGE_SANITY_CHECK_CACHE,
    SCHEDULER_STAGE_IDLE,
)
_SCHEDULER_STAGE_CATEGORY_SET = frozenset(SCHEDULER_STAGE_CATEGORIES)


@dataclass(slots=True)
class SchedulerStageMetricsRecorder:
    """Accumulate mutually exclusive scheduler wall time by stage.

    Nested stages temporarily replace their parent, and uncategorized time is
    assigned to ``other``. Summing all categories therefore recovers elapsed
    scheduler-loop wall time. Active torch profilers receive matching
    ``scheduler.<stage>`` ranges without requiring Python stacks.
    """

    enabled: bool
    _current_stage: str = SCHEDULER_STAGE_OTHER
    _trace_stage: str | None = None
    _last_wall_ns: int | None = None
    _wall_ns: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def start(self, wall_ns: int) -> None:
        if not self.enabled:
            return
        self._current_stage = SCHEDULER_STAGE_OTHER
        self._last_wall_ns = wall_ns
        self._wall_ns.clear()

    def enter(self, stage: str) -> str | None:
        if (
            not self.enabled
            or self._last_wall_ns is None
            or self._current_stage == stage
        ):
            return None
        self._sample(time.monotonic_ns())
        previous_stage = self._current_stage
        self._current_stage = stage
        return previous_stage

    def exit(self, previous_stage: str | None) -> None:
        if previous_stage is None:
            return
        self._sample(time.monotonic_ns())
        self._current_stage = previous_stage

    @contextmanager
    def record(self, stage: str) -> Iterator[None]:
        if stage not in _SCHEDULER_STAGE_CATEGORY_SET:
            raise ValueError(f"Unknown scheduler stage: {stage}")
        previous_stage = self.enter(stage)
        previous_trace_stage = self._trace_stage
        trace_stage_changed = previous_trace_stage != stage
        if trace_stage_changed:
            self._trace_stage = stage
        try:
            if trace_stage_changed:
                with profile_range(
                    f"scheduler.{stage}", nvtx_enabled=NVTX_SCHEDULER_ENABLED
                ):
                    yield
            else:
                yield
        finally:
            if trace_stage_changed:
                self._trace_stage = previous_trace_stage
            self.exit(previous_stage)

    def drain(self, wall_ns: int) -> dict[str, int]:
        if not self.enabled or self._last_wall_ns is None:
            return {}
        self._sample(wall_ns)
        wall_ns_by_stage = dict(self._wall_ns)
        self._wall_ns.clear()
        return wall_ns_by_stage

    def _sample(self, wall_ns: int) -> None:
        assert self._last_wall_ns is not None
        self._wall_ns[self._current_stage] += wall_ns - self._last_wall_ns
        self._last_wall_ns = wall_ns


_F = TypeVar("_F", bound=Callable)


def scheduler_stage_method(stage: str) -> Callable[[_F], _F]:
    if stage not in _SCHEDULER_STAGE_CATEGORY_SET:
        raise ValueError(f"Unknown scheduler stage: {stage}")
    trace_name = f"scheduler.{stage}"

    def decorator(func: _F) -> _F:
        profiled_func = scheduler_nvtx_method(trace_name)(func)

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            recorder = self.scheduler_stage_metrics
            if recorder is None:
                return profiled_func(self, *args, **kwargs)

            previous_stage = recorder.enter(stage)
            previous_trace_stage = recorder._trace_stage
            trace_stage_changed = previous_trace_stage != stage
            if trace_stage_changed:
                recorder._trace_stage = stage
            try:
                return profiled_func(self, *args, **kwargs)
            finally:
                if trace_stage_changed:
                    recorder._trace_stage = previous_trace_stage
                recorder.exit(previous_stage)

        return cast(_F, wrapper)

    return decorator
