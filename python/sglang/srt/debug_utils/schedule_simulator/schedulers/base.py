from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

from sglang.srt.debug_utils.schedule_simulator.request import SimRequest


@dataclass
class ScheduleDecision:
    to_run: List[SimRequest] = field(default_factory=list)
    to_preempt: List[SimRequest] = field(default_factory=list)


class SchedulerPolicy(ABC):
    @abstractmethod
    def schedule(self, gpu_state: "GPUState") -> ScheduleDecision: ...


from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState
