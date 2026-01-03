from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from sglang.srt.debug_utils.schedule_simulator.request import SimRequest

if TYPE_CHECKING:
    from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState


@dataclass
class ScheduleDecision:
    to_run: List[SimRequest] = field(default_factory=list)


class SchedulerPolicy(ABC):
    @abstractmethod
    def schedule(self, gpu_state: "GPUState") -> ScheduleDecision:
        """Decide which pending requests to run."""
        ...

    @abstractmethod
    def select_victim(self, gpu_state: "GPUState") -> Optional[SimRequest]:
        """Select a running request to evict. Returns None if nothing to evict."""
        ...
