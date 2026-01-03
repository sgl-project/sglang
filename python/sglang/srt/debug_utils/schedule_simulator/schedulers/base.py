from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState


class SchedulerPolicy(ABC):
    @abstractmethod
    def schedule(self, gpu_state: "GPUState") -> None: ...
