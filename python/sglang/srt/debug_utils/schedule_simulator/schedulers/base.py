from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState


class SchedulerPolicy(ABC):
    @abstractmethod
    def schedule(self, gpu_state: "GPUState", max_total_tokens: int) -> None:
        """Schedule requests on the GPU, modifying gpu_state in place.

        The scheduler should:
        1. Evict running requests if over budget (via gpu_state.evict_request)
        2. Run pending requests that fit (via gpu_state.run_request)

        After this method returns, gpu_state.is_valid(max_total_tokens) must be True.
        """
        ...
