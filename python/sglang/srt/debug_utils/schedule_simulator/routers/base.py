from abc import ABC, abstractmethod
from typing import List

from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState
from sglang.srt.debug_utils.schedule_simulator.request import SimRequest


class RouterPolicy(ABC):
    @abstractmethod
    def route(
        self,
        incoming_request: SimRequest,
        gpu_states: List[GPUState],
    ) -> int:
        ...

