from typing import List

from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState
from sglang.srt.debug_utils.schedule_simulator.request import SimRequest
from sglang.srt.debug_utils.schedule_simulator.routers.base import RouterPolicy


class RoundRobinRouter(RouterPolicy):
    def __init__(self):
        self._counter = 0

    def route(
        self,
        incoming_request: SimRequest,
        gpu_states: List[GPUState],
    ) -> int:
        gpu_id = self._counter % len(gpu_states)
        self._counter += 1
        return gpu_id

