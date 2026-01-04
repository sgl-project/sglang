import random
from typing import List

from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState
from sglang.srt.debug_utils.schedule_simulator.request import SimRequest
from sglang.srt.debug_utils.schedule_simulator.routers.base import RouterPolicy


class RandomRouter(RouterPolicy):
    def route(
        self,
        incoming_request: SimRequest,
        gpu_states: List[GPUState],
    ) -> int:
        return random.randint(0, len(gpu_states) - 1)
