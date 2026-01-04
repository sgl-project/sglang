import random
from typing import Dict, List

from sglang.srt.debug_utils.schedule_simulator.gpu_state import GPUState
from sglang.srt.debug_utils.schedule_simulator.request import SimRequest
from sglang.srt.debug_utils.schedule_simulator.routers.base import RouterPolicy


class StickyRouter(RouterPolicy):
    def __init__(self):
        self._group_to_gpu: Dict[str, int] = {}

    def route(
        self,
        incoming_request: SimRequest,
        gpu_states: List[GPUState],
    ) -> int:
        group_id = incoming_request.group_id
        if group_id is None:
            return random.randint(0, len(gpu_states) - 1)
        if group_id not in self._group_to_gpu:
            self._group_to_gpu[group_id] = random.randint(0, len(gpu_states) - 1)
        return self._group_to_gpu[group_id]

