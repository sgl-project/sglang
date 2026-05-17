import random
from collections import defaultdict

from sglang.srt.debug_utils.schedule_simulator.request import SimRequest
from sglang.srt.debug_utils.schedule_simulator.routers.base import RouterPolicy


class StickyRouter(RouterPolicy):
    def __init__(self, num_gpus: int):
        self._num_gpus = num_gpus
        self._group_to_gpu = defaultdict(self._assign_gpu)

    def _assign_gpu(self) -> int:
        return random.randint(0, self._num_gpus - 1)

    def route(self, incoming_request: SimRequest) -> int:
        group_id = incoming_request.group_id
        if group_id is None:
            return random.randint(0, self._num_gpus - 1)
        return self._group_to_gpu[group_id]
