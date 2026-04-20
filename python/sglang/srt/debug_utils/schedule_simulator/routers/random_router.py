import random

from sglang.srt.debug_utils.schedule_simulator.request import SimRequest
from sglang.srt.debug_utils.schedule_simulator.routers.base import RouterPolicy


class RandomRouter(RouterPolicy):
    def __init__(self, num_gpus: int):
        self._num_gpus = num_gpus

    def route(self, incoming_request: SimRequest) -> int:
        return random.randint(0, self._num_gpus - 1)
