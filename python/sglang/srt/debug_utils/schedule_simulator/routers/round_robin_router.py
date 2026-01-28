from sglang.srt.debug_utils.schedule_simulator.request import SimRequest
from sglang.srt.debug_utils.schedule_simulator.routers.base import RouterPolicy


class RoundRobinRouter(RouterPolicy):
    def __init__(self, num_gpus: int):
        self._num_gpus = num_gpus
        self._counter = 0

    def route(self, incoming_request: SimRequest) -> int:
        gpu_id = self._counter % self._num_gpus
        self._counter += 1
        return gpu_id
