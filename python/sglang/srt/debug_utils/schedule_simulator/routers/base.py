from abc import ABC, abstractmethod

from sglang.srt.debug_utils.schedule_simulator.request import SimRequest


class RouterPolicy(ABC):
    @abstractmethod
    def route(self, incoming_request: SimRequest) -> int: ...
