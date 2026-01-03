from dataclasses import dataclass, field
from typing import List

from sglang.srt.debug_utils.schedule_simulator.request import SimRequest


@dataclass
class GPUState:
    gpu_id: int
    pending_requests: List[SimRequest] = field(default_factory=list)
    running_requests: List[SimRequest] = field(default_factory=list)

    def batch_size(self) -> int:
        return len(self.running_requests)

    def total_seq_len(self) -> int:
        return sum(req.seq_len() for req in self.running_requests)
