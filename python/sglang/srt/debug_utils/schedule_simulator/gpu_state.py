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

    def is_valid(self, max_total_tokens: int) -> bool:
        return self.total_seq_len() <= max_total_tokens

    def run_request(self, req: SimRequest) -> None:
        """Move a request from pending to running."""
        assert req in self.pending_requests
        self.pending_requests.remove(req)
        self.running_requests.append(req)

    def evict_request(self, req: SimRequest) -> None:
        """Move a request from running back to pending (at front)."""
        assert req in self.running_requests
        self.running_requests.remove(req)
        self.pending_requests.insert(0, req)
