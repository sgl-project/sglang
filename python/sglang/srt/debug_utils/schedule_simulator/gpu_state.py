from dataclasses import dataclass, field
from typing import List

from sglang.srt.debug_utils.schedule_simulator.request import SimRequest


@dataclass
class StepRecord:
    step: int
    gpu_id: int
    running_count: int
    pending_count: int
    total_seq_len: int
    running_req_ids: List[str] = field(default_factory=list)
    pending_req_ids: List[str] = field(default_factory=list)


@dataclass
class GPUState:
    gpu_id: int
    max_total_tokens: int
    pending_requests: List[SimRequest] = field(default_factory=list)
    running_requests: List[SimRequest] = field(default_factory=list)

    def batch_size(self) -> int:
        return len(self.running_requests)

    def total_seq_len(self) -> int:
        return sum(req.seq_len() for req in self.running_requests)

    def is_valid(self) -> bool:
        return self.total_seq_len() <= self.max_total_tokens

    def start_request(self, req: SimRequest) -> None:
        assert req in self.pending_requests
        self.pending_requests.remove(req)
        self.running_requests.append(req)

    def evict_request(self, req: SimRequest) -> None:
        assert req in self.running_requests
        self.running_requests.remove(req)
        self.pending_requests.insert(0, req)

    def execute_step(self) -> None:
        finished = []
        for req in self.running_requests:
            req.decoded_tokens += 1
            if req.is_finished():
                finished.append(req)
        for req in finished:
            self.running_requests.remove(req)

    def get_step_record(self, step: int) -> StepRecord:
        return StepRecord(
            step=step,
            gpu_id=self.gpu_id,
            running_count=len(self.running_requests),
            pending_count=len(self.pending_requests),
            total_seq_len=self.total_seq_len(),
            running_req_ids=[r.request_id for r in self.running_requests],
            pending_req_ids=[r.request_id for r in self.pending_requests],
        )
