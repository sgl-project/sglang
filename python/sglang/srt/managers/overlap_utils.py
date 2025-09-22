import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.utils import get_compiler_backend


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _resolve_future_token_ids(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


class FutureMap:
    def __init__(
        self,
        max_running_requests: int,
        device: torch.device,
    ):
        self.future_limit = max_running_requests * 3
        self.future_buffer_len = max_running_requests * 5
        self.device = device

        # Circular buffer pointers
        self.current_ct = None
        self.next_ct = 0

        self.token_ids_buf = torch.empty(
            (self.future_buffer_len,), dtype=torch.int64, device=self.device
        )

    def allocate(self, bs: int):
        """Allocate bs future slots"""
        self.current_ct = self.next_ct
        self.next_ct = (self.next_ct + bs) % self.future_limit

    def resolve_future(self, model_worker_batch: ModelWorkerBatch):
        input_ids = model_worker_batch.input_ids
        _resolve_future_token_ids(input_ids, self.token_ids_buf)

    def update_next_future(self, bs: int):
        return torch.arange(
            -(self.current_ct + 1),
            -(self.current_ct + 1 + bs),
            -1,
            dtype=torch.int64,
            device=self.device,
        )

    def store_to_map(self, bs: int, next_token_ids: torch.Tensor):
        self.token_ids_buf[self.current_ct + 1 : self.current_ct + bs + 1] = (
            next_token_ids
        )
