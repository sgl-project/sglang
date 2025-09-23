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
        self.future_ct = 0
        # A factor of 3 is used to avoid collision in the circular buffer.
        self.future_limit = max_running_requests * 3
        # A factor of 5 is used to ensure the buffer is large enough.
        self.future_buffer_len = max_running_requests * 5
        self.device = device

        self.token_ids_buf = torch.empty(
            (self.future_buffer_len,), dtype=torch.int64, device=self.device
        )

    def update_ct(self, bs: int) -> int:
        """Update the circular buffer pointer and return the current pointer."""
        cur_future_ct = self.future_ct
        self.future_ct = (cur_future_ct + bs) % self.future_limit
        return cur_future_ct

    def resolve_future(self, model_worker_batch: ModelWorkerBatch):
        input_ids = model_worker_batch.input_ids
        _resolve_future_token_ids(input_ids, self.token_ids_buf)

    def update_next_future(self, future_ct: int, bs: int):
        return torch.arange(
            -(future_ct + 1),
            -(future_ct + 1 + bs),
            -1,
            dtype=torch.int64,
            device=self.device,
        )

    def store_to_map(self, future_ct: int, bs: int, next_token_ids: torch.Tensor):
        self.token_ids_buf[future_ct + 1 : future_ct + bs + 1] = next_token_ids
