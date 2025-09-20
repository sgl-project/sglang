import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatchOutput
from sglang.srt.speculative.eagle_utils_v2 import EagleDraftInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class FutureMap:
    def __init__(
        self,
        spec_algorithm: SpeculativeAlgorithm,
        max_running_requests: int,
        device: torch.device,
    ):
        self.spec_algorithm = spec_algorithm
        self.future_ct = 0
        self.future_limit = max_running_requests * 3
        self.future_buffer_len = (
            max_running_requests * 5
        )  # 2 * max_running_requests for overlap schedule
        self.device = device

        if self.spec_algorithm.is_eagle():
            self.future_topk_p_map = [None] * (self.future_buffer_len)
            self.future_topk_index_map = [None] * (self.future_buffer_len)
            self.future_hidden_states_map = [None] * (self.future_buffer_len)
            self.future_verified_id_map = [None] * (self.future_buffer_len)
            self.future_new_seq_lens_map = [None] * (self.future_buffer_len)
        else:
            self.future_next_token_ids_map = torch.empty(
                (self.future_buffer_len,), dtype=torch.int64, device=self.device
            )

    def get_next_future_ct(self, bs: int) -> int:
        cur_future_ct = self.future_ct
        # increment the future_ct, circular buffer pointer
        self.future_ct = (cur_future_ct + bs) % self.future_limit
        return cur_future_ct

    def resolve_future(self, model_worker_batch: ModelWorkerBatch):
        # TODO(lsyin) too much overhead here, optimize use a triton kernel
        if self.spec_algorithm.is_eagle():
            spec_info = model_worker_batch.spec_info
            if spec_info is None:
                return

            ids = [-idx for idx in spec_info.topk_p.tolist()]

            topk_p_stacked = [self.future_topk_p_map[idx] for idx in ids]
            topk_index_stacked = [self.future_topk_index_map[idx] for idx in ids]
            hidden_states_stacked = [self.future_hidden_states_map[idx] for idx in ids]
            verified_id_stacked = [self.future_verified_id_map[idx] for idx in ids]
            new_seq_lens_stacked = [self.future_new_seq_lens_map[idx] for idx in ids]

            # TODO: think if there's a good way to hide the stack overhead
            # e.g. when the ids are continuous, we can just use a pointer to the first element
            spec_info.topk_p = torch.stack(topk_p_stacked, dim=0)
            spec_info.topk_index = torch.stack(topk_index_stacked, dim=0)
            spec_info.hidden_states = torch.stack(hidden_states_stacked, dim=0)
            spec_info.verified_id = torch.stack(verified_id_stacked, dim=0)
            spec_info.new_seq_lens = torch.stack(new_seq_lens_stacked, dim=0)
        else:
            input_ids = model_worker_batch.input_ids
            input_ids[:] = torch.where(
                input_ids < 0,
                self.future_next_token_ids_map[torch.clamp(-input_ids, min=0)],
                input_ids,
            )

    def update_next_future(
        self,
        schedule_batch: ScheduleBatch,
        future_ct: int,
        bs: int,
        allocate_lens: torch.Tensor,
    ):
        if self.spec_algorithm.is_eagle():
            # future_spec_info_indices is a reference to the next spec_info fields, val is a reference id stored as a negative index.
            future_spec_info_indices = torch.arange(
                -(future_ct + 1),
                -(future_ct + 1 + bs),
                -1,
                dtype=torch.int64,
                device=self.device,
            )

            schedule_batch.spec_info = EagleDraftInput(
                topk_p=future_spec_info_indices,
                topk_index=future_spec_info_indices,
                hidden_states=future_spec_info_indices,
                verified_id=future_spec_info_indices,
                new_seq_lens=future_spec_info_indices,
                allocate_lens=allocate_lens,  # allocate_lens is never a future
            )
        else:
            return torch.arange(
                -(future_ct + 1),
                -(future_ct + 1 + bs),
                -1,
                dtype=torch.int64,
                device=self.device,
            )

    def store_to_map(self, future_ct: int, bs: int, forward_output: ForwardBatchOutput):
        if self.spec_algorithm.is_eagle():
            spec_info = forward_output.spec_info
            # Store references (no clone) for each request into circular buffers
            # NOTE: self.future_allocate_lens_map is not needed here because it's assigned to batch.spec_info.allocate_lens, not a future
            for i in range(bs):
                slot = future_ct + i + 1
                self.future_topk_p_map[slot] = spec_info.topk_p[i]
                self.future_topk_index_map[slot] = spec_info.topk_index[i]
                self.future_hidden_states_map[slot] = spec_info.hidden_states[i]
                self.future_verified_id_map[slot] = spec_info.verified_id[i]
                self.future_new_seq_lens_map[slot] = spec_info.new_seq_lens[i]
        else:
            self.future_next_token_ids_map[future_ct + 1 : future_ct + bs + 1] = (
                forward_output.next_token_ids
            )
