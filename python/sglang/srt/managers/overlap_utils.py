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
        # 2 * max_running_requests for overlap schedule
        self.future_buffer_len = max_running_requests * 5
        self.device = device

        self.buf_inited = False

        if self.spec_algorithm.is_eagle():
            pass
        else:
            self.future_next_token_ids_map = torch.empty(
                (self.future_buffer_len,), dtype=torch.int64, device=self.device
            )

    def _lazy_init_buf(self, spec_info: EagleDraftInput):
        if self.buf_inited or not self.spec_algorithm.is_eagle():
            return

        self.buf_inited = True

        # get the template for each tensor
        topk_p0 = spec_info.topk_p[0]
        topk_index0 = spec_info.topk_index[0]
        hidden_states0 = spec_info.hidden_states[0]
        verified_id0 = spec_info.verified_id[0]
        new_seq_lens0 = spec_info.new_seq_lens[0]

        self.topk_p_buf = torch.empty(
            (self.future_buffer_len, *topk_p0.shape),
            dtype=topk_p0.dtype,
            device=self.device,
        )
        self.topk_index_buf = torch.empty(
            (self.future_buffer_len, *topk_index0.shape),
            dtype=topk_index0.dtype,
            device=self.device,
        )
        self.hidden_states_buf = torch.empty(
            (self.future_buffer_len, *hidden_states0.shape),
            dtype=hidden_states0.dtype,
            device=self.device,
        )
        self.verified_id_buf = torch.empty(
            (self.future_buffer_len, *verified_id0.shape),
            dtype=verified_id0.dtype,
            device=self.device,
        )
        self.new_seq_lens_buf = torch.empty(
            (self.future_buffer_len, *new_seq_lens0.shape),
            dtype=new_seq_lens0.dtype,
            device=self.device,
        )

    def get_next_future_ct(self, bs: int) -> int:
        cur_future_ct = self.future_ct
        # increment the future_ct, circular buffer pointer
        self.future_ct = (cur_future_ct + bs) % self.future_limit
        return cur_future_ct

    def resolve_future(self, model_worker_batch: ModelWorkerBatch):
        if self.spec_algorithm.is_eagle():
            spec_info: EagleDraftInput = model_worker_batch.spec_info
            if spec_info is None:
                return

            ids = -spec_info.topk_p

            spec_info.topk_p = self.topk_p_buf[ids]
            spec_info.topk_index = self.topk_index_buf[ids]
            spec_info.hidden_states = self.hidden_states_buf[ids]
            spec_info.verified_id = self.verified_id_buf[ids]
            spec_info.new_seq_lens = self.new_seq_lens_buf[ids]
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
            spec_info: EagleDraftInput = forward_output.spec_info
            self._lazy_init_buf(spec_info)
            for i in range(bs):
                slot = future_ct + i + 1
                self.topk_p_buf[slot] = spec_info.topk_p[i]
                self.topk_index_buf[slot] = spec_info.topk_index[i]
                self.hidden_states_buf[slot] = spec_info.hidden_states[i]
                self.verified_id_buf[slot] = spec_info.verified_id[i]
                self.new_seq_lens_buf[slot] = spec_info.new_seq_lens[i]
        else:
            self.future_next_token_ids_map[future_ct + 1 : future_ct + bs + 1] = (
                forward_output.next_token_ids
            )
