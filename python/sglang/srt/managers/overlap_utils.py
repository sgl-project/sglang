from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.speculative.spec_utils import spec_need_hidden_states
from sglang.srt.utils import get_compiler_backend

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ModelWorkerBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.ngram_info import NgramVerifyInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


@torch.compile(dynamic=True, backend=get_compiler_backend())
def _resolve_future_token_ids(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


@dataclass
class FutureIndices:
    indices: torch.Tensor
    interval: Optional[slice] = None


class FutureMap:
    def __init__(
        self,
        max_running_requests: int,
        chunked_prefill_size: int,
        context_len: int,
        device: torch.device,
        spec_algo: Optional[SpeculativeAlgorithm] = None,
    ):
        # FIXME: the calculation of future_limit and future_buffer_len maybe too conservative
        self.future_ct = 0

        # Circular buffer layout (wraps in this order):
        # Running decode batch -> Prefill chunk 1 -> ... -> Prefill chunk N
        # A running decode batch's result will be resolved after all prefill chunks are done.
        # reserve `max_num_chunks` extra future slots on top of `max_running_requests * 3`.
        max_num_chunks = (
            (context_len + chunked_prefill_size - 1) // chunked_prefill_size
            if chunked_prefill_size
            else 0
        )
        self.future_limit = max_running_requests * (3 + max_num_chunks)
        # Adding 2 * max_running_requests to future_limit ensures the buffer is sufficiently large.
        self.future_buffer_len = self.future_limit + 2 * max_running_requests
        self.device = device
        self.spec_algo = spec_algo

        if self.spec_algo.is_none():
            # For non-speculative decoding, we only need to store the token ids.
            self.buf_initialized = True
            self.token_ids_buf = torch.empty(
                (self.future_buffer_len,), dtype=torch.int64, device=self.device
            )
        else:
            # For speculative decoding, we lazily initialize the buffers
            # This is to make the shape derivation easier.
            self.buf_initialized = False

    def _lazy_init_buf(self, draft_input: EagleDraftInput):
        self.buf_initialized = True
        if self.spec_algo.is_ngram():
            draft_token = draft_input.draft_token
            custom_mask = draft_input.custom_mask
            positions = draft_input.positions
            retrive_index0 = draft_input.retrive_index[0]
            retrive_next_token0 = draft_input.retrive_next_token[0]
            retrive_next_sibling0 = draft_input.retrive_next_sibling[0]
            draft_token_num = draft_input.draft_token_num
            self.draft_token_buf = torch.empty(
                (self.future_buffer_len, *draft_token.shape),
                dtype=draft_token.dtype,
                device=self.device,
            )
            self.custom_mask_buf = torch.empty(
                (self.future_buffer_len, *custom_mask.shape),
                dtype=custom_mask.dtype,
                device=self.device,
            )
            self.positions_buf = torch.empty(
                (self.future_buffer_len, *positions.shape),
                dtype=positions.dtype,
                device=self.device,
            )
            self.retrive_index0_buf = torch.empty(
                (self.future_buffer_len, *retrive_index0.shape),
                dtype=retrive_index0.dtype,
                device=self.device,
            )
            self.retrive_next_token0_buf = torch.empty(
                (self.future_buffer_len, *retrive_next_token0.shape),
                dtype=retrive_next_token0.dtype,
                device=self.device,
            )
            self.retrive_next_sibling0_buf = torch.empty(
                (self.future_buffer_len, *retrive_next_sibling0.shape),
                dtype=retrive_next_sibling0.dtype,
                device=self.device,
            )
            self.draft_token_num_buf = torch.empty(
                (self.future_buffer_len,),
                dtype=torch.int64,
                device=self.device,
            )
            return

        # Get a reference for each tensor
        topk_p0 = draft_input.topk_p[0]
        topk_index0 = draft_input.topk_index[0]
        verified_id0 = draft_input.verified_id[0]
        new_seq_lens0 = draft_input.new_seq_lens[0]

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

        if spec_need_hidden_states():
            hidden_states0 = draft_input.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.future_buffer_len, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

    def alloc_future_indices(self, bs: int) -> FutureIndices:
        """Update the circular buffer pointer and allocate future indices."""
        cur_future_ct = self.future_ct
        self.future_ct = (cur_future_ct + bs) % self.future_limit
        start = cur_future_ct + 1
        end = cur_future_ct + 1 + bs
        indices = torch.arange(start, end, dtype=torch.int64, device=self.device)
        return FutureIndices(indices=indices, interval=slice(start, end))

    def set_draft_input_ngram(
        self, draft_input: NgramVerifyInput, future_indices: FutureIndices
    ):
        draft_input.draft_token = self.draft_token_buf[future_indices.indices]
        draft_input.custom_mask = self.custom_mask_buf[future_indices.indices]
        draft_input.positions = self.positions_buf[future_indices.indices]
        draft_input.retrive_index = self.retrive_index0_buf[future_indices.indices]
        draft_input.retrive_next_token = self.retrive_next_token0_buf[
            future_indices.indices
        ]
        draft_input.retrive_next_sibling = self.retrive_next_sibling0_buf[
            future_indices.indices
        ]
        draft_input.draft_token_num = self.draft_token_num_buf[future_indices.indices]

    def resolve_future(self, model_worker_batch: ModelWorkerBatch):
        if self.spec_algo.is_none():
            _resolve_future_token_ids(model_worker_batch.input_ids, self.token_ids_buf)
        else:
            # TODO(lsyin): write future indices into spec_info.future_indices
            draft_input: EagleDraftInput = model_worker_batch.spec_info
            if draft_input is None:
                # FIXME(lsyin): No future exists, only for prefill batch, not compatible with mixed mode
                return
            if self.spec_algo.is_ngram():
                self.set_draft_input_ngram(draft_input)
                return
            indices = draft_input.future_indices.indices
            draft_input.topk_p = self.topk_p_buf[indices]
            draft_input.topk_index = self.topk_index_buf[indices]
            draft_input.verified_id = self.verified_id_buf[indices]
            draft_input.new_seq_lens = self.new_seq_lens_buf[indices]
            if spec_need_hidden_states():
                draft_input.hidden_states = self.hidden_states_buf[indices]

    def is_empty_slice(self, s: slice) -> bool:
        start, stop, step = s.indices(self.future_buffer_len)
        if step > 0:
            return start >= stop
        else:
            return start <= stop

    def store_to_map(
        self, future_indices: FutureIndices, batch_result: GenerationBatchResult
    ):
        if self.spec_algo.is_none():
            intv = future_indices.interval
            self.token_ids_buf[intv] = batch_result.next_token_ids
        else:
            draft_input: EagleDraftInput = batch_result.next_draft_input
            self.store_to_map_for_new_batch(future_indices, draft_input)

    def store_to_map_for_new_batch(
        self, future_indices: FutureIndices, draft_input: EagleDraftInput
    ):
        intv = future_indices.interval
        if self.is_empty_slice(intv):
            # idle indices in dp attention do not need store info
            return

        if not self.buf_initialized:
            self._lazy_init_buf(draft_input)

        if self.spec_algo.is_ngram():
            self.draft_token_buf[intv] = draft_input.draft_token
            self.custom_mask_buf[intv] = draft_input.custom_mask
            self.positions_buf[intv] = draft_input.positions
            self.retrive_index0_buf[intv] = draft_input.retrive_index
            self.retrive_next_token0_buf[intv] = draft_input.retrive_next
            self.retrive_next_sibling0_buf[intv] = draft_input.retrive_next_sibling
            self.draft_token_num_buf[intv] = draft_input.draft_token_num
            return

        self.topk_p_buf[intv] = draft_input.topk_p
        self.topk_index_buf[intv] = draft_input.topk_index
        self.verified_id_buf[intv] = draft_input.verified_id
        self.new_seq_lens_buf[intv] = draft_input.new_seq_lens
        if spec_need_hidden_states():
            self.hidden_states_buf[intv] = draft_input.hidden_states
