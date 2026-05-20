from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from sglang.srt.speculative.spec_utils import spec_need_hidden_states
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

_is_cuda = is_cuda()
_is_hip = is_hip()


def _resolve_future_token_ids_native(input_ids, future_token_ids_map):
    input_ids[:] = torch.where(
        input_ids < 0,
        future_token_ids_map[torch.clamp(-input_ids, min=0)],
        input_ids,
    )


if _is_cuda or _is_hip:
    from sglang.jit_kernel.resolve_future_token_ids import (
        resolve_future_token_ids_cuda,
    )

    _resolve_future_token_ids = resolve_future_token_ids_cuda
else:
    _resolve_future_token_ids = _resolve_future_token_ids_native


@dataclass
class FutureIndices:
    indices: torch.Tensor


class FutureMap:
    def __init__(
        self,
        device: torch.device,
        spec_algo: SpeculativeAlgorithm,
        req_to_token_pool: ReqToTokenPool,
    ):
        # All buffers are indexed by req_pool_idx. Slot 0 mirrors the KV cache
        # pool's padding row, so CUDA-graph padded batches (req_pool_idx == 0)
        # read/write here harmlessly.
        self.device = device
        self.spec_algo = spec_algo
        self.req_pool_size = req_to_token_pool.req_to_token.shape[0]

        if self.spec_algo.is_none():
            self.buf_initialized = True
            self.token_ids_buf = torch.empty(
                (self.req_pool_size,), dtype=torch.int64, device=self.device
            )
        else:
            # Spec v2 buffer shapes follow the first draft input we see.
            self.buf_initialized = False

    def _lazy_init_buf(self, draft_input: EagleDraftInput):
        self.buf_initialized = True

        topk_p0 = draft_input.topk_p[0]
        topk_index0 = draft_input.topk_index[0]
        bonus_token0 = draft_input.bonus_tokens[0]
        new_seq_lens0 = draft_input.new_seq_lens[0]

        self.topk_p_buf = torch.empty(
            (self.req_pool_size, *topk_p0.shape),
            dtype=topk_p0.dtype,
            device=self.device,
        )
        self.topk_index_buf = torch.empty(
            (self.req_pool_size, *topk_index0.shape),
            dtype=topk_index0.dtype,
            device=self.device,
        )
        self.bonus_tokens_buf = torch.empty(
            (self.req_pool_size, *bonus_token0.shape),
            dtype=bonus_token0.dtype,
            device=self.device,
        )
        self.new_seq_lens_buf = torch.empty(
            (self.req_pool_size, *new_seq_lens0.shape),
            dtype=new_seq_lens0.dtype,
            device=self.device,
        )

        if spec_need_hidden_states():
            hidden_states0 = draft_input.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.req_pool_size, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

    def resolve_future(self, batch: ScheduleBatch):
        if self.spec_algo.is_none():
            _resolve_future_token_ids(batch.input_ids, self.token_ids_buf)
        else:
            draft_input: EagleDraftInput = batch.spec_info
            if draft_input is None:
                # FIXME(lsyin): No future exists, only for prefill batch, not compatible with mixed mode
                return
            indices = draft_input.future_indices.indices
            draft_input.topk_p = self.topk_p_buf[indices]
            draft_input.topk_index = self.topk_index_buf[indices]
            draft_input.bonus_tokens = self.bonus_tokens_buf[indices]
            draft_input.new_seq_lens = self.new_seq_lens_buf[indices]
            if spec_need_hidden_states():
                draft_input.hidden_states = self.hidden_states_buf[indices]

    def store_to_map(
        self, future_indices: FutureIndices, batch_result: GenerationBatchResult
    ):
        if self.spec_algo.is_none():
            self.token_ids_buf[future_indices.indices] = batch_result.next_token_ids
        else:
            draft_input: EagleDraftInput = batch_result.next_draft_input
            self.store_to_map_for_new_batch(future_indices, draft_input)

    def store_to_map_for_new_batch(
        self, future_indices: FutureIndices, draft_input: EagleDraftInput
    ):
        if not self.buf_initialized:
            self._lazy_init_buf(draft_input)

        indices = future_indices.indices
        self.topk_p_buf[indices] = draft_input.topk_p
        self.topk_index_buf[indices] = draft_input.topk_index
        self.bonus_tokens_buf[indices] = draft_input.bonus_tokens
        self.new_seq_lens_buf[indices] = draft_input.new_seq_lens
        if spec_need_hidden_states():
            self.hidden_states_buf[indices] = draft_input.hidden_states
