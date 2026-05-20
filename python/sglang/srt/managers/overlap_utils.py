from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

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
    interval: Optional[slice] = None


class FutureMap:
    def __init__(
        self,
        max_running_requests: int,
        chunked_prefill_size: int,
        context_len: int,
        device: torch.device,
        spec_algo: SpeculativeAlgorithm,
        req_to_token_pool: ReqToTokenPool,
    ):
        self.device = device
        self.spec_algo = spec_algo

        if self.spec_algo.is_none():
            # token_ids_buf is addressed directly by req_pool_idx (see
            # alloc_future_indices below). Sizing follows ReqToTokenPool's
            # allocated rows (req_to_token.shape[0] == size + 1) so slot 0
            # is the padding row shared with the KV cache pool: CUDA-graph
            # padded batches default req_pool_idx to 0, and dummy reads/
            # writes here land harmlessly on that same row. Real req_pool_idx
            # in [1, size] => `-req_pool_idx` is strictly negative and
            # unambiguous as the resolve_future placeholder.
            self.buf_initialized = True
            self.token_ids_buf = torch.empty(
                (req_to_token_pool.req_to_token.shape[0],),
                dtype=torch.int64,
                device=self.device,
            )
        else:
            # Spec v2 buffers still use a ring slot allocator; sizing follows
            # the historical worst-case formula. Follow-up will migrate spec v2
            # buffers to the same req-indexed scheme as token_ids_buf.
            self.future_ct = 0
            max_num_chunks = (
                (context_len + chunked_prefill_size - 1) // chunked_prefill_size
                if chunked_prefill_size
                else 0
            )
            self.future_limit = max_running_requests * (3 + max_num_chunks)
            self.future_buffer_len = self.future_limit + 2 * max_running_requests
            self.buf_initialized = False

    def _lazy_init_buf(self, draft_input: EagleDraftInput):
        self.buf_initialized = True

        # Get a reference for each tensor
        topk_p0 = draft_input.topk_p[0]
        topk_index0 = draft_input.topk_index[0]
        bonus_token0 = draft_input.bonus_tokens[0]
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
        self.bonus_tokens_buf = torch.empty(
            (self.future_buffer_len, *bonus_token0.shape),
            dtype=bonus_token0.dtype,
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

    def alloc_future_indices(self, batch: ScheduleBatch) -> FutureIndices:
        """Allocate future-token slot handles for this batch.

        For non-spec, the slot identity is the request's req_pool_idx itself:
        we reuse ``batch.req_pool_indices`` as the indices into
        ``token_ids_buf``. No fresh tensor is allocated, so there is no
        cross-stream lifetime concern for the indices tensor (the SB owns it).

        For spec v2, the ring allocator is preserved.
        """
        if self.spec_algo.is_none():
            return FutureIndices(indices=batch.req_pool_indices)

        bs = len(batch.seq_lens)
        cur_future_ct = self.future_ct
        self.future_ct = (cur_future_ct + bs) % self.future_limit
        start = cur_future_ct + 1
        end = cur_future_ct + 1 + bs
        indices = torch.arange(start, end, dtype=torch.int64, device=self.device)
        return FutureIndices(indices=indices, interval=slice(start, end))

    def resolve_future(self, batch: ScheduleBatch):
        if self.spec_algo.is_none():
            _resolve_future_token_ids(batch.input_ids, self.token_ids_buf)
        else:
            # TODO(lsyin): write future indices into spec_info.future_indices
            draft_input: EagleDraftInput = batch.spec_info
            if draft_input is None:
                # FIXME(lsyin): No future exists, only for prefill batch, not compatible with mixed mode
                return
            indices = draft_input.future_indices.indices
            # The indices tensor was allocated on the default stream but is
            # used here on the forward stream. Meanwhile, the old spec_info
            # holding this tensor will lose all Python references (replaced at
            # batch.spec_info), so the caching allocator (torch GC) could
            # reclaim the memory before the GPU finishes reading it.
            indices.record_stream(torch.get_device_module(self.device).current_stream())
            draft_input.topk_p = self.topk_p_buf[indices]
            draft_input.topk_index = self.topk_index_buf[indices]
            draft_input.bonus_tokens = self.bonus_tokens_buf[indices]
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
            # Empty indices (e.g. DP attention idle rank) advanced-index into
            # token_ids_buf as a natural no-op; no explicit guard needed.
            self.token_ids_buf[future_indices.indices] = batch_result.next_token_ids
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

        self.topk_p_buf[intv] = draft_input.topk_p
        self.topk_index_buf[intv] = draft_input.topk_index
        self.bonus_tokens_buf[intv] = draft_input.bonus_tokens
        self.new_seq_lens_buf[intv] = draft_input.new_seq_lens
        if spec_need_hidden_states():
            self.hidden_states_buf[intv] = draft_input.hidden_states
