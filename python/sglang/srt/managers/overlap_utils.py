from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import torch

from sglang.srt.speculative.spec_utils import spec_need_hidden_states
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
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
        # Bufs indexed by req_pool_idx; slot 0 mirrors KV padding row so
        # CUDA-graph padded batches (req_pool_idx == 0) are harmless.
        self.device = device
        self.spec_algo = spec_algo
        self.req_pool_size = req_to_token_pool.req_to_token.shape[0]

        self.output_tokens_buf = torch.empty(
            (self.req_pool_size,), dtype=torch.int64, device=self.device
        )
        self.new_seq_lens_buf = torch.empty(
            (self.req_pool_size,), dtype=torch.int64, device=self.device
        )
        if self.spec_algo.is_some():
            self._forward_buf_initialized = False

        self.publish_ready = None  # lazy device.Event(); only spec_v2 needs it

    def _lazy_init_forward_buf(self, draft_input: EagleDraftInput):
        self._forward_buf_initialized = True

        topk_p0 = draft_input.topk_p[0]
        topk_index0 = draft_input.topk_index[0]
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
        if spec_need_hidden_states():
            hidden_states0 = draft_input.hidden_states[0]
            self.hidden_states_buf = torch.empty(
                (self.req_pool_size, *hidden_states0.shape),
                dtype=hidden_states0.dtype,
                device=self.device,
            )

    def resolve_future(self, batch: ScheduleBatch):
        if batch.forward_mode.is_decode():
            batch.seq_lens = self.new_seq_lens_buf[batch.req_pool_indices]
            torch._assert_async((batch.seq_lens > 0).all())

        if self.spec_algo.is_none():
            _resolve_future_token_ids(batch.input_ids, self.output_tokens_buf)
        else:
            self._resolve_spec_extras(batch)

    def _resolve_spec_extras(self, batch: ScheduleBatch) -> None:
        draft_input: EagleDraftInput = batch.spec_info
        if draft_input is None:
            # FIXME(lsyin): only prefill; not compatible with mixed mode
            return
        indices = draft_input.future_indices.indices
        # FIXME: indices = batch.req_pool_indices, pinned 2 iters via
        # record_batch_in_overlap; record_stream here is redundant.
        indices.record_stream(torch.get_device_module(self.device).current_stream())
        draft_input.topk_p = self.topk_p_buf[indices]
        draft_input.topk_index = self.topk_index_buf[indices]
        draft_input.bonus_tokens = self.output_tokens_buf[indices]
        if spec_need_hidden_states():
            draft_input.hidden_states = self.hidden_states_buf[indices]

    def invalidate(self, batch: ScheduleBatch, future_indices: FutureIndices) -> None:
        sentinel = -future_indices.indices
        batch.input_ids = sentinel
        batch.seq_lens = sentinel

    def resolve_seq_lens_cpu(self, batch: ScheduleBatch) -> None:
        fi = batch.spec_info.future_indices if batch.spec_info is not None else None
        if fi is None:
            return
        if self.publish_ready is not None:
            self.publish_ready.wait()
        batch.seq_lens_cpu = self.new_seq_lens_buf[fi.indices].cpu()
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())

    def publish(
        self, future_indices: FutureIndices, new_seq_lens: torch.Tensor
    ) -> None:
        indices = future_indices.indices
        if indices.shape[0] == 0:
            return  # DP idle
        self.new_seq_lens_buf[indices] = new_seq_lens.to(self.new_seq_lens_buf.dtype)
        # Fast path: only spec_v2 needs the event (schedule-stream D2H sync).
        if self.spec_algo.is_some():
            if self.publish_ready is None:
                self.publish_ready = torch.get_device_module(self.device).Event()
            self.publish_ready.record()

    def stash(
        self,
        future_indices: FutureIndices,
        payload: Union[torch.Tensor, EagleDraftInput],
    ) -> None:
        indices = future_indices.indices
        if indices.shape[0] == 0:
            # DP idle: payload is empty stub; lazy-init shape peek would IndexError.
            return
        if self.spec_algo.is_none():
            self.output_tokens_buf[indices] = payload.to(torch.int64)
            return

        draft_input: EagleDraftInput = payload
        if not self._forward_buf_initialized:
            self._lazy_init_forward_buf(draft_input)
        self.output_tokens_buf[indices] = draft_input.bonus_tokens.to(
            self.output_tokens_buf.dtype
        )
        self.topk_p_buf[indices] = draft_input.topk_p.to(self.topk_p_buf.dtype)
        self.topk_index_buf[indices] = draft_input.topk_index.to(
            self.topk_index_buf.dtype
        )
        if spec_need_hidden_states():
            self.hidden_states_buf[indices] = draft_input.hidden_states.to(
                self.hidden_states_buf.dtype
            )
