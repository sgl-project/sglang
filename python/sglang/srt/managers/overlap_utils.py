from __future__ import annotations

import os
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

# Token-buf consume tracking: init to -1, assert non-negative on gather,
# write -1 back. Catches "gather without intermediate stash" bugs. CI enables
# via the existing SGLANG_IS_IN_CI; off in production.
_DEBUG_ASSERT = os.getenv("SGLANG_IS_IN_CI", "").lower() == "true"


@torch.compile(dynamic=True)
def _assert_nonneg_and_invalidate(
    values: torch.Tensor, buf: torch.Tensor, indices: torch.Tensor
) -> None:
    """Fused: assert all `values >= 0` and scatter -1 into `buf[indices]`.
    Compiled so the reduction + assert + scatter run as one kernel launch."""
    torch._assert_async((values >= 0).all())
    buf[indices] = -1


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


class FutureMap:
    """Cross-iter relay buffer for values the next iter's schedule cannot
    compute locally (e.g. spec_v2 seq_lens after accept_lens, sampled tokens).

    Forward stream publishes into a buf; next iter's schedule pulls lazily.
    Schedule-deterministic values (e.g. non-spec seq_lens via +1) stay
    maintained by SB directly and do not need the relay.

    SB.seq_lens GPU is always a faithful seq_lens_cpu mirror; forward path
    treats it as read-only, spec mutations land on forward_batch.seq_lens.
    """

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

        self.output_tokens_buf = (
            torch.full((self.req_pool_size,), -1, dtype=torch.int64, device=self.device)
            if _DEBUG_ASSERT
            else torch.empty(
                (self.req_pool_size,), dtype=torch.int64, device=self.device
            )
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
        # seq_lens is already real on entry (SB +1 for non-spec;
        # resolve_seq_lens_cpu pulled from buf for spec_v2). Only resolve
        # input_ids tokens / spec extras here.
        if self.spec_algo.is_none():
            _resolve_future_token_ids(batch.input_ids, self.output_tokens_buf)
            if _DEBUG_ASSERT:
                _assert_nonneg_and_invalidate(
                    batch.input_ids, self.output_tokens_buf, batch.req_pool_indices
                )
        else:
            self._resolve_spec_extras(batch)

    def _resolve_spec_extras(self, batch: ScheduleBatch) -> None:
        draft_input: EagleDraftInput = batch.spec_info
        if draft_input is None:
            # FIXME(lsyin): only prefill; not compatible with mixed mode
            return
        indices = draft_input.future_indices
        # FIXME: indices = batch.req_pool_indices, pinned 2 iters via
        # record_batch_in_overlap; record_stream here is redundant.
        indices.record_stream(torch.get_device_module(self.device).current_stream())
        draft_input.topk_p = self.topk_p_buf[indices]
        draft_input.topk_index = self.topk_index_buf[indices]
        draft_input.bonus_tokens = self.output_tokens_buf[indices]
        if _DEBUG_ASSERT:
            _assert_nonneg_and_invalidate(
                draft_input.bonus_tokens, self.output_tokens_buf, indices
            )
        if spec_need_hidden_states():
            draft_input.hidden_states = self.hidden_states_buf[indices]

    def set_input_ids_sentinel(
        self, batch: ScheduleBatch, future_indices: torch.Tensor
    ) -> None:
        # Sentinel for the decode portion so mixed batches can cat extend
        # (positive real tokens) + decode (negative sentinels) into one
        # input_ids; resolve_future translates negatives via output_tokens_buf.
        batch.input_ids = -future_indices

    def resolve_seq_lens_cpu(self, batch: ScheduleBatch) -> None:
        # Lazy pull from new_seq_lens_buf for spec_v2 (accept_lens not known to
        # schedule). Write into both CPU and GPU so SB.seq_lens stays a faithful
        # seq_lens_cpu mirror.
        fi = batch.spec_info.future_indices if batch.spec_info is not None else None
        if fi is None:
            return
        if self.publish_ready is not None:
            self.publish_ready.wait()
        new_seq_lens = self.new_seq_lens_buf[fi]
        batch.seq_lens = new_seq_lens
        batch.seq_lens_cpu = new_seq_lens.cpu()
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())

    def publish(self, future_indices: torch.Tensor, new_seq_lens: torch.Tensor) -> None:
        indices = future_indices
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
        future_indices: torch.Tensor,
        payload: Union[torch.Tensor, EagleDraftInput],
    ) -> None:
        indices = future_indices
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
