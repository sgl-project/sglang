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
            self.buf_initialized = False

        # Single cross-stream fence covering "all buf writes dispatched so far".
        # Recorded on forward stream at the tail of every store_to_map; waited
        # on by schedule-stream consumers (e.g. resolve_seq_lens_cpu) before
        # reading the buf. Lives on FutureMap so callers can't forget to
        # propagate it through filter/merge/etc.
        self._last_store_done: Optional[torch.cuda.Event] = None
        # Per-iter flag: did store_post_verify record the fence already? If so,
        # store_to_map skips its own record so the fence stays at verify-end
        # (preserving schedule prep / draft_extend overlap). Reset by every
        # store_to_map call.
        self._fence_done_by_post_verify: bool = False

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
            # FIXME: redundant. `indices` = batch.req_pool_indices, pinned via
            # record_batch_in_overlap's attr_snapshot for 2 iters; refcount > 0
            # across forward's read, allocator can't reclaim. Safe to remove.
            indices.record_stream(torch.get_device_module(self.device).current_stream())
            draft_input.topk_p = self.topk_p_buf[indices]
            draft_input.topk_index = self.topk_index_buf[indices]
            draft_input.bonus_tokens = self.bonus_tokens_buf[indices]
            draft_input.new_seq_lens = self.new_seq_lens_buf[indices]
            # Resolve placeholder batch.seq_lens (set to -indices at end of
            # previous run_batch) to post-verify GPU values from the buf.
            # Mirrors the input_ids placeholder pattern.
            batch.seq_lens = draft_input.new_seq_lens
            if spec_need_hidden_states():
                draft_input.hidden_states = self.hidden_states_buf[indices]

    def resolve_seq_lens_cpu(self, batch: ScheduleBatch) -> None:
        """Schedule-stream counterpart of resolve_future for the CPU mirror.

        Reads post-verify seq_lens from new_seq_lens_buf into batch.seq_lens_cpu
        and recomputes batch.seq_lens_sum. Waits on _last_store_done so the
        D2H sees all forward-stream buf writes dispatched so far. No-op for
        paths without future state (first iter / no spec_info).
        """
        fi = batch.spec_info.future_indices if batch.spec_info is not None else None
        if fi is None:
            return
        if self._last_store_done is not None:
            self._last_store_done.wait()
        batch.seq_lens_cpu = self.new_seq_lens_buf[fi.indices].cpu()
        batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())

    def _record_store_done(self) -> None:
        # Must be called on forward stream right after a buf write that
        # produces values consumed by schedule stream. Reuses one Event across
        # iters; record() repositions it to the current forward-stream point.
        # Forward-stream FIFO guarantees the new position is at-or-past every
        # prior write recorded on this event.
        if self._last_store_done is None:
            self._last_store_done = torch.get_device_module(self.device).Event()
        self._last_store_done.record()

    def store_post_verify(
        self,
        future_indices: FutureIndices,
        new_seq_lens: torch.Tensor,
        bonus_tokens: torch.Tensor,
    ) -> None:
        """Forward stream. Writes the buf fields produced at verify-end
        (new_seq_lens, bonus_tokens) and records the cross-stream fence here.

        The fence lands at verify-end rather than at store_to_map (which runs
        after draft_extend), so schedule-stream consumers (resolve_seq_lens_cpu)
        unblock as soon as verify is done — preserving the schedule prep /
        draft_extend overlap window on forward stream.

        First iter (buf not yet allocated): no-op. The subsequent store_to_map
        will write these fields and record the fence.
        """
        if self.spec_algo.is_none() or not self.buf_initialized:
            return
        indices = future_indices.indices
        if indices.shape[0] == 0:
            # DP idle: nothing to store for this rank.
            return
        # Advanced indexing requires explicit dtype cast (slice assignment
        # used to coerce implicitly).
        self.new_seq_lens_buf[indices] = new_seq_lens.to(self.new_seq_lens_buf.dtype)
        self.bonus_tokens_buf[indices] = bonus_tokens.to(self.bonus_tokens_buf.dtype)
        self._record_store_done()
        self._fence_done_by_post_verify = True

    def store_to_map(
        self, future_indices: FutureIndices, batch_result: GenerationBatchResult
    ):
        """Forward stream. Writes all buf fields. Records the cross-stream
        fence only if store_post_verify did not already record it for this
        iter (decode-branch). Extend-branch iters (prefill / mixed) never fire
        the post_verify callback, so this method must write new_seq_lens /
        bonus_tokens here too — otherwise their buf slots stay uninitialized
        (`torch.empty` garbage) and a downstream `.cpu()` overflows int32.
        """
        if self.spec_algo.is_none():
            indices = future_indices.indices
            if indices.shape[0] == 0:
                # DP attention idle rank: indices is empty but next_token_ids
                # may carry padded values from sibling ranks. Nothing to store
                # for this rank.
                return
            # next_token_ids is int32; buf is int64. Slice assignment used to
            # cast implicitly, but advanced indexing requires an explicit match.
            self.token_ids_buf[indices] = batch_result.next_token_ids.to(torch.int64)
            if not self._fence_done_by_post_verify:
                self._record_store_done()
            self._fence_done_by_post_verify = False
            return

        draft_input: EagleDraftInput = batch_result.next_draft_input
        indices = future_indices.indices
        if indices.shape[0] == 0:
            # DP idle rank: draft_input fields are empty stubs without a usable
            # shape, so _lazy_init_buf's shape peek (draft_input.topk_p[0])
            # would IndexError. Defer init until a real batch arrives.
            return

        if not self.buf_initialized:
            self._lazy_init_buf(draft_input)

        # Slice assignment used to coerce src dtype to buf dtype implicitly;
        # advanced index requires an explicit cast. bonus_tokens / new_seq_lens
        # in particular differ across disagg (int64) and forward (int32) paths.
        # topk_p / topk_index / hidden_states are forward-only buf fields
        # (consumed by next iter's resolve_future on forward stream — FIFO
        # ordering covers them, no fence needed).
        self.topk_p_buf[indices] = draft_input.topk_p.to(self.topk_p_buf.dtype)
        self.topk_index_buf[indices] = draft_input.topk_index.to(
            self.topk_index_buf.dtype
        )
        if spec_need_hidden_states():
            self.hidden_states_buf[indices] = draft_input.hidden_states.to(
                self.hidden_states_buf.dtype
            )
        # new_seq_lens / bonus_tokens are schedule-consumed (resolve_seq_lens_cpu
        # reads new_seq_lens_buf via cross-stream D2H). Two write sites:
        #   - decode-branch iter: store_post_verify wrote them already; the
        #     fence is already at verify-end. Skip here to avoid a redundant
        #     write that would race with schedule-stream reads guarded only by
        #     the verify-end fence.
        #   - extend-branch iter (and disagg bootstrap): no post_verify ran,
        #     so write them here and record the fence.
        if not self._fence_done_by_post_verify:
            self.new_seq_lens_buf[indices] = draft_input.new_seq_lens.to(
                self.new_seq_lens_buf.dtype
            )
            self.bonus_tokens_buf[indices] = draft_input.bonus_tokens.to(
                self.bonus_tokens_buf.dtype
            )
            self._record_store_done()
        self._fence_done_by_post_verify = False

    def store_to_map_for_new_batch(
        self, future_indices: FutureIndices, draft_input: EagleDraftInput
    ) -> None:
        """Bootstrap helper for disagg-decode prebuilt batches. The caller
        constructs a fully-populated EagleDraftInput (all post-verify and
        post-draft_extend fields) and asks FutureMap to seed the buf in one
        shot, recording the fence so subsequent resolve_seq_lens_cpu sees
        valid data. Equivalent to the first-iter path in store_to_map."""
        indices = future_indices.indices
        if indices.shape[0] == 0:
            return
        if not self.buf_initialized:
            self._lazy_init_buf(draft_input)
        self.topk_p_buf[indices] = draft_input.topk_p.to(self.topk_p_buf.dtype)
        self.topk_index_buf[indices] = draft_input.topk_index.to(
            self.topk_index_buf.dtype
        )
        self.new_seq_lens_buf[indices] = draft_input.new_seq_lens.to(
            self.new_seq_lens_buf.dtype
        )
        self.bonus_tokens_buf[indices] = draft_input.bonus_tokens.to(
            self.bonus_tokens_buf.dtype
        )
        if spec_need_hidden_states():
            self.hidden_states_buf[indices] = draft_input.hidden_states.to(
                self.hidden_states_buf.dtype
            )
        self._record_store_done()
