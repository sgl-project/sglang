from __future__ import annotations

from typing import TYPE_CHECKING, Union

import torch

from sglang.srt.speculative.spec_utils import spec_need_hidden_states

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def resolve_forward_inputs(batch: ScheduleBatch, future_map: FutureMap) -> None:
    """Materialize input_ids at forward entry. Prefill H2D from pinned CPU
    staging; decode gather from future_map.output_tokens_buf. Spec_v2 sets
    input_ids inside the worker (draft tokens); only extras are gathered here.
    """
    if batch.prefill_input_ids_cpu is not None:
        prefill_dev = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
        if batch.mix_running_indices is not None:
            decode_dev = future_map.output_tokens_buf[batch.mix_running_indices]
            batch.input_ids = torch.cat([prefill_dev, decode_dev])
        else:
            batch.input_ids = prefill_dev
        batch.prefill_input_ids_cpu = None
        batch.mix_running_indices = None
    elif batch.input_ids is None and future_map.spec_algo.is_none():
        batch.input_ids = future_map.output_tokens_buf[batch.req_pool_indices]

    # spec_v1 (non-overlap spec) doesn't relay extras; only spec_v2 does.
    if batch.is_spec_v2:
        future_map._resolve_spec_extras(batch)


class FutureMap:
    """Always-on pool-indexed relay for cross-iter values. Forward writes via
    publish/stash; next iter reads via resolve_forward_inputs / resolve_seq_lens_cpu.
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
        if spec_need_hidden_states():
            draft_input.hidden_states = self.hidden_states_buf[indices]

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
        # Dispatch by payload type, not spec_algo: spec_v1 (non-overlap spec)
        # also passes a token Tensor here.
        if isinstance(payload, torch.Tensor):
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
