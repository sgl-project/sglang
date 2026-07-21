"""DFLASH spec-v2 overlap scheduling data structures."""

import contextlib
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.allocation import alloc_for_spec_decode
from sglang.srt.runtime_context import get_server_args
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.utils.common import is_pin_memory_available

_OVERLAP_PLAN_STREAMS: dict[str, torch.cuda.Stream] = {}


def _get_overlap_plan_stream(
    device: torch.device | str,
) -> tuple[Optional[torch.cuda.Stream], contextlib.AbstractContextManager]:
    """Return an optional plan stream/context for overlap scheduling prep kernels."""
    if not envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        return None, contextlib.nullcontext()

    device_str = str(device)
    stream = _OVERLAP_PLAN_STREAMS.get(device_str)
    if stream is None:
        stream = torch.get_device_module(device_str).Stream()
        _OVERLAP_PLAN_STREAMS[device_str] = stream
    return stream, torch.get_device_module(device_str).stream(stream)


class DFlashDraftInputV2(SpecInput):
    """Draft-side state carried across overlap iterations (spec-v2)."""

    def __init__(
        self,
        *,
        topk_p: torch.Tensor,
        topk_index: torch.Tensor,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        hidden_states: torch.Tensor,
        max_top_k: int = 1,
        uniform_top_k_value: Optional[int] = None,
        prefill_tail_hidden_states: Optional[torch.Tensor] = None,
        prefill_tail_valid_mask: Optional[torch.Tensor] = None,
        prefill_tail_start_positions: Optional[torch.Tensor] = None,
        prefill_tail_hidden_projected: bool = True,
        reserved_seq_lens_cpu: Optional[torch.Tensor] = None,
        reserved_seq_lens_sum: Optional[int] = None,
        future_indices: Optional[torch.Tensor] = None,
        verify_token_budget: Optional[int] = None,
    ):
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)
        # Legacy Eagle-shaped fields; DFLASH relays via FutureMap so these are unused.
        self.topk_p = topk_p
        self.topk_index = topk_index
        self.bonus_tokens = bonus_tokens
        self.new_seq_lens = new_seq_lens
        self.hidden_states = hidden_states
        self.max_top_k = max_top_k
        self.uniform_top_k_value = uniform_top_k_value
        self.prefill_tail_hidden_states = prefill_tail_hidden_states
        self.prefill_tail_valid_mask = prefill_tail_valid_mask
        self.prefill_tail_start_positions = prefill_tail_start_positions
        self.prefill_tail_hidden_projected = prefill_tail_hidden_projected
        self.reserved_seq_lens_cpu = reserved_seq_lens_cpu
        self.reserved_seq_lens_sum = reserved_seq_lens_sum
        self._prepare_batch_seq_lens_cpu_buf = None
        self._prepare_cur_kv_lens_cpu_buf = None
        self._prepare_nxt_kv_lens_cpu_buf = None
        self._prepare_cur_kv_lens_gpu_buf = None
        self._prepare_nxt_kv_lens_gpu_buf = None
        self.future_indices = future_indices
        self.verify_token_budget = verify_token_budget
        # Spec v2 draft state itself does not change token accounting.
        self.num_tokens_per_req = 1
        self.num_tokens_for_logprob_per_req = 1

    def _ensure_prepare_length_buffers(
        self, bs: int, device: torch.device | str
    ) -> None:
        pin_memory = is_pin_memory_available(device)

        def needs_cpu_alloc(buf: Optional[torch.Tensor]) -> bool:
            return buf is None or buf.numel() < bs

        def needs_gpu_alloc(buf: Optional[torch.Tensor]) -> bool:
            return buf is None or buf.numel() < bs or str(buf.device) != str(device)

        def grown_capacity(buf: Optional[torch.Tensor]) -> int:
            current = 0 if buf is None else int(buf.numel())
            return max(bs, 32, current * 2 if current > 0 else 0)

        # The three CPU scratch buffers grow together; capacity is the only
        # invariant (batch is int64 non-pinned, cur/nxt are int32 pinned).
        if needs_cpu_alloc(self._prepare_batch_seq_lens_cpu_buf):
            capacity = grown_capacity(self._prepare_batch_seq_lens_cpu_buf)
            self._prepare_batch_seq_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int64, device="cpu"
            )
            self._prepare_cur_kv_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )
            self._prepare_nxt_kv_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )

        if needs_gpu_alloc(self._prepare_cur_kv_lens_gpu_buf):
            capacity = grown_capacity(self._prepare_cur_kv_lens_gpu_buf)
            self._prepare_cur_kv_lens_gpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device=device
            )
            self._prepare_nxt_kv_lens_gpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device=device
            )

    @classmethod
    def create_idle_input(cls, device: torch.device) -> "DFlashDraftInputV2":
        return cls(
            topk_p=torch.empty((0, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, 0), device=device, dtype=torch.int64),
            bonus_tokens=torch.empty((0,), device=device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int64),
            hidden_states=torch.empty((0, 0), device=device, dtype=torch.float16),
        )

    def prepare_for_decode(self, batch: ScheduleBatch):
        """Allocate headroom in the shared req_to_token pool for the next DFLASH step.

        DFLASH spec-v2 uses overlap scheduling's "over-allocation" approach: we reserve
        future KV slots ahead of time so the worker can gather `out_cache_loc` directly
        from `req_to_token` without allocator backup/restore. CPU metadata intentionally
        lags by one iteration; keep it separate from the reserved upper bound that backs
        the overallocated mapping.
        """
        plan_stream, plan_stream_ctx = _get_overlap_plan_stream(batch.device)

        bs = batch.batch_size()
        if bs == 0:
            return
        self._ensure_prepare_length_buffers(bs, batch.device)
        assert self._prepare_batch_seq_lens_cpu_buf is not None
        assert self._prepare_cur_kv_lens_cpu_buf is not None
        assert self._prepare_nxt_kv_lens_cpu_buf is not None
        assert self._prepare_cur_kv_lens_gpu_buf is not None
        assert self._prepare_nxt_kv_lens_gpu_buf is not None
        batch_seq_lens_cpu_t = self._prepare_batch_seq_lens_cpu_buf[:bs]
        cur_kv_lens_cpu_t = self._prepare_cur_kv_lens_cpu_buf[:bs]

        # For DFLASH, each decode step needs a fixed-size verify block.
        block_size = int(get_server_args().speculative_num_draft_tokens)
        if block_size <= 0:
            raise ValueError(
                f"DFLASH invalid speculative_num_draft_tokens={block_size}."
            )
        page_size = batch.token_to_kv_pool_allocator.page_size
        nxt_kv_lens_cpu_t = self._prepare_nxt_kv_lens_cpu_buf[:bs]
        committed_seq_lens_sum = 0
        reserved_seq_lens_sum = 0
        num_needed_tokens = 0
        max_top_k = 1
        uniform_top_k_value = None
        uniform_top_k = True
        for i, req in enumerate(batch.reqs):
            committed_len = int(req.kv_committed_len)
            # Read the allocation watermark from the req object like EAGLE.
            cur_alloc_len = int(req.kv.kv_allocated_len)
            reserved_len = max(cur_alloc_len, committed_len + 2 * block_size)
            top_k = int(req.sampling_params.top_k)

            batch_seq_lens_cpu_t[i] = committed_len
            cur_kv_lens_cpu_t[i] = cur_alloc_len
            nxt_kv_lens_cpu_t[i] = reserved_len

            committed_seq_lens_sum += committed_len
            reserved_seq_lens_sum += reserved_len
            num_needed_tokens += reserved_len - cur_alloc_len

            if top_k > max_top_k:
                max_top_k = top_k
            if i == 0:
                uniform_top_k_value = top_k
            elif uniform_top_k and top_k != uniform_top_k_value:
                uniform_top_k = False

        self.max_top_k = max(max_top_k, 1)
        self.uniform_top_k_value = uniform_top_k_value if uniform_top_k else None

        caller_stream = None
        if plan_stream is not None:
            caller_stream = torch.get_device_module(batch.device).current_stream()

        with plan_stream_ctx:
            if plan_stream is not None and caller_stream is not None:
                # `batch.seq_lens`, `batch.req_pool_indices`, and related tensors may
                # have just been rebuilt on the scheduler stream by filter/merge ops.
                # The plan stream must wait for those writes before reading them.
                plan_stream.wait_stream(caller_stream)

            cur_kv_lens = self._prepare_cur_kv_lens_gpu_buf[:bs]
            nxt_kv_lens = self._prepare_nxt_kv_lens_gpu_buf[:bs]
            cur_kv_lens.copy_(cur_kv_lens_cpu_t, non_blocking=True)
            nxt_kv_lens.copy_(nxt_kv_lens_cpu_t, non_blocking=True)

            alloc_for_spec_decode(
                batch.tree_cache,
                batch.req_to_token_pool,
                reqs=batch.reqs,
                req_pool_indices=batch.req_pool_indices,
                cur_kv_lens=cur_kv_lens,
                cur_kv_lens_cpu=cur_kv_lens_cpu_t,
                nxt_kv_lens=nxt_kv_lens,
                nxt_kv_lens_cpu=nxt_kv_lens_cpu_t,
                num_needed_tokens=num_needed_tokens,
                batch=batch,
            )
        if caller_stream is not None:
            # Enqueue the dependency on the caller's stream, not inside the
            # plan-stream context, so forward work cannot observe partially
            # prepared req_to_token / KV allocation state.
            caller_stream.wait_stream(plan_stream)

        # Seed committed; overlap's resolve overwrites it with the published value.
        batch.seq_lens_cpu = batch_seq_lens_cpu_t
        batch.seq_lens_sum = committed_seq_lens_sum
        self.reserved_seq_lens_cpu = nxt_kv_lens_cpu_t
        self.reserved_seq_lens_sum = reserved_seq_lens_sum

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.reserved_seq_lens_cpu is not None:
            self.reserved_seq_lens_cpu = self.reserved_seq_lens_cpu[new_indices.cpu()]
            self.reserved_seq_lens_sum = int(self.reserved_seq_lens_cpu.sum().item())

        if (
            self.prefill_tail_hidden_states is not None
            and self.prefill_tail_hidden_states.numel() > 0
        ):
            lengths = self.prefill_tail_valid_mask.to(torch.int64)
            selected = torch.zeros(
                lengths.shape[0], dtype=torch.bool, device=lengths.device
            )
            selected[new_indices] = True
            row_mask = torch.repeat_interleave(selected, lengths)
            self.prefill_tail_hidden_states = self.prefill_tail_hidden_states[row_mask]
        if (
            self.prefill_tail_valid_mask is not None
            and self.prefill_tail_valid_mask.numel() > 0
        ):
            self.prefill_tail_valid_mask = self.prefill_tail_valid_mask[new_indices]
        if (
            self.prefill_tail_start_positions is not None
            and self.prefill_tail_start_positions.numel() > 0
        ):
            self.prefill_tail_start_positions = self.prefill_tail_start_positions[
                new_indices
            ]

        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
            return

        self.topk_p = self.topk_p[new_indices]
        self.topk_index = self.topk_index[new_indices]
        self.bonus_tokens = self.bonus_tokens[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]
        self.hidden_states = self.hidden_states[new_indices]

    def merge_batch(self, spec_info: "DFlashDraftInputV2"):
        lhs_bs = self._batch_size()
        rhs_bs = spec_info._batch_size()

        if self.reserved_seq_lens_cpu is not None:
            assert spec_info.reserved_seq_lens_cpu is not None
            self.reserved_seq_lens_cpu = torch.cat(
                [self.reserved_seq_lens_cpu, spec_info.reserved_seq_lens_cpu]
            )
            self.reserved_seq_lens_sum = int(self.reserved_seq_lens_cpu.sum().item())
        elif spec_info.reserved_seq_lens_cpu is not None:
            self.reserved_seq_lens_cpu = spec_info.reserved_seq_lens_cpu
            self.reserved_seq_lens_sum = spec_info.reserved_seq_lens_sum

        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = torch.cat(
                [self.future_indices, spec_info.future_indices]
            )
            self._merge_prefill_tail(spec_info, lhs_bs, rhs_bs)
            return

        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p], dim=0)
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index], dim=0)
        self.bonus_tokens = torch.cat(
            [self.bonus_tokens, spec_info.bonus_tokens], dim=0
        )
        self.new_seq_lens = torch.cat(
            [self.new_seq_lens, spec_info.new_seq_lens], dim=0
        )
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], dim=0
        )
        self._merge_prefill_tail(spec_info, lhs_bs, rhs_bs)

    def _batch_size(self) -> int:
        if self.future_indices is not None:
            return int(self.future_indices.shape[0])
        return int(self.bonus_tokens.shape[0])

    def _merge_prefill_tail(
        self, spec_info: "DFlashDraftInputV2", lhs_bs: int, rhs_bs: int
    ) -> None:
        self.prefill_tail_hidden_projected = (
            self.prefill_tail_hidden_projected
            and spec_info.prefill_tail_hidden_projected
        )
        lhs_hidden = self.prefill_tail_hidden_states
        rhs_hidden = spec_info.prefill_tail_hidden_states
        lhs_mask = self.prefill_tail_valid_mask
        rhs_mask = spec_info.prefill_tail_valid_mask
        lhs_start = self.prefill_tail_start_positions
        rhs_start = spec_info.prefill_tail_start_positions
        if lhs_hidden is None and rhs_hidden is None:
            return
        hidden_template = rhs_hidden if lhs_hidden is None else lhs_hidden
        if lhs_hidden is None:
            lhs_hidden = hidden_template.new_empty((0, hidden_template.shape[-1]))
            lhs_mask = torch.zeros(lhs_bs, dtype=torch.int64, device=rhs_mask.device)
            lhs_start = torch.zeros(
                (lhs_bs,), dtype=rhs_start.dtype, device=rhs_start.device
            )
        if rhs_hidden is None:
            rhs_hidden = hidden_template.new_empty((0, hidden_template.shape[-1]))
            rhs_mask = torch.zeros(rhs_bs, dtype=torch.int64, device=lhs_mask.device)
            rhs_start = torch.zeros(
                (rhs_bs,), dtype=lhs_start.dtype, device=lhs_start.device
            )

        self.prefill_tail_hidden_states = torch.cat([lhs_hidden, rhs_hidden], dim=0)
        self.prefill_tail_valid_mask = torch.cat([lhs_mask, rhs_mask], dim=0)
        self.prefill_tail_start_positions = torch.cat([lhs_start, rhs_start], dim=0)
