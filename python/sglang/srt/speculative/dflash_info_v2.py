"""DFLASH spec-v2 overlap scheduling data structures."""

import contextlib
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.overlap_utils import FutureIndices
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
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


@dataclass
class DFlashDraftInputV2(SpecInput):
    """Draft-side state carried across overlap iterations (spec-v2)."""

    # Legacy Eagle-shaped fields kept only for dataclass compatibility. DFLASH
    # overlap only relays verified_id/new_seq_lens through FutureMap.
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    verified_id: torch.Tensor
    new_seq_lens: torch.Tensor
    hidden_states: torch.Tensor
    verify_done: Optional[torch.cuda.Event] = None
    max_top_k: int = 1
    uniform_top_k_value: Optional[int] = None
    cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None
    planning_seq_lens_cpu: Optional[torch.Tensor] = None
    planning_seq_lens_sum: Optional[int] = None
    reserved_seq_lens_cpu: Optional[torch.Tensor] = None
    reserved_seq_lens_sum: Optional[int] = None
    _prepare_committed_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_planning_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_batch_seq_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_cur_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_nxt_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_cur_kv_lens_gpu_buf: Optional[torch.Tensor] = None
    _prepare_nxt_kv_lens_gpu_buf: Optional[torch.Tensor] = None

    # Filled by scheduler after dispatch.
    future_indices: Optional[FutureIndices] = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DFLASH_DRAFT)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        # Spec v2 draft state itself does not change token accounting.
        return (1, 1)

    def _ensure_prepare_length_buffers(
        self, bs: int, device: torch.device | str
    ) -> None:
        pin_memory = is_pin_memory_available(device)

        def needs_cpu_alloc(buf: Optional[torch.Tensor]) -> bool:
            return buf is None or buf.numel() < bs or buf.is_pinned() != pin_memory

        def needs_gpu_alloc(buf: Optional[torch.Tensor]) -> bool:
            return buf is None or buf.numel() < bs or str(buf.device) != str(device)

        def grown_capacity(buf: Optional[torch.Tensor]) -> int:
            current = 0 if buf is None else int(buf.numel())
            return max(bs, 32, current * 2 if current > 0 else 0)

        if needs_cpu_alloc(self._prepare_committed_kv_lens_cpu_buf):
            capacity = grown_capacity(self._prepare_committed_kv_lens_cpu_buf)
            self._prepare_committed_kv_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )
            self._prepare_planning_kv_lens_cpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device="cpu", pin_memory=pin_memory
            )
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
            verified_id=torch.empty((0,), device=device, dtype=torch.int32),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int32),
            hidden_states=torch.empty((0, 0), device=device, dtype=torch.float16),
            verify_done=None,
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
        if plan_stream is None:
            # Ensure previous forward is completed before mutating shared buffers.
            batch.maybe_wait_verify_done()

        bs = batch.batch_size()
        if bs == 0:
            return
        self._ensure_prepare_length_buffers(bs, batch.device)
        assert self._prepare_committed_kv_lens_cpu_buf is not None
        assert self._prepare_planning_kv_lens_cpu_buf is not None
        assert self._prepare_batch_seq_lens_cpu_buf is not None
        assert self._prepare_cur_kv_lens_cpu_buf is not None
        assert self._prepare_nxt_kv_lens_cpu_buf is not None
        assert self._prepare_cur_kv_lens_gpu_buf is not None
        assert self._prepare_nxt_kv_lens_gpu_buf is not None
        committed_kv_lens_cpu_t = self._prepare_committed_kv_lens_cpu_buf[:bs]
        planning_kv_lens_cpu_t = self._prepare_planning_kv_lens_cpu_buf[:bs]
        batch_seq_lens_cpu_t = self._prepare_batch_seq_lens_cpu_buf[:bs]
        cur_kv_lens_cpu_t = self._prepare_cur_kv_lens_cpu_buf[:bs]
        cur_allocated_seq_lens_cpu = self.cur_allocated_seq_lens_cpu

        # For DFLASH, each decode step needs a fixed-size verify block.
        block_size = int(get_global_server_args().speculative_num_draft_tokens)
        if block_size <= 0:
            raise ValueError(
                f"DFLASH invalid speculative_num_draft_tokens={block_size}."
            )

        page_size = batch.token_to_kv_pool_allocator.page_size
        nxt_kv_lens_cpu_t = self._prepare_nxt_kv_lens_cpu_buf[:bs]
        committed_seq_lens_sum = 0
        planning_seq_lens_sum = 0
        reserved_seq_lens_sum = 0
        num_needed_tokens = 0
        max_top_k = 1
        uniform_top_k_value = None
        uniform_top_k = True
        for i, req in enumerate(batch.reqs):
            committed_len = int(req.kv_committed_len)
            if cur_allocated_seq_lens_cpu is not None and i < len(
                cur_allocated_seq_lens_cpu
            ):
                cur_alloc_len = int(cur_allocated_seq_lens_cpu[i])
            else:
                cur_alloc_len = int(req.kv_allocated_len)
            planning_len = committed_len + block_size
            reserved_len = max(cur_alloc_len, committed_len + 2 * block_size)
            top_k = int(req.sampling_params.top_k)

            committed_kv_lens_cpu_t[i] = committed_len
            batch_seq_lens_cpu_t[i] = committed_len
            cur_kv_lens_cpu_t[i] = cur_alloc_len
            planning_kv_lens_cpu_t[i] = planning_len
            nxt_kv_lens_cpu_t[i] = reserved_len

            committed_seq_lens_sum += committed_len
            planning_seq_lens_sum += planning_len
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

            if plan_stream is not None and self.verify_done is not None:
                plan_stream.wait_event(self.verify_done)

            cur_kv_lens = self._prepare_cur_kv_lens_gpu_buf[:bs]
            nxt_kv_lens = self._prepare_nxt_kv_lens_gpu_buf[:bs]
            cur_kv_lens.copy_(cur_kv_lens_cpu_t, non_blocking=True)
            nxt_kv_lens.copy_(nxt_kv_lens_cpu_t, non_blocking=True)

            if num_needed_tokens > 0:
                if page_size == 1:
                    out_cache_loc = alloc_token_slots(
                        batch.tree_cache, num_needed_tokens
                    )
                else:
                    last_loc = get_last_loc(
                        batch.req_to_token_pool.req_to_token,
                        batch.req_pool_indices,
                        cur_kv_lens,
                    )
                    out_cache_loc = alloc_paged_token_slots_extend(
                        batch.tree_cache,
                        cur_kv_lens,
                        cur_kv_lens_cpu_t,
                        nxt_kv_lens,
                        nxt_kv_lens_cpu_t,
                        last_loc,
                        num_needed_tokens,
                    )

                # Updating req_to_token is a write to a shared tensor: it must not overlap
                # with the previous batch's forward, which also reads req_to_token.
                assign_req_to_token_pool_func(
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    cur_kv_lens,
                    nxt_kv_lens,
                    out_cache_loc,
                    bs,
                )
        if caller_stream is not None:
            # Enqueue the dependency on the caller's stream, not inside the
            # plan-stream context, so forward work cannot observe partially
            # prepared req_to_token / KV allocation state.
            caller_stream.wait_stream(plan_stream)

        for i, req in enumerate(batch.reqs):
            req.kv_allocated_len = int(nxt_kv_lens_cpu_t[i])

        # Preserve the lagging committed CPU view on the batch and carry the
        # tighter host-side planning bound separately from the full reserved
        # allocator upper bound. Overlap scheduling only drifts by at most one
        # DFlash block on the committed prefix lengths.
        batch.seq_lens_cpu = batch_seq_lens_cpu_t
        batch.seq_lens_sum = committed_seq_lens_sum
        self.planning_seq_lens_cpu = planning_kv_lens_cpu_t
        self.planning_seq_lens_sum = planning_seq_lens_sum
        self.reserved_seq_lens_cpu = nxt_kv_lens_cpu_t
        self.reserved_seq_lens_sum = reserved_seq_lens_sum

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        if self.cur_allocated_seq_lens_cpu is not None:
            self.cur_allocated_seq_lens_cpu = self.cur_allocated_seq_lens_cpu[
                new_indices.cpu()
            ]
        if self.planning_seq_lens_cpu is not None:
            self.planning_seq_lens_cpu = self.planning_seq_lens_cpu[new_indices.cpu()]
            self.planning_seq_lens_sum = int(self.planning_seq_lens_cpu.sum().item())
        if self.reserved_seq_lens_cpu is not None:
            self.reserved_seq_lens_cpu = self.reserved_seq_lens_cpu[new_indices.cpu()]
            self.reserved_seq_lens_sum = int(self.reserved_seq_lens_cpu.sum().item())

        if self.future_indices is not None:
            self.future_indices.indices = self.future_indices.indices[new_indices]
            return

        self.topk_p = self.topk_p[new_indices]
        self.topk_index = self.topk_index[new_indices]
        self.verified_id = self.verified_id[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]
        self.hidden_states = self.hidden_states[new_indices]

    def merge_batch(self, spec_info: "DFlashDraftInputV2"):
        if self.cur_allocated_seq_lens_cpu is not None:
            assert spec_info.cur_allocated_seq_lens_cpu is not None
            self.cur_allocated_seq_lens_cpu = torch.cat(
                [self.cur_allocated_seq_lens_cpu, spec_info.cur_allocated_seq_lens_cpu]
            )
        elif spec_info.cur_allocated_seq_lens_cpu is not None:
            self.cur_allocated_seq_lens_cpu = spec_info.cur_allocated_seq_lens_cpu

        if self.planning_seq_lens_cpu is not None:
            assert spec_info.planning_seq_lens_cpu is not None
            self.planning_seq_lens_cpu = torch.cat(
                [self.planning_seq_lens_cpu, spec_info.planning_seq_lens_cpu]
            )
            self.planning_seq_lens_sum = int(self.planning_seq_lens_cpu.sum().item())
        elif spec_info.planning_seq_lens_cpu is not None:
            self.planning_seq_lens_cpu = spec_info.planning_seq_lens_cpu
            self.planning_seq_lens_sum = spec_info.planning_seq_lens_sum

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
            self.future_indices = FutureIndices(
                indices=torch.cat(
                    [self.future_indices.indices, spec_info.future_indices.indices]
                )
            )
            return

        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p], dim=0)
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index], dim=0)
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], dim=0)
        self.new_seq_lens = torch.cat(
            [self.new_seq_lens, spec_info.new_seq_lens], dim=0
        )
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], dim=0
        )
