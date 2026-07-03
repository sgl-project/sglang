from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.spec_info import (
    SpecInput,
    SpecInputType,
    SpeculativeAlgorithm,
)
from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.managers.tp_worker import TpModelWorker


@dataclass
class _DSparkBlockInputBase(SpecInput):
    draft_token: torch.Tensor
    positions: torch.Tensor
    draft_token_num: int
    topk: int = 1
    custom_mask: torch.Tensor | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL
    num_tokens_per_batch: int = -1
    num_tokens_per_req: int = -1

    def _init_spec_input(self, spec_input_type: SpecInputType):
        SpecInput.__init__(self, spec_input_type=spec_input_type)
        if self.num_tokens_per_batch == -1:
            self.num_tokens_per_batch = int(self.draft_token_num)
        if self.num_tokens_per_req == -1:
            self.num_tokens_per_req = int(self.draft_token_num)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
        kv_start_idx: Optional[torch.Tensor] = None,
    ):
        device = req_pool_indices.device
        bs = len(req_pool_indices)

        qo_indptr = torch.arange(
            0,
            (bs + 1) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device=device,
        )

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * bs,
            dtype=torch.int32,
            device=device,
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            kv_start_idx,
            kv_indices,
            req_to_token.size(1),
        )
        mask = self.custom_mask
        if mask is not None:
            mask_numel = (
                paged_kernel_lens_sum * self.draft_token_num
                + (self.draft_token_num**2) * bs
            )
            if mask.numel() < mask_numel:
                mask = torch.cat(
                    [
                        mask,
                        torch.full(
                            (mask_numel - mask.numel(),),
                            True,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ],
                    dim=0,
                )
                self.custom_mask = mask
        return kv_indices, cum_kv_seq_len, qo_indptr, mask


@dataclass
class DSparkDraftBlockInput(_DSparkBlockInputBase):
    def __post_init__(self):
        self._init_spec_input(SpecInputType.DSPARK_DRAFT_BLOCK)

    def prepare_for_draft_block(
        self,
        batch: ScheduleBatch,
        draft_model_runner: ModelRunner,
        out_cache_loc: torch.Tensor,
        dp_decode_global_num_tokens: Optional[list[int]] = None,
    ) -> ForwardBatch:
        batch.input_ids = self.draft_token
        batch.out_cache_loc = out_cache_loc
        batch.spec_info = self
        batch.spec_algorithm = SpeculativeAlgorithm.DSPARK
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.capture_hidden_mode = self.capture_hidden_mode

        draft_forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        draft_forward_batch.lora_ids = [None] * draft_forward_batch.batch_size

        server_args = get_global_server_args()
        if (
            server_args.enable_dp_attention
            and dp_decode_global_num_tokens is not None
        ):
            draft_global_num_tokens = [
                int(x) * int(self.draft_token_num) for x in dp_decode_global_num_tokens
            ]
            draft_forward_batch.global_num_tokens_cpu = draft_global_num_tokens
            draft_forward_batch.global_num_tokens_gpu = torch.tensor(
                draft_global_num_tokens,
                dtype=torch.int64,
                device=draft_model_runner.device,
            )
            draft_forward_batch.global_num_tokens_for_logprob_cpu = (
                draft_global_num_tokens
            )
            draft_forward_batch.global_num_tokens_for_logprob_gpu = torch.tensor(
                draft_global_num_tokens,
                dtype=torch.int64,
                device=draft_model_runner.device,
            )

        return draft_forward_batch


@dataclass
class DSparkVerifyInput(_DSparkBlockInputBase):
    def __post_init__(self):
        self._init_spec_input(SpecInputType.DSPARK_VERIFY)

    def prepare_for_verify(
        self,
        batch: ScheduleBatch,
        target_worker: TpModelWorker,
    ) -> tuple[ForwardBatch, bool]:
        batch.input_ids = self.draft_token
        batch.spec_info = self
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = self.capture_hidden_mode
        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

        can_run_cuda_graph = bool(
            target_worker.model_runner.decode_cuda_graph_runner
            and target_worker.model_runner.decode_cuda_graph_runner.can_run_graph(
                verify_forward_batch
            )
        )
        if can_run_cuda_graph:
            target_worker.model_runner.decode_cuda_graph_runner.load_batch(
                verify_forward_batch
            )
            verify_forward_batch.mark_forward_metadata_ready()
        # Non-cuda-graph: defer metadata init to the forward path so DP attention
        # padding in prepare_mlp_sync_batch is reflected in the backend plan.

        return verify_forward_batch, can_run_cuda_graph


@dataclass
class DSparkDraftInputV2(SpecInput):
    bonus_tokens: torch.Tensor
    new_seq_lens: torch.Tensor
    main_hidden: Optional[torch.Tensor] = None
    confidence: Optional[torch.Tensor] = None
    verify_done: Optional[torch.cuda.Event] = None
    cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None
    reserved_seq_lens_cpu: Optional[torch.Tensor] = None
    reserved_seq_lens_sum: Optional[int] = None
    topk_p: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 0), dtype=torch.float32)
    )
    topk_index: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 0), dtype=torch.int64)
    )
    hidden_states: torch.Tensor = field(
        default_factory=lambda: torch.empty((0, 0), dtype=torch.float16)
    )
    transfer_warmup_rounds: torch.Tensor = field(
        default_factory=lambda: torch.empty((0,), dtype=torch.int32)
    )
    direct_carry_valid: bool = True
    future_indices: Optional[torch.Tensor] = None
    _prepare_batch_seq_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_cur_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_nxt_kv_lens_cpu_buf: Optional[torch.Tensor] = None
    _prepare_cur_kv_lens_gpu_buf: Optional[torch.Tensor] = None
    _prepare_nxt_kv_lens_gpu_buf: Optional[torch.Tensor] = None

    def __post_init__(self):
        super().__init__(spec_input_type=SpecInputType.DSPARK_DRAFT)
        device = self.bonus_tokens.device

        def move_empty_to_device(tensor: Optional[torch.Tensor]):
            if tensor is None or tensor.numel() != 0 or tensor.device == device:
                return tensor
            return torch.empty(tensor.shape, dtype=tensor.dtype, device=device)

        self.topk_p = move_empty_to_device(self.topk_p)
        self.topk_index = move_empty_to_device(self.topk_index)
        self.hidden_states = move_empty_to_device(self.hidden_states)
        self.transfer_warmup_rounds = move_empty_to_device(
            self.transfer_warmup_rounds
        )

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return 1, 1

    def carry_prepare_buffers_from(self, other: DSparkDraftInputV2) -> None:
        self._prepare_batch_seq_lens_cpu_buf = other._prepare_batch_seq_lens_cpu_buf
        self._prepare_cur_kv_lens_cpu_buf = other._prepare_cur_kv_lens_cpu_buf
        self._prepare_nxt_kv_lens_cpu_buf = other._prepare_nxt_kv_lens_cpu_buf
        self._prepare_cur_kv_lens_gpu_buf = other._prepare_cur_kv_lens_gpu_buf
        self._prepare_nxt_kv_lens_gpu_buf = other._prepare_nxt_kv_lens_gpu_buf

    def _ensure_prepare_length_buffers(
        self, bs: int, device: torch.device | str, need_gpu: bool = False
    ) -> None:
        pin_memory = is_pin_memory_available(device)

        def needs_cpu_alloc(buf: Optional[torch.Tensor]) -> bool:
            return buf is None or buf.numel() < bs

        def needs_gpu_alloc(buf: Optional[torch.Tensor]) -> bool:
            return buf is None or buf.numel() < bs or str(buf.device) != str(device)

        def grown_capacity(buf: Optional[torch.Tensor]) -> int:
            current = 0 if buf is None else int(buf.numel())
            return max(bs, 32, current * 2 if current > 0 else 0)

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

        if need_gpu and needs_gpu_alloc(self._prepare_cur_kv_lens_gpu_buf):
            capacity = grown_capacity(self._prepare_cur_kv_lens_gpu_buf)
            self._prepare_cur_kv_lens_gpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device=device
            )
            self._prepare_nxt_kv_lens_gpu_buf = torch.empty(
                (capacity,), dtype=torch.int32, device=device
            )

    @classmethod
    def create_idle_input(cls, device: torch.device) -> DSparkDraftInputV2:
        return cls(
            bonus_tokens=torch.empty((0,), device=device, dtype=torch.int64),
            new_seq_lens=torch.empty((0,), device=device, dtype=torch.int64),
            topk_p=torch.empty((0, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, 0), device=device, dtype=torch.int64),
            hidden_states=torch.empty((0, 0), device=device, dtype=torch.float16),
            transfer_warmup_rounds=torch.empty((0,), device=device, dtype=torch.int32),
            verify_done=None,
        )

    def prepare_for_decode(self, batch: ScheduleBatch):
        if self.verify_done is not None:
            torch.get_device_module(batch.device).current_stream().wait_event(
                self.verify_done
            )

        bs = batch.batch_size()
        if bs == 0:
            return

        block_size = int(get_global_server_args().speculative_num_draft_tokens)
        if block_size <= 0:
            raise ValueError(
                f"DSpark invalid speculative_num_draft_tokens={block_size}."
            )

        device = batch.device
        page_size = batch.token_to_kv_pool_allocator.page_size
        cur_alloc = self.cur_allocated_seq_lens_cpu

        self._ensure_prepare_length_buffers(bs, device, need_gpu=False)
        assert self._prepare_batch_seq_lens_cpu_buf is not None
        assert self._prepare_cur_kv_lens_cpu_buf is not None
        assert self._prepare_nxt_kv_lens_cpu_buf is not None
        committed_cpu = self._prepare_batch_seq_lens_cpu_buf[:bs]
        cur_kv_lens_cpu = self._prepare_cur_kv_lens_cpu_buf[:bs]
        nxt_kv_lens_cpu = self._prepare_nxt_kv_lens_cpu_buf[:bs]
        committed_sum = 0
        reserved_sum = 0
        num_needed_tokens = 0
        for i, req in enumerate(batch.reqs):
            committed_len = int(req.kv_committed_len)
            if cur_alloc is not None and i < len(cur_alloc):
                cur_alloc_len = int(cur_alloc[i])
            else:
                cur_alloc_len = int(req.kv_allocated_len)
            reserved_len = max(cur_alloc_len, committed_len + 2 * block_size)
            cur_kv_lens_cpu[i] = cur_alloc_len
            nxt_kv_lens_cpu[i] = reserved_len
            committed_cpu[i] = committed_len
            committed_sum += committed_len
            reserved_sum += reserved_len
            num_needed_tokens += reserved_len - cur_alloc_len

        if num_needed_tokens > 0:
            self._ensure_prepare_length_buffers(bs, device, need_gpu=True)
            assert self._prepare_cur_kv_lens_gpu_buf is not None
            assert self._prepare_nxt_kv_lens_gpu_buf is not None
            cur_kv_lens = self._prepare_cur_kv_lens_gpu_buf[:bs]
            nxt_kv_lens = self._prepare_nxt_kv_lens_gpu_buf[:bs]
            cur_kv_lens.copy_(cur_kv_lens_cpu, non_blocking=True)
            nxt_kv_lens.copy_(nxt_kv_lens_cpu, non_blocking=True)

            if page_size == 1:
                out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
            else:
                last_loc = get_last_loc(
                    batch.req_to_token_pool.req_to_token,
                    batch.req_pool_indices,
                    cur_kv_lens,
                )
                out_cache_loc = alloc_paged_token_slots_extend(
                    batch.tree_cache,
                    cur_kv_lens,
                    cur_kv_lens_cpu,
                    nxt_kv_lens,
                    nxt_kv_lens_cpu,
                    last_loc,
                    num_needed_tokens,
                )
            assign_req_to_token_pool_func(
                batch.req_pool_indices,
                batch.req_to_token_pool.req_to_token,
                cur_kv_lens,
                nxt_kv_lens,
                out_cache_loc,
                bs,
            )

        for i, req in enumerate(batch.reqs):
            req.kv_allocated_len = max(req.kv_allocated_len, int(nxt_kv_lens_cpu[i]))

        batch.seq_lens_cpu = committed_cpu
        batch.seq_lens_sum = committed_sum
        self.reserved_seq_lens_cpu = nxt_kv_lens_cpu
        self.reserved_seq_lens_sum = reserved_sum

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        cpu_indices = new_indices.cpu()
        if self.cur_allocated_seq_lens_cpu is not None:
            self.cur_allocated_seq_lens_cpu = self.cur_allocated_seq_lens_cpu[
                cpu_indices
            ]
        if self.reserved_seq_lens_cpu is not None:
            self.reserved_seq_lens_cpu = self.reserved_seq_lens_cpu[cpu_indices]
            self.reserved_seq_lens_sum = int(self.reserved_seq_lens_cpu.sum().item())

        if self.future_indices is not None:
            self.future_indices = self.future_indices[new_indices]
            if self.hidden_states.numel() > 0:
                self.hidden_states = self.hidden_states[new_indices]
            if self.transfer_warmup_rounds.numel() > 0:
                self.transfer_warmup_rounds = self.transfer_warmup_rounds[new_indices]
            self.direct_carry_valid = False
            return

        self.bonus_tokens = self.bonus_tokens[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]
        if self.main_hidden is not None:
            self.main_hidden = self.main_hidden[new_indices]
        if self.confidence is not None:
            self.confidence = self.confidence[new_indices]
        if self.topk_p.numel() > 0:
            self.topk_p = self.topk_p[new_indices]
        if self.topk_index.numel() > 0:
            self.topk_index = self.topk_index[new_indices]
        if self.hidden_states.numel() > 0:
            self.hidden_states = self.hidden_states[new_indices]
        if self.transfer_warmup_rounds.numel() > 0:
            self.transfer_warmup_rounds = self.transfer_warmup_rounds[new_indices]

    def merge_batch(self, spec_info: DSparkDraftInputV2):
        if (
            self.cur_allocated_seq_lens_cpu is not None
            and spec_info.cur_allocated_seq_lens_cpu is not None
        ):
            self.cur_allocated_seq_lens_cpu = torch.cat(
                [self.cur_allocated_seq_lens_cpu, spec_info.cur_allocated_seq_lens_cpu]
            )
        else:
            self.cur_allocated_seq_lens_cpu = None

        if (
            self.reserved_seq_lens_cpu is not None
            and spec_info.reserved_seq_lens_cpu is not None
        ):
            self.reserved_seq_lens_cpu = torch.cat(
                [self.reserved_seq_lens_cpu, spec_info.reserved_seq_lens_cpu]
            )
            self.reserved_seq_lens_sum = int(self.reserved_seq_lens_cpu.sum().item())
        else:
            self.reserved_seq_lens_cpu = None
            self.reserved_seq_lens_sum = None

        if self.future_indices is not None:
            assert spec_info.future_indices is not None
            self.future_indices = torch.cat(
                [self.future_indices, spec_info.future_indices]
            )
            if self.hidden_states.numel() == 0:
                self.hidden_states = spec_info.hidden_states
            elif spec_info.hidden_states.numel() > 0:
                self.hidden_states = torch.cat(
                    [self.hidden_states, spec_info.hidden_states], dim=0
                )
            self.transfer_warmup_rounds = torch.cat(
                [self.transfer_warmup_rounds, spec_info.transfer_warmup_rounds],
                dim=0,
            )
            self.direct_carry_valid = False
            return

        self.bonus_tokens = torch.cat(
            [self.bonus_tokens, spec_info.bonus_tokens], dim=0
        )
        self.new_seq_lens = torch.cat(
            [self.new_seq_lens, spec_info.new_seq_lens], dim=0
        )
        if self.main_hidden is not None and spec_info.main_hidden is not None:
            self.main_hidden = torch.cat(
                [self.main_hidden, spec_info.main_hidden], dim=0
            )
        if self.confidence is not None and spec_info.confidence is not None:
            self.confidence = torch.cat([self.confidence, spec_info.confidence], dim=0)
        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p], dim=0)
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index], dim=0)
        if self.hidden_states.numel() == 0:
            self.hidden_states = spec_info.hidden_states
        elif spec_info.hidden_states.numel() == 0:
            pass
        else:
            self.hidden_states = torch.cat(
                [self.hidden_states, spec_info.hidden_states], dim=0
            )
        self.transfer_warmup_rounds = torch.cat(
            [self.transfer_warmup_rounds, spec_info.transfer_warmup_rounds], dim=0
        )
