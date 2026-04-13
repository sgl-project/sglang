from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.distributed import get_tp_group
from sglang.srt.layers.dp_attention import (
    get_attention_tp_group,
    is_dp_attention_enabled,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.utils import get_alloc_len_per_decode
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ExtendNumTokens,
    ForwardBatch,
    ForwardMode,
    KvLen,
    LastLoc,
    OutCacheLoc,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.penaltylib.repetition_penalty import apply_scaling_penalties
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    generate_simulated_accept_index,
)
from sglang.srt.utils.common import is_cuda, is_hip, is_npu, next_power_of_2

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )
    from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
    )


@triton.jit
def assign_draft_cache_locs_page_size_1(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    copy_len = topk * speculative_num_steps
    out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps

    # Copy from req_to_token to out_cache_loc
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(token_pool + kv_start + copy_offset, mask=mask)
        tl.store(out_cache_ptr + copy_offset, data, mask=mask)


@dataclass
class EagleDraftInputV2Mixin:
    def prepare_for_decode(self: EagleDraftInput, batch: ScheduleBatch):
        batch.maybe_evict_swa()

        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = batch.batch_size()

        # Now seq_lens is correct
        batch.maybe_wait_verify_done()

        # Accumulate penalty
        # This is a relaxed version of penalties for speculative decoding.
        if batch.sampling_info.penalizer_orchestrator.is_required:
            output_ids = torch.tensor(
                [
                    (
                        req.output_ids[-1]
                        if len(req.output_ids)
                        else req.origin_input_ids[-1]
                    )
                    for req in batch.reqs
                ],
                dtype=torch.int64,
                device=batch.device,
            )
            batch.sampling_info.penalizer_orchestrator.cumulate_output_tokens(
                output_ids
            )

        page_size = batch.token_to_kv_pool_allocator.page_size

        # DSV4
        cur_full_kv_lens_cpu = []
        nxt_full_kv_lens_cpu = []
        cur_c4_kv_lens_cpu = []
        nxt_c4_kv_lens_cpu = []
        cur_c128_kv_lens_cpu = []
        nxt_c128_kv_lens_cpu = []
        num_needed_tokens = []
        full_num_needed_token = 0
        c4_num_needed_token = 0
        c128_num_needed_token = 0

        alloc_len_per_decode = get_alloc_len_per_decode()
        for r in batch.reqs:
            # Over-allocation happens here
            x = r.kv_committed_len + 2 * alloc_len_per_decode - r.kv_allocated_len
            cur_full_kv_lens_cpu.append(r.kv_allocated_len)
            nxt_full_kv_lens_cpu.append(r.kv_allocated_len + x)
            full_num_needed_token += x
            r.kv_allocated_len += x

            x = r.kv_allocated_len // 4 - r.c4_kv_allocated_len
            cur_c4_kv_lens_cpu.append(r.c4_kv_allocated_len)
            nxt_c4_kv_lens_cpu.append(r.c4_kv_allocated_len + x)
            c4_num_needed_token += x
            r.c4_kv_allocated_len += x

            x = r.kv_allocated_len // 128 - r.c128_kv_allocated_len
            cur_c128_kv_lens_cpu.append(r.c128_kv_allocated_len)
            nxt_c128_kv_lens_cpu.append(r.c128_kv_allocated_len + x)
            c128_num_needed_token += x
            r.c128_kv_allocated_len += x

            r.decode_batch_idx += 1
        num_needed_tokens = ExtendNumTokens(
            full_num_needed_token,
            full_num_needed_token,
            c4_num_needed_token,
            c128_num_needed_token,
            full_num_needed_token,
            full_num_needed_token,
        )

        cur_kv_lens_cpu_list = [
            cur_full_kv_lens_cpu,
            cur_full_kv_lens_cpu,
            cur_c4_kv_lens_cpu,
            cur_c128_kv_lens_cpu,
            cur_full_kv_lens_cpu,
            cur_full_kv_lens_cpu,
        ]

        nxt_kv_lens_cpu_list = [
            nxt_full_kv_lens_cpu,
            nxt_full_kv_lens_cpu,
            nxt_c4_kv_lens_cpu,
            nxt_c128_kv_lens_cpu,
            nxt_full_kv_lens_cpu,
            nxt_full_kv_lens_cpu,
        ]

        cur_kv_lens = None
        nxt_kv_lens = None

        if page_size == 1:
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
        else:
            cur_kv_lens_cpu = KvLen.from_list(
                cur_kv_lens_cpu_list,
                dtype=torch.int64,
            )
            cur_kv_lens = cur_kv_lens_cpu.to(
                device=batch.device,
            )

            nxt_kv_lens_cpu = KvLen.from_list(
                nxt_kv_lens_cpu_list,
                dtype=torch.int64,
            )
            nxt_kv_lens = nxt_kv_lens_cpu.to(
                device=batch.device,
            )

            last_full_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                cur_kv_lens.full_kv_len,
            )
            last_swa_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token_swa,
                batch.req_pool_indices,
                cur_kv_lens.swa_kv_len,
            )
            last_c4_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token_c4,
                batch.req_pool_indices,
                cur_kv_lens.c4_kv_len,
            )
            last_c128_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token_c128,
                batch.req_pool_indices,
                cur_kv_lens.c128_kv_len,
            )
            last_c4_state_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token_c4_state,
                batch.req_pool_indices,
                cur_kv_lens.c4_state_kv_len,
            )
            last_c128_state_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token_c128_state,
                batch.req_pool_indices,
                cur_kv_lens.c128_state_kv_len,
            )
            last_loc = LastLoc(
                last_full_loc,
                last_swa_loc,
                last_c4_loc,
                last_c128_loc,
                last_c4_state_loc,
                last_c128_state_loc,
            )
            out_alloc_meta_data = alloc_paged_token_slots_extend(
                batch.tree_cache,
                cur_kv_lens,  # pre
                cur_kv_lens_cpu,
                nxt_kv_lens,  # cur
                nxt_kv_lens_cpu,
                last_loc,  # Las
                num_needed_tokens,  # List
            )

        if cur_kv_lens is None:
            cur_kv_lens = cur_kv_lens_cpu.pin_memory().to(
                device=batch.device, non_blocking=True
            )
            nxt_kv_lens = nxt_kv_lens_cpu.pin_memory().to(
                device=batch.device, non_blocking=True
            )

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            cur_kv_lens.full_kv_len,
            nxt_kv_lens.full_kv_len,
            out_alloc_meta_data.out_full_loc,
            bs,
        )

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token_swa,
            cur_kv_lens.swa_kv_len,
            nxt_kv_lens.swa_kv_len,
            out_alloc_meta_data.out_swa_loc,
            bs,
        )

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token_c4,
            cur_kv_lens.c4_kv_len,
            nxt_kv_lens.c4_kv_len,
            out_alloc_meta_data.out_c4_loc,
            bs,
        )

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token_c128,
            cur_kv_lens.c128_kv_len,
            nxt_kv_lens.c128_kv_len,
            out_alloc_meta_data.out_c128_loc,
            bs,
        )

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token_c4_state,
            cur_kv_lens.c4_state_kv_len,
            nxt_kv_lens.c4_state_kv_len,
            out_alloc_meta_data.out_c4_state_loc,
            bs,
        )

        assign_req_to_token_pool_func(
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token_c128_state,
            cur_kv_lens.c128_state_kv_len,
            nxt_kv_lens.c128_state_kv_len,
            out_alloc_meta_data.out_c128_state_loc,
            bs,
        )

        # FIXME(lsyin): make this sync optional
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()

        batch.kv_seq_lens_cpu.full_kv_len = batch.kv_seq_lens.full_kv_len.cpu()
        batch.kv_seq_lens_cpu.c4_kv_len = batch.kv_seq_lens.c4_kv_len.cpu()
        batch.kv_seq_lens_cpu.c128_kv_len = batch.kv_seq_lens.c128_kv_len.cpu()

    def prepare_for_v2_draft(
        self: EagleDraftInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        cuda_graph_runner: EAGLEDraftCudaGraphRunner,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ):
        if not batch.forward_mode.is_idle():
            bs = len(batch.seq_lens)

            # Assign cache locations
            batch.out_cache_loc_dsv4 = OutCacheLoc.from_data(
                bs * topk * num_steps,
                dtype=torch.int64,
                device=batch.input_ids.device,
            )
            batch.out_cache_loc = batch.out_cache_loc_dsv4.out_full_loc
            # FIXME(lsyin): align with the default code path
            assign_draft_cache_locs_page_size_1[(bs,)](
                batch.req_pool_indices,
                req_to_token_pool.req_to_token,
                batch.seq_lens,
                batch.out_cache_loc,
                req_to_token_pool.req_to_token.shape[1],
                topk,
                num_steps,
            )
            assign_draft_cache_locs_page_size_1[(bs,)](
                batch.req_pool_indices,
                req_to_token_pool.req_to_token_swa,
                batch.seq_lens,
                batch.out_cache_loc_dsv4.out_swa_loc,
                req_to_token_pool.req_to_token.shape[1],
                topk,
                num_steps,
            )
            extend_c4_token = (
                ((batch.seq_lens_cpu + num_steps) // 4 - (batch.seq_lens_cpu) // 4)
                .sum()
                .item()
            )

            if extend_c4_token > 0:
                batch.out_cache_loc_dsv4.out_c4_loc = assign_extend_cache_locs_func(
                    req_pool_indices=batch.req_pool_indices,
                    req_to_token=req_to_token_pool.req_to_token_c4,
                    start_offset=batch.seq_lens // 4,  # 1
                    end_offset=(batch.seq_lens + num_steps) // 4,
                    batch_size=bs,
                    extend_token_nums=num_steps * bs * topk,
                    device=batch.input_ids.device,
                )
            else:
                batch.out_cache_loc_dsv4.out_c4_loc = torch.empty(
                    num_steps * bs * topk,
                    dtype=torch.int32,
                    device=batch.input_ids.device,
                )

            extend_c128_token = (
                ((batch.seq_lens_cpu + num_steps) // 128 - (batch.seq_lens_cpu) // 128)
                .sum()
                .item()
            )
            if extend_c128_token > 0:
                batch.out_cache_loc_dsv4.out_c128_loc = assign_extend_cache_locs_func(
                    req_pool_indices=batch.req_pool_indices,
                    req_to_token=req_to_token_pool.req_to_token_c128,
                    start_offset=batch.seq_lens // 128,
                    end_offset=(batch.seq_lens + num_steps) // 128,
                    batch_size=bs,
                    extend_token_nums=num_steps * bs * topk,
                    device=batch.input_ids.device,
                )
            else:
                batch.out_cache_loc_dsv4.out_c128_loc = torch.empty(
                    num_steps * bs * topk,
                    dtype=torch.int32,
                    device=batch.input_ids.device,
                )

            assign_draft_cache_locs_page_size_1[(bs,)](
                batch.req_pool_indices,
                req_to_token_pool.req_to_token_c4_state,
                batch.seq_lens,
                batch.out_cache_loc_dsv4.out_c4_state_loc,
                req_to_token_pool.req_to_token.shape[1],
                topk,
                num_steps,
            )
            assign_draft_cache_locs_page_size_1[(bs,)](
                batch.req_pool_indices,
                req_to_token_pool.req_to_token_c128_state,
                batch.seq_lens,
                batch.out_cache_loc_dsv4.out_c128_state_loc,
                req_to_token_pool.req_to_token.shape[1],
                topk,
                num_steps,
            )

        # Get a forward batch
        self.num_tokens_per_req = topk
        self.num_tokens_for_logprob_per_req = topk
        batch.capture_hidden_mode = CaptureHiddenMode.LAST
        self.positions = batch.seq_lens.repeat_interleave(topk, dim=0)
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        return forward_batch, can_cuda_graph

    def prepare_for_extend_to_fill_draft_kvcache(
        self,
        batch: ModelWorkerBatch,
        predict: torch.Tensor,
        num_draft_tokens: int,
        draft_model_runner: Any,
        cuda_graph_runner: Any,
    ):
        seq_lens_cpu_ = batch.seq_lens_cpu
        extend_num_tokens = len(batch.seq_lens) * num_draft_tokens

        batch.spec_info = self
        batch.input_ids = predict
        batch.seq_lens = batch.seq_lens + num_draft_tokens
        batch.seq_lens_cpu = batch.seq_lens_cpu + num_draft_tokens
        batch.seq_lens_sum += extend_num_tokens
        batch.extend_seq_lens = [num_draft_tokens for _ in range(len(batch.seq_lens))]
        batch.extend_prefix_lens = seq_lens_cpu_.tolist()
        batch.extend_num_tokens = extend_num_tokens
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.DRAFT_EXTEND_V2
        )
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        can_cuda_graph = cuda_graph_runner and cuda_graph_runner.can_run(forward_batch)
        if not batch.forward_mode.is_idle() and not can_cuda_graph:
            draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
        return forward_batch


@dataclass
class EagleVerifyInputV2Mixin:
    def prepare_for_v2_verify(
        self: EagleVerifyInput,
        req_to_token_pool: ReqToTokenPool,
        batch: ModelWorkerBatch,
        target_worker: TpModelWorker,
    ):
        if not batch.forward_mode.is_idle():
            # Assign cache locations
            bs = len(batch.req_pool_indices)
            batch.input_ids = self.draft_token
            device = batch.input_ids.device
            batch.out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=req_to_token_pool.req_to_token,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens + self.draft_token_num,
                batch_size=bs,
                extend_token_nums=self.draft_token_num * bs,
                device=device,
            )

            batch.out_cache_loc_dsv4.out_full_loc = batch.out_cache_loc

            batch.out_cache_loc_dsv4.out_swa_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,  # int 64
                req_to_token=req_to_token_pool.req_to_token_swa,  # int 32
                start_offset=batch.seq_lens,  # int 64
                end_offset=batch.seq_lens + self.draft_token_num,  # int 64
                batch_size=bs,
                extend_token_nums=self.draft_token_num * bs,
                device=device,
            )
            extend_c4_token = (
                (
                    (batch.seq_lens_cpu + self.draft_token_num) // 4
                    - (batch.seq_lens_cpu) // 4
                )
                .sum()
                .item()
            )
            if extend_c4_token > 0:
                batch.out_cache_loc_dsv4.out_c4_loc = assign_extend_cache_locs_func(
                    req_pool_indices=batch.req_pool_indices,
                    req_to_token=req_to_token_pool.req_to_token_c4,
                    start_offset=batch.seq_lens // 4,  # 1
                    end_offset=(batch.seq_lens + self.draft_token_num) // 4,
                    batch_size=bs,
                    extend_token_nums=extend_c4_token,
                    device=device,
                )
            else:
                batch.out_cache_loc_dsv4.out_c4_loc = torch.empty(
                    0, dtype=torch.int32, device=device
                )
            extend_c128_token = (
                (
                    (batch.seq_lens_cpu + self.draft_token_num) // 128
                    - (batch.seq_lens_cpu) // 128
                )
                .sum()
                .item()
            )
            if extend_c128_token > 0:
                batch.out_cache_loc_dsv4.out_c128_loc = assign_extend_cache_locs_func(
                    req_pool_indices=batch.req_pool_indices,
                    req_to_token=req_to_token_pool.req_to_token_c128,
                    start_offset=batch.seq_lens // 128,
                    end_offset=(batch.seq_lens + self.draft_token_num) // 128,
                    batch_size=bs,
                    extend_token_nums=extend_c128_token,
                    device=device,
                )
            else:
                batch.out_cache_loc_dsv4.out_c128_loc = torch.empty(
                    0, dtype=torch.int32, device=device
                )

            batch.out_cache_loc_dsv4.out_c4_state_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=req_to_token_pool.req_to_token_c4_state,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens + self.draft_token_num,
                batch_size=bs,
                extend_token_nums=self.draft_token_num * bs,
                device=device,
            )
            batch.out_cache_loc_dsv4.out_c128_state_loc = assign_extend_cache_locs_func(
                req_pool_indices=batch.req_pool_indices,
                req_to_token=req_to_token_pool.req_to_token_c128_state,
                start_offset=batch.seq_lens,
                end_offset=batch.seq_lens + self.draft_token_num,
                batch_size=bs,
                extend_token_nums=self.draft_token_num * bs,
                device=device,
            )

            # Set mamba_track_indices for mamba prefix-cache state tracking
            if get_global_server_args().enable_mamba_extra_buffer():
                batch.mamba_track_indices = torch.stack(
                    [
                        req.mamba_ping_pong_track_buffer[req.mamba_next_track_idx]
                        for req in batch.reqs
                    ]
                ).to(torch.int64)
                batch.mamba_track_mask = None
                batch.mamba_track_seqlens = None

        # Get a forward batch
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

        # Run attention backend plan and cuda graph preparation
        can_run_cuda_graph = bool(
            target_worker.model_runner.graph_runner
            and target_worker.model_runner.graph_runner.can_run(verify_forward_batch)
        )
        if can_run_cuda_graph:
            target_worker.model_runner.graph_runner.replay_prepare(verify_forward_batch)
        else:
            if not batch.forward_mode.is_idle():
                target_worker.model_runner.attn_backend.init_forward_metadata(
                    verify_forward_batch
                )

        return verify_forward_batch, can_run_cuda_graph

    def sample(
        self: EagleVerifyInput,
        batch: ModelWorkerBatch,
        logits_output: LogitsProcessorOutput,
        vocab_mask: torch.Tensor = None,
    ):
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).
        """
        if batch.forward_mode.is_idle():
            predict = torch.empty(0, dtype=torch.int32, device=batch.input_ids.device)
            accept_length = torch.empty(
                0, dtype=torch.int32, device=batch.input_ids.device
            )
            accept_index = torch.empty(
                0, dtype=torch.int32, device=batch.input_ids.device
            )
            return predict, accept_length, accept_index

        bs = len(batch.seq_lens)
        sampling_info = batch.sampling_info
        next_token_logits = logits_output.next_token_logits
        device = batch.input_ids.device

        # Apply penalty
        # This is a relaxed version of penalties for speculative decoding.
        if sampling_info.acc_additive_penalties is not None:
            next_token_logits.add_(
                torch.repeat_interleave(
                    sampling_info.acc_additive_penalties, self.draft_token_num, dim=0
                )
            )
        if sampling_info.acc_scaling_penalties is not None:
            apply_scaling_penalties(
                next_token_logits,
                torch.repeat_interleave(
                    sampling_info.acc_scaling_penalties, self.draft_token_num, dim=0
                ),
            )
        if sampling_info.logit_bias is not None:
            next_token_logits.add_(
                torch.repeat_interleave(
                    sampling_info.logit_bias, self.draft_token_num, dim=0
                )
            )

        # Apply grammar mask if provided
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=next_token_logits, vocab_mask=vocab_mask
            )

        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict_shape = list(next_token_logits.shape)[:-1]
        predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device=device
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device=device)

        # Sample tokens
        if sampling_info.is_all_greedy or _is_npu or _is_hip:
            target_predict = torch.argmax(next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)
            predict, accept_index, accept_length = verify_tree_greedy_func(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                target_predict=target_predict,
                topk=self.topk,
            )
        else:
            # Apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * num_draft_tokens, 1)

            target_probs = F.softmax(
                next_token_logits / expanded_temperature, dim=-1
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)
            draft_probs = torch.zeros_like(target_probs)

            # coins for rejection sampling
            coins = torch.rand_like(candidates, dtype=torch.float32, device=device)
            # coins for final sampling
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device=device
            )

            tree_speculative_sampling_target_only(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                uniform_samples=coins,
                uniform_samples_for_final_sampling=coins_for_final_sampling,
                target_probs=target_probs,
                draft_probs=draft_probs,
                threshold_single=get_global_server_args().speculative_accept_threshold_single,
                threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
                deterministic=True,
            )

            # Sync sampling results across TP ranks: different GPUs may
            # produce slightly different target_probs due to floating-point
            # non-determinism in softmax/top_k/top_p, causing different
            # sampled tokens. Broadcast from rank 0 to ensure consistency.
            tp_group = (
                get_attention_tp_group()
                if is_dp_attention_enabled()
                else get_tp_group()
            )
            if tp_group.world_size > 1:
                tp_group.broadcast(predict, src=0)
                tp_group.broadcast(accept_index, src=0)
                tp_group.broadcast(accept_length, src=0)

        if SIMULATE_ACC_LEN > 0:
            # Do simulation
            accept_index = generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                accept_length=accept_length,  # mutable
                simulate_acc_len=SIMULATE_ACC_LEN,
                bs=bs,
                spec_steps=self.spec_steps,
            )

        # Include the bonus token
        accept_length.add_(1)
        return predict, accept_length, accept_index


@triton.jit
def fill_new_verified_id(
    verified_id,
    accept_lens,
    new_verified_id,
    num_draft_tokens: tl.constexpr,
):
    # NOTE: we cannot fuse any in-place operations of `accept_lens` inside this kernel
    # because this kernel reads accept_lens
    pid = tl.program_id(axis=0)
    accept_length = tl.load(accept_lens + pid)

    verified_id_idx = num_draft_tokens * pid + accept_length - 1
    verified_id_data = tl.load(verified_id + verified_id_idx)
    tl.store(new_verified_id + pid, verified_id_data)


@triton.jit
def fill_accepted_out_cache_loc(
    accept_index,
    out_cache_loc,
    accepted_out_cache_loc,
    size_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = tl.arange(0, size_upper)

    masks = (tl.load(accept_index + offset, offset < pid, other=-1) != -1).to(tl.int64)
    dst = tl.sum(masks)
    src = tl.load(accept_index + pid)
    if src > -1:
        value = tl.load(out_cache_loc + src)
        tl.store(accepted_out_cache_loc + dst, value)


@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE


def assign_extend_cache_locs_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    batch_size: int,
    extend_token_nums: int,
    device,
) -> torch.Tensor:
    if _is_cuda or _is_hip:
        out_cache_loc = torch.empty(
            (extend_token_nums,),
            dtype=torch.int64,
            device=device,
        )
        assign_extend_cache_locs[(batch_size,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
            next_power_of_2(batch_size),
        )

        return out_cache_loc

    elif _is_npu:
        out_cache_loc = torch.empty(
            (extend_token_nums,),
            dtype=torch.int32,
            device=device,
        )
        torch.ops.npu.cache_loc_update(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
        )

        return out_cache_loc
