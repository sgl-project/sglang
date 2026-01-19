from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.model_runner import ModelRunner
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
        if batch.tree_cache.supports_swa() and batch.tree_cache.is_chunk_cache():
            for req in batch.reqs:
                batch.tree_cache.evict_swa(req, req.seqlen - 1)

        from sglang.srt.speculative.spec_utils import assign_req_to_token_pool_func

        bs = batch.batch_size()

        # Now seq_lens is correct
        batch.maybe_wait_verify_done()

        page_size = batch.token_to_kv_pool_allocator.page_size
        cur_kv_lens_cpu = []
        nxt_kv_lens_cpu = []
        num_needed_tokens = 0

        # paged(topk>1) alloc lower bound
        # TODO Simply Getattr... Think of optimizing it later
        seq_lens_cpu = getattr(batch, "seq_lens_cpu", batch.seq_lens.cpu())
        page_size = batch.token_to_kv_pool_allocator.page_size
        server_args = get_global_server_args()
        topk = int(getattr(server_args, "speculative_eagle_topk", 1))
        num_steps = int(getattr(server_args, "speculative_num_steps", 0))
        self.page_size = page_size  # for later use...

        for i, r in enumerate(batch.reqs):
            x_dense = int(
                r.kv_committed_len + 2 * self.ALLOC_LEN_PER_DECODE - r.kv_allocated_len
            )
            if x_dense < 0:
                x_dense = 0

            x_paged_min = 0
            if page_size > 1 and topk > 1:
                prefix_len = int(seq_lens_cpu[i])
                last = prefix_len % page_size
                base = prefix_len - last
                num_new_pages = (last + num_steps + page_size - 1) // page_size

                # abs lower_bound ：token_pool covers for base + topk*num_new_pages*page_size
                required_min = base + topk * num_new_pages * page_size

                x_paged_min = required_min - int(r.kv_allocated_len)
                if x_paged_min < 0:
                    x_paged_min = 0

            x = x_dense if x_dense > x_paged_min else x_paged_min

            cur_kv_lens_cpu.append(int(r.kv_allocated_len))
            r.kv_allocated_len = int(r.kv_allocated_len) + x
            nxt_kv_lens_cpu.append(int(r.kv_allocated_len))
            num_needed_tokens += x

        cur_kv_lens_cpu = torch.tensor(cur_kv_lens_cpu, dtype=torch.int32, device="cpu")
        nxt_kv_lens_cpu = torch.tensor(nxt_kv_lens_cpu, dtype=torch.int32, device="cpu")

        if page_size == 1:
            out_cache_loc = alloc_token_slots(batch.tree_cache, num_needed_tokens)
        else:
            cur_kv_lens = cur_kv_lens_cpu.to(device=batch.device)
            nxt_kv_lens = nxt_kv_lens_cpu.to(device=batch.device)
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
            cur_kv_lens_cpu.to(device=batch.device),
            nxt_kv_lens_cpu.to(device=batch.device),
            out_cache_loc,
            bs,
        )

        # FIXME(lsyin): make this sync optional
        batch.seq_lens_cpu = batch.seq_lens.cpu()
        batch.seq_lens_sum = batch.seq_lens_cpu.sum().item()

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

            # Assign cache locations (page_size==1 or topk==1 => dense; else paged gather)
            page_size = int(self.page_size)
            topk = int(topk)
            num_steps = int(num_steps)

            if page_size == 1 or topk == 1:
                batch.out_cache_loc = torch.empty(
                    (bs * topk * num_steps,),
                    device=batch.device,
                    dtype=torch.int64,
                )
                assign_draft_cache_locs_page_size_1[(bs,)](
                    batch.req_pool_indices,
                    req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.out_cache_loc,
                    req_to_token_pool.req_to_token.shape[1],
                    topk,
                    num_steps,
                )

            else:
                # --- paged(topk>1) gather out_cache_loc from req_to_token ---
                idx = batch.req_pool_indices.to(torch.int64)
                rows = req_to_token_pool.req_to_token[idx]
                pool_len = rows.size(1)

                seq_lens = batch.seq_lens.to(rows.device, dtype=torch.int64)
                last_page = seq_lens % page_size
                prefix_base = seq_lens - last_page
                num_new_pages = (
                    last_page + num_steps + page_size - 1
                ) // page_size  # (bs,)

                topk_ids = torch.arange(
                    topk, device=rows.device, dtype=torch.int64
                ).view(
                    1, topk
                )  # (1,topk)
                starts = (
                    prefix_base.view(bs, 1)
                    + topk_ids * (num_new_pages.view(bs, 1) * page_size)
                    + last_page.view(bs, 1)
                )
                steps = torch.arange(
                    num_steps, device=rows.device, dtype=torch.int64
                ).view(1, 1, num_steps)

                pos = (starts.view(bs, topk, 1) + steps).reshape(bs, topk * num_steps)

                # TODO assertion should be removed into unit test
                # assert int(pos.max()) < int(pool_len), f"paged out_cache_loc OOB: max_pos={int(pos.max())}, pool_len={int(pool_len)}"
                # assert int(pos.min()) >= 0, f"paged out_cache_loc negative pos: min_pos={int(pos.min())}"

                out = torch.gather(rows, 1, pos)  # (bs, topk*S)
                batch.out_cache_loc = out.to(torch.int64).reshape(-1)

        # Get a forward batch
        self.num_tokens_per_batch = topk
        self.num_tokens_for_logprob_per_batch = topk
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
                draft_token_num=self.draft_token_num,
                device=device,
            )

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
            predict = torch.empty(0, dtype=torch.long, device=batch.input_ids.device)
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
        if sampling_info.is_all_greedy or _is_npu:
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
def build_compact_kv_src_tgt_cache_loc(
    accept_index,
    accept_lens,
    out_cache_loc,
    src_cache_loc,
    tgt_cache_loc,
    draft_token_num: tl.constexpr,
    accept_index_len: tl.constexpr,
    accept_index_upper: tl.constexpr,
):
    """
    Build (src_cache_loc, tgt_cache_loc) pairs to compact accepted KV cache to the
    front of the per-request verify slots without any dynamic allocation.

    Layout:
    - accept_index: [bs, accept_index_len] (padded with -1)
    - out_cache_loc: [bs, draft_token_num]
    - src/tgt_cache_loc: [bs, accept_index_len]

    For each request i and position j in [0, accept_index_len):
    - tgt = out_cache_loc[i, j]
    - src = out_cache_loc[accept_index[i, j]] if j < accept_lens[i] and accept_index[i, j] >= 0
      else tgt (no-op copy)
    """
    bid = tl.program_id(axis=0)
    offsets = tl.arange(0, accept_index_upper)
    mask = offsets < accept_index_len

    accept_len = tl.load(accept_lens + bid)

    tgt_pos = bid * draft_token_num + offsets
    tgt_vals = tl.load(out_cache_loc + tgt_pos, mask=mask, other=0)
    src_vals = tgt_vals

    acc_idx = tl.load(
        accept_index + bid * accept_index_len + offsets, mask=mask, other=-1
    )
    copy_mask = mask & (offsets < accept_len) & (acc_idx >= 0)
    src_candidate = tl.load(out_cache_loc + acc_idx, mask=copy_mask, other=0)
    src_vals = tl.where(copy_mask, src_candidate, src_vals)

    out_pos = bid * accept_index_len + offsets
    tl.store(src_cache_loc + out_pos, src_vals, mask=mask)
    tl.store(tgt_cache_loc + out_pos, tgt_vals, mask=mask)


@triton.jit
def compact_data_tensors_kernel(
    accept_index,
    accept_length,
    predict,
    hidden_states,
    out_cache_loc,
    out_predict,
    out_hidden,
    out_cache,
    stride: tl.constexpr,
    max_accept: tl.constexpr,
    hidden_dim: tl.constexpr,
):
    """Compact predict, hidden_states, and out_cache_loc into fixed-stride buffers."""
    BLOCK_H: tl.constexpr = 128
    pid = tl.program_id(axis=0)
    acc_len = tl.load(accept_length + pid)

    accept_row = accept_index + pid * max_accept
    out_base = pid * stride

    for col in tl.static_range(max_accept):
        if col < acc_len:
            src_idx = tl.load(accept_row + col)
            # Skip -1 padding to avoid OOB access
            if src_idx >= 0:
                dst_idx = out_base + col

                tok = tl.load(predict + src_idx)
                tl.store(out_predict + dst_idx, tok)

                cache_loc = tl.load(out_cache_loc + src_idx)
                tl.store(out_cache + dst_idx, cache_loc)

                # Blocked copy over hidden dimension
                for h_start in tl.static_range(0, hidden_dim, BLOCK_H):
                    h_offsets = h_start + tl.arange(0, BLOCK_H)
                    h_mask = h_offsets < hidden_dim
                    h_vals = tl.load(
                        hidden_states + src_idx * hidden_dim + h_offsets, mask=h_mask
                    )
                    tl.store(
                        out_hidden + dst_idx * hidden_dim + h_offsets,
                        h_vals,
                        mask=h_mask,
                    )


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
    draft_token_num: int,
    device,
) -> torch.Tensor:
    if _is_cuda or _is_hip:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
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
            (batch_size * draft_token_num,),
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


def compact_data_tensors_func(
    accept_index: torch.Tensor,
    accept_length: torch.Tensor,
    tree_size: int,
    predict: torch.Tensor,
    hidden_states: torch.Tensor,
    out_cache_loc: torch.Tensor,
):
    """Compact scattered acceptance into prefix form using fused Triton kernel."""
    bs = accept_index.shape[0]
    max_accept = accept_index.shape[1]
    stride = tree_size
    hidden_dim = hidden_states.shape[-1]

    assert stride >= max_accept, "tree_size must be >= max accepted tokens"

    packed_predict = torch.zeros(
        (bs * stride,), device=predict.device, dtype=predict.dtype
    )
    packed_hidden = torch.zeros(
        (bs * stride, hidden_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    packed_cache = torch.zeros(
        (bs * stride,), device=out_cache_loc.device, dtype=out_cache_loc.dtype
    )

    compact_data_tensors_kernel[(bs,)](
        accept_index,
        accept_length,
        predict,
        hidden_states,
        out_cache_loc,
        packed_predict,
        packed_hidden,
        packed_cache,
        stride=stride,
        max_accept=max_accept,
        hidden_dim=hidden_dim,
    )

    return packed_predict, packed_hidden, packed_cache
