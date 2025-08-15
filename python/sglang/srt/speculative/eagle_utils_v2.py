from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch, global_server_args_dict
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.utils import is_cuda, is_hip, next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
        EAGLEDraftCudaGraphRunner,
    )

if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
        verify_tree_greedy,
    )
    from sgl_kernel.top_k import fast_topk
elif is_hip():
    from sgl_kernel import verify_tree_greedy


logger = logging.getLogger(__name__)


# Simulate acceptance length for benchmarking purposes
SIMULATE_ACC_LEN = os.environ.get("SIMULATE_ACC_LEN")
SIMULATE_ACC_METHOD = os.environ.get("SIMULATE_ACC_METHOD", "multinomial")


@dataclass
class EagleDraftInput:
    # The inputs for decode
    # shape: (b, topk)
    topk_p: torch.Tensor = None  # future when overlap
    topk_index: torch.Tensor = None  # future when overlap
    # shape: (b, hidden_size)
    hidden_states: torch.Tensor = None  # future when overlap

    # Inputs for extend
    # shape: (b,)
    verified_id: torch.Tensor = None  # future when overlap

    # Metadata for seq_lens
    new_seq_lens: torch.Tensor = None  # future when overlap
    allocate_lens: torch.Tensor = None  # never a future
    verify_done: torch.cuda.Event = None  # never a future

    # for overlap schedule, these are references, so can be indexed without race condition
    def filter_batch(self, new_indices: torch.Tensor):
        self.topk_p = self.topk_p[new_indices]
        self.topk_index = self.topk_index[new_indices]
        self.hidden_states = self.hidden_states[new_indices]
        self.verified_id = self.verified_id[new_indices]
        self.allocate_lens = self.allocate_lens[new_indices]
        self.new_seq_lens = self.new_seq_lens[new_indices]

    def merge_batch(self, spec_info: EagleDraftInput):
        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p])
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index])
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], axis=0
        )
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], axis=0)
        self.allocate_lens = torch.cat(
            [self.allocate_lens, spec_info.allocate_lens], axis=0
        )
        self.new_seq_lens = torch.cat(
            [self.new_seq_lens, spec_info.new_seq_lens], axis=0
        )

    def prepare_for_draft(
        self,
        batch: ModelWorkerBatch,
        cuda_graph_runner: EAGLEDraftCudaGraphRunner,
        draft_model_runner: ModelRunner,
        topk: int,
        num_steps: int,
    ):
        bs = len(batch.seq_lens)

        # Assign cache locations
        batch.out_cache_loc = torch.empty(
            (bs * topk * num_steps,),
            dtype=torch.int64,
            device=batch.input_ids.device,
        )
        assign_draft_cache_locs[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            topk,
            num_steps,
        )

        # Get a forward batch
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
    ):
        seq_lens_backup = batch.seq_lens
        seq_lens_cpu_backup = batch.seq_lens_cpu
        extend_num_tokens = len(batch.seq_lens) * num_draft_tokens

        batch.spec_info = self
        batch.input_ids = predict
        batch.seq_lens = batch.seq_lens + num_draft_tokens
        batch.seq_lens_cpu = batch.seq_lens_cpu + num_draft_tokens
        batch.seq_lens_sum += extend_num_tokens
        batch.extend_seq_lens = torch.full_like(batch.seq_lens, num_draft_tokens)
        batch.extend_prefix_lens = seq_lens_backup
        batch.extend_prefix_lens_cpu = seq_lens_cpu_backup
        batch.extend_num_tokens = extend_num_tokens
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        forward_batch = ForwardBatch.init_new(batch, draft_model_runner)
        draft_model_runner.attn_backend.init_forward_metadata(forward_batch)
        return forward_batch


@dataclass
class EagleVerifyInput:
    draft_token: torch.Tensor
    custom_mask: torch.Tensor
    positions: torch.Tensor
    retrive_index: torch.Tensor
    retrive_next_token: torch.Tensor
    retrive_next_sibling: torch.Tensor
    retrive_cum_len: torch.Tensor
    num_steps: int
    topk: int
    num_draft_tokens: int

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        batch_size = len(req_pool_indices)
        qo_indptr = torch.arange(
            0,
            (1 + batch_size) * self.num_draft_tokens,
            step=self.num_draft_tokens,
            dtype=torch.int32,
            device="cuda",
        )
        paged_kernel_lens = paged_kernel_lens + self.num_draft_tokens

        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device="cuda"
        )
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.num_draft_tokens * batch_size,
            dtype=torch.int32,
            device="cuda",
        )
        create_flashinfer_kv_indices_triton[(batch_size,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask

    def prepare_for_verify(
        self,
        batch: ModelWorkerBatch,
        target_worker: TpModelWorker,
    ):
        # Assign cache locations
        bs = len(batch.req_pool_indices)
        batch.input_ids = self.draft_token
        device = batch.input_ids.device
        batch.out_cache_loc = torch.empty(
            (bs * self.num_draft_tokens,),
            dtype=torch.int64,
            device=device,
        )

        assign_extend_cache_locs[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + self.num_draft_tokens,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )

        # Get a forward batch
        batch.spec_info = self
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.capture_hidden_mode = CaptureHiddenMode.FULL
        verify_forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)

        # Run attention backend plan and cuda graph preparation
        can_run_cuda_graph = bool(
            target_worker.model_runner.cuda_graph_runner
            and target_worker.model_runner.cuda_graph_runner.can_run(
                verify_forward_batch
            )
        )
        if can_run_cuda_graph:
            target_worker.model_runner.cuda_graph_runner.replay_prepare(
                verify_forward_batch
            )
        else:
            target_worker.model_runner.attn_backend.init_forward_metadata(
                verify_forward_batch
            )

        return verify_forward_batch, can_run_cuda_graph

    def sample(
        self,
        batch: ModelWorkerBatch,
        logits_output: LogitsProcessorOutput,
    ):
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).
        """
        bs = len(batch.seq_lens)
        sampling_info = batch.sampling_info
        next_token_logits = logits_output.next_token_logits
        device = batch.input_ids.device

        candidates = self.draft_token.reshape(bs, self.num_draft_tokens)
        predict = torch.zeros(
            (bs * (self.num_steps + 1),), dtype=torch.int32, device=device
        )
        accept_index = torch.full(
            (bs, self.num_steps + 1), -1, dtype=torch.int32, device=device
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device=device)

        # Sample tokens
        if sampling_info.is_all_greedy:
            target_predict = torch.argmax(next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.num_draft_tokens)

            verify_tree_greedy(
                predicts=predict,  # mutable
                accept_index=accept_index,  # mutable
                accept_token_num=accept_length,  # mutable
                candidates=candidates,
                retrive_index=self.retrive_index,
                retrive_next_token=self.retrive_next_token,
                retrive_next_sibling=self.retrive_next_sibling,
                target_predict=target_predict,
            )
        else:
            # Apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.num_draft_tokens, dim=0
            )  # (bs * num_draft_tokens, 1)

            target_probs = F.softmax(
                next_token_logits / expanded_temperature, dim=-1
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.num_draft_tokens, dim=0
                ),
            )  # (bs * num_draft_tokens, vocab_size)
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.num_draft_tokens, dim=0
                ),
            )
            target_probs = target_probs.reshape(bs, self.num_draft_tokens, -1)

            # This is currently not used
            draft_probs = torch.empty_like(target_probs)

            all_coins = torch.rand(
                (bs * self.num_draft_tokens + bs), dtype=torch.float32, device=device
            )
            # coins for rejection sampling
            coins = all_coins[:-bs]
            # coins for final sampling
            coins_for_final_sampling = all_coins[-bs:]

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
                threshold_single=global_server_args_dict[
                    "speculative_accept_threshold_single"
                ],
                threshold_acc=global_server_args_dict[
                    "speculative_accept_threshold_acc"
                ],
                deterministic=True,
            )

        if SIMULATE_ACC_LEN:
            # Do simulation
            accept_index = _generate_simulated_accept_index(
                accept_index=accept_index,
                predict=predict,  # mutable
                accept_length=accept_length,  # mutable
                simulate_acc_len=SIMULATE_ACC_LEN,
                bs=bs,
                num_steps=self.num_steps,
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
def assign_draft_cache_locs(
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


@torch.compile(dynamic=True)
def select_top_k_tokens(
    i: int,
    topk_p: torch.Tensor,
    topk_index: torch.Tensor,
    hidden_states: torch.Tensor,
    scores: torch.Tensor,
    topk: int,
):
    if i == 0:
        # The first step after extend
        input_ids = topk_index.flatten()
        hidden_states = hidden_states.repeat_interleave(topk, dim=0)
        scores = topk_p  # shape: (b, topk)

        tree_info = (
            topk_p.unsqueeze(1),  # shape: (b, 1, topk)
            topk_index,  # shape: (b, topk)
            torch.arange(-1, topk, dtype=torch.long, device=hidden_states.device)
            .unsqueeze(0)
            .repeat(topk_p.shape[0], 1),  # shape: (b, topk + 1)
        )
    else:
        # The later decode steps
        expand_scores = torch.mul(
            scores.unsqueeze(2), topk_p.reshape(-1, topk, topk)
        )  # (b, topk, 1) x (b, topk ,topk) -> (b, topk, topk)
        topk_cs_p, topk_cs_index = fast_topk(
            expand_scores.flatten(start_dim=1), topk, dim=-1
        )  # (b, topk)
        scores = topk_cs_p  # shape: (b, topk)

        topk_index = topk_index.reshape(-1, topk**2)
        input_ids = torch.gather(topk_index, index=topk_cs_index, dim=1).flatten()

        selected_input_index = topk_cs_index.flatten() // topk + torch.arange(
            0, hidden_states.shape[0], step=topk, device=hidden_states.device
        ).repeat_interleave(topk)
        hidden_states = hidden_states[selected_input_index, :]

        tree_info = (
            expand_scores,  # shape: (b, topk, topk)
            topk_index,  # shape: (b, topk * topk)
            topk_cs_index + (topk**2 * (i - 1) + topk),  # shape: (b, topk)
        )

    return input_ids, hidden_states, scores, tree_info


def fast_topk_torch(values, topk, dim):
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        max_value, max_index = torch.max(values, dim=dim)
        return max_value.unsqueeze(1), max_index.unsqueeze(1)
    else:
        # Use topk for efficiency with larger k values
        return torch.topk(values, topk, dim=dim)


def _generate_simulated_accept_index(
    accept_index,
    predict,
    accept_length,
    simulate_acc_len,
    bs,
    num_steps,
):
    simulate_acc_len_float = float(simulate_acc_len)
    if SIMULATE_ACC_METHOD == "multinomial":
        simulated_values = torch.normal(
            mean=simulate_acc_len_float,
            std=1.0,
            size=(1,),
            device="cpu",
        )
        # clamp simulated values to be between 1 and self.num_steps
        simulated_values = torch.clamp(simulated_values, min=1.0, max=num_steps + 1)
        simulate_acc_len = int(simulated_values.round().item())
    elif SIMULATE_ACC_METHOD == "match-expected":
        # multinomial sampling does not match the expected length
        # we keep it for the sake of compatibility of existing tests
        # but it's better to use "match-expected" for the cases that need to
        # match the expected length, One caveat is that this will only sample
        # either round down or round up of the expected length
        simulate_acc_len_float = max(1.0, min(num_steps + 1, simulate_acc_len_float))
        lower = int(simulate_acc_len_float // 1)
        upper = lower + 1 if lower < num_steps + 1 else lower
        if lower == upper:
            simulate_acc_len = lower
        else:
            weight_upper = simulate_acc_len_float - lower
            weight_lower = 1.0 - weight_upper
            probs = torch.tensor([weight_lower, weight_upper], device="cpu")
            sampled_index = torch.multinomial(probs, num_samples=1)
            simulate_acc_len = lower if sampled_index == 0 else upper
    else:
        raise ValueError(f"Invalid simulate_acc_method: {SIMULATE_ACC_METHOD}")

    accept_indx_first_col = accept_index[:, 0].view(-1, 1)
    sim_accept_index = torch.full(
        (bs, num_steps + 1), -1, dtype=torch.int32, device="cuda"
    )
    sim_accept_index[:, :simulate_acc_len] = accept_indx_first_col + torch.arange(
        simulate_acc_len, device=accept_index.device
    )
    accept_length.fill_(simulate_acc_len - 1)
    predict.fill_(100)  # some legit token id
    return sim_accept_index
