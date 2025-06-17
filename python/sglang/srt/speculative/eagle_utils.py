from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from sglang.srt.constrained.base_grammar_backend import BaseGrammarObject
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import (
    Req,
    ScheduleBatch,
    get_last_loc,
    global_server_args_dict,
)
from sglang.srt.mem_cache.memory_pool import TokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.utils import is_cuda, is_hip, next_power_of_2

logger = logging.getLogger(__name__)

if is_cuda():
    from sgl_kernel import (
        fast_topk,
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
        verify_tree_greedy,
    )
elif is_hip():
    from sgl_kernel import fast_topk, verify_tree_greedy


logger = logging.getLogger(__name__)


# Simulate acceptance length for benchmarking purposes
SIMULATE_ACC_LEN = os.environ.get("SIMULATE_ACC_LEN")
SIMULATE_ACC_METHOD = os.environ.get("SIMULATE_ACC_METHOD", "multinomial")

TREE_TRAVERSE_TIME_THRESHOLD = 1  # TODO: set this properly


@dataclass
class EagleDraftInput:
    # The inputs for decode
    # shape: (b, topk)
    topk_p: torch.Tensor = None
    topk_index: torch.Tensor = None
    # shape: (b, hidden_size)
    hidden_states: torch.Tensor = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    # Inputs for extend
    # shape: (b,)
    verified_id: torch.Tensor = None
    accept_length: torch.Tensor = None
    accept_length_cpu: List[int] = None

    # Inputs for the attention backends
    # shape: (b + 1,)
    kv_indptr: torch.Tensor = None
    kv_indices: torch.Tensor = None

    def prepare_for_extend(self, batch: ScheduleBatch):
        if batch.forward_mode.is_idle():
            return
        # Prefill only generate 1 token.
        assert len(self.verified_id) == len(batch.seq_lens)

        pt = 0
        for i, extend_len in enumerate(batch.extend_lens):
            input_ids = batch.input_ids[pt : pt + extend_len]
            batch.input_ids[pt : pt + extend_len] = torch.cat(
                (input_ids[1:], self.verified_id[i].reshape(1))
            )
            pt += extend_len

    @classmethod
    def create_idle_input(
        cls,
        device: torch.device,
        hidden_size: int,
        topk: int,
        capture_hidden_mode: CaptureHiddenMode,
    ):
        return cls(
            verified_id=None,
            hidden_states=torch.empty(
                (0, hidden_size), device=device, dtype=torch.float32
            ),
            topk_p=torch.empty((0, topk), device=device, dtype=torch.float32),
            topk_index=torch.empty((0, topk), device=device, dtype=torch.int64),
            capture_hidden_mode=capture_hidden_mode,
        )

    def prepare_extend_after_decode(
        self,
        batch: ScheduleBatch,
        speculative_num_steps: int,
    ):
        batch.forward_mode = ForwardMode.DRAFT_EXTEND
        batch.input_ids = self.verified_id
        batch.extend_lens = [x + 1 for x in batch.spec_info.accept_length_cpu]
        batch.extend_num_tokens = sum(batch.extend_lens)
        batch.seq_lens = batch.spec_info.seq_lens_for_draft_extend
        batch.req_pool_indices = batch.spec_info.req_pool_indices_for_draft_extend
        batch.return_logprob = False
        batch.return_hidden_states = False

        self.capture_hidden_mode = CaptureHiddenMode.LAST
        self.accept_length.add_(1)
        self.positions = torch.empty_like(batch.input_ids, dtype=torch.long)
        self.verified_id = torch.empty_like(self.accept_length, dtype=torch.int32)

        create_extend_after_decode_spec_info[(len(batch.seq_lens),)](
            batch.input_ids,
            batch.seq_lens,
            self.accept_length,
            self.positions,
            self.verified_id,
            next_power_of_2(max(speculative_num_steps + 1, len(batch.seq_lens))),
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        bs = self.accept_length.numel()
        qo_indptr = torch.zeros((bs + 1,), dtype=torch.int32, device="cuda")
        qo_indptr[1:] = torch.cumsum(self.accept_length, dim=0)
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device="cuda")
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        if paged_kernel_lens_sum is None:
            paged_kernel_lens_sum = cum_kv_seq_len[-1]

        kv_indices = torch.empty(
            paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
        )

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.size(1),
        )
        return kv_indices, cum_kv_seq_len, qo_indptr, None

    def filter_batch(self, new_indices: torch.Tensor):
        self.topk_p = self.topk_p[: len(new_indices)]
        self.topk_index = self.topk_index[: len(new_indices)]
        self.hidden_states = self.hidden_states[: len(new_indices)]
        self.verified_id = self.verified_id[: len(new_indices)]

    def merge_batch(self, spec_info: EagleDraftInput):
        if self.hidden_states is None:
            self.hidden_states = spec_info.hidden_states
            self.verified_id = spec_info.verified_id
            self.topk_p = spec_info.topk_p
            self.topk_index = spec_info.topk_index
            return
        if spec_info.hidden_states is None:
            return
        self.hidden_states = torch.cat(
            [self.hidden_states, spec_info.hidden_states], axis=0
        )
        self.verified_id = torch.cat([self.verified_id, spec_info.verified_id], axis=0)
        self.topk_p = torch.cat([self.topk_p, spec_info.topk_p])
        self.topk_index = torch.cat([self.topk_index, spec_info.topk_index])


@dataclass
class EagleVerifyOutput:
    # Draft input batch
    draft_input: EagleDraftInput
    # Logit outputs from target worker
    logits_output: LogitsProcessorOutput
    # Accepted token ids including the bonus token
    verified_id: torch.Tensor
    # Accepted token length per sequence in a batch in CPU.
    accept_length_per_req_cpu: List[int]
    # Accepted indices from logits_output.next_token_logits
    accepted_indices: torch.Tensor


@dataclass
class EagleVerifyInput:
    draft_token: torch.Tensor
    custom_mask: torch.Tensor
    positions: torch.Tensor
    retrive_index: torch.Tensor
    retrive_next_token: torch.Tensor
    retrive_next_sibling: torch.Tensor
    retrive_cum_len: torch.Tensor
    spec_steps: int
    topk: int
    draft_token_num: int
    capture_hidden_mode: CaptureHiddenMode
    seq_lens_sum: int
    seq_lens_cpu: torch.Tensor
    grammar: BaseGrammarObject = None

    @classmethod
    def create_idle_input(cls, topk: int, spec_steps: int, num_verify_tokens: int):
        return cls(
            draft_token=torch.empty((0,), dtype=torch.long, device="cuda"),
            custom_mask=torch.full((0,), True, dtype=torch.bool, device="cuda"),
            positions=torch.empty((0,), dtype=torch.int64, device="cuda"),
            retrive_index=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrive_next_token=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrive_next_sibling=torch.full(
                (0, num_verify_tokens), -1, dtype=torch.long, device="cuda"
            ),
            retrive_cum_len=None,
            topk=topk,
            draft_token_num=num_verify_tokens,
            spec_steps=spec_steps,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=0,
            seq_lens_cpu=torch.empty((0,), dtype=torch.int32),
        )

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):

        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token

        if page_size == 1:
            batch.out_cache_loc = batch.alloc_token_slots(len(batch.input_ids))
            end_offset = batch.seq_lens + self.draft_token_num
        else:
            prefix_lens = batch.seq_lens
            end_offset = prefix_lens + self.draft_token_num
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = batch.alloc_paged_token_slots_extend(
                prefix_lens, end_offset, last_loc, len(batch.input_ids)
            )
            self.last_loc = last_loc

        bs = batch.batch_size()
        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            end_offset,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            next_power_of_2(bs),
        )

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
            (1 + batch_size) * self.draft_token_num,
            step=self.draft_token_num,
            dtype=torch.int32,
            device="cuda",
        )
        cum_kv_seq_len = torch.zeros(
            (batch_size + 1,), dtype=torch.int32, device="cuda"
        )

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        kv_indices = torch.empty(
            paged_kernel_lens_sum + self.draft_token_num * batch_size,
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

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: torch.Tensor,
        token_to_kv_pool_allocator: TokenToKVPoolAllocator,
        page_size: int,
        vocab_mask: Optional[torch.Tensor] = None,  # For grammar
    ) -> torch.Tensor:
        """
        Verify and find accepted tokens based on logits output and batch
        (which contains spec decoding information).

        WARNING: This API in-place modifies the states of logits_output

        This API updates values inside logits_output based on the accepted
        tokens. I.e., logits_output.next_token_logits only contains
        accepted token logits.
        """
        if batch.forward_mode.is_idle():
            return EagleVerifyOutput(
                draft_input=EagleDraftInput.create_idle_input(
                    device=batch.device,
                    hidden_size=batch.model_config.hidden_size,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                ),
                logits_output=logits_output,
                verified_id=torch.empty(0, dtype=torch.long, device=batch.device),
                accept_length_per_req_cpu=[],
                accepted_indices=torch.full(
                    (0, self.spec_steps + 1),
                    -1,
                    dtype=torch.int32,
                    device=batch.device,
                ),
            )

        bs = self.retrive_index.shape[0]
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        sampling_info = batch.sampling_info

        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        predict = torch.empty(predict_shape, dtype=torch.int32, device="cuda")
        accept_index = torch.full(
            (bs, self.spec_steps + 1), -1, dtype=torch.int32, device="cuda"
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device="cuda")

        # Apply the custom logit processors if registered in the sampling info.
        if sampling_info.has_custom_logit_processor:
            apply_custom_logit_processor(
                logits_output.next_token_logits,
                sampling_info,
                num_tokens_in_batch=self.draft_token_num,
            )

        # Apply penalty
        if sampling_info.penalizer_orchestrator.is_required:
            # This is a relaxed version of penalties for speculative decoding.
            linear_penalty = torch.zeros(
                (bs, logits_output.next_token_logits.shape[1]),
                dtype=torch.float32,
                device="cuda",
            )
            sampling_info.apply_logits_bias(linear_penalty)
            logits_output.next_token_logits.add_(
                torch.repeat_interleave(linear_penalty, self.draft_token_num, dim=0)
            )

        # Apply grammar mask
        if vocab_mask is not None:
            assert self.grammar is not None
            self.grammar.apply_vocab_mask(
                logits=logits_output.next_token_logits, vocab_mask=vocab_mask
            )

        # Sample tokens
        if batch.sampling_info.is_all_greedy:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
            target_predict = target_predict.reshape(bs, self.draft_token_num)

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
            # apply temperature and get target probs
            expanded_temperature = torch.repeat_interleave(
                sampling_info.temperatures, self.draft_token_num, dim=0
            )  # (bs * draft_token_num, 1)

            target_probs = F.softmax(
                logits_output.next_token_logits / expanded_temperature, dim=-1
            )  # (bs * draft_token_num, vocab_size)
            target_probs = top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ks, self.draft_token_num, dim=0
                ),
            )  # (bs * draft_token_num, vocab_size)
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )
            target_probs = target_probs.reshape(bs, self.draft_token_num, -1)

            draft_probs = torch.zeros(
                target_probs.shape, dtype=torch.float32, device="cuda"
            )

            # coins for rejection sampling
            coins = torch.rand_like(candidates, dtype=torch.float32, device="cuda")
            # coins for final sampling
            coins_for_final_sampling = torch.rand(
                (bs,), dtype=torch.float32, device="cuda"
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
                spec_steps=self.spec_steps,
            )

        unfinished_index = []
        unfinished_accept_index = []
        accept_index_cpu = accept_index.tolist()
        predict_cpu = predict.tolist()
        has_finished = False

        # Iterate every accepted token and check if req has finished after append the token
        # should be checked BEFORE free kv cache slots
        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                id = predict_cpu[idx]
                req.output_ids.append(id)
                req.check_finished()
                if req.finished():
                    has_finished = True
                    # set all tokens after finished token to -1 and break
                    accept_index[i, j + 1 :] = -1
                    break
                else:
                    if req.grammar is not None:
                        try:
                            req.grammar.accept_token(id)
                        except ValueError as e:
                            logger.info(
                                f"{i=}, {req=}\n" f"{accept_index=}\n" f"{predict=}\n"
                            )
                            raise e
            if not req.finished():
                unfinished_index.append(i)
                if idx == -1:
                    unfinished_accept_index.append(accept_index[i, :j])
                else:
                    unfinished_accept_index.append(accept_index[i])
            req.spec_verify_ct += 1

        if has_finished:
            accept_length = (accept_index != -1).sum(dim=1) - 1

        # Free the KV cache for unaccepted tokens
        # TODO: fuse them
        accept_index = accept_index[accept_index != -1]
        verified_id = predict[accept_index]
        evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
        evict_mask[accept_index] = False

        if page_size == 1:
            # TODO: boolean array index leads to a device sync. Remove it.
            token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
        else:
            if self.topk == 1:
                # Only evict full empty page. Do not evict partial empty page
                align_evict_mask_to_page_size[len(batch.seq_lens),](
                    batch.seq_lens,
                    evict_mask,
                    page_size,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                )
                token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            else:
                # Shift the accepted tokens to the beginning.
                # Only evict the last part
                src_cache_loc, tgt_cache_loc, to_free_num_slots = get_src_tgt_cache_loc(
                    batch.seq_lens,
                    batch.out_cache_loc,
                    accept_index,
                    accept_length,
                    self.draft_token_num,
                    page_size,
                )
                to_free_slots = torch.empty(
                    (to_free_num_slots.sum().item(),),
                    dtype=torch.int64,
                    device=to_free_num_slots.device,
                )

                # out_cache_loc: [0  1  2,  3  4  5,  6  7  8]
                # accept_index:  [0 -1  2,  3  4 -1,  6 -1 -1]
                # tgt_cache_loc: [0  1   ,  3  4   ,  6      ]
                # to_free_slots: [      2,        5,     7  8]
                # to_free_slots also needs to be page-aligned without the first partial page
                #
                # split each row of out_cache_loc into two parts.
                # 1. the first part goes to tgt_cache_loc. length = accept_length[i] + 1
                # 2. the second part goes to to_free_slots.
                get_target_cache_loc[(bs,)](
                    tgt_cache_loc,
                    to_free_slots,
                    accept_length,
                    to_free_num_slots,
                    batch.out_cache_loc,
                    self.draft_token_num,
                    next_power_of_2(self.draft_token_num),
                    next_power_of_2(bs),
                )

                # Free the kv cache
                token_to_kv_pool_allocator.free(to_free_slots)

                # Copy the kv cache
                batch.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                    tgt_cache_loc, src_cache_loc
                )

        # Construct EagleVerifyOutput
        if not has_finished:
            if page_size == 1 or self.topk == 1:
                batch.out_cache_loc = batch.out_cache_loc[accept_index]
                assign_req_to_token_pool[(bs,)](
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + accept_length + 1,
                    batch.out_cache_loc,
                    batch.req_to_token_pool.req_to_token.shape[1],
                    next_power_of_2(bs),
                )
            else:
                batch.out_cache_loc = tgt_cache_loc
            batch.seq_lens.add_(accept_length + 1)

            draft_input = EagleDraftInput()
            draft_input.hidden_states = batch.spec_info.hidden_states[accept_index]
            draft_input.verified_id = verified_id
            draft_input.accept_length = accept_length
            draft_input.accept_length_cpu = accept_length.tolist()
            draft_input.seq_lens_for_draft_extend = batch.seq_lens
            draft_input.req_pool_indices_for_draft_extend = batch.req_pool_indices

            return EagleVerifyOutput(
                draft_input=draft_input,
                logits_output=logits_output,
                verified_id=verified_id,
                accept_length_per_req_cpu=draft_input.accept_length_cpu,
                accepted_indices=accept_index,
            )
        else:
            if page_size == 1 or self.topk == 1:
                assign_req_to_token_pool[(bs,)](
                    batch.req_pool_indices,
                    batch.req_to_token_pool.req_to_token,
                    batch.seq_lens,
                    batch.seq_lens + accept_length + 1,
                    batch.out_cache_loc[accept_index],
                    batch.req_to_token_pool.req_to_token.shape[1],
                    next_power_of_2(bs),
                )
                batch.seq_lens.add_(accept_length + 1)

            accept_length_cpu = accept_length.tolist()
            draft_input = EagleDraftInput()
            if len(unfinished_accept_index) > 0:
                unfinished_accept_index = torch.cat(unfinished_accept_index)
                unfinished_index_device = torch.tensor(
                    unfinished_index, dtype=torch.int64, device=predict.device
                )
                draft_input_accept_length_cpu = [
                    accept_length_cpu[i] for i in unfinished_index
                ]
                if page_size == 1 or self.topk == 1:
                    batch.out_cache_loc = batch.out_cache_loc[unfinished_accept_index]
                else:
                    batch.out_cache_loc = torch.empty(
                        len(unfinished_index) + sum(draft_input_accept_length_cpu),
                        dtype=torch.int64,
                        device=predict.device,
                    )
                    accept_length_filter = create_accept_length_filter(
                        accept_length,
                        unfinished_index_device,
                        batch.seq_lens,
                    )
                    filter_finished_cache_loc_kernel[(bs,)](
                        batch.out_cache_loc,
                        tgt_cache_loc,
                        accept_length,
                        accept_length_filter,
                        next_power_of_2(bs),
                        next_power_of_2(self.draft_token_num),
                    )

                draft_input.hidden_states = batch.spec_info.hidden_states[
                    unfinished_accept_index
                ]
                draft_input.verified_id = predict[unfinished_accept_index]
                draft_input.accept_length_cpu = draft_input_accept_length_cpu
                draft_input.accept_length = accept_length[unfinished_index_device]
                draft_input.seq_lens_for_draft_extend = batch.seq_lens[
                    unfinished_index_device
                ]
                draft_input.req_pool_indices_for_draft_extend = batch.req_pool_indices[
                    unfinished_index_device
                ]

            return EagleVerifyOutput(
                draft_input=draft_input,
                logits_output=logits_output,
                verified_id=verified_id,
                accept_length_per_req_cpu=accept_length_cpu,
                accepted_indices=accept_index,
            )


@triton.jit
def create_extend_after_decode_spec_info(
    verified_id,
    seq_lens,
    accept_lens,
    positions,
    new_verified_id,
    bs_upper: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = tl.arange(0, bs_upper)
    seq_length = tl.load(seq_lens + pid)
    accept_length = tl.load(accept_lens + pid)

    accept_len_cumsum = tl.sum(
        tl.load(accept_lens + offsets, mask=offsets < pid, other=0)
    )
    positions_ptr = positions + accept_len_cumsum
    mask = offsets < accept_length
    tl.store(positions_ptr + offsets, seq_length - accept_length + offsets, mask)

    accept_len_cumsum += accept_length - 1
    verified_id_data = tl.load(verified_id + accept_len_cumsum)
    tl.store(new_verified_id + pid, verified_id_data)


@triton.jit
def assign_req_to_token_pool(
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

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE


@triton.jit
def assign_draft_cache_locs(
    req_pool_indices,
    req_to_token,
    seq_lens,
    extend_lens,
    num_new_pages_per_topk,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
    page_size: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    if page_size == 1 or topk == 1:
        copy_len = topk * speculative_num_steps
        out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps
    else:
        bs_offset = tl.arange(0, bs_upper)
        copy_len = tl.load(extend_lens + pid)
        cum_copy_len = tl.sum(tl.load(extend_lens + bs_offset, mask=bs_offset < pid))
        out_cache_ptr = out_cache_loc + cum_copy_len

    # Part 1: Copy from out_cache_loc to req_to_token
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(out_cache_ptr + copy_offset, mask=mask)
        tl.store(token_pool + kv_start + copy_offset, data, mask=mask)

    if page_size == 1 or topk == 1:
        return

    # Part 2: Copy the indices for the last partial page
    prefix_len = tl.load(seq_lens + pid)
    last_page_len = prefix_len % page_size
    offsets = tl.arange(0, page_size)
    mask = offsets < last_page_len
    num_new_pages_per_topk_ = tl.load(num_new_pages_per_topk + pid)
    prefix_base = token_pool + prefix_len - last_page_len

    for topk_id in range(topk):
        value = tl.load(prefix_base + offsets, mask=mask)
        tl.store(
            prefix_base + topk_id * num_new_pages_per_topk_ * page_size + offsets,
            value,
            mask=mask,
        )

    # Part 3: Remove the padding in out_cache_loc
    iter_offest = tl.arange(0, iter_upper)
    for topk_id in range(topk):
        indices = tl.load(
            prefix_base
            + topk_id * num_new_pages_per_topk_ * page_size
            + last_page_len
            + iter_offest,
            mask=iter_offest < speculative_num_steps,
        )
        tl.store(
            out_cache_loc
            + pid * topk * speculative_num_steps
            + topk_id * speculative_num_steps
            + iter_offest,
            indices,
            mask=iter_offest < speculative_num_steps,
        )


@triton.jit
def generate_draft_decode_kv_indices(
    req_pool_indices,
    req_to_token,
    paged_kernel_lens,
    kv_indices,
    kv_indptr,
    positions,
    pool_len: tl.constexpr,
    kv_indices_stride: tl.constexpr,
    kv_indptr_stride: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
    num_tokens_upper: tl.constexpr,
    page_size: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    iters = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    topk_id = tl.program_id(axis=2)

    num_steps = tl.num_programs(axis=0)
    num_seqs = tl.num_programs(axis=1)
    topk = tl.num_programs(axis=2)

    kv_indices += kv_indices_stride * iters
    kv_indptr += kv_indptr_stride * iters
    iters += 1

    load_offset = tl.arange(0, bs_upper)
    seq_lens = tl.load(paged_kernel_lens + load_offset, mask=load_offset < bid, other=0)
    seq_len = tl.load(paged_kernel_lens + bid)
    cum_seq_len = tl.sum(seq_lens)

    # Update kv_indices
    kv_offset = cum_seq_len * topk + bid * iters * topk + topk_id * (seq_len + iters)
    kv_ptr = kv_indices + kv_offset
    token_pool_ptr = req_to_token + tl.load(req_pool_indices + bid) * pool_len

    kv_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(seq_len, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = kv_offset < seq_len
        data = tl.load(token_pool_ptr + kv_offset, mask=mask)
        tl.store(kv_ptr + kv_offset, data, mask=mask)
        kv_offset += BLOCK_SIZE

    extend_offset = tl.arange(0, iter_upper)
    if page_size == 1 or topk == 1:
        extend_data = tl.load(
            token_pool_ptr + seq_len + topk_id * num_steps + tl.arange(0, iter_upper),
            mask=extend_offset < iters,
        )
    else:
        prefix_len = seq_len
        last_page_len = prefix_len % page_size
        num_new_pages_per_topk = (
            last_page_len + num_steps + page_size - 1
        ) // page_size
        prefix_base = seq_len // page_size * page_size
        start = (
            prefix_base + topk_id * num_new_pages_per_topk * page_size + last_page_len
        )
        extend_data = tl.load(
            token_pool_ptr + start + extend_offset,
            mask=extend_offset < iters,
        )

    tl.store(kv_ptr + seq_len + extend_offset, extend_data, mask=extend_offset < iters)

    # Update kv_indptr
    bs_offset = tl.arange(0, num_tokens_upper)

    zid = bid * topk + topk_id
    if zid == 0:
        zid = num_seqs * topk
    positions = tl.load(positions + bs_offset, mask=bs_offset < zid, other=0)
    base = tl.sum(positions)
    tl.store(kv_indptr + zid, base + zid * iters)


@triton.jit
def align_evict_mask_to_page_size(
    seq_lens,
    evict_mask,
    page_size: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    t_range = tl.arange(0, BLOCK_SIZE)

    bid = tl.program_id(axis=0)
    seq_len = tl.load(seq_lens + bid)
    io_mask = t_range < num_draft_tokens
    mask_row = tl.load(
        evict_mask + bid * num_draft_tokens + t_range, mask=io_mask, other=0
    )

    num_trues = tl.sum(mask_row)
    num_false = num_draft_tokens - num_trues

    start = (seq_len + num_false - 1) // page_size * page_size - seq_len
    for i in range(max(start, 0), min(start + page_size, num_draft_tokens)):
        tl.store(evict_mask + bid * num_draft_tokens + i, False)


@triton.jit
def get_target_cache_loc(
    tgt_cache_loc,
    to_free_slots,
    accept_length,
    to_free_num_slots,
    out_cache_loc,
    num_verify_tokens: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
    bs_upper: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    offset = tl.arange(0, num_verify_tokens_upper)
    bs_offset = tl.arange(0, bs_upper)

    # write the first part to tgt_cache_loc
    accept_len_all = tl.load(accept_length + bs_offset, mask=bs_offset < bid)
    tgt_cache_loc_start = tl.sum(accept_len_all) + bid
    copy_len = tl.load(accept_length + bid) + 1
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + offset, mask=offset < copy_len
    )
    tl.store(
        tgt_cache_loc + tgt_cache_loc_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )

    # write the second part to to_free_num_pages
    to_free_num_slots_all = tl.load(to_free_num_slots + bs_offset, mask=bs_offset < bid)
    to_free_num_slots_cur = tl.load(to_free_num_slots + bid)
    out_cache_loc_start = num_verify_tokens - to_free_num_slots_cur
    to_free_slots_start = tl.sum(to_free_num_slots_all)

    copy_len = to_free_num_slots_cur
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + out_cache_loc_start + offset,
        mask=offset < copy_len,
    )
    tl.store(
        to_free_slots + to_free_slots_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )


@torch.compile(dynamic=True)
def get_src_tgt_cache_loc(
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accept_index: torch.Tensor,
    accept_length: torch.Tensor,
    draft_token_num: int,
    page_size: int,
):
    src_cache_loc = out_cache_loc[accept_index]
    tgt_cache_loc = torch.empty_like(src_cache_loc)
    extended_len = seq_lens + draft_token_num
    keep_len = torch.minimum(
        (seq_lens + accept_length + 1 + page_size - 1) // page_size * page_size,
        extended_len,
    )
    to_free_num_slots = extended_len - keep_len
    return src_cache_loc, tgt_cache_loc, to_free_num_slots


@triton.jit
def filter_finished_cache_loc_kernel(
    out_cache_loc,
    tgt_cache_loc,
    accept_length,
    accept_length_filter,
    bs_upper: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
):
    bid = tl.program_id(0)
    bs_offset = tl.arange(0, bs_upper)

    accept_length_all = tl.load(accept_length + bs_offset, mask=bs_offset < bid)
    old_start = tl.sum(accept_length_all) + bid

    accept_length_filter_all = tl.load(
        accept_length_filter + bs_offset, mask=bs_offset < bid
    )
    new_start = tl.sum(accept_length_filter_all)

    copy_len = tl.load(accept_length_filter + bid)
    copy_offset = tl.arange(0, num_verify_tokens_upper)
    value = tl.load(
        tgt_cache_loc + old_start + copy_offset, mask=copy_offset < copy_len
    )
    tl.store(
        out_cache_loc + new_start + copy_offset, value, mask=copy_offset < copy_len
    )


@torch.compile(dynamic=True)
def create_accept_length_filter(
    accept_length: torch.Tensor,
    unfinished_index_device: torch.Tensor,
    seq_lens: torch.Tensor,
):
    accept_length_filter = torch.zeros_like(accept_length)
    accept_length_filter[unfinished_index_device] = (
        accept_length[unfinished_index_device] + 1
    )
    seq_lens.add_(accept_length + 1)
    return accept_length_filter


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
            torch.arange(-1, topk, dtype=torch.long, device="cuda")
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

        if hidden_states.shape[0] > 0:
            selected_input_index = topk_cs_index.flatten() // topk + torch.arange(
                0, hidden_states.shape[0], step=topk, device="cuda"
            ).repeat_interleave(topk)
            hidden_states = hidden_states[selected_input_index, :]

        tree_info = (
            expand_scores,  # shape: (b, topk, topk)
            topk_index,  # shape: (b, topk * topk)
            topk_cs_index + (topk**2 * (i - 1) + topk),  # shape: (b, topk)
        )

    return input_ids, hidden_states, scores, tree_info


def _generate_simulated_accept_index(
    accept_index,
    predict,
    accept_length,
    simulate_acc_len,
    bs,
    spec_steps,
):
    simulate_acc_len_float = float(simulate_acc_len)
    if SIMULATE_ACC_METHOD == "multinomial":
        simulated_values = torch.normal(
            mean=simulate_acc_len_float,
            std=1.0,
            size=(1,),
            device="cpu",
        )
        # clamp simulated values to be between 1 and self.spec_steps
        simulated_values = torch.clamp(simulated_values, min=1.0, max=spec_steps + 1)
        simulate_acc_len = int(simulated_values.round().item())
    elif SIMULATE_ACC_METHOD == "match-expected":
        # multinomial sampling does not match the expected length
        # we keep it for the sake of compatibility of existing tests
        # but it's better to use "match-expected" for the cases that need to
        # match the expected length, One caveat is that this will only sample
        # either round down or round up of the expected length
        simulate_acc_len_float = max(1.0, min(spec_steps + 1, simulate_acc_len_float))
        lower = int(simulate_acc_len_float // 1)
        upper = lower + 1 if lower < spec_steps + 1 else lower
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
        (bs, spec_steps + 1), -1, dtype=torch.int32, device="cuda"
    )
    sim_accept_index[:, :simulate_acc_len] = accept_indx_first_col + torch.arange(
        simulate_acc_len, device=accept_index.device
    )
    accept_length.fill_(simulate_acc_len - 1)
    predict.fill_(100)  # some legit token id
    return sim_accept_index


def traverse_tree(
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    draft_tokens: torch.Tensor,
    grammar: BaseGrammarObject,
    allocate_token_bitmask: torch.Tensor,
):
    """
    Traverse the tree constructed by the draft model to generate the logits mask.
    """
    assert (
        retrieve_next_token.shape == retrieve_next_sibling.shape == draft_tokens.shape
    )

    allocate_token_bitmask.fill_(0)

    def dfs(
        curr: int,
        retrieve_next_token: torch.Tensor,
        retrieve_next_sibling: torch.Tensor,
        parent_pos: int,
    ):
        if curr == 0:
            # the first token generated by the target model, and thus it is always
            # accepted from the previous iteration
            accepted = True
        else:
            parent_bitmask = allocate_token_bitmask[parent_pos]
            curr_token_id = draft_tokens[curr]
            # 32 boolean bitmask values are packed into 32-bit integers
            accepted = (
                parent_bitmask[curr_token_id // 32] & (1 << (curr_token_id % 32))
            ) != 0

        if accepted:
            if curr != 0:
                # Accept the current token
                grammar.accept_token(draft_tokens[curr])
            if not grammar.is_terminated():
                # Generate the bitmask for the current token
                grammar.fill_vocab_mask(allocate_token_bitmask, curr)
                if retrieve_next_token[curr] != -1:
                    # Visit the child node
                    dfs(
                        retrieve_next_token[curr],
                        retrieve_next_token,
                        retrieve_next_sibling,
                        curr,
                    )

            if curr != 0:
                # Rollback the current token
                grammar.rollback(1)

        if retrieve_next_sibling[curr] != -1:
            # Visit the sibling node
            dfs(
                retrieve_next_sibling[curr],
                retrieve_next_token,
                retrieve_next_sibling,
                parent_pos,
            )

    dfs(0, retrieve_next_token, retrieve_next_sibling, -1)


def generate_token_bitmask(
    reqs: List[Req],
    verify_input: EagleVerifyInput,
    retrieve_next_token_cpu: torch.Tensor,
    retrieve_next_sibling_cpu: torch.Tensor,
    draft_tokens_cpu: torch.Tensor,
    vocab_size: int,
):
    """
    Generate the logit mask for structured output.
    Draft model's token can be either valid or invalid with respect to the grammar.
    We need to perform DFS to
    1. figure out which tokens are accepted by the grammar.
    2. if so, what is the corresponding logit mask.
    """

    num_draft_tokens = draft_tokens_cpu.shape[-1]

    allocate_token_bitmask = None
    assert len(reqs) == retrieve_next_token_cpu.shape[0]
    grammar = None
    for i, req in enumerate(reqs):
        if req.grammar is not None:
            if allocate_token_bitmask is None:
                allocate_token_bitmask = req.grammar.allocate_vocab_mask(
                    vocab_size=vocab_size,
                    batch_size=draft_tokens_cpu.numel(),
                    device="cpu",
                )
            grammar = req.grammar
            s = time.perf_counter()
            traverse_tree(
                retrieve_next_token_cpu[i],
                retrieve_next_sibling_cpu[i],
                draft_tokens_cpu[i],
                req.grammar,
                allocate_token_bitmask[
                    i * num_draft_tokens : (i + 1) * num_draft_tokens
                ],
            )
            tree_traverse_time = time.perf_counter() - s
            if tree_traverse_time > TREE_TRAVERSE_TIME_THRESHOLD:
                logger.warning(
                    f"Bit mask generation took {tree_traverse_time} seconds with "
                    f"grammar: {req.grammar}"
                )

    verify_input.grammar = grammar
    return allocate_token_bitmask
