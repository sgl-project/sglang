from __future__ import annotations

import copy
import logging
from typing import Optional, Tuple

import torch
import triton

from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)

from dataclasses import dataclass

import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import apply_custom_logit_processor
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import (
    alloc_paged_token_slots_extend,
    alloc_token_slots,
    get_last_loc,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    TREE_SPEC_KERNEL_AVAILABLE,
    assign_req_to_token_pool,
    get_src_tgt_cache_loc,
    get_target_cache_loc,
)
from sglang.srt.utils import is_cuda, is_hip, next_power_of_2

if is_cuda():
    from sgl_kernel import (
        top_k_renorm_prob,
        top_p_renorm_prob,
        tree_speculative_sampling_target_only,
        verify_tree_greedy,
    )
elif is_hip():
    from sgl_kernel import verify_tree_greedy


@dataclass
class NgramVerifyInput(SpecInput):
    def __init__(
        self,
        draft_token: torch.Tensor,
        tree_mask: torch.Tensor,
        positions: torch.Tensor,
        retrive_index: torch.Tensor,
        retrive_next_token: torch.Tensor,
        retrive_next_sibling: torch.Tensor,
        draft_token_num: int,
    ):
        super().__init__(SpecInputType.NGRAM_VERIFY)
        self.draft_token = draft_token
        self.custom_mask = tree_mask
        self.positions = positions
        self.retrive_index = retrive_index
        self.retrive_next_token = retrive_next_token
        self.retrive_next_sibling = retrive_next_sibling
        self.draft_token_num = draft_token_num
        self.device = self.custom_mask.device

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.draft_token_num, self.draft_token_num

    def prepare_for_verify(self, batch: ScheduleBatch, page_size: int):
        if batch.forward_mode.is_idle():
            return

        batch.input_ids = self.draft_token

        if page_size == 1:
            batch.out_cache_loc = alloc_token_slots(
                batch.tree_cache,
                len(batch.input_ids),
            )
            end_offset = batch.seq_lens + self.draft_token_num
        else:
            # TODO(lsyin): add prefix lens cpu here to support page size > 1
            prefix_lens = batch.seq_lens
            prefix_lens_cpu = batch.seq_lens_cpu
            end_offset = prefix_lens + self.draft_token_num
            end_offset_cpu = prefix_lens_cpu + self.draft_token_num
            last_loc = get_last_loc(
                batch.req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            batch.out_cache_loc = alloc_paged_token_slots_extend(
                batch.tree_cache,
                prefix_lens,
                prefix_lens_cpu,
                end_offset,
                end_offset_cpu,
                last_loc,
                len(batch.input_ids),
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
            triton.next_power_of_2(bs),
        )

    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        bs = len(req_pool_indices)

        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=self.device)

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        self.qo_indptr = (
            torch.arange(0, bs + 1, dtype=torch.int32, device=self.device)
            * self.draft_token_num
        )

        kv_indices = torch.empty(
            cum_kv_seq_len[-1], dtype=torch.int32, device=self.device
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
        return kv_indices, cum_kv_seq_len, self.qo_indptr, self.custom_mask

    def _fill_requests(
        self,
        batch: ScheduleBatch,
        logits_output: torch.Tensor,
    ):
        accept_index_cpu = self.accept_index.tolist()
        predict_cpu = self.predict.tolist()
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
                    self.accept_index[i, j + 1 :] = -1
                    break
                else:
                    if req.grammar is not None:
                        try:
                            req.grammar.accept_token(id)
                        except ValueError as e:
                            logger.info(
                                f"{i=}, {req=}\n"
                                f"{self.accept_index=}\n"
                                f"{self.predict=}\n"
                            )
                            raise e
            req.spec_verify_ct += 1
        if has_finished:
            self.accept_length = (self.accept_index != -1).sum(dim=1) - 1
        self.accept_index = self.accept_index[self.accept_index != -1]

        logits_output.next_token_logits = logits_output.next_token_logits[
            self.accept_index
        ]
        if logits_output.hidden_states:
            logits_output.hidden_states = logits_output.hidden_states[self.accept_index]
        self.verified_id = self.predict[self.accept_index]

    def _free_cache(
        self, batch: ScheduleBatch, page_size: int, accept_length_cpu: torch.Tensor
    ):
        bs = batch.batch_size()
        # Free the KV cache for unaccepted tokens
        if page_size == 1:
            # TODO: boolean array index leads to a device sync. Remove it.
            evict_mask = torch.full_like(self.draft_token, True, dtype=torch.bool)
            evict_mask[self.accept_index] = False
            batch.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            batch.out_cache_loc = batch.out_cache_loc[self.accept_index]
        else:
            # Shift the accepted tokens to the beginning.
            # Only evict the last part
            src_cache_loc, tgt_cache_loc, to_free_num_slots = get_src_tgt_cache_loc(
                batch.seq_lens,
                batch.out_cache_loc,
                self.accept_index,
                self.accept_length,
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
                self.accept_length,
                to_free_num_slots,
                batch.out_cache_loc,
                self.draft_token_num,
                next_power_of_2(self.draft_token_num),
                next_power_of_2(bs),
            )

            # Free the kv cache
            batch.token_to_kv_pool_allocator.free(to_free_slots)

            # Copy the kv cache
            batch.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                tgt_cache_loc, src_cache_loc
            )
            batch.out_cache_loc = tgt_cache_loc

        accept_length_list = accept_length_cpu.tolist()
        for i, req in enumerate(batch.reqs):
            req.kv_committed_len += accept_length_list[i] + 1
            req.kv_allocated_len = req.kv_committed_len

        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + self.accept_length + 1,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )

    def _greedy_verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
    ):
        bs = batch.batch_size()
        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1)
        target_predict = target_predict.reshape(bs, self.draft_token_num)

        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        self.predict = torch.empty(predict_shape, dtype=torch.int32, device=self.device)
        self.accept_index = torch.full(
            (bs, self.draft_token_num), -1, dtype=torch.int32, device=self.device
        )
        self.accept_length = torch.empty((bs,), dtype=torch.int32, device=self.device)

        verify_tree_greedy(
            predicts=self.predict,  # mutable
            accept_index=self.accept_index,  # mutable
            accept_token_num=self.accept_length,  # mutable
            candidates=candidates,
            retrive_index=self.retrive_index,
            retrive_next_token=self.retrive_next_token,
            retrive_next_sibling=self.retrive_next_sibling,
            target_predict=target_predict,
        )

    def _sampling_verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ):
        bs = batch.batch_size()
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        self.predict = torch.empty(predict_shape, dtype=torch.int32, device=self.device)
        self.accept_index = torch.full(
            (bs, self.draft_token_num), -1, dtype=torch.int32, device=self.device
        )
        self.accept_length = torch.empty((bs,), dtype=torch.int32, device=self.device)
        # apply temperature and get target probs
        expanded_temperature = torch.repeat_interleave(
            sampling_info.temperatures, self.draft_token_num, dim=0
        )  # (bs * draft_token_num, 1)

        target_probs = F.softmax(
            logits_output.next_token_logits / expanded_temperature, dim=-1
        )  # (bs * draft_token_num, vocab_size)

        # NOTE: The test shows that top_p_renorm_prob and top_k_renorm_prob are the key factors
        # contributing to the poor performance of _sampling_verify.
        target_probs = top_k_renorm_prob(
            target_probs,
            torch.repeat_interleave(sampling_info.top_ks, self.draft_token_num, dim=0),
        )  # (bs * draft_token_num, vocab_size)

        if sampling_info.need_top_p_sampling:
            # logger.info("Using top-p sampling in speculative decoding verification.")
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(
                    sampling_info.top_ps, self.draft_token_num, dim=0
                ),
            )

        target_probs = target_probs.reshape(bs, self.draft_token_num, -1)
        draft_probs = torch.zeros(
            target_probs.shape, dtype=torch.float32, device=self.device
        )

        # coins for rejection sampling
        coins = torch.rand_like(candidates, dtype=torch.float32, device=self.device)
        # coins for final sampling
        coins_for_final_sampling = torch.rand(
            (bs,), dtype=torch.float32, device=self.device
        )
        tree_speculative_sampling_target_only(
            predicts=self.predict,  # mutable
            accept_index=self.accept_index,  # mutable
            accept_token_num=self.accept_length,  # mutable
            candidates=candidates.to(torch.int64),
            retrive_index=self.retrive_index.to(torch.int64),
            retrive_next_token=self.retrive_next_token.to(torch.int64),
            retrive_next_sibling=self.retrive_next_sibling.to(torch.int64),
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_for_final_sampling,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=get_global_server_args().speculative_accept_threshold_single,
            threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
            deterministic=True,
        )

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
        vocab_mask: Optional[torch.Tensor] = None,  # For grammar
    ) -> torch.Tensor:
        bs = self.retrive_index.shape[0]
        sampling_info = batch.sampling_info

        if bs != len(sampling_info):
            sampling_info = copy.deepcopy(sampling_info)
            # NOTE: retrive_index are the indices of the requests that are kept.
            sampling_info.filter_batch(self.retrive_index.tolist(), self.retrive_index)

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
                device=self.device,
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

        # Sample tokens. Force greedy sampling on AMD
        is_all_greedy = (
            sampling_info.is_all_greedy or envs.SGLANG_NGRAM_FORCE_GREEDY_VERIFY.get()
        )
        if (not is_all_greedy) and (not TREE_SPEC_KERNEL_AVAILABLE):
            logger.warning(
                "Tree speculative sampling kernel unavailable (likely AMD/HIP build). "
                "Falling back to greedy verification."
            )

        if is_all_greedy or not TREE_SPEC_KERNEL_AVAILABLE:
            self._greedy_verify(batch, logits_output)
        else:
            # NOTE: Compared with greedy_verify, the performance of _sampling_verify is relatively poor.
            self._sampling_verify(batch, logits_output, sampling_info)

        self._fill_requests(batch, logits_output)

        accept_length_cpu = self.accept_length.cpu()
        num_accepted_tokens = accept_length_cpu.sum().item()

        self._free_cache(batch, page_size, accept_length_cpu)

        batch.seq_lens.add_(self.accept_length + 1)
        batch.seq_lens_cpu.add_(accept_length_cpu + 1)

        return logits_output, self.verified_id, num_accepted_tokens

    def filter_batch(self, new_indices: torch.Tensor, has_been_filtered: bool = True):
        pass

    def merge_batch(self, spec_info: NgramVerifyInput):
        pass
