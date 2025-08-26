from __future__ import annotations

from typing import TYPE_CHECKING, List, Type, Callable

import numpy as np
import torch
import triton
import threading

import logging

logger = logging.getLogger(__name__)


from dataclasses import dataclass

from sglang.srt.managers.schedule_batch import (
    ScheduleBatch,
    get_last_loc,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.speculative.eagle_utils import (
    assign_req_to_token_pool,
    create_flashinfer_kv_indices_triton,
    get_src_tgt_cache_loc,
    get_target_cache_loc,
)

from sgl_kernel import lookahead_verify_tree_greedy
from sglang.srt.utils import next_power_of_2

@dataclass
class LookaheadVerifyInput:
    def __init__(
        self,
        draft_token: torch.Tensor,
        tree_mask: torch.Tensor,
        positions: torch.Tensor,
        retrive_index: torch.Tensor,
        retrive_next_token: torch.Tensor,
        retrive_next_sibling: torch.Tensor,

        accept_length: torch.Tensor,
        accept_token_ids: torch.Tensor,
        last_verified_ids: torch.Tensor,
        flatten_index: torch.Tensor,
        total_accept_num: torch.Tensor,

        draft_token_num: int,
    ):
        self.draft_token = draft_token
        self.custom_mask = tree_mask
        self.positions = positions
        self.retrive_index = retrive_index
        self.retrive_next_token = retrive_next_token
        self.retrive_next_sibling = retrive_next_sibling

        self.accept_length = accept_length
        self.accept_token_ids = accept_token_ids
        self.last_verified_ids = last_verified_ids
        self.flatten_index = flatten_index
        self.total_accept_num = total_accept_num

        self.draft_token_num = draft_token_num
        self.device = self.custom_mask.device

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

        cum_kv_seq_len = torch.zeros(
            (bs + 1,), dtype=torch.int32, device=self.device
        )

        paged_kernel_lens = paged_kernel_lens + self.draft_token_num
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        self.qo_indptr = torch.arange(
            0, bs + 1, dtype=torch.int32, device=self.device
        ) * self.draft_token_num

        kv_indices = torch.empty(cum_kv_seq_len[-1], dtype=torch.int32, device=self.device)

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
        accept_index_flatten: torch.Tensor,
    ):
        logits_output.next_token_logits = logits_output.next_token_logits[
            accept_index_flatten
        ]

        accept_token_ids_cpu = self.accept_token_ids.tolist()
        for req, accept_token_ids in zip(batch.reqs, accept_token_ids_cpu):
            for accept_token_id in accept_token_ids:
                if accept_token_id < 0:
                    break
                req.output_ids.append(accept_token_id)
                batch.seq_lens_sum += 1
            req.check_finished()


    def _free_cache(
        self, 
        batch: ScheduleBatch, 
        page_size: int, 
        accept_index_flatten: torch.Tensor
    ):
        bs = batch.batch_size()
        if page_size == 1:
            evict_index_flatten = self.flatten_index[self.total_accept_num:]
            batch.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_index_flatten])
            batch.out_cache_loc = batch.out_cache_loc[accept_index_flatten]
        else:
            # Shift the accepted tokens to the beginning.
            # Only evict the last part
            src_cache_loc, tgt_cache_loc, to_free_num_slots = get_src_tgt_cache_loc(
                batch.seq_lens,
                batch.out_cache_loc,
                accept_index_flatten,
                self.accept_length - 1,
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
                self.accept_length - 1,
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

        assign_req_to_token_pool[(bs,)](
            batch.req_pool_indices,
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + self.accept_length,
            batch.out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            triton.next_power_of_2(bs),
        )

    def verify(
        self,
        batch: ScheduleBatch,
        logits_output: LogitsProcessorOutput,
        page_size: int,
    ) -> torch.Tensor:
        target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).to(torch.int32)
        lookahead_verify_tree_greedy(
            accept_token_num=self.accept_length,  # mutable, at least 1
            accept_token_ids=self.accept_token_ids, # mutable
            last_verified_ids=self.last_verified_ids, # mutable
            flatten_index=self.flatten_index, # mutable
            total_accept_num=self.total_accept_num, # mutable
            candidates=self.draft_token,
            retrive_index=self.retrive_index,
            retrive_next_token=self.retrive_next_token,
            retrive_next_sibling=self.retrive_next_sibling,
            target_predict=target_predict,
            # TODO: eos_ids
            eos_token_id=batch.eos_id,
        )

        accept_index_flatten = self.flatten_index[:self.total_accept_num]

        self._free_cache(batch, page_size, accept_index_flatten)
        self._fill_requests(batch, logits_output, accept_index_flatten)

        batch.seq_lens.add_(self.accept_length)

        return logits_output, self.last_verified_ids, self.accept_length.sum().item()

    def filter_batch(self, new_indices: torch.Tensor):
        pass

    def merge_batch(self, spec_info: LookaheadVerifyInput):
        pass