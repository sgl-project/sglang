import logging
import os
import threading
import time
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.cpp_lookahead.lookahead_cache import LookaheadCache
from sglang.srt.speculative.lookahead_utils import LookaheadVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import broadcast_pyobj

logger = logging.getLogger(__name__)

USE_FULL_MASK = True


class LOOKAHEADWorker:
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        self.branch_length: int = server_args.speculative_lookahead_branch_length
        self.max_match_window_size: int = (
            server_args.speculative_lookahead_max_match_window_size
        )

        self.max_batch_size = target_worker.max_running_requests
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"

        self._init_preallocated_tensors()

        self.lookahead_cache = LookaheadCache(
            min_match_window_size=server_args.speculative_lookahead_min_match_window_size,
            max_match_window_size=server_args.speculative_lookahead_max_match_window_size,
            min_bfs_breadth=server_args.speculative_lookahead_min_bfs_breadth,
            max_bfs_breadth=server_args.speculative_lookahead_max_bfs_breadth,
            capacity=server_args.speculative_lookahead_capacity,
            branch_length=server_args.speculative_lookahead_branch_length,
            draft_token_num=server_args.speculative_num_draft_tokens,
        )

    def clear_cache_pool(self):
        self.lookahead_cache.reset()

    def _efficient_concat_last_n(self, seq1: List[int], seq2: List[int], n: int):
        seq2_len = len(seq2)
        if seq2_len >= n:
            return seq2[-n:]

        need_from_seq1 = n - seq2_len
        return seq1[-need_from_seq1:] + seq2

    def _init_preallocated_tensors(self):
        max_total_drafts = self.max_batch_size * self.draft_token_num
        max_total_mask_size = (
            self.max_batch_size * self.draft_token_num * self.draft_token_num
        )

        self.draft_tokens = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.retrieve_indexes = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrive_next_token = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.retrive_next_sibling = torch.empty(
            (self.max_batch_size, self.draft_token_num),
            dtype=torch.int64,
            device=self.device,
        )
        self.positions = torch.empty(
            (max_total_drafts,), dtype=torch.int64, device=self.device
        )
        self.tree_mask = torch.empty(
            (max_total_mask_size,), dtype=torch.bool, device=self.device
        )

        self.draft_tokens_batch = []
        self.tree_mask_batch = []
        self.retrieve_indexes_batch = []
        self.retrive_next_token_batch = []
        self.retrive_next_sibling_batch = []
        self.positions_batch = []

        for bs in range(0, self.max_batch_size + 1):
            self.retrieve_indexes_batch.append(self.retrieve_indexes[:bs, :])
            self.retrive_next_token_batch.append(self.retrive_next_token[:bs, :])
            self.retrive_next_sibling_batch.append(self.retrive_next_sibling[:bs, :])
            self.positions_batch.append(self.positions[: bs * self.draft_token_num])
            self.draft_tokens_batch.append(
                self.draft_tokens[: bs * self.draft_token_num]
            )
            self.tree_mask_batch.append(
                self.tree_mask[: bs * self.draft_token_num * self.draft_token_num]
            )

    def _prepare_draft_tokens(
        self, batch: ScheduleBatch
    ) -> tuple[np.ndarray, np.ndarray]:
        bs = batch.batch_size()

        self.lookahead_cache.synchronize()
        batch_tokens = []
        for req in batch.reqs:
            check_token = self._efficient_concat_last_n(
                req.origin_input_ids, req.output_ids, self.max_match_window_size
            )
            batch_tokens.append(check_token)
        req_drafts, mask = self.lookahead_cache.batch_get(batch_tokens)
        total_draft_token_num = len(req_drafts)

        # Check if speculative decoding is needed; here we always enforce it
        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"
        return req_drafts, mask

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
        if batch.forward_mode.is_extend():
            return

        bs = batch.batch_size()

        retrive_index = self.retrieve_indexes_batch[bs]
        retrive_next_token = self.retrive_next_token_batch[bs]
        retrive_next_sibling = self.retrive_next_sibling_batch[bs]
        positions = self.positions_batch[bs]
        tree_mask = self.tree_mask_batch[bs]
        draft_tokens = self.draft_tokens_batch[bs]

        req_drafts, mask = self._prepare_draft_tokens(batch)
        tree_mask.copy_(torch.from_numpy(mask), non_blocking=True)
        draft_tokens.copy_(torch.from_numpy(req_drafts), non_blocking=True)

        reconstruct_indices_from_tree_mask(
            tree_mask,
            batch.seq_lens,
            positions,  # mutable
            retrive_index,  # mutable
            retrive_next_token,  # mutable
            retrive_next_sibling,  # mutable
            bs,
            self.draft_token_num,
        )

        # NOTE: QLEN_MASK is faster than FULL_MASK, but requires corresponding changes in flashinfer.
        # Testing shows about 8% performance improvement (the effect is roughly proportional to batch size).
        if USE_FULL_MASK:
            tree_mask = []
            mask = mask.reshape(
                batch.batch_size(), self.draft_token_num, self.draft_token_num
            )
            for i, req in enumerate(batch.reqs):
                seq_len = len(req.origin_input_ids) + len(req.output_ids)
                req_mask = torch.ones((self.draft_token_num, seq_len - 1)).cuda()
                req_mask = torch.cat(
                    (req_mask, torch.from_numpy(mask[i]).cuda()), dim=1
                ).to(torch.bool)
                tree_mask.append(req_mask.flatten())
            tree_mask = torch.cat(tree_mask, dim=0)

        batch.spec_algorithm = SpeculativeAlgorithm.LOOKAHEAD
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = LookaheadVerifyInput(
            draft_tokens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            self.draft_token_num,
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)

    def _update_lookahead_cache(self, batch: ScheduleBatch):
        batch_tokens = []
        for req in batch.reqs:
            # FIXME: Whether to insert 'extend' into the cache or not, after testing,
            # there is not much difference, so we will not insert it for now.
            # if batch.forward_mode.is_extend():
            #     put_ids = req.origin_input_ids + req.output_ids
            # else:
            put_ids = self._efficient_concat_last_n(
                req.origin_input_ids, req.output_ids, self.branch_length
            )
            batch_tokens.append(put_ids)
        self.lookahead_cache.batch_put(batch_tokens)

    def forward_batch_speculative_generation(self, batch: ScheduleBatch):
        self._prepare_for_speculative_decoding(batch)
        model_worker_batch = batch.get_model_worker_batch()
        bid = model_worker_batch.bid
        num_accepted_tokens = 0

        if model_worker_batch.forward_mode.is_target_verify():
            logits_output, _, can_run_cuda_graph = (
                self.target_worker.forward_batch_generation(
                    model_worker_batch, skip_sample=True
                )
            )
            verify_input = model_worker_batch.spec_info
            logits_output, next_token_ids, num_accepted_tokens = verify_input.verify(
                batch, logits_output, self.page_size
            )
            self._update_lookahead_cache(batch)
            batch.forward_mode = ForwardMode.DECODE

        else:
            logits_output, next_token_ids, can_run_cuda_graph = (
                self.target_worker.forward_batch_generation(model_worker_batch)
            )

        return (
            logits_output,
            next_token_ids,
            bid,
            num_accepted_tokens,
            can_run_cuda_graph,
        )
