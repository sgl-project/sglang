"""
Suffix decoding worker that reuses NGRAMWorker with a cache adapter.

This is a thin wrapper that replaces NgramCache with SuffixCacheAdapter,
allowing all the tree-based verification logic to be reused.

Performance optimizations over the base implementation:
- Numpy-based full-mask construction (single CPU→GPU transfer instead of
  per-request torch.ones().cuda() + torch.cat() allocations)
- O(n) BFS ancestor propagation in tree mask building (replaces O(n×d)
  while-loops)
- Throttled inactive-request cleanup in the cache adapter
"""

import logging
from typing import Optional

import numpy as np
import torch
from sgl_kernel.speculative import reconstruct_indices_from_tree_mask

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_worker import USE_FULL_MASK, NGRAMWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.suffix_cache_adapter import SuffixCacheAdapter
from sglang.srt.speculative.suffix_info import SuffixVerifyInput

logger = logging.getLogger(__name__)


class SuffixWorker(NGRAMWorker):
    """
    Suffix decoding worker that inherits from NGRAMWorker.

    The only difference is using SuffixCacheAdapter instead of NgramCache.
    All tree-based verification logic is inherited from NGRAMWorker,
    including forward_batch_generation.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Call parent __init__ which sets up all the infrastructure
        super().__init__(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        self.ngram_cache = SuffixCacheAdapter(
            draft_token_num=server_args.speculative_num_draft_tokens,
            max_batch_size=self.max_batch_size,
            max_tree_depth=server_args.speculative_suffix_max_tree_depth,
            max_cached_requests=server_args.speculative_suffix_max_cached_requests,
            max_spec_factor=server_args.speculative_suffix_max_spec_factor,
            min_token_prob=server_args.speculative_suffix_min_token_prob,
        )

    def _prepare_draft_tokens(self, batch):
        """
        Override to pass token sequences to the cache adapter.

        NGRAMWorker passes only last N tokens, but the suffix cache needs:
        1. Full prompt for start_request()
        2. Output IDs for incremental cache updates
        3. Request identity tracking

        Passes origin_input_ids and output_ids *separately* to avoid
        O(seq_len) list concatenation per request per step.
        """
        bs = batch.batch_size()

        self.ngram_cache.synchronize()
        batch_req_ids = []
        batch_prompts = []
        batch_output_ids = []
        for req in batch.reqs:
            batch_req_ids.append(req.rid)
            batch_prompts.append(req.origin_input_ids)
            batch_output_ids.append(req.output_ids)

        req_drafts, mask = self.ngram_cache.batch_get(
            batch_req_ids, batch_prompts, batch_output_ids
        )
        total_draft_token_num = len(req_drafts)

        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"

        return req_drafts, mask

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch):
        """
        Override to set batch.spec_algorithm to SUFFIX and use SuffixVerifyInput.

        Builds the custom attention mask for the triton extend_attention_fwd
        kernel used on AMD/ROCm (non-MLA path).

        Performance: the full-mask is built with numpy on CPU and transferred
        to GPU in a single copy, avoiding per-request torch.ones().cuda() +
        torch.cat() allocations.
        """
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
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            bs,
            self.draft_token_num,
        )

        if USE_FULL_MASK:
            # Build [ones(draft_num, seq_len-1) | tree_mask(draft_num, draft_num)]
            # using numpy on CPU, then do a single transfer to GPU.
            # This avoids per-request torch.ones().cuda() + torch.cat() allocations.
            mask_reshaped = mask.reshape(bs, self.draft_token_num, self.draft_token_num)
            parts = []
            for i, req in enumerate(batch.reqs):
                seq_len = len(req.origin_input_ids) + len(req.output_ids)
                width = (seq_len - 1) + self.draft_token_num
                full_mask = np.ones((self.draft_token_num, width), dtype=np.bool_)
                full_mask[:, seq_len - 1 :] = mask_reshaped[i]
                parts.append(full_mask.ravel())
            tree_mask = torch.from_numpy(np.concatenate(parts)).to(
                device=batch.seq_lens.device
            )

        batch.spec_algorithm = SpeculativeAlgorithm.SUFFIX
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.spec_info = SuffixVerifyInput(
            draft_tokens,
            tree_mask,
            positions,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            self.draft_token_num,
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)

    def _update_ngram_cache(self, batch):
        """
        No-op override.  The suffix cache is updated incrementally inside
        batch_get() (before speculation), so there is nothing to do here.
        Skipping avoids an O(bs × seq_len) list construction per step.
        """
        pass
