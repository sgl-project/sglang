"""SUFFIX speculative-decoding worker (no pipeline parallelism).

SUFFIX is *model-free* speculative decoding (Snowflake ArcticInference's
SuffixDecoding): a CPU suffix tree built over each request's prompt + generated
tokens proposes draft continuations, which are verified in a single target
forward pass. It is especially effective on agentic / repetitive workloads and
adds no draft-model GPU cost.

Implementation reuses :class:`NGRAMWorker` unchanged for the verify + KV-cache
machinery; the only difference is the *draft source*. ``NGRAMWorker`` reads
draft candidates from ``self.ngram_corpus`` (an n-gram corpus); here we replace
that attribute with a :class:`SuffixCacheAdapter` wrapping
``arctic_inference``'s ``SuffixDecodingCache``, and override the two hooks that
talk to it so the adapter is fed each request's full token history (the suffix
tree needs prompt + output, not just the recent n-gram pattern).
"""

from __future__ import annotations

import logging
from typing import List, Optional

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_worker import NGRAMWorker
from sglang.srt.speculative.suffix_cache_adapter import SuffixCacheAdapter

logger = logging.getLogger(__name__)


class SuffixWorker(NGRAMWorker):
    """``NGRAMWorker`` whose draft source is an Arctic suffix tree.

    Tensor parallelism is supported as-is (inherited from ``NGRAMWorker``: the
    suffix draft is built per-rank on CPU and the target forward is TP-sharded
    transparently). Pipeline parallelism and overlap-scheduling (spec-v2) are
    added by follow-on workers. ``forward_batch_generation`` and all verify/KV
    logic are inherited from ``NGRAMWorker`` verbatim.
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
        super().__init__(
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        # Swap the n-gram corpus for the suffix-tree adapter. The inherited
        # _prepare_draft_tokens / _update_ngram_corpus call self.ngram_corpus,
        # which we override below to pass the adapter full request history.
        self.ngram_corpus = SuffixCacheAdapter(
            draft_token_num=server_args.speculative_num_draft_tokens,
            max_batch_size=self.max_batch_size,
            max_tree_depth=server_args.speculative_suffix_max_tree_depth,
            max_cached_requests=server_args.speculative_suffix_max_cached_requests,
            max_spec_factor=server_args.speculative_suffix_max_spec_factor,
            min_token_prob=server_args.speculative_suffix_min_token_prob,
        )

    def _prepare_draft_tokens(self, batch: ScheduleBatch):
        bs = batch.batch_size()
        self.ngram_corpus.synchronize()
        req_ids: List[str] = []
        prompts: List[List[int]] = []
        tokens: List[List[int]] = []
        for req in batch.reqs:
            req_ids.append(req.rid)
            prompts.append(req.origin_input_ids)
            tokens.append(req.origin_input_ids + req.output_ids)
        req_drafts, mask = self.ngram_corpus.batch_get(req_ids, prompts, tokens)
        total_draft_token_num = len(req_drafts)
        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"
        return req_drafts, mask

    def _update_ngram_corpus(self, batch: ScheduleBatch):
        req_ids = [req.rid for req in batch.reqs]
        tokens = [req.origin_input_ids + req.output_ids for req in batch.reqs]
        self.ngram_corpus.batch_put(req_ids, tokens)
