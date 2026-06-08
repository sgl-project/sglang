"""SUFFIX speculative-decoding worker with pipeline-parallel (pp_size > 1) support.

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

For ``pp_size == 1`` this is byte-for-byte the base SUFFIX behavior (it
delegates to NGRAMWorker.forward_batch_generation). The PP path is only taken
when ``pp_size > 1``.

PP design (Approach A — replicate drafts, propagate accept):
  - SUFFIX's draft comes from a CPU suffix tree over origin_input_ids +
    output_ids, not from a draft-model forward. Every PP rank runs that tree
    independently and produces an IDENTICAL draft (requests are broadcast
    around the PP ring so every rank has the prompt; each rank maintains its
    own output_ids). Draft, tree mask, and KV-slot allocation are therefore
    identical on all ranks with zero cross-rank communication.
  - The LAST rank has logits: it runs the normal NGRAM verify (compute accept
    + free its own KV + append its own output_ids), and ships
    ``{accept_indices, num_accept}`` up the EXISTING last->first PP output
    ring alongside ``next_token_ids``.
  - NON-LAST ranks forward-only (return hidden states), leaving
    ``batch.spec_info`` populated. When the ring delivers the accept (during
    the scheduler's result-preprocess step), :meth:`apply_deferred_accept`
    replays the CPU side of verify: append output_ids, free rejected KV,
    compact ``out_cache_loc``, advance ``seq_lens``.

Why the per-rank KV allocators stay in lockstep (requires
``--pp-async-batch-depth 0``, ``pp_loop_size == 2``): alloc happens at forward
(identical program point on all ranks); the last rank frees during verify (end
of its launch), the non-last rank frees during result-preprocess of that
micro-batch (which sits right before the NEXT micro-batch's launch / alloc).
In alloc/free program order both ranks emit
``...alloc(A) free(A) alloc(B) free(B)...`` identically, so the free-lists
evolve identically. We also canonicalize (merge + sort) the free list before
each verify alloc as belt-and-suspenders.

Constraints on the PP path: ``page_size == 1`` for the canonicalized free
list, ``pp_async_batch_depth == 0``, greedy verify (no logprob across ranks
since logits live only on the last rank).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.ngram_worker import NGRAMWorker
from sglang.srt.speculative.suffix_cache_adapter import SuffixCacheAdapter

logger = logging.getLogger(__name__)


class SuffixWorker(NGRAMWorker):
    """``NGRAMWorker`` whose draft source is an Arctic suffix tree.

    Tensor parallelism is supported as-is (inherited from ``NGRAMWorker``: the
    suffix draft is built per-rank on CPU and the target forward is TP-sharded
    transparently). Overlap-scheduling (spec-v2) is provided by
    :class:`SuffixWorkerV2`. Pipeline parallelism is supported below.
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
        self.pp_group = target_worker.pp_group

    # ---- suffix-tree overrides (apply to both PP and non-PP) ---------------

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

    # ---- forward dispatch -------------------------------------------------

    def forward_batch_generation(self, batch: ScheduleBatch, pp_proxy_tensors=None):
        # Non-PP: unchanged base NGRAM/SUFFIX path (with the suffix-tree draft
        # override above). The PP-specific machinery below is skipped entirely.
        if self.pp_group.world_size == 1:
            return super().forward_batch_generation(batch)

        is_decode = (
            not batch.forward_mode.is_extend() and not batch.forward_mode.is_idle()
        )
        if is_decode:
            self._canonicalize_allocator(batch)

        self._prepare_for_speculative_decoding(batch)
        # The target worker takes the ScheduleBatch directly (it builds the
        # ForwardBatch internally); spec_info / forward_mode were set on
        # ``batch`` by _prepare_for_speculative_decoding above.

        if not batch.forward_mode.is_target_verify():
            # Extend / idle — plain forward, thread pp_proxy.
            return self.target_worker.forward_batch_generation(
                batch, pp_proxy_tensors=pp_proxy_tensors
            )

        if self.pp_group.is_last_rank:
            return self._verify_last_rank(batch, pp_proxy_tensors)
        return self._forward_non_last_rank(batch, pp_proxy_tensors)

    # ---- allocator canonicalization (cross-rank determinism) --------------

    @staticmethod
    def _canonicalize_allocator(batch: ScheduleBatch) -> None:
        alloc = batch.token_to_kv_pool_allocator
        # DSv4 (and other paged / hybrid-SWA models) use an allocator whose
        # ``free_pages`` is None (it keeps separate full / SWA free lists), so
        # the simple ``page_size == 1`` free-list canonicalization does not
        # apply. The cross-rank free-list determinism it provides is only
        # needed for the plain TokenToKVPoolAllocator; skip it otherwise.
        if getattr(alloc, "free_pages", None) is None:
            return
        rel = getattr(alloc, "release_pages", None)
        if rel is not None and rel.numel() > 0:
            alloc.free_pages = torch.cat((alloc.free_pages, rel))
            alloc.release_pages = torch.empty(
                (0,), dtype=alloc.free_pages.dtype, device=alloc.free_pages.device
            )
        alloc.free_pages, _ = torch.sort(alloc.free_pages)

    # ---- last rank: run verify, ship accept up the ring -------------------

    def _verify_last_rank(self, batch, pp_proxy_tensors):
        batch_result = self.target_worker.forward_batch_generation(
            batch, pp_proxy_tensors=pp_proxy_tensors, is_verify=True
        )
        logits_output = batch_result.logits_output
        verify_input: NgramVerifyInput = batch.spec_info

        # Defer the rejected-KV free + seq_lens advance to the result-preprocess
        # step (apply_deferred_free), mirroring the non-last rank's
        # apply_deferred_accept timing. Otherwise the last rank frees rejected
        # draft KV *immediately* (here) while non-last ranks free it one
        # micro-batch later — during that window non-last ranks see a smaller
        # token_to_kv_pool_allocator.available_size(), check_decode_mem()
        # diverges across ranks, ranks reach DIFFERENT KV-full retraction
        # decisions, divergent per-rank batch shapes, and finally a CUDA
        # device-side assert (paged free OOB) on the deferred-apply rank, or a
        # ring deadlock / hang. ``_fill_requests`` (output_ids append) still
        # runs immediately inside verify(); only the KV / seq_lens side is
        # deferred, so output handling is unchanged.
        logits_output, next_token_ids, num_correct = verify_input.verify(
            batch, logits_output, self.page_size, vocab_mask=None, defer_apply=True
        )
        # Snapshot out_cache_loc / draft_token now (forward has run, values are
        # final) so the deferred ``_free_cache`` reads a private copy, not a
        # cuda-graph static buffer the next micro-batch overwrites. (Same race
        # the non-last rank guards.)
        if self.page_size > 1:
            if getattr(batch, "out_cache_loc", None) is not None:
                batch.out_cache_loc = batch.out_cache_loc.clone()
            if getattr(verify_input, "draft_token", None) is not None:
                verify_input.draft_token = verify_input.draft_token.clone()
        # ``batch.spec_info`` now carries ``accept_indices`` (1D) +
        # ``num_accept_tokens`` — the scheduler's _pp_prepare_tensor_dict reads
        # these to ship up the ring, and _pp_process_batch_result calls
        # apply_deferred_free on this rank.
        batch._pp_pending_spec_free = True
        self._update_ngram_corpus(batch)
        finished = [r.rid for r in batch.reqs if r.finished() or r.is_retracted]
        if finished:
            self.ngram_corpus.erase_match_state(finished)
        batch.forward_mode = ForwardMode.DECODE

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_correct_drafts=num_correct,
            num_correct_drafts_per_req_cpu=verify_input.num_correct_drafts.cpu().tolist(),
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
            accept_lens=verify_input.num_accept_tokens,
        )

    # ---- non-last rank: forward only, defer the apply ---------------------

    def _forward_non_last_rank(self, batch, pp_proxy_tensors):
        # Forward only — return hidden states. KV free / output_ids append are
        # deferred to apply_deferred_accept when the ring delivers the accept.
        batch_result = self.target_worker.forward_batch_generation(
            batch, pp_proxy_tensors=pp_proxy_tensors, is_verify=True
        )
        # With a paged KV pool (page_size > 1, e.g. DSv4), ``out_cache_loc`` and
        # the verify ``draft_token`` can alias cuda-graph static buffers that
        # the NEXT micro-batch overwrites before this micro-batch's *deferred*
        # ``_free_cache`` runs — the paged free indexes garbage slots, raising
        # a CUDA "index out of bounds" device assert. Snapshot them into
        # private tensors now (the forward has run, so the values are final on
        # this stream). This is the actual race fix; a sync before
        # ``_free_cache`` is insufficient because the clobber is concurrent,
        # not merely earlier. (``page_size == 1`` is unaffected: the snapshot
        # is a cheap clone of two small tensors.)
        if self.page_size > 1:
            if getattr(batch, "out_cache_loc", None) is not None:
                batch.out_cache_loc = batch.out_cache_loc.clone()
            vi = batch.spec_info
            if vi is not None and getattr(vi, "draft_token", None) is not None:
                vi.draft_token = vi.draft_token.clone()
        # Keep batch.spec_info (the NgramVerifyInput) intact for the deferred
        # apply — it holds draft_token, out_cache_loc, retrieve_*.
        batch.forward_mode = ForwardMode.DECODE
        return GenerationBatchResult(
            logits_output=None,
            pp_hidden_states_proxy_tensors=batch_result.pp_hidden_states_proxy_tensors,
            can_run_cuda_graph=batch_result.can_run_cuda_graph,
        )

    # ---- deferred apply on the non-last rank ------------------------------

    def apply_deferred_accept(
        self,
        batch: ScheduleBatch,
        accept_indices: torch.Tensor,
        accept_tokens: torch.Tensor,
        num_accept_tokens: torch.Tensor,
        page_size: int,
    ) -> None:
        """Replay the CPU side of NgramVerifyInput.verify on a non-last rank,
        given the accept result computed by the last rank. Must run during the
        result-preprocess step (before the next micro-batch's alloc) so the
        free-list stays in lockstep with the last rank."""
        verify_input: NgramVerifyInput = batch.spec_info
        verify_input.accept_indices = accept_indices
        verify_input.num_accept_tokens = num_accept_tokens

        num_accept_cpu = num_accept_tokens.cpu().tolist()
        accept_tokens_cpu = accept_tokens.cpu().tolist()
        num_correct_drafts_cpu = (num_accept_tokens - 1).cpu()

        # 1. append output_ids (mirror _fill_requests CPU side)
        think_end_id = batch.model_config.think_end_id
        cursor = 0
        for i, req in enumerate(batch.reqs):
            n = num_accept_cpu[i]
            for tok in accept_tokens_cpu[cursor : cursor + n]:
                req.output_ids.append(tok)
                if req.require_reasoning and think_end_id is not None:
                    req.update_reasoning_tokens(tok, think_end_id)
                req.update_finish_state()
                if req.finished():
                    break
            cursor += n
            req.spec_verify_ct += 1
            ncd = num_accept_cpu[i] - 1
            req.spec_num_correct_drafts += ncd
            if hasattr(req, "update_spec_correct_drafts_histogram"):
                req.update_spec_correct_drafts_histogram(ncd)

        # 2. free rejected KV + compact out_cache_loc + req_to_token + committed.
        # With a paged KV pool (page_size > 1, e.g. DSv4's SWA allocator) the
        # free kernel page-aligns over out_cache_loc; if the producing async
        # work (cuda-graph forward / ring recv) for out_cache_loc /
        # accept_indices has not completed, the free indexes garbage slots and
        # raises a CUDA "index out of bounds" device assert. Make it
        # deterministic with a sync. This is cheap (~0.03ms): by the
        # deferred-apply point the GPU work is normally already done, so the
        # sync is a near-no-op except in the rare racing window.
        if page_size > 1 and torch.cuda.is_available():
            torch.cuda.synchronize()
        verify_input._free_cache(batch, page_size, num_correct_drafts_cpu)

        # 3. advance seq_lens
        batch.seq_lens.add_(num_accept_tokens)
        batch.seq_lens_cpu.add_(num_accept_tokens.cpu())

        self._update_ngram_corpus(batch)
        finished = [r.rid for r in batch.reqs if r.finished() or r.is_retracted]
        if finished:
            self.ngram_corpus.erase_match_state(finished)

    # ---- deferred free on the LAST rank (symmetry with non-last) ----------

    def apply_deferred_free(self, batch: ScheduleBatch, page_size: int) -> None:
        """Last-rank counterpart to :meth:`apply_deferred_accept`: run the
        rejected-KV free + ``seq_lens`` advance that ``verify(defer_apply=True)``
        skipped, at the same loop position the non-last rank frees.
        ``_fill_requests`` already appended the accepted output_ids inside
        verify(), so only the KV / seq_lens side is replayed here. Deferring
        the free on *every* rank makes per-rank KV occupancy identical at
        ``get_next_batch_to_run``, so KV-full retraction decisions stay in
        lockstep across the pipeline (see :meth:`_verify_last_rank`)."""
        if not getattr(batch, "_pp_pending_spec_free", False):
            return
        batch._pp_pending_spec_free = False
        verify_input: NgramVerifyInput = batch.spec_info
        if (
            verify_input is None
            or getattr(verify_input, "num_accept_tokens", None) is None
        ):
            return
        num_accept_tokens = verify_input.num_accept_tokens
        num_correct_drafts_cpu = (num_accept_tokens - 1).cpu()
        # Same race guard as apply_deferred_accept: with a paged KV pool the
        # free kernel page-aligns over ``out_cache_loc``; make the producing
        # GPU work deterministic before indexing (cheap — usually already
        # complete).
        if page_size > 1 and torch.cuda.is_available():
            torch.cuda.synchronize()
        verify_input._free_cache(batch, page_size, num_correct_drafts_cpu)
        batch.seq_lens.add_(num_accept_tokens)
        batch.seq_lens_cpu.add_(num_accept_tokens.cpu())
