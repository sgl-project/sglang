"""SUFFIX speculative-decoding worker.

SUFFIX is *model-free* speculative decoding (Snowflake ArcticInference's
SuffixDecoding): a CPU suffix tree built over each request's prompt +
generated tokens proposes draft continuations, which are verified in a
single target forward pass. It is especially effective on agentic /
repetitive workloads and adds no draft-model GPU cost.

Implementation reuses :class:`NGRAMWorker` unchanged for the entire spec-v2
machinery (draft slot assignment, target verify, ``NgramVerifyInput.sample``,
overlap ``on_publish`` fence, corpus lifecycle); the only difference is the
*draft source*. ``_create_draft_source`` swaps the C++ n-gram corpus for a
:class:`SuffixCacheAdapter` wrapping ``arctic_inference``'s
``SuffixDecodingCache``, and we override the two hooks that talk to it:

  - ``_prepare_draft_tokens`` — the suffix tree matches against the request's
    *full* history (prompt + output), not just a recent n-gram window, and
    needs per-request lifecycle (``start_request``) keyed by rid.
  - ``_update_ngram_corpus`` — feed the realized tokens back into the tree.

Unlike the base class, the previous round's accepts are NOT spliced
unconditionally under overlap: result processing usually lags one iteration,
but whenever another batch (e.g. a prefill chunk) runs between two decode
steps of this batch, the lagging result HAS been processed into
``req.output_ids`` by the time we are called again — an unconditional splice
then duplicates the whole accept run. The base trie's bounded-window insert
shrugs that off; the suffix adapter feeds the tree by absolute-length delta,
so a duplicated splice both corrupts the tree content and desynchronizes the
delta cursor, degrading match length from then on. We therefore reconstruct
the exact frontier from ``seq_lens`` (invariant: the generated text is always
``seq_lens + 1`` tokens long — the trailing bonus token's KV is committed by
the *next* verify) and splice only the genuinely missing tail.

SUFFIX drafts are linear chains (arctic ``use_tree_spec=False``), so accepted
tokens always form a contiguous prefix of the canonical verify slots —
``drafts_are_linear_chains = True`` skips the post-verify KV move (an
identity for chains, and unsupported by DeepSeek-V4's NSA paginated pool).

Per-request draft length is additionally capped at
``context_len - seq_len - 1`` so a verify step can never overshoot the
context window (which would produce an out-of-range zombie verify under
overlap scheduling).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_worker import NGRAMWorker
from sglang.srt.speculative.suffix_cache_adapter import (
    SegmentedTokens,
    SuffixCacheAdapter,
)

logger = logging.getLogger(__name__)


class SuffixWorker(NGRAMWorker):
    """``NGRAMWorker`` whose draft source is an Arctic suffix tree."""

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
        # Arctic chains: accepted KV needs no post-verify move (see base flag).
        self.drafts_are_linear_chains = True
        # The adapter manages request lifecycle itself with a grace window
        # (absence from the batch is NOT departure under PP micro-batch
        # rotation; immediate erase forces a full local-tree rebuild from the
        # prompt on every return). See SuffixCacheAdapter._cleanup_inactive_requests.
        self.corpus_gc_on_departure = False
        # Cap per-request draft length so verify can't overshoot the context
        # window (the -1 leaves room for the bonus token).
        self._max_context_len = int(target_worker.model_runner.model_config.context_len)
        # PD disaggregation (decode instance): prebuilt requests whose suffix
        # tree should be rebuilt while their KV transfer is still in flight
        # (rid -> Req). Drained by prewarm_step(), enqueued by
        # prewarm_disagg_requests() from the decode scheduler's transfer-queue
        # polling loop. Empty (and unused) on colocated / prefill instances.
        self._prewarm_pending: Dict[str, object] = {}

    # ------------------------------------------------------------------
    # PD disaggregation: decode-side suffix-tree prewarm
    #
    # Correctness does NOT depend on prewarm: a prebuilt request that reaches
    # its first decode step un-prewarmed is cold-started lazily inside the
    # adapter's batch_get (_get_or_create_cache_req_id), and the transferred
    # output tail (the prefill-produced bonus token) is appended there by the
    # normal absolute-length delta. Prewarm only moves the tens-of-ms prompt
    # tree build off the first decode step, overlapping it with KV transfer.
    # ------------------------------------------------------------------
    def prewarm_disagg_requests(self, reqs) -> None:
        """Enqueue prebuilt requests for suffix-tree prewarm. Idempotent per
        rid; skips requests whose tree the adapter already tracks."""
        for req in reqs:
            rid = req.rid
            if rid in self.ngram_corpus.req_state or rid in self._prewarm_pending:
                continue
            self._prewarm_pending[rid] = req

    def prewarm_step(self, token_budget: int = 32768) -> None:
        """Build at most ``token_budget`` prompt tokens worth of pending
        prewarm trees. Runs on the scheduler thread (arctic is not
        thread-safe); the budget bounds the per-iteration stall. The adapter's
        own grace window keeps a prewarmed tree alive until its first decode."""
        spent = 0
        while self._prewarm_pending and spent < token_budget:
            rid, req = self._prewarm_pending.popitem()
            if rid in self.ngram_corpus.req_state:
                continue
            prompt = list(req.origin_input_ids)
            # Seed the backend tree with the prompt; the transferred output
            # tail is appended later as the batch_get delta on first decode.
            self.ngram_corpus._get_or_create_cache_req_id(rid, prompt)
            spent += len(prompt)

    def _create_draft_source(self, server_args: ServerArgs):
        # The inherited forward flow reaches self.ngram_corpus only through
        # the hooks we override below, plus synchronize() /
        # erase_match_state(), which the adapter provides with matching
        # signatures.
        return SuffixCacheAdapter(
            draft_token_num=server_args.speculative_num_draft_tokens,
            max_batch_size=self.max_batch_size,
            max_tree_depth=server_args.speculative_suffix_max_tree_depth,
            max_cached_requests=server_args.speculative_suffix_max_cached_requests,
            max_spec_factor=server_args.speculative_suffix_max_spec_factor,
            min_token_prob=server_args.speculative_suffix_min_token_prob,
        )

    def _missing_tail(self, batch: ScheduleBatch, i: int, target_len: int) -> List[int]:
        """The tail of request i's history that is not yet in
        ``req.output_ids``. Under overlap, ``output_ids`` lags by the last
        round's accepts ONLY when that result has not been processed yet
        (another batch running in between processes it); splice exactly the
        missing tail of the staged accept run — never unconditionally (see
        module docstring)."""
        req = batch.reqs[i]
        missing = target_len - len(req.origin_input_ids) - len(req.output_ids)
        if missing <= 0:
            # Result already processed; an unconditional splice would
            # duplicate the accept run here.
            return []
        a = self.prev_accept_lens[i]
        stride = self.draft_token_num
        prev = self.prev_token_ids[i * stride : i * stride + a]
        if missing > a:
            # Should not happen (result processing lags at most one round);
            # feed what we have rather than corrupting the tree with a hole.
            logger.warning(
                "SUFFIX frontier gap exceeds staged accepts "
                "(missing=%d staged=%d rid=%s)",
                missing,
                a,
                req.rid,
            )
            return prev
        return prev[a - missing :]

    def _prepare_draft_tokens(self, batch: ScheduleBatch):
        bs = len(batch.reqs)

        # Stage the previous round's accepts (mirrors the base class); they
        # are consumed by _request_tokens only for requests whose output_ids
        # actually lag.
        prev_token_ids, prev_accept_lens = (
            batch.spec_info.accept_tokens,
            batch.spec_info.accept_lens,
        )
        if not prev_token_ids.is_cpu:
            prev_token_ids = prev_token_ids.cpu()
            prev_accept_lens = prev_accept_lens.cpu()
        self.prev_token_ids = prev_token_ids.tolist()
        self.prev_accept_lens = prev_accept_lens.tolist()

        self.ngram_corpus.synchronize()
        req_ids: List[str] = []
        prompts: List[List[int]] = []
        tokens: List[List[int]] = []
        max_lens: List[int] = []
        seq_lens_cpu = batch.seq_lens_cpu.tolist()
        for i, req in enumerate(batch.reqs):
            req_ids.append(req.rid)
            # Pass references, not copies — the adapter materializes only
            # the short slices it needs (per-step O(seq_len) copies otherwise
            # dominate the decode-step CPU budget at agent context lengths).
            prompts.append(req.origin_input_ids)
            # Full history — the suffix tree matches arbitrarily long
            # patterns, unlike the trie's bounded window. seq_lens + 1 is the
            # exact text frontier (see module docstring).
            tokens.append(
                SegmentedTokens(
                    req.origin_input_ids,
                    req.output_ids,
                    self._missing_tail(batch, i, seq_lens_cpu[i] + 1),
                )
            )
            max_lens.append(max(0, self._max_context_len - seq_lens_cpu[i] - 1))
        req_drafts, mask = self.ngram_corpus.batch_get(
            req_ids, prompts, tokens, max_lens=max_lens
        )
        total_draft_token_num = len(req_drafts)
        assert (
            total_draft_token_num == bs * self.draft_token_num
        ), f"{total_draft_token_num=}, {bs=}, {self.draft_token_num=}"
        return req_drafts, mask

    def _update_ngram_corpus(self, batch: ScheduleBatch):
        # The adapter ingests tokens incrementally inside batch_get;
        # batch_put is a per-request sanity hook only (no token payload).
        self.ngram_corpus.batch_put([req.rid for req in batch.reqs])
