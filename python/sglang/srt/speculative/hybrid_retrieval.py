"""NGRAM retrieval engine for the hybrid MTP + retrieval chain (hybrid retrieval).

Wraps the existing :class:`NgramCorpus` (an online, per-process trie over each
request's own running tokens) and exposes a CHAIN-ONLY query for the EAGLE
hybrid path: given each request's verified context, return a fixed-width
``[bs, L]`` continuation chain ``[root, c1, ..., c_{L-1}]`` (root = the
context's last token; unmatched positions zero-padded) that
``hybrid_chain.build_hybrid_verify_input`` splices onto the kept MTP prefix
(agreement-gated) and verifies.

Design notes:
  * Chain-only: the corpus is built with bfs breadth fixed to 1. (breadth=1
    caps children-per-node but the corpus may still seed BFS from several
    suffix anchors, so the flattened result can occasionally branch; this only
    costs accept length -- the agreement gate and verify reject any off-path
    token, so greedy output and TP determinism are unaffected.)
  * The corpus is updated (``insert``) only with VERIFIED tokens, AFTER each
    verify step (mirroring ``NGRAMWorker``: query first, insert after -- so a
    query matches earlier occurrences of the suffix, not the just-written leaf).
  * Greedy output is independent of trie content: a retrieved token is accepted
    only if it equals the target's argmax, so retrieval changes accept LENGTH,
    never correctness.
  * Per-TP-rank: each worker process holds its own corpus. Every rank inserts
    the same verified tokens (the batch is replicated) and issues the same
    query, so ``batch_get`` is deterministic and returns an identical chain on
    every rank. The spliced chain is additionally broadcast from rank 0 (see
    ``hybrid_chain``) because the truncation threshold reads a continuous
    logprob.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import msgspec
import numpy as np
import torch

from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus


class PendingAccepted(msgspec.Struct):
    """One batch's accepted tokens, copied D2H on a side stream (overlap mode).

    ``cpu_tokens`` is ``[bs, stride]`` pinned host memory, ``cpu_lens`` is
    ``[bs]`` pinned host accept lengths (incl. bonus), ``done`` is the CUDA
    event marking copy completion (None on the synchronous/CPU fallback path).
    """

    rids: List[str]
    cpu_tokens: torch.Tensor
    cpu_lens: torch.Tensor
    done: Optional[torch.cuda.Event] = None


def verified_context_tail(req, n: int) -> List[int]:
    """Last ``n`` tokens of ``req.origin_input_ids ++ req.output_ids``.

    Slices before materializing: ``output_ids`` is an ``array``, and both
    sequences can be tens of thousands of tokens long, so copying either in full
    once per request per decode step is a real cost for an ``n`` of ~18.
    """
    output_tail = list(req.output_ids[-n:])
    if len(output_tail) >= n:
        return output_tail
    need = n - len(output_tail)
    return list(req.origin_input_ids[-need:]) + output_tail


class HybridRetriever:
    # Cap on the closed-rid set (see erase); relays are at most a couple of
    # batches deep, so this is far more headroom than the drain ever needs.
    _MAX_CLOSED_RIDS = 4096

    def __init__(self, server_args, device: str):
        self.device = device
        # = L, the verify width; also the retrieval chain width.
        self.draft_token_num: int = server_args.speculative_num_draft_tokens
        self.max_trie_depth: int = server_args.speculative_ngram_max_trie_depth
        # Chain-only retrieval: breadth fixed to 1. draft_token_num = L so the
        # returned chain is [root + up to L-1 continuation tokens], zero-padded.
        self.corpus = NgramCorpus(
            min_bfs_breadth=1,
            max_bfs_breadth=1,
            match_type=server_args.speculative_ngram_match_type,
            capacity=server_args.speculative_ngram_capacity,
            max_trie_depth=self.max_trie_depth,
            draft_token_num=self.draft_token_num,
        )
        # Run the (host-side, CPU) trie walk on a worker thread so it overlaps
        # the GPU draft_forward. launch_query() submits before draft();
        # collect() joins in the verify-input build. Safe because the walk is
        # joined before verify, so no insert/publish runs concurrently with it.
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._fut = None

        # Overlap mode: the retriever stops reading req.output_ids (which lags
        # one batch under the overlap scheduler) and maintains its OWN per-req
        # verified context, fed by an async accepted-token relay from verify.
        self.overlap: bool = server_args.speculative_hybrid_overlap
        # Shadow verified context: last max_trie_depth verified tokens per req
        # (query/insert source) + running total length (for batch_get position).
        self._ctx_tail: Dict[str, List[int]] = {}
        self._ctx_total_len: Dict[str, int] = {}
        # The exact per-req context we queried this step; inserted after verify
        # (query-first / insert-after), so a query matches earlier occurrences.
        self._last_query_snapshot: Optional[List[List[int]]] = None
        # Async accepted-token relay: verify enqueues a D2H copy on a side
        # stream and appends a PendingAccepted; the next launch_query drains it
        # (sync the event, fold tokens into the shadow context). Keeps the D2H
        # off the forward hot path. FIFO of at most a couple of batches.
        self._pending: deque[PendingAccepted] = deque()
        self._closed_rids: set = set()  # departed; the relay drain skips them
        # rids of the last decode batch, for the departed-request eviction.
        self._prev_rids: set = set()
        self._accept_copy_stream = (
            torch.cuda.Stream(device=device)
            if (self.overlap and torch.cuda.is_available())
            else None
        )

    def reset(self) -> None:
        """Clear the online corpus and all per-request retrieval state.

        The scheduler only calls this through ``/flush_cache`` while the
        engine is fully idle.  Idle does not, however, imply that the CPU trie
        walk, overlap D2H relay, or asynchronous corpus inserts have finished,
        so drain them in that order before resetting the trie.  In particular,
        ``NgramCorpus.reset()`` does not drain inserts that are already queued.
        """
        fut, self._fut = self._fut, None
        if fut is not None:
            fut.result()

        # Keep every pending pinned destination alive until its D2H copy has
        # completed.  Synchronizing the dedicated stream covers all queued
        # events; the fallback path synchronizes them individually.
        if self._accept_copy_stream is not None:
            self._accept_copy_stream.synchronize()
        else:
            for pending in self._pending:
                if pending.done is not None:
                    pending.done.synchronize()
        self._pending.clear()

        # This ordering is critical: otherwise a pre-reset queued insert can
        # land after reset and leak corpus data into the next experiment arm.
        self.corpus.synchronize()
        self.corpus.reset()

        self._ctx_tail.clear()
        self._ctx_total_len.clear()
        self._last_query_snapshot = None
        self._prev_rids.clear()
        self._closed_rids.clear()

    # ---------------------------- query / collect ----------------------------

    def launch_query(self, reqs) -> None:
        """Submit the trie walk for the current verified context to the worker
        thread (call at the start of the decode step, before draft_forward)."""
        if self.overlap:
            self._launch_query_overlap(reqs)
            return
        req_ids: List[str] = []
        batch_tokens: List[List[int]] = []
        total_lens: List[int] = []
        for req in reqs:
            req_ids.append(req.rid)
            batch_tokens.append(verified_context_tail(req, self.max_trie_depth))
            total_lens.append(len(req.origin_input_ids) + len(req.output_ids))
        self._fut = self._pool.submit(
            self._trie_walk, req_ids, batch_tokens, total_lens, len(reqs)
        )

    def collect(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Join the worker thread and return ``(chains, lens)`` on device:
        ``[bs, L]`` retrieved chains and the ``[bs]`` TRUE continuation length of
        each (excluding the root). Returns ``(None, None)`` when launch_query was
        never called.

        The length is reported explicitly rather than recovered from the chain,
        because there is no value that can mean "no token": token id 0 is a real
        vocabulary entry (``<|begin_of_sentence|>`` in DSV4, and it IS in the
        trie since the full prompt is indexed). Counting non-zeros would confuse
        a genuine 0 with tail padding.
        """
        fut, self._fut = self._fut, None
        if fut is None:
            return None, None
        chains, lens = fut.result()
        return (
            torch.from_numpy(chains).to(self.device, non_blocking=True),
            torch.from_numpy(lens).to(self.device, non_blocking=True),
        )

    def query(self, reqs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Synchronous launch+collect (kept for non-threaded callers/tests)."""
        self.launch_query(reqs)
        return self.collect()

    def _trie_walk(
        self, req_ids, batch_tokens, total_lens, bs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CPU trie match + linear-chain extraction.

        Returns ``(chains[bs, L], lens[bs])``. Column 0 of a chain is the
        context's last token (root); columns ``1..lens[r]`` are the retrieved
        continuation and the rest is zero padding. ``lens`` is produced here, not
        inferred downstream, because the padding value 0 is also a legal token.
        """
        self.corpus.synchronize()  # drain pending async inserts before matching
        # valid_lens[r] = how many entries of request r's block are REAL nodes
        # (anchor included). It comes from the C++ producer because it is not
        # recoverable afterwards: fillResult zero-pads the tail and gives each
        # pad prev=0, making a pad's mask row identical to a real depth-1 child
        # whose token is 0 -- and token 0 is legal (DSV4's
        # <|begin_of_sentence|>, present in the trie because the prompt is
        # indexed). Counting non-zeros or walking leaf paths over the padded
        # arrays reports a NO-MATCH result as a length-1 continuation of token 0,
        # which fakes a graft and throws away the whole MTP fallback.
        req_drafts, mask, valid_lens = self.corpus.batch_get_with_lens(
            req_ids, batch_tokens, total_lens
        )
        num_draft_tokens = self.draft_token_num
        drafts = np.asarray(req_drafts).reshape(bs, num_draft_tokens)
        masks = np.asarray(mask).reshape(bs, num_draft_tokens, num_draft_tokens)
        valid = np.asarray(valid_lens).reshape(bs)
        # NGRAM with breadth=1 can still return a BRANCHING tree: BFS is seeded
        # from every expandable suffix anchor, so the root may have several
        # children and the result is flattened in BFS order (siblings first) --
        # i.e. column j is NOT chain depth j. The hybrid splice + agreement gate
        # require a single LINEAR chain, so extract the longest root-to-leaf
        # path (deepest suffix match = best continuation) and lay it out as
        # [root, c1, c2, ...] padded with 0. Without this the retrieved chain
        # never aligns with the MTP chain and is never grafted.
        out = np.zeros((bs, num_draft_tokens), dtype=np.int64)
        lens = np.zeros((bs,), dtype=np.int32)
        for r in range(bs):
            out[r, 0] = int(drafts[r, 0])  # root = verified-context last token
            # Cut the padding off FIRST, using the producer's count, then find
            # the longest path over what is left. Padding is removed
            # structurally, so nothing here infers a length from token values.
            # k == 1 (anchor only) means no match: the loop below leaves
            # lens[r] == 0, which is what makes the agreement gate reject.
            k = int(valid[r])
            if k <= 1:
                continue
            sub_tokens = drafts[r, :k].tolist()
            sub_mask = [row[:k] for row in masks[r, :k].tolist()]
            # Still the longest ROOT-TO-LEAF path, not k - 1: breadth is pinned
            # to 1 but BFS is seeded from every expandable suffix anchor, so the
            # real tree can branch and k counts all of its nodes.
            paths = self.corpus.leaf_paths_from_mask(sub_tokens, sub_mask)
            if paths:
                longest = max(paths, key=len)  # [root, c1, ..., leaf]
                m = min(len(longest), num_draft_tokens)
                out[r, :m] = longest[:m]
                # m counts the root, so the continuation is m - 1 tokens long.
                lens[r] = m - 1
        return out, lens  # collect() moves both to device

    # -------------------------- corpus maintenance ---------------------------

    def insert_prompt(self, reqs) -> None:
        """Insert each request's FULL prompt into the trie at prefill so
        retrieval can copy from the prompt (RAG / long-context / code-with-
        context). The decode-time insert() only adds sliding windows of
        generated text, so without this the prompt body is never matchable."""
        batch_tokens = [
            list(req.origin_input_ids) for req in reqs if req.origin_input_ids
        ]
        if batch_tokens:
            self.corpus.batch_put(batch_tokens)

    def insert(self, reqs) -> None:
        """Insert each request's verified-context tail into the trie (after verify)."""
        if self.overlap:
            # Overlap: insert exactly the (fresh, shadow-context) tail we queried.
            snap, self._last_query_snapshot = self._last_query_snapshot, None
            if snap:
                self.corpus.batch_put(snap)
            return
        self.corpus.batch_put(
            [verified_context_tail(req, self.max_trie_depth) for req in reqs]
        )

    def erase(self, reqs) -> None:
        """Drop per-request state for requests that LEFT the decode batch.

        ``req.finished()`` is unusable here: under overlap it flips at result
        processing, one iteration after the request left the batch. Mirrors the
        upstream NGRAM worker's set-difference policy; the last batch's entries
        persist while the engine is idle (bounded and small).
        """
        current_rids = {req.rid for req in reqs}
        departed_rids = self._prev_rids - current_rids
        self._prev_rids = current_rids
        if not departed_rids:
            return
        for rid in departed_rids:
            self._ctx_tail.pop(rid, None)
            self._ctx_total_len.pop(rid, None)
        # Close the rids so a still-in-flight accepted-token relay for them is
        # dropped instead of resurrecting a shadow context nobody will read.
        # Bound the set: it only has to cover relays still queued, which is at
        # most a couple of batches deep.
        if len(self._closed_rids) > self._MAX_CLOSED_RIDS:
            self._closed_rids.clear()
        self._closed_rids |= departed_rids
        self.corpus.erase_match_state(list(departed_rids))

    # ------------- overlap: shadow context + async accepted relay -------------

    def publish_accepted(
        self,
        reqs,
        next_token_ids: torch.Tensor,
        accept_lens: torch.Tensor,
        stride: int,
    ) -> None:
        """After verify, relay this step's accepted tokens into the shadow
        context WITHOUT stalling the forward stream.

        ``next_token_ids`` is the flat ``[bs*stride]`` committed-token buffer
        (the same one the scheduler slices with
        ``stride = speculative_num_draft_tokens``); row i's first
        ``accept_lens[i]`` entries are req i's accepted tokens (incl. the bonus
        root). We copy them D2H on a side stream and drain at the next
        launch_query.
        """
        if not reqs:
            return
        rids = [req.rid for req in reqs]
        bs = len(rids)
        # Clone on the CURRENT (forward) stream first, so the async D2H reads a
        # PRIVATE tensor -- never a cuda-graph-reused output buffer that the
        # next verify replay could overwrite before the copy finishes. The
        # clones are tiny int tensors; record_stream keeps them alive until the
        # copy stream is done. Slice to exactly the region the scheduler commits
        # from; the buffer may be over-allocated, so never rely on
        # numel == bs*stride.
        tokens_2d = next_token_ids[: bs * stride].reshape(bs, stride).clone()
        lens = accept_lens[:bs].reshape(bs).clone()
        if self._accept_copy_stream is None:
            # Non-cuda / synchronous fallback: resolve immediately.
            self._pending.append(
                PendingAccepted(
                    rids=rids, cpu_tokens=tokens_2d.to("cpu"), cpu_lens=lens.to("cpu")
                )
            )
            return
        current = torch.cuda.current_stream(next_token_ids.device)
        copy_stream = self._accept_copy_stream
        # Copy waits for verify+clone; the forward does NOT wait for the copy.
        copy_stream.wait_stream(current)
        cpu_tokens = torch.empty((bs, stride), dtype=tokens_2d.dtype, pin_memory=True)
        cpu_lens = torch.empty((bs,), dtype=lens.dtype, pin_memory=True)
        done = torch.cuda.Event()
        with torch.cuda.stream(copy_stream):
            cpu_tokens.copy_(tokens_2d, non_blocking=True)
            cpu_lens.copy_(lens, non_blocking=True)
            done.record(copy_stream)
        tokens_2d.record_stream(copy_stream)
        lens.record_stream(copy_stream)
        self._pending.append(
            PendingAccepted(
                rids=rids, cpu_tokens=cpu_tokens, cpu_lens=cpu_lens, done=done
            )
        )

    def _drain_pending_accepts(self) -> None:
        """Fold every enqueued batch of accepted tokens into the shadow context.

        Called at the start of launch_query so that, by the time we build this
        step's query, all prior verifies are reflected (the D2H has typically
        finished at least one iteration ago; ``event.synchronize()`` is the only
        potential wait).
        """
        while self._pending:
            pending = self._pending.popleft()
            if pending.done is not None:
                pending.done.synchronize()
            lens = pending.cpu_lens.tolist()
            rows = pending.cpu_tokens.tolist()
            for rid, num_accept_tokens, row in zip(pending.rids, lens, rows):
                num_accept_tokens = int(num_accept_tokens)
                if num_accept_tokens <= 0 or rid in self._closed_rids:
                    continue
                self._append_ctx(rid, row[:num_accept_tokens])

    def _launch_query_overlap(self, reqs) -> None:
        self._drain_pending_accepts()
        req_ids: List[str] = []
        batch_tokens: List[List[int]] = []
        total_lens: List[int] = []
        for req in reqs:
            rid = req.rid
            self._ensure_req(rid, req)
            req_ids.append(rid)
            # Copy the tail so the worker thread and the post-verify insert both
            # read an immutable snapshot (never a list a later _append_ctx
            # mutates).
            batch_tokens.append(list(self._ctx_tail[rid]))
            total_lens.append(self._ctx_total_len[rid])
        self._last_query_snapshot = batch_tokens
        self._fut = self._pool.submit(
            self._trie_walk, req_ids, batch_tokens, total_lens, len(reqs)
        )

    def _ensure_req(self, rid: str, req) -> None:
        """Seed the shadow context for a first-seen req from its prompt only.

        The prefill's first generated token is NOT added here: it arrives as
        column 0 (the bonus root) of the FIRST decode step's accepted tokens, so
        publish_accepted folds it in exactly once. (Seeding from output_ids
        would risk double-counting it, since under overlap it may or may not
        have landed there yet.)
        """
        if rid in self._ctx_total_len:
            return
        # A rid can come back after being retracted; un-close it so its relays
        # are folded in again.
        self._closed_rids.discard(rid)
        self._ctx_tail[rid] = list(req.origin_input_ids[-self.max_trie_depth :])
        self._ctx_total_len[rid] = len(req.origin_input_ids)

    def _append_ctx(self, rid: str, tokens: List[int]) -> None:
        if not tokens:
            return
        tail = self._ctx_tail.get(rid)
        if tail is None:
            tail = []
            self._ctx_tail[rid] = tail
        tail.extend(int(t) for t in tokens)
        overflow = len(tail) - self.max_trie_depth
        if overflow > 0:
            del tail[:overflow]
        self._ctx_total_len[rid] = self._ctx_total_len.get(rid, 0) + len(tokens)
