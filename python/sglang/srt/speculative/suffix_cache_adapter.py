"""
Suffix decoding cache adapter for SGLang v0.5.9.

This wraps arctic_inference.suffix_decoding.SuffixDecodingCache and exposes an
NgramCache-like interface used by NGRAMWorker:
  - batch_get(...) -> flat draft tokens + flat tree mask
  - batch_put(...) -> no-op / sanity check
  - synchronize(), reset()

The important difference from NGRAM is that suffix decoding needs full request
history (prompt + generated tokens) and a stable request id.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SuffixCacheAdapter:
    def __init__(
        self,
        draft_token_num: int,
        max_batch_size: int,
        max_tree_depth: int = 24,
        max_cached_requests: int = 10000,
        max_spec_factor: float = 1.0,
        min_token_prob: float = 0.1,
    ):
        # Lazy import so normal SGLang startup does not require arctic-inference.
        try:
            from arctic_inference.suffix_decoding import SuffixDecodingCache
        except ImportError as exc:
            raise ImportError(
                "SUFFIX speculative decoding requires Arctic Inference. "
                "Install it with: pip install arctic-inference"
            ) from exc

        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            max_cached_requests=max_cached_requests,
        )
        self.draft_token_num = draft_token_num
        self.max_batch_size = max_batch_size
        self.max_tree_depth = max_tree_depth
        self.max_spec_factor = max_spec_factor
        self.min_token_prob = min_token_prob

        # Optional debug: SUFFIX_DEBUG_TREE=1 dumps the first generated tree.
        self.debug_tree_dump_remaining = int(os.environ.get("SUFFIX_DEBUG_TREE", "0"))

        # SGLang request id -> [arctic request id, last token length added]
        self.req_state: dict[str, list] = {}
        # rid -> number of consecutive batch_get/batch_peek_scores calls the
        # request has been ABSENT from the batch. Drives grace-period teardown
        # in _cleanup_inactive_requests (see there for why).
        self._absent_count: dict[str, int] = {}
        self._cleanup_grace = int(os.environ.get("SUFFIX_CLEANUP_GRACE", "32"))

        max_total_drafts = self.max_batch_size * self.draft_token_num
        self.draft_buffer = np.empty((max_total_drafts,), dtype=np.int64)
        self.mask_buffer = np.empty(
            (self.max_batch_size, self.draft_token_num, self.draft_token_num),
            dtype=bool,
        )

        # Last batch's per-request raw SuffixDecodingDraft results from
        # arctic, populated by batch_get. Cleared at the start of each call.
        self.last_drafts: list = []

    def _cleanup_inactive_requests(self, active_req_ids: set[str]) -> None:
        # Release a request's suffix tree only after it has been ABSENT from the
        # batch for `_cleanup_grace` consecutive calls -- NOT immediately when it
        # is missing from the current batch.
        #
        # Why: under pipeline parallelism (pp_loop_size > 1 micro-batching) and
        # DP-attention, the per-step batch passed here is only a SUBSET of the
        # active requests -- they rotate between micro-batches. The naive
        # "tear down everything not in this batch" then deletes the local suffix
        # tree of a request that is merely waiting its turn, forcing a full
        # start_request() prompt-tree REBUILD (tens of ms for long prompts) when
        # it returns next step. Measured ~26 rebuilds/request under PP, which
        # made the suffix-tree build (not the C++ speculate, which is ~0.1ms)
        # the dominant decode-step cost. The grace window spans micro-batch
        # rotation, so still-active requests are never torn down; genuinely
        # finished requests simply stop appearing and are released once the
        # window elapses.
        #
        # This change only ever makes teardown LESS aggressive, so it cannot
        # regress correctness -- worst case is freeing a finished request's local
        # tree a few steps later. In non-PP / single-batch mode every active
        # request is present every call, so this behaves like the old code
        # (release on finish) just delayed by `_cleanup_grace` steps.
        grace = self._cleanup_grace
        absent = self._absent_count
        active_backend_reqs = getattr(self.suffix_cache, "active_requests", set())
        to_release = []
        for rid in list(self.req_state):
            if rid in active_req_ids:
                absent.pop(rid, None)
            elif absent.get(rid, 0) + 1 >= grace:
                to_release.append(rid)
            else:
                absent[rid] = absent.get(rid, 0) + 1
        for rid in to_release:
            cache_req_id, _ = self.req_state.pop(rid)
            absent.pop(rid, None)
            if cache_req_id in active_backend_reqs:
                self.suffix_cache.stop_request(cache_req_id)

    def _get_or_create_cache_req_id(
        self, sglang_req_id: str, prompt: List[int]
    ) -> Tuple[str, int]:
        if sglang_req_id not in self.req_state:
            cache_req_id = sglang_req_id
            self.suffix_cache.start_request(cache_req_id, prompt)
            # The prompt is already loaded into the backend cache.
            self.req_state[sglang_req_id] = [cache_req_id, len(prompt)]
        cache_req_id, last_length = self.req_state[sglang_req_id]
        return cache_req_id, last_length

    def batch_get(
        self,
        batch_req_ids: List[str],
        batch_prompts: List[List[int]],
        batch_tokens: List[List[int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = len(batch_req_ids)
        if batch_size == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=bool)
        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds max_batch_size={self.max_batch_size}"
            )

        total_draft_tokens = batch_size * self.draft_token_num
        draft_view = self.draft_buffer[:total_draft_tokens]
        mask_view = self.mask_buffer[:batch_size]
        mask_view.fill(False)

        self._cleanup_inactive_requests(set(batch_req_ids))
        self.last_drafts = []

        for row, (sglang_req_id, prompt, tokens) in enumerate(
            zip(batch_req_ids, batch_prompts, batch_tokens)
        ):
            cache_req_id, last_length = self._get_or_create_cache_req_id(
                sglang_req_id, prompt
            )

            # Add only the verified delta since the previous step.
            current_length = len(tokens)
            if current_length > last_length:
                new_tokens = tokens[last_length:current_length]
                if cache_req_id in getattr(self.suffix_cache, "active_requests", set()):
                    self.suffix_cache.add_active_response(cache_req_id, new_tokens)
                    self.req_state[sglang_req_id][1] = current_length
                else:
                    logger.warning(
                        "Suffix cache request %s is not active when updating",
                        cache_req_id,
                    )

            # Arctic suffix decoding matches from the current suffix pattern.
            pattern_start = max(0, len(tokens) - self.max_tree_depth)
            pattern = tokens[pattern_start:]
            draft = self.suffix_cache.speculate(
                cache_req_id,
                pattern,
                max_spec_tokens=self.draft_token_num,
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )
            self.last_drafts.append(draft)

            draft_ids = list(draft.token_ids)
            draft_parents = list(draft.parents)
            draft_ids, draft_parents = self._reorder_tree_bfs(draft_ids, draft_parents)
            context_token = tokens[-1] if tokens else 0
            draft_ids, draft_parents = self._inject_root_node(
                draft_ids, draft_parents, context_token
            )

            original_len = len(draft_ids)
            if original_len < self.draft_token_num:
                pad_len = self.draft_token_num - original_len
                draft_ids.extend([0] * pad_len)
                draft_parents.extend([0] * pad_len)
            elif original_len > self.draft_token_num:
                draft_ids = draft_ids[: self.draft_token_num]
                draft_parents = draft_parents[: self.draft_token_num]
                original_len = self.draft_token_num

            start = row * self.draft_token_num
            end = start + self.draft_token_num
            draft_view[start:end] = draft_ids

            # Build ancestry mask: token i can attend to its ancestors and itself.
            mask = mask_view[row]
            for i in range(original_len):
                mask[i, i] = True
                parent = draft_parents[i]
                while 0 <= parent < self.draft_token_num:
                    mask[i, parent] = True
                    parent = draft_parents[parent]

            if self.debug_tree_dump_remaining > 0 and original_len > 0:
                logger.warning(
                    "[SUFFIX DEBUG] req=%s len=%d ids=%s",
                    sglang_req_id,
                    original_len,
                    draft_ids,
                )
                logger.warning("[SUFFIX DEBUG] mask=\n%s", mask.astype(int))
                self.debug_tree_dump_remaining -= 1

        tree_mask = mask_view.reshape(-1)[: total_draft_tokens * self.draft_token_num]
        return draft_view, tree_mask

    def batch_put(
        self, batch_req_ids: List[str], batch_tokens: List[List[int]]
    ) -> None:
        # Cache update is performed in batch_get before speculation, because the
        # suffix backend needs a per-request incremental state.
        for rid in batch_req_ids:
            if rid not in self.req_state:
                logger.debug("batch_put called before batch_get for request %s", rid)

    def erase_match_state(self, req_ids: List[str]) -> None:
        # Release finished requests from the suffix cache (NGRAMWorker calls this
        # with finished req ids after each step). Mirrors NgramCorpus.erase_match_state.
        active = getattr(self.suffix_cache, "active_requests", set())
        for rid in req_ids:
            state = self.req_state.pop(rid, None)
            self._absent_count.pop(rid, None)
            if state is not None and state[0] in active:
                self.suffix_cache.stop_request(state[0])

    # External-corpus management is an NGRAM-only feature; SUFFIX has no SAM.
    # Stubbed so the duck-typed NGRAMWorker / HTTP handlers fail clearly if used.
    def commit_external_corpus_load(self, *args, **kwargs):
        raise NotImplementedError("SUFFIX speculative decoding has no external corpus.")

    def list_external_corpora(self, *args, **kwargs):
        return []

    def load_external_corpus_named(self, *args, **kwargs):
        raise NotImplementedError("SUFFIX speculative decoding has no external corpus.")

    def remove_external_corpus(self, *args, **kwargs):
        raise NotImplementedError("SUFFIX speculative decoding has no external corpus.")

    def synchronize(self) -> None:
        pass

    def reset(self) -> None:
        for cache_req_id in list(getattr(self.suffix_cache, "active_requests", set())):
            self.suffix_cache.stop_request(cache_req_id)
        self.req_state.clear()
        self._absent_count.clear()

    def _reorder_tree_bfs(
        self, token_ids: List[int], parents: List[Optional[int]]
    ) -> Tuple[List[int], List[int]]:
        """Ensure parents precede descendants; SGLang reconstruct assumes this."""
        n = len(token_ids)
        if n <= 1:
            return token_ids, [(-1 if p is None else p) for p in parents]

        children: List[List[int]] = [[] for _ in range(n)]
        roots: List[int] = []
        for idx, parent in enumerate(parents):
            if parent is None or parent < 0 or parent >= n:
                roots.append(idx)
            else:
                children[parent].append(idx)
        if not roots:
            roots = [0]

        order: List[int] = []
        visited = [False] * n
        for root in roots:
            q = deque([root])
            while q:
                node = q.popleft()
                if visited[node]:
                    continue
                visited[node] = True
                order.append(node)
                q.extend(children[node])
        for idx in range(n):
            if not visited[idx]:
                order.append(idx)

        if order == list(range(n)):
            return token_ids, [(-1 if p is None else p) for p in parents]

        remap = {old: new for new, old in enumerate(order)}
        reordered_ids = [token_ids[old] for old in order]
        reordered_parents: List[int] = []
        for old in order:
            parent = parents[old]
            if parent is None or parent < 0:
                reordered_parents.append(-1)
            else:
                reordered_parents.append(remap.get(parent, -1))
        return reordered_ids, reordered_parents

    def _inject_root_node(
        self, token_ids: List[int], parents: List[int], context_token: int
    ) -> Tuple[List[int], List[int]]:
        """Insert latest verified token as root node to match NGRAM layout."""
        rooted_ids = [context_token]
        rooted_parents = [-1]
        for parent in parents:
            rooted_parents.append(0 if parent < 0 else parent + 1)
        rooted_ids.extend(token_ids)
        return rooted_ids, rooted_parents
