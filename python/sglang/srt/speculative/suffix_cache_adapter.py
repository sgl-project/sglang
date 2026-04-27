"""
Cache adapter that wraps the suffix decoding backend cache to provide
the same interface as NgramCache.

This allows NGRAMWorker to use suffix decoding without modification.

Performance optimizations:
- Adaptive fallback: when the suffix tree produces no useful speculation
  (only root tokens), the caller can skip expensive tree verification and
  fall back to normal single-token decoding.  This avoids paying the
  verification overhead (draft_token_num × batch_size forward tokens) for
  zero benefit.
- Vectorised mask construction: BFS-ordered parent arrays allow O(n) numpy
  propagation instead of O(n × depth) Python while-loops.
- Reduced cleanup frequency: inactive-request pruning runs every N calls
  instead of every batch_get().
"""

import logging
import os
from collections import deque
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# How often (in batch_get calls) to scan for inactive requests.
_CLEANUP_INTERVAL = 8


class SuffixCacheAdapter:
    """
    Adapter that wraps SuffixDecodingCache to match NgramCache interface.

    NGRAMWorker expects:
    - batch_get(batch_tokens: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]
      Returns (draft_tokens, tree_mask) as flat numpy arrays
    - batch_put(batch_tokens: List[List[int]]) -> None
      Updates cache with verified tokens
    - synchronize() -> None
      No-op for suffix cache
    - reset() -> None
      Clears all cached data

    After each batch_get(), the caller can inspect:
    - last_total_speculated: total number of draft tokens (excluding root) that
      were produced by the suffix tree across the batch.  When this is 0, the
      caller should skip tree verification and fall back to normal decode.
    """

    def __init__(
        self,
        draft_token_num: int,
        max_batch_size: int,
        max_tree_depth: int = 24,
        max_cached_requests: int = 10000,
        max_spec_factor: float = 1.0,
        min_token_prob: float = 0.1,
    ):
        """
        Args:
            draft_token_num: Fixed number of draft tokens (for padding)
            max_tree_depth: Maximum depth for suffix tree
            max_cached_requests: Maximum number of cached requests
            max_spec_factor: Maximum speculation factor
            min_token_prob: Minimum token probability threshold
        """
        # Lazy import to avoid error when Suffix Decoding is not used
        from arctic_inference.suffix_decoding import SuffixDecodingCache

        self.suffix_cache = SuffixDecodingCache(
            max_tree_depth=max_tree_depth,
            max_cached_requests=max_cached_requests,
        )
        self.draft_token_num = draft_token_num
        self.max_batch_size = max_batch_size
        self.max_tree_depth = max_tree_depth
        self.max_spec_factor = max_spec_factor
        self.min_token_prob = min_token_prob

        # Debug toggles (set env e.g. SUFFIX_DEBUG_TREE=1 to dump first batch)
        self.debug_tree_dump_remaining = int(os.environ.get("SUFFIX_DEBUG_TREE", "0"))

        # Track state by SGlang request ID (stable identifier)
        # Map: sglang_req_id → [arctic_req_id, last_length]
        self.req_state = {}

        # Preallocate buffers to avoid per-step allocations
        self.max_total_drafts = self.max_batch_size * self.draft_token_num
        self.draft_buffer = np.empty((self.max_total_drafts,), dtype=np.int64)
        self.mask_buffer = np.empty(
            (self.max_batch_size, self.draft_token_num, self.draft_token_num),
            dtype=bool,
        )

        # --- Adaptive fallback bookkeeping ---
        # After each batch_get(), this holds the total number of speculated
        # tokens (excluding the root node) across the whole batch.
        # When this is 0 the caller should skip verification.
        self.last_total_speculated: int = 0

        # Counter for throttling _cleanup_inactive_requests()
        self._cleanup_counter: int = 0

    def _cleanup_inactive_requests(self, active_req_ids: set):
        """Stop backend requests that are no longer active in SGlang.

        Only runs every _CLEANUP_INTERVAL calls to amortise the scan cost.
        """
        self._cleanup_counter += 1
        if self._cleanup_counter < _CLEANUP_INTERVAL:
            return
        self._cleanup_counter = 0

        inactive_req_ids = [rid for rid in self.req_state if rid not in active_req_ids]
        for rid in inactive_req_ids:
            cache_req_id, _ = self.req_state.pop(rid)
            if cache_req_id in getattr(self.suffix_cache, "active_requests", set()):
                self.suffix_cache.stop_request(cache_req_id)

    def _get_or_create_cache_req_id(
        self, sglang_req_id: str, prompt: List[int], output_ids: List[int]
    ) -> tuple:
        """Get or create a backend request ID for the given SGlang request.

        Args:
            sglang_req_id: Stable request ID from SGlang
            prompt: Prompt tokens only (no generated tokens)
            output_ids: Generated output tokens (not used for creation,
                only prompt is needed for start_request)

        Returns: (arctic_req_id, last_length)
        """
        if sglang_req_id not in self.req_state:
            # Use SGlang request ID directly as backend request ID
            cache_req_id = sglang_req_id

            # Initialize the request in suffix cache with ONLY the prompt
            self.suffix_cache.start_request(cache_req_id, prompt)

            # Track: [arctic_req_id, last_length]
            # IMPORTANT: Set last_length to prompt length since the backend already has the prompt
            self.req_state[sglang_req_id] = [cache_req_id, len(prompt)]

        cache_req_id, last_length = self.req_state[sglang_req_id]
        return cache_req_id, last_length

    def batch_get(
        self,
        batch_req_ids: List[str],
        batch_prompts: List[List[int]],
        batch_output_ids: List[List[int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get draft tokens for a batch of token sequences.

        This is called BEFORE verification with the current state.
        We speculate based on the current tokens.

        After this call, ``self.last_total_speculated`` holds the total number
        of *useful* speculated tokens (excluding the root node) across the
        batch.  When this is 0 the caller should skip tree verification and
        fall back to normal single-token decoding.

        Args:
            batch_req_ids: List of SGlang request IDs (stable)
            batch_prompts: List of prompt tokens (no generated tokens)
            batch_output_ids: List of generated output tokens per request.
                Passed separately from prompts to avoid O(seq_len) list
                concatenation per request per step.

        Returns:
            Tuple of:
            - draft_tokens: np.ndarray of shape (batch_size * draft_token_num,)
            - tree_mask: np.ndarray of shape (batch_size * draft_token_num * draft_token_num,)
        """
        batch_size = len(batch_req_ids)
        if batch_size == 0:
            self.last_total_speculated = 0
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=bool)

        if batch_size > self.max_batch_size:
            raise ValueError(
                f"Batch size {batch_size} exceeds configured max_batch_size={self.max_batch_size}"
            )

        total_draft_tokens = batch_size * self.draft_token_num
        draft_view = self.draft_buffer[:total_draft_tokens]
        mask_view = self.mask_buffer[:batch_size]
        mask_view.fill(False)

        active_req_ids = set(batch_req_ids)
        self._cleanup_inactive_requests(active_req_ids)

        total_speculated = 0  # track useful drafts across batch

        for idx, (sglang_req_id, prompt, output_ids) in enumerate(
            zip(batch_req_ids, batch_prompts, batch_output_ids)
        ):
            cache_req_id, last_length = self._get_or_create_cache_req_id(
                sglang_req_id, prompt, output_ids
            )

            # Ensure cache includes the latest verified tokens before speculation.
            # last_length is always >= len(prompt) (set on start_request), so
            # new tokens are always within output_ids.
            prompt_len = len(prompt)
            current_length = prompt_len + len(output_ids)
            if current_length > last_length:
                new_start = last_length - prompt_len
                new_tokens = output_ids[new_start:]
                if cache_req_id in self.suffix_cache.active_requests:
                    self.suffix_cache.add_active_response(cache_req_id, new_tokens)
                    self.req_state[sglang_req_id][1] = current_length
                else:
                    logger.warning(
                        "[BATCH_GET %d] Suffix cache req %s not active when updating!",
                        idx,
                        cache_req_id,
                    )

            # Extract pattern from end of sequence (up to max_tree_depth)
            # without building the full prompt+output_ids list.
            out_len = len(output_ids)
            if out_len >= self.max_tree_depth:
                pattern = output_ids[-self.max_tree_depth :]
            else:
                need = self.max_tree_depth - out_len
                pattern = prompt[-need:] + output_ids

            # Speculate using suffix cache
            draft = self.suffix_cache.speculate(
                cache_req_id,
                pattern,
                max_spec_tokens=self.draft_token_num,
                max_spec_factor=self.max_spec_factor,
                min_token_prob=self.min_token_prob,
            )

            # Convert to fixed-size arrays
            draft_ids = list(draft.token_ids)
            draft_parents = list(draft.parents)
            draft_ids, draft_parents = self._reorder_tree_bfs(draft_ids, draft_parents)

            context_token = (
                output_ids[-1] if output_ids else (prompt[-1] if prompt else 0)
            )
            draft_ids, draft_parents = self._inject_root_node(
                draft_ids, draft_parents, context_token
            )

            # original_draft_len includes the root node at index 0.
            # Useful speculation = original_draft_len - 1  (root is not speculation).
            original_draft_len = len(draft_ids)
            useful = max(0, original_draft_len - 1)
            total_speculated += useful

            # Pad or truncate to match draft_token_num
            if original_draft_len < self.draft_token_num:
                pad_len = self.draft_token_num - original_draft_len
                draft_ids.extend([0] * pad_len)
                draft_parents.extend([0] * pad_len)
            elif original_draft_len > self.draft_token_num:
                draft_ids = draft_ids[: self.draft_token_num]
                draft_parents = draft_parents[: self.draft_token_num]
                original_draft_len = self.draft_token_num

            start = idx * self.draft_token_num
            end = start + self.draft_token_num
            draft_view[start:end] = draft_ids

            # ----------------------------------------------------------
            # Build tree mask from parent structure.
            #
            # Because _reorder_tree_bfs + _inject_root_node guarantee BFS
            # order (every parent index < child index), we can propagate
            # ancestor masks in a single forward pass: node i inherits all
            # mask bits of its parent.  This replaces the previous O(n*d)
            # nested while-loop with O(n) numpy boolean-OR operations.
            # ----------------------------------------------------------
            mask = mask_view[idx]
            if original_draft_len > 0:
                # Vectorised self-attention diagonal
                diag = np.arange(original_draft_len)
                mask[diag, diag] = True
                # Propagate ancestors (BFS order ⇒ parent already processed).
                # Full-row OR is safe: padded columns are False and stay False.
                for i in range(1, original_draft_len):
                    p = draft_parents[i]
                    if 0 <= p < original_draft_len:
                        mask[i] |= mask[p]

            if self.debug_tree_dump_remaining > 0 and original_draft_len > 0:
                logger.warning(
                    "[SUFFIX DEBUG] req=%s, original_draft_len=%d, masked_len=%d, draft_ids=%s",
                    sglang_req_id,
                    original_draft_len,
                    len(draft_ids),
                    draft_ids,
                )
                logger.warning(
                    "[SUFFIX DEBUG] mask=\n%s",
                    mask.astype(int),
                )
                self.debug_tree_dump_remaining -= 1

        self.last_total_speculated = total_speculated
        tree_mask = mask_view.reshape(-1)[: total_draft_tokens * self.draft_token_num]

        return draft_view, tree_mask

    def batch_put(self, *args, **kwargs):
        """No-op: cache updates happen inside batch_get before speculation."""
        pass

    def synchronize(self):
        """No-op for suffix cache (no async operations)."""
        pass

    def reset(self):
        """Clear all cached data."""
        # Stop all active requests
        for cache_req_id in list(self.suffix_cache.active_requests):
            self.suffix_cache.stop_request(cache_req_id)
        # Clear tracking
        self.req_state.clear()
        logger.info("[SUFFIX ADAPTER] Cache reset")

    def _reorder_tree_bfs(
        self, token_ids: List[int], parents: List[Optional[int]]
    ) -> Tuple[List[int], List[int]]:
        """
        Reorder nodes so parents always precede their descendants.

        reconstruct_indices_from_tree_mask assumes this layout; the backend emits
        score-ordered nodes, so we re-topologize the list before building masks.
        """
        n = len(token_ids)
        if n <= 1:
            return token_ids, parents

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
            if visited[root]:
                continue
            queue = deque([root])
            while queue:
                node = queue.popleft()
                if visited[node]:
                    continue
                visited[node] = True
                order.append(node)
                for child in children[node]:
                    if not visited[child]:
                        queue.append(child)

        # Append any detached nodes (should not happen, but keep deterministic order).
        for idx in range(n):
            if not visited[idx]:
                order.append(idx)

        if order == list(range(n)):
            return token_ids, parents

        remap = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
        reordered_ids = [token_ids[old_idx] for old_idx in order]
        reordered_parents: List[int] = []
        for old_idx in order:
            parent = parents[old_idx]
            if parent is None or parent < 0:
                reordered_parents.append(-1)
            else:
                reordered_parents.append(remap.get(parent, -1))

        return reordered_ids, reordered_parents

    def _inject_root_node(
        self, token_ids: List[int], parents: List[int], context_token: int
    ) -> Tuple[List[int], List[int]]:
        """
        Insert the latest verified token as index 0 so the layout matches NGRAM.
        """
        rooted_ids = [context_token]
        rooted_parents = [-1]
        for parent_idx in parents:
            if parent_idx < 0:
                rooted_parents.append(0)
            else:
                rooted_parents.append(parent_idx + 1)
        rooted_ids.extend(token_ids)
        return rooted_ids, rooted_parents
