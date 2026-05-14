# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Token-level matching provider with block hash acceleration.

This provider implements fuzzy matching by finding exact token-identical
sequences at any position (not just prefix position), accelerated by
non-overlapping block hashing.

Example:
    Cached:    [A, B, C, D, E, F, G, H]
    Prompt:    [X, Y, Z, W, A, B, C, D, E, F, M, N]
    
    Exact prefix match: 4 tokens [X, Y, Z, W]
    Remaining: [A, B, C, D, E, F, M, N]
    
    Fuzzy matching finds [A, B, C, D, E, F] (6 tokens) matches
    Total matched: 4 (exact) + 6 (fuzzy) = 10 tokens
"""

import logging
from typing import List, Optional

import torch

from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.non_prefix_store import (
    NodeRef,
    NonPrefixEntry,
    NonPrefixKVStore,
)
from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import (
    FuzzyMatchProvider,
    FuzzyMatchResult,
)

logger = logging.getLogger(__name__)


class TokenBlockMatchProvider(FuzzyMatchProvider):
    """Token-level matching accelerated by non-overlapping block hashing.
    
    Core logic: from the prompt position after exact prefix match,
    find the cached sequence that gives the longest consecutive
    token-identical extension.
    
    Acceleration: divide sequences into non-overlapping blocks of fixed size.
    Compare block-by-block via hash starting from the first unmatched position.
    When a block mismatches or an incomplete trailing block is reached,
    fall back to per-token comparison within that boundary to find the
    exact match length.
    
    Block hashing is purely a filter — the actual match boundary is
    determined by per-token comparison.
    """
    
    def __init__(self, config: FuzzyMatchConfig):
        """Initialize the token block match provider.
        
        Args:
            config: Fuzzy match configuration.
        """
        super().__init__(config)
        self.block_size = config.fuzzy_block_size
        
        # Non-prefix store for all cached segments (both prefix-complete and mid-prompt)
        self.non_prefix_store = NonPrefixKVStore(
            max_entries=config.fuzzy_non_prefix_max_entries,
            block_size=self.block_size,
        )
        
        logger.info(
            f"TokenBlockMatchProvider initialized with block_size={self.block_size}"
        )
    
    def cache_on_request_finished(
        self,
        request,
        token_ids: List[int],
        kv_cache: torch.Tensor,
        cache_start_pos: int,
        cache_end_pos: int,
        radix_tree=None,
    ) -> bool:
        """Cache request for future fuzzy matching.
        
        Args:
            request: The completed request object.
            token_ids: Full token sequence of the request.
            kv_cache: KV cache tensor for this request.
            cache_start_pos: Starting position of the cacheable segment.
            cache_end_pos: Ending position (exclusive) of the cacheable segment.
            radix_tree: Radix tree instance for creating node references.
            
        Returns:
            True if cached, False otherwise.
        """
        if not self.config.cache_fuzzy_results:
            logger.debug("Fuzzy caching disabled by config")
            return False
        
        segment_tokens = token_ids[cache_start_pos:cache_end_pos]
        logger.info(
            f"[FUZZY CACHE] Segment tokens (first 20): {segment_tokens[:20]}"
        )
        logger.info(
            f"[FUZZY CACHE] Segment tokens (last 20): {segment_tokens[-20:]}"
        )
        
        if len(segment_tokens) < self.min_match_length:
            logger.warning(
                f"[FUZZY CACHE] Segment too short for caching: "
                f"{len(segment_tokens)} < {self.min_match_length}, skipping"
            )
            return False
        
        # All segments are cached to non_prefix_store with node references.
        # This unifies prefix-complete sequences (cache_start_pos=0) and
        # mid-prompt segments (cache_start_pos>0) into a single store.
        # Node references avoid double-counting pool indices.
        node_refs = self._create_node_refs_from_tree(
            radix_tree=radix_tree,
            cache_start_pos=cache_start_pos,
            cache_end_pos=cache_end_pos,
        )
        
        self.non_prefix_store.insert(
            token_ids=segment_tokens,
            node_refs=node_refs,
            extra_key=request.extra_key,
            start_pos=cache_start_pos,
            radix_tree=radix_tree,
        )
        
        logger.info(
            f"[FUZZY CACHE] ✓ Cached to NON_PREFIX_STORE: "
            f"tokens={len(segment_tokens)}, "
            f"range=[{cache_start_pos}:{cache_end_pos}], "
            f"total_entries={len(self.non_prefix_store.entries)}"
        )
        logger.info(
            f"[FUZZY CACHE] Block index size: {len(self.non_prefix_store.block_index)}"
        )
        
        return True
    
    def _create_node_refs_from_tree(
        self,
        radix_tree,
        cache_start_pos: int,
        cache_end_pos: int,
    ) -> List["NodeRef"]:
        """Create NodeRef objects by finding nodes in the radix tree.
        
        This method traverses the radix tree to find nodes that contain
        the tokens in the range [cache_start_pos, cache_end_pos).
        
        Args:
            radix_tree: The RadixCache instance.
            cache_start_pos: Start position of the segment.
            cache_end_pos: End position of the segment.
            
        Returns:
            List of NodeRef objects pointing to the radix tree nodes.
        """
        from sglang.srt.mem_cache.fuzzy_match.non_prefix_store import NodeRef
        
        node_refs = []
        
        # Traverse the tree to find nodes covering [cache_start_pos, cache_end_pos)
        # We start from root and walk down, keeping track of the cumulative token count
        current_pos = 0
        remaining = cache_end_pos - cache_start_pos
        
        # Walk the tree from root to find nodes in the target range
        node = radix_tree.root_node
        stack = [(node, 0)]  # (node, start_offset_in_tokens)
        
        while stack and remaining > 0:
            node, offset = stack.pop()
            
            # Add children to stack FIRST (before potentially skipping this node)
            # This ensures we traverse the entire tree even if this node has no value
            node_value_len = len(node.value) if node.value is not None else 0
            for child_key, child in node.children.items():
                child_offset = offset + node_value_len
                stack.append((child, child_offset))
            
            if node.value is None or node_value_len == 0:
                continue
            
            # Check if this node's range overlaps with [cache_start_pos, cache_end_pos)
            node_start = offset
            node_end = offset + node_value_len
            
            overlap_start = max(node_start, cache_start_pos)
            overlap_end = min(node_end, cache_end_pos)
            
            if overlap_start < overlap_end:
                # This node overlaps with the target range
                ref_offset = overlap_start - node_start
                ref_length = overlap_end - overlap_start
                node_refs.append(NodeRef(
                    node_id=node.id,
                    offset=ref_offset,
                    length=ref_length,
                ))
                remaining -= ref_length
        
        logger.debug(
            f"[FUZZY CACHE] Created {len(node_refs)} node refs for range [{cache_start_pos}:{cache_end_pos}]"
        )
        
        return node_refs
    
    def match_on_prefix_miss(
        self,
        prompt_token_ids: List[int],
        already_matched_len: int,
    ) -> Optional[FuzzyMatchResult]:
        """Find fuzzy match for the remaining prompt tokens.
        
        Args:
            prompt_token_ids: Complete token IDs of the current prompt.
            already_matched_len: Number of tokens already matched by exact prefix matching.
            
        Returns:
            FuzzyMatchResult if found, None otherwise.
        """
        remaining = prompt_token_ids[already_matched_len:]
        
        logger.info(
            f"[FUZZY MATCH] Request: prompt_len={len(prompt_token_ids)}, "
            f"already_matched_len={already_matched_len}, "
            f"remaining_len={len(remaining)}"
        )
        logger.info(
            f"[FUZZY MATCH] Remaining tokens (first 30): {remaining[:30]}"
        )
        
        if len(remaining) < self.min_match_length:
            logger.info(
                f"[FUZZY MATCH] ✗ Remaining tokens too short: "
                f"{len(remaining)} < {self.min_match_length}"
            )
            return None
        
        # Search in non_prefix_store for fuzzy matches.
        # This store contains both prefix-complete sequences (cache_start_pos=0)
        # and mid-prompt segments (cache_start_pos>0).
        logger.info(
            f"[FUZZY MATCH] Searching non_prefix_store: {len(self.non_prefix_store.entries)} entries, "
            f"block_index size: {len(self.non_prefix_store.block_index)}"
        )
        candidates = self.non_prefix_store.find_by_block_hash(
            query_tokens=remaining,
            min_length=self.min_match_length,
            extra_key=None,
        )
        
        logger.info(
            f"[FUZZY MATCH] Non-prefix store candidates: {len(candidates)}"
        )
        
        if not candidates:
            logger.info(
                f"[FUZZY MATCH] ✗ No match found (best_len=0, "
                f"min_match_length={self.min_match_length})"
            )
            return None

        # Filter to candidates anchored at the variant's prefix boundary
        # (query_start == 0). SGLang's device_indices contract is
        # contiguous from the exact-prefix boundary, so a match that
        # starts mid-query would silently claim positions [0..N-1] are
        # cached when only [query_start..query_start+N-1] really are.
        prefix_anchored = [c for c in candidates if c[2] == 0]
        if not prefix_anchored:
            logger.info(
                f"[FUZZY MATCH] ✗ No prefix-anchored match (longest match "
                f"started mid-query at offset>0; dropped to keep KV reuse "
                f"semantically correct)"
            )
            return None

        matched_len, entry, _, donor_start = prefix_anchored[0]

        if matched_len < self.min_match_length:
            logger.info(
                f"[FUZZY MATCH] ✗ No match found (best_len={matched_len}, "
                f"min_match_length={self.min_match_length})"
            )
            return None

        logger.info(
            f"[FUZZY MATCH] ✓ Match found! "
            f"cached_token_count={matched_len}, donor_start={donor_start}, "
            f"token_ids (first 20): {entry.token_ids[donor_start:donor_start + 20]}"
        )

        # The slot indices returned to the engine must correspond to the
        # donor positions implied by cached_start_pos below. Pass
        # donor_start so resolve_kv_cache slices from the matched region
        # rather than the entry's first matched_len slots.
        kv_cache_indices = self.non_prefix_store.resolve_kv_cache(
            entry, matched_len, donor_offset=donor_start,
        )

        # Surface the donor's final TreeNode id so RadixCache.match_prefix can
        # inc_lock_ref it. Without this, the donor's KV-pool slots can be
        # LRU-evicted while the recipient is still consuming them, tripping
        # the runtime pool-leak detector (~240 slots leaked on the first
        # non-trivial fuzzy hit on Qwen-7B-AWQ / A10G, verified 2026-05-14).
        # The donor lock is released in RadixCache.cache_finished_req.
        donor_last_node_id = (
            entry.node_refs[-1].node_id if entry.node_refs else None
        )

        # cached_start_pos reflects the donor position where the matched
        # block actually begins. When the match is non-prefix in the
        # donor (donor_start > 0), this triggers
        # `needs_realization = (cached_start_pos != exact_matched_len)`
        # in radix_cache: realized_locs get pre-allocated and donor KV
        # is copied with RoPE correction into fresh slots. Without it,
        # donor and recipient share physical pool slots and the pool
        # accounting silently double-counts (over-count leak proportional
        # to matched_len per fuzzy hit).
        return FuzzyMatchResult(
            cached_token_count=matched_len,
            cached_token_ids=entry.token_ids[donor_start:donor_start + matched_len],
            prompt_token_count=matched_len,  # 1:1 for token-level matching
            kv_cache_indices=kv_cache_indices,
            position_offset=already_matched_len,
            cached_start_pos=entry.start_pos + donor_start,
            donor_last_node_id=donor_last_node_id,
        )
