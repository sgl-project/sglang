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

"""Non-prefix KV cache store for fuzzy matching.

This module stores references to radix tree nodes instead of duplicating pool indices.
When a fuzzy match hits, the pool indices are resolved from the radix tree nodes.
This eliminates the double-counting problem because pool indices are only stored
in the radix tree, not duplicated here.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class NodeRef:
    """Reference to a segment within a radix tree node.
    
    Instead of storing pool indices directly, we store a reference to
    the radix tree node and the range within it. This way, the actual
    pool indices are only stored in the radix tree, avoiding double-counting.
    
    Attributes:
        node_id: ID of the TreeNode in the radix tree.
        offset: Starting offset within the node's value array.
        length: Number of tokens this reference covers.
    """
    node_id: int
    offset: int
    length: int


@dataclass
class NonPrefixEntry:
    """Entry for cached segment in fuzzy matching storage.
    
    This entry stores token IDs (for matching) and node references (for
    resolving pool indices from the radix tree). The actual pool indices
    are only stored in the radix tree, so there's no double-counting.
    
    For token-level matching (TokenBlockMatch):
    - token_ids: used for exact token comparison
    - block_hashes: used for accelerated block-hash matching
    - node_refs: used to resolve pool indices from radix tree
    
    For semantic matching (future SemanticEmbedding):
    - token_ids: used for reference and fallback token comparison
    - semantic_metadata: embedding vectors or other semantic signatures
    - node_refs: used to resolve pool indices from radix tree
    
    Attributes:
        id: Unique identifier for this entry.
        token_ids: Token IDs of the segment (for matching).
        node_refs: List of NodeRef objects pointing to radix tree nodes.
        extra_key: Extra key (e.g., lora_id, cache_salt).
        timestamp: Last access time for LRU eviction.
        block_hashes: Precomputed block hashes for accelerated matching.
        semantic_metadata: Optional semantic data (e.g., embeddings) for
            semantic matching providers. TokenBlockMatch leaves this None.
        start_pos: The original starting position where this KV cache segment
            was computed. Used for RoPE reversal when reusing at a new position.
        num_full_blocks: Number of complete blocks (for matching optimization).
    """
    id: int
    token_ids: List[int]
    node_refs: List[NodeRef]
    extra_key: Optional[str]
    timestamp: float = field(default_factory=time.monotonic)
    block_hashes: Optional[List[int]] = None
    semantic_metadata: Optional[Dict[str, any]] = None
    start_pos: int = 0
    num_full_blocks: int = 0  # For matching optimization


class NonPrefixKVStore:
    """Flat store for non-prefix cached segments.
    
    This store holds token IDs (for matching) and node references to the
    radix tree. The actual pool indices remain in the radix tree only,
    so there's no double-counting problem.
    
    Features:
    - Block hash indexing for accelerated token-level matching
    - LRU eviction when capacity is reached
    - Support for extra_key namespace filtering
    - Optional semantic metadata storage for semantic matching providers
    
    Design for multiple matching strategies:
    - TokenBlockMatch: uses token_ids + block_hashes + node_refs
    - SemanticEmbedding (future): uses semantic_metadata + node_refs
      Token IDs are still stored for reference and fallback comparison,
      but the primary matching uses embedding similarity.
    
    Usage:
        store = NonPrefixKVStore(max_entries=10000, block_size=32)
        
        # Insert a segment with node references (TokenBlockMatch)
        store.insert(
            token_ids=[1, 2, 3, 4, 5],
            node_refs=[NodeRef(node_id=42, offset=0, length=5)],
            extra_key="lora_1",
        )
        
        # Insert a segment with semantic metadata (SemanticEmbedding)
        store.insert(
            token_ids=[1, 2, 3, 4, 5],
            node_refs=[NodeRef(node_id=42, offset=0, length=5)],
            extra_key="lora_1",
            semantic_metadata={"embedding": np.array([...]), "cluster_id": 7},
        )
        
        # Find matches by block hash
        candidates = store.find_by_block_hash(
            query_tokens=[1, 2, 3, 6, 7],
            min_length=16,
        )
    """
    
    def __init__(self, max_entries: int = 10000, block_size: int = 32):
        """Initialize the non-prefix store.
        
        Args:
            max_entries: Maximum number of entries before eviction.
            block_size: Size of blocks for hash indexing (tokens per block).
        """
        self.max_entries = max_entries
        self.block_size = block_size
        self.entries: List[NonPrefixEntry] = []
        self.block_index: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self._entry_id_counter = 0
        
        # Node registry: maps node_id -> TreeNode
        # Set by RadixCache via set_node_registry()
        self._node_registry: Dict[int, any] = {}
    
    def set_node_registry(self, registry: Dict[int, any]):
        """Set the node registry from RadixCache for resolving node references."""
        self._node_registry = registry
    
    def update_node_refs_on_split(self, old_node_id: int, new_node_id: int, split_len: int):
        """Update NodeRefs when a node is split.
        
        When _split_node divides a child node into a new parent (prefix) and
        the modified child (suffix), NodeRefs pointing to the old node need
        to be redirected or adjusted.
        
        Args:
            old_node_id: ID of the node that was split (now holds the suffix).
            new_node_id: ID of the newly created parent node (holds the prefix).
            split_len: Number of tokens in the prefix (new parent node).
        """
        for entry in self.entries:
            for ref in entry.node_refs:
                if ref.node_id == old_node_id:
                    # This NodeRef points to the split node.
                    # Determine which part it references: prefix or suffix.
                    ref_start = ref.offset
                    ref_end = ref.offset + ref.length
                    
                    if ref_end <= split_len:
                        # Entirely in the prefix portion -> redirect to new node
                        ref.node_id = new_node_id
                        # offset stays the same (it's relative to the original node start)
                    elif ref_start >= split_len:
                        # Entirely in the suffix portion -> adjust offset
                        ref.offset = ref_start - split_len
                        # node_id stays the same (still points to old_node_id which is now the suffix)
                    else:
                        # Spans both prefix and suffix -> split the NodeRef
                        # Truncate this ref to the suffix portion only
                        # The prefix portion should get a new NodeRef
                        prefix_length = split_len - ref_start
                        ref.node_id = new_node_id
                        # ref.offset stays the same, length becomes prefix portion
                        ref.length = prefix_length
                        
                        # Add a new NodeRef for the suffix portion
                        entry.node_refs.append(NodeRef(
                            node_id=old_node_id,
                            offset=0,
                            length=ref_end - split_len,
                        ))
    
    def insert(
        self,
        token_ids: List[int],
        node_refs: List[NodeRef],
        extra_key: Optional[str],
        semantic_metadata: Optional[Dict[str, any]] = None,
        start_pos: int = 0,
        radix_tree=None,
    ) -> int:
        """Insert a non-prefix segment into the store.
        
        Args:
            token_ids: Token IDs of the segment (for matching).
            node_refs: List of NodeRef objects pointing to radix tree nodes.
            extra_key: Extra key for namespace filtering.
            semantic_metadata: Optional semantic data for semantic matching
                providers (e.g., embedding vectors, cluster IDs).
                TokenBlockMatch passes None; future SemanticEmbedding providers
                can store their semantic signatures here.
            start_pos: The original starting position where this KV cache segment
                was computed. Used for RoPE reversal when reusing at a new position.
            radix_tree: Reserved for future use (currently not used).
            
        Returns:
            The ID of the inserted entry.
        """
        # Compute block hashes for indexing
        block_hashes = self._compute_block_hashes(token_ids)
        
        # Deduplication: check if an entry with the same first block hash already exists.
        if block_hashes:
            first_block_hash = block_hashes[0]
            for existing_entry in self.entries:
                if (existing_entry.block_hashes 
                    and existing_entry.block_hashes[0] == first_block_hash
                    and len(existing_entry.token_ids) == len(token_ids)
                    and existing_entry.extra_key == extra_key):
                    logger.info(
                        f"[FUZZY CACHE] Duplicate entry skipped: "
                        f"{len(token_ids)} tokens"
                    )
                    return
        
        if len(self.entries) >= self.max_entries:
            self._evict(radix_tree=radix_tree)
        
        entry_id = self._entry_id_counter
        self._entry_id_counter += 1
        
        entry = NonPrefixEntry(
            id=entry_id,
            token_ids=token_ids,
            node_refs=node_refs,
            extra_key=extra_key,
            timestamp=time.monotonic(),
            block_hashes=block_hashes,
            semantic_metadata=semantic_metadata,
            start_pos=start_pos,
        )
        
        self.entries.append(entry)
        
        # Index by block hashes
        for i, block_hash in enumerate(block_hashes):
            self.block_index[block_hash].append((entry_id, i * self.block_size))
 
        logger.debug(
            f"Inserted non-prefix entry {entry_id} with {len(token_ids)} tokens, "
            f"{len(node_refs)} node refs, {len(block_hashes)} blocks"
        )
        
        return entry_id
    
    def resolve_kv_cache(
        self,
        entry: NonPrefixEntry,
        matched_len: int,
        donor_offset: int = 0,
    ) -> torch.Tensor:
        """Resolve pool indices from node references for the matched portion.

        Args:
            entry: The NonPrefixEntry containing node references.
            matched_len: Number of matched tokens from ``donor_offset``.
            donor_offset: Number of tokens to skip into the entry before
                collecting slots. Set to the donor-side offset where the
                block-hash match actually started; ``0`` reproduces the
                legacy "first N slots" behavior. Required so the slot
                indices returned to the engine correspond to the actual
                matched region in the donor (the radix_cache then does
                RoPE correction from those donor positions). Without this,
                a match found at e.g. donor positions [16..115] would
                return slots [0..99] and SGLang would silently use the
                donor's first 100 slots → semantic mismatch + pool leak.

        Returns:
            Tensor of pool indices for tokens
            ``[donor_offset .. donor_offset + matched_len)``.
        """
        indices = []
        skip = donor_offset
        remaining = matched_len
        for ref in entry.node_refs:
            if remaining <= 0:
                break

            node = self._node_registry.get(ref.node_id)
            if node is None or node.value is None:
                logger.warning(
                    f"Node {ref.node_id} not found or evicted during resolution"
                )
                continue

            # Skip refs that lie entirely before the donor_offset.
            if skip >= ref.length:
                skip -= ref.length
                continue

            # Slice from (ref.offset + skip) to keep alignment with the
            # donor's actual matched region.
            start = ref.offset + skip
            end = min(ref.offset + ref.length, start + remaining)
            actual_len = end - start
            if actual_len > 0:
                indices.append(node.value[start:end])
                remaining -= actual_len
            skip = 0

        if not indices:
            return torch.empty(0, dtype=torch.int64)

        return torch.cat(indices)
    
    def find_by_block_hash(
        self,
        query_tokens: List[int],
        min_length: int,
        extra_key: Optional[str] = None,
    ) -> List[Tuple[int, NonPrefixEntry, int, int]]:
        """Find entries that share a block with the query.

        This is for token-level fuzzy matching only. Semantic matching
        should use find_by_semantic() instead.

        Args:
            query_tokens: Query token sequence.
            min_length: Minimum match length in tokens.
            extra_key: Optional extra key for namespace filtering.

        Returns:
            List of ``(matched_len, entry, query_start, donor_start)``
            tuples, sorted by matched_len in descending order.

            ``query_start`` is the offset *into the query* where the match
            begins. Callers that must surface a contiguous prefix to
            SGLang's device_indices contract should filter to ``query_start
            == 0`` and either reject the rest or handle the gap upstream.

            ``donor_start`` is the offset *into the entry* (the donor's
            stored token sequence) where the matched block begins. Callers
            must pass this to ``resolve_kv_cache(entry, matched_len,
            donor_offset=donor_start)`` and surface it via
            ``FuzzyMatchResult.cached_start_pos = entry.start_pos +
            donor_start`` so RadixCache's RoPE correction picks up.
        """
        candidates = []

        # Compute query block hashes
        num_query_blocks = len(query_tokens) // self.block_size

        logger.info(
            f"[NON_PREFIX_STORE] find_by_block_hash: query_len={len(query_tokens)}, "
            f"block_size={self.block_size}, num_query_blocks={num_query_blocks}, "
            f"total_entries={len(self.entries)}, block_index_size={len(self.block_index)}"
        )

        if num_query_blocks == 0:
            logger.warning(
                f"[NON_PREFIX_STORE] Query too short for block hash matching: "
                f"query_len={len(query_tokens)} < block_size={self.block_size}. "
                f"Falling back to linear scan of all entries."
            )
            return self._find_by_linear_scan(query_tokens, min_length, extra_key)

        for query_block_idx in range(num_query_blocks):
            start = query_block_idx * self.block_size
            block = tuple(query_tokens[start:start + self.block_size])
            block_hash = hash(block)

            if block_hash not in self.block_index:
                continue

            for entry_id, cached_pos in self.block_index[block_hash]:
                entry = self.entries[entry_id]

                # Filter by extra_key if specified
                if extra_key is not None and entry.extra_key != extra_key:
                    continue

                # Verify block match and extend per-token
                count = self._count_match_from_position(
                    entry.token_ids,
                    cached_pos,
                    query_tokens,
                    start,
                )

                if count >= min_length:
                    candidates.append((count, entry, start, cached_pos))

        # Sort by matched_len descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        return candidates

    def find_by_semantic(
        self,
        query_tokens: List[int],
        min_similarity: float,
        extra_key: Optional[str] = None,
    ) -> List[Tuple[float, NonPrefixEntry]]:
        """Find entries by semantic similarity.

        TODO: Not yet implemented. This method is reserved for future
        semantic embedding-based fuzzy matching. When implemented, it
        should convert query_tokens to embeddings, compare against stored
        semantic_metadata, and return entries ranked by similarity.

        Args:
            query_tokens: Query token sequence.
            min_similarity: Minimum similarity threshold (0.0 - 1.0).
            extra_key: Optional extra key for namespace filtering.

        Returns:
            List of (similarity_score, entry) tuples, sorted by similarity
            in descending order.
        """
        raise NotImplementedError(
            "Semantic matching is not yet implemented. "
            "Use find_by_block_hash() for token-level matching instead."
        )

    def _find_by_linear_scan(
        self,
        query_tokens: List[int],
        min_length: int,
        extra_key: Optional[str] = None,
    ) -> List[Tuple[int, NonPrefixEntry, int, int]]:
        """Linear scan fallback for short queries (token-level matching only).

        Returns ``(matched_len, entry, query_start=0, donor_start=cached_pos)``
        tuples, matching the shape of ``find_by_block_hash`` so callers
        don't have to special-case the short-query fallback.
        """
        candidates = []

        for entry in self.entries:
            if extra_key is not None and entry.extra_key != extra_key:
                continue

            # Try to match from each position in the entry
            for cached_pos in range(len(entry.token_ids)):
                count = self._count_match_from_position(
                    entry.token_ids,
                    cached_pos,
                    query_tokens,
                    0,
                )
                if count >= min_length:
                    candidates.append((count, entry, 0, cached_pos))
                    break

        candidates.sort(key=lambda x: x[0], reverse=True)

        logger.info(
            f"[NON_PREFIX_STORE] Linear scan complete: found {len(candidates)} candidates"
        )

        return candidates
    
    def _count_match_from_position(
        self,
        entry_tokens: List[int],
        entry_pos: int,
        query_tokens: List[int],
        query_pos: int,
    ) -> int:
        """Count matching tokens from a given position."""
        count = 0
        max_count = min(
            len(entry_tokens) - entry_pos,
            len(query_tokens) - query_pos,
        )
        
        for i in range(max_count):
            if entry_tokens[entry_pos + i] != query_tokens[query_pos + i]:
                break
            count += 1
        
        return count
    
    def _compute_block_hashes(self, token_ids: List[int]) -> List[int]:
        """Compute block hashes for indexing.
        
        When token_ids length is less than block_size, hash the entire sequence
        as a single block to enable deduplication via block_hashes[0].
        """
        block_hashes = []
        
        if len(token_ids) < self.block_size:
            # Token sequence shorter than block_size - hash the entire sequence
            block_hashes.append(hash(tuple(token_ids)))
        else:
            num_blocks = len(token_ids) // self.block_size
            for i in range(num_blocks):
                start = i * self.block_size
                block = tuple(token_ids[start:start + self.block_size])
                block_hashes.append(hash(block))
        
        return block_hashes
    
    def _evict(self, radix_tree=None):
        """Evict the least recently used entry.
        
        Args:
            radix_tree: Reserved for future use (currently not used).
        """
        if not self.entries:
            return
        
        lru_idx = 0
        lru_time = self.entries[0].timestamp
        for i, entry in enumerate(self.entries):
            if entry.timestamp < lru_time:
                lru_time = entry.timestamp
                lru_idx = i
        
        evicted = self.entries.pop(lru_idx)
        
        # Remove from block index
        for block_hash in (evicted.block_hashes or []):
            if block_hash in self.block_index:
                self.block_index[block_hash] = [
                    (eid, pos) for eid, pos in self.block_index[block_hash]
                    if eid != evicted.id
                ]
                if not self.block_index[block_hash]:
                    del self.block_index[block_hash]

        logger.debug(
            f"Evicted non-prefix entry {evicted.id} with {len(evicted.token_ids)} tokens"
        )
    
    @property
    def total_entries(self) -> int:
        return len(self.entries)
