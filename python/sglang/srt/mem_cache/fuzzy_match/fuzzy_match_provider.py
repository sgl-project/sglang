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

"""Abstract interface for fuzzy/semantic prefix matching providers."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional

import torch

from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig

logger = logging.getLogger(__name__)


@dataclass
class FuzzyMatchSegment:
    """One contiguous span of a multi-segment / multi-donor fuzzy match.

    Used by ``SemanticEmbeddingProvider`` and other providers that produce
    N:M token alignments (REORDER, PARAPHRASE, multi-donor scatter) where a
    single ``cached_start_pos`` is insufficient to describe the layout.

    When ``FuzzyMatchResult.segments`` is set, ``model_runner._correct_fuzzy_kv_rope``
    iterates per-segment, allocating new pool slots and applying RoPE
    correction with deltas derived from ``donor_positions -> target_positions``.
    Providers that produce a single contiguous span (``TokenBlockMatchProvider``)
    set ``segments = None`` and the legacy contiguous path runs unchanged.

    Two addressing modes:

    * **NodeRef** (preferred for new providers): the segment
      points at a TreeNode in the radix tree's ``_node_registry`` via
      ``donor_node_id`` + ``donor_offset`` + ``length``. ``model_runner``
      resolves to pool indices at consume time:

          node = radix_tree._node_registry[donor_node_id]
          donor_kv_slots = node.value[donor_offset : donor_offset + length]

      This preserves the "no double-counting" principle (the radix tree is
      the single owner of pool indices) and ties slot lifetime to the donor
      TreeNode's lifetime - paired with ``FuzzyMatchResult.donor_last_node_id``
      ``inc_lock_ref`` protection.

    * **Legacy pool-indices** (``donor_kv_indices``): a raw tensor of pool
      indices. Used by ``TokenBlockMatchProvider``'s contiguous-span path.

    Attributes:
        target_positions: Absolute token positions in the new prompt.
        donor_positions: Source positions in the donor (used to compute the
            RoPE delta per token).
        donor_node_id: Optional ID of the donor TreeNode in the radix tree's
            ``_node_registry``. NodeRef-based addressing.
        donor_offset: Optional offset into the donor TreeNode's ``value``
            tensor where this segment's slots begin.
        length: Optional number of slots this segment covers in the donor.
        donor_kv_indices: Optional raw pool-indices tensor (legacy).
        donor_req_id: Optional identifier of the source donor (multi-donor).
        layer_recompute_mask: Optional per-segment override of the global
            ``FuzzyMatchResult.layer_recompute_mask``.
    """

    target_positions: torch.Tensor
    donor_positions: torch.Tensor

    # NodeRef-based addressing (preferred for new providers).
    donor_node_id: Optional[int] = None
    donor_offset: Optional[int] = None
    length: Optional[int] = None

    # Legacy pool-indices tensor.
    donor_kv_indices: Optional[torch.Tensor] = None

    donor_req_id: Optional[str] = None
    layer_recompute_mask: Optional[List[bool]] = None


@dataclass
class QualitySignals:
    """Provider-visible quality signals attached to a fuzzy match.

    Used by the scheduler and observability layer to log/monitor why a match
    fired (or to gate behavior on confidence). Optional - providers that don't
    track these set ``FuzzyMatchResult.quality_signals = None``.

    Attributes:
        cosine_similarity: Cosine similarity between query and donor embeddings.
        reuse_ratio: Fraction of the prompt covered by donor KV.
        confidence_tier: Provider-defined string label (e.g. "exact", "fuzzy",
            "verified_reuse", "fast_reuse").
        passed_quality_gate: Whether the provider's internal quality gate
            accepted this match.
        rejection_reason: Optional human-readable reason if a gate rejected
            (still surfaced for telemetry on near-misses).
    """

    cosine_similarity: float
    reuse_ratio: float
    confidence_tier: str
    passed_quality_gate: bool
    rejection_reason: Optional[str] = None


@dataclass
class FuzzyMatchResult:
    """Result returned by fuzzy matching.

    Attributes:
        cached_token_count: Number of tokens found in cache that can be reused.
        cached_token_ids: The token IDs of the cached sequence.
        prompt_token_count: Number of tokens in the current prompt that this cache replaces.
            May differ from cached_token_count for semantic matching.
        kv_cache_indices: KV cache indices to reuse (after-RoPE, from the memory pool).
            For multi-segment results (``segments`` populated), this may be empty;
            callers should iterate segments instead.
        position_offset: Position offset to apply when reusing this cache.
        cached_start_pos: The original starting position where the cached KV was
            computed (for RoPE reversal). When reusing cached KV, we need to
            reverse RoPE at cached_start_pos and re-apply RoPE at the new position.
        segments: Optional list of segments for N:M / multi-donor alignment.
            When set, the model executor honors per-segment positions instead
            of the single ``cached_start_pos`` field.
        layer_recompute_mask: Optional per-layer recomputation flags. When
            element ``i`` is True, the model executor recomputes layer ``i``
            for the matched tokens instead of reusing cached KV. Used by
            providers that apply selective recomputation curves to preserve
            quality on imperfect matches.
        quality_signals: Optional provider-visible quality metrics.
    """

    cached_token_count: int
    cached_token_ids: List[int]
    prompt_token_count: int
    kv_cache_indices: torch.Tensor
    position_offset: int
    cached_start_pos: int = 0
    # --- Optional extensions (Pull request: SemanticEmbeddingProvider) ---
    segments: Optional[List[FuzzyMatchSegment]] = None
    layer_recompute_mask: Optional[List[bool]] = None
    quality_signals: Optional[QualitySignals] = None
    # ID of the donor's TreeNode in radix_tree._node_registry. When set, the
    # caller (RadixCache.match_prefix) is responsible for inc_lock_ref'ing the
    # donor node so its KV-cache slots can't be LRU-evicted while the new
    # request is consuming them, and dec_lock_ref'ing on request finish.
    # Without this, sustained fuzzy traffic causes the SGLang runtime checker's
    # "pool memory leak detected!" assertion to fire (~19k slots leaked per
    # ~50-75 fuzzy-mode requests on Qwen-1.5B / A10G).
    donor_last_node_id: Optional[int] = None
    # Free-form provider-private payload (avoid name collisions with future fields).
    _match_entry: Any = None


class FuzzyMatchProvider(ABC):
    """Abstract interface for fuzzy/semantic prefix matching.

    This provider is ONLY invoked when exact prefix matching fails to cover
    the full prompt (matched n < L-1). Its sole responsibility is to manage
    fuzzy KV cache: lookup, read, and write. It does NOT handle exact-match
    caching -- that remains in SGLang's original RadixCache.

    Community implementations can provide:
    - Embedding-based semantic similarity matching (token IDs/counts may differ)
    - Token-level block-matching (token IDs identical, but at non-prefix positions)
    - Hybrid approaches

    Usage:
        1. Subclass FuzzyMatchProvider
        2. Implement cache_on_request_finished() and match_on_prefix_miss()
        3. Register with RadixCache via init_fuzzy_match()
    """
    
    def __init__(self, config):
        """Initialize the semantic provider.
        
        Args:
            config: FuzzyMatchConfig instance containing configuration parameters.
        """
        self.config = config
        self.min_match_length = config.fuzzy_min_match_length
    
    def set_min_match_length(self, length: int) -> None:
        """Set the minimum number of tokens required for a fuzzy match.
        
        Args:
            length: Minimum match length in tokens.
        """
        self.min_match_length = length
    
    @abstractmethod
    def cache_on_request_finished(
        self,
        request,
        token_ids: List[int],
        kv_cache: torch.Tensor,
        cache_start_pos: int,
        cache_end_pos: int,
        radix_tree=None,
    ) -> bool:
        """Called when a request completes. Decides whether to cache its KV
        into the fuzzy-matching storage structures.
        
        Args:
            request: The completed request object (Req).
            token_ids: Full token sequence of the request.
            kv_cache: KV cache tensor (after-RoPE) for this request.
            cache_start_pos: Starting position of the cacheable segment.
            cache_end_pos: Ending position (exclusive) of the cacheable segment.
            radix_tree: Radix tree instance for creating node references.
            
        Returns:
            True if the KV cache was stored, False otherwise.
        """
        pass
    
    def on_donor_inserted(self, request, donor_last_node_id: int) -> None:
        """Optional hook: called by RadixCache.cache_finished_req AFTER the
        donor's KV has been inserted into the radix tree, with the resulting
        TreeNode id from ``radix_tree._node_registry``.

        Providers that need to inc_lock_ref the donor TreeNode at match time
        (i.e. all providers that surface real KV reuse via cached_token_count > 0)
        should record this id keyed on the request and surface it as
        ``FuzzyMatchResult.donor_last_node_id`` from ``match_on_prefix_miss``.

        Default is a no-op for backward compatibility. Override in providers
        that need donor-node locking.
        """
        return None

    @abstractmethod
    def match_on_prefix_miss(
        self,
        prompt_token_ids: List[int],
        already_matched_len: int,
    ) -> Optional[FuzzyMatchResult]:
        """Called when exact prefix matching falls short.
        
        Args:
            prompt_token_ids: Complete token IDs of the current prompt.
            already_matched_len: Number of tokens already matched by
                the exact radix tree (n in the design).
                
        Returns:
            FuzzyMatchResult if a match is found, None otherwise.
            
        Implementation Notes:
            - Should check FuzzyRadixTree first, then NonPrefixKVStore.
            - Must respect min_match_length threshold.
            - For semantic matching, the provider may use its own
              embedding store and return mappings between prompt
              tokens and cached tokens.
        """
        pass


def create_fuzzy_match_provider(config: FuzzyMatchConfig) -> Optional["FuzzyMatchProvider"]:
    """Create a fuzzy match provider based on configuration.

    Args:
        config: Fuzzy matching configuration.

    Returns:
        FuzzyMatchProvider instance or None if fuzzy matching is disabled.

    Raises:
        ValueError: If the provider name is unknown.
        NotImplementedError: If the provider is not yet implemented.

    Example:
        >>> config = FuzzyMatchConfig(
        ...     enable_fuzzy_match=True,
        ...     fuzzy_match_provider="TokenBlockMatch",
        ... )
        >>> provider = create_fuzzy_match_provider(config)
        >>> print(type(provider))
        <class 'TokenBlockMatchProvider'>
    """
    if not config.enable_fuzzy_match:
        logger.debug("Fuzzy matching is disabled")
        return None
    
    provider_name = config.fuzzy_match_provider
    logger.info(f"Creating semantic provider: {provider_name}")
    
    if provider_name == "TokenBlockMatch":
        from sglang.srt.mem_cache.fuzzy_match.token_block_match import (
            TokenBlockMatchProvider,
        )
        return TokenBlockMatchProvider(config)
    elif provider_name == "SemanticEmbedding":
        from sglang.srt.mem_cache.fuzzy_match.semantic_embedding import (
            SemanticEmbeddingProvider,
        )
        return SemanticEmbeddingProvider(config)
    else:
        raise ValueError(
            f"Unknown fuzzy match provider: {provider_name}. "
            f"Supported providers: 'TokenBlockMatch', 'SemanticEmbedding'"
        )
