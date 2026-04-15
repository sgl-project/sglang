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
from typing import List, Optional

import torch

from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig

logger = logging.getLogger(__name__)


@dataclass
class FuzzyMatchResult:
    """Result returned by fuzzy matching.
    
    Attributes:
        cached_token_count: Number of tokens found in cache that can be reused.
        cached_token_ids: The token IDs of the cached sequence.
        prompt_token_count: Number of tokens in the current prompt that this cache replaces.
            May differ from cached_token_count for semantic matching.
        kv_cache_indices: KV cache indices to reuse (after-RoPE, from the memory pool).
        position_offset: Position offset to apply when reusing this cache.
        cached_start_pos: The original starting position where the cached KV was
            computed (for RoPE reversal). When reusing cached KV, we need to
            reverse RoPE at cached_start_pos and re-apply RoPE at the new position.
    """
    cached_token_count: int
    cached_token_ids: List[int]
    prompt_token_count: int
    kv_cache_indices: torch.Tensor
    position_offset: int
    cached_start_pos: int = 0


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
        # TODO: Implement SemanticEmbeddingProvider
        raise NotImplementedError(
            "SemanticEmbeddingProvider is not yet implemented. "
            "This provider requires FAISS and SentenceTransformer dependencies."
        )
    else:
        raise ValueError(
            f"Unknown fuzzy match provider: {provider_name}. "
            f"Supported providers: 'TokenBlockMatch', 'SemanticEmbedding'"
        )
