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
from typing import Any, List, Optional

import msgspec
import torch

from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig

logger = logging.getLogger(__name__)


class FuzzyMatchSegment(msgspec.Struct):
    """One contiguous span in a provider-supplied fuzzy match plan."""

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


class QualitySignals(msgspec.Struct):
    """Optional provider metrics used for logging and debugging."""

    cosine_similarity: float
    reuse_ratio: float
    confidence_tier: str
    passed_quality_gate: bool
    rejection_reason: Optional[str] = None


class FuzzyMatchResult(msgspec.Struct):
    """Fuzzy match candidate returned to RadixCache."""

    cached_token_count: int
    cached_token_ids: List[int]
    prompt_token_count: int
    kv_cache_indices: torch.Tensor
    position_offset: int
    cached_start_pos: int = 0
    # Optional fields used by SemanticEmbedding.
    segments: Optional[List[FuzzyMatchSegment]] = None
    layer_recompute_mask: Optional[List[bool]] = None
    quality_signals: Optional[QualitySignals] = None
    donor_last_node_id: Optional[int] = None
    # Provider-internal handle to the matched donor entry.
    match_entry: Any = None


class FuzzyMatchProvider(ABC):
    """Provider interface used when exact prefix matching misses."""

    def __init__(self, config):
        self.config = config
        self.min_match_length = config.fuzzy_min_match_length

    def set_min_match_length(self, length: int) -> None:
        """Set the minimum match length in tokens."""
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
        """Register a completed request as a future donor."""
        pass

    def on_donor_inserted(self, request, donor_last_node_id: int) -> None:
        """Receive the TreeNode id created by RadixCache insertion."""
        return None

    def on_cache_reset(self) -> None:
        """Clear provider-side state after the owning cache resets."""
        return None

    @abstractmethod
    def match_on_prefix_miss(
        self,
        prompt_token_ids: List[int],
        already_matched_len: int,
        request=None,
        extra_key=None,
    ) -> Optional[FuzzyMatchResult]:
        """Return a fuzzy match for a request whose exact prefix missed."""
        pass


def create_fuzzy_match_provider(
    config: FuzzyMatchConfig,
) -> Optional["FuzzyMatchProvider"]:
    """Create the configured fuzzy match provider."""
    if not config.enable_fuzzy_match:
        logger.debug("Fuzzy matching is disabled")
        return None

    provider_name = config.fuzzy_match_provider
    logger.info(f"Creating semantic provider: {provider_name}")

    if provider_name == "SemanticEmbedding":
        from sglang.srt.mem_cache.fuzzy_match.semantic_embedding import (
            SemanticEmbeddingProvider,
        )

        return SemanticEmbeddingProvider(config)
    else:
        raise ValueError(
            f"Unknown fuzzy match provider: {provider_name}. "
            f"Supported providers: 'SemanticEmbedding'"
        )
