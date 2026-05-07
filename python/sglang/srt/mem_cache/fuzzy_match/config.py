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

"""Configuration for fuzzy prefix matching."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FuzzyMatchConfig:
    """Configuration for fuzzy prefix matching.
    
    This configuration controls the behavior of fuzzy/semantic prefix matching
    in RadixCache. All fuzzy matching features are opt-in and disabled by default.
    
    Attributes:
        enable_fuzzy_match: Whether to enable fuzzy prefix matching.
        fuzzy_min_match_length: Minimum number of tokens that must be matched
            by exact prefix matching before attempting fuzzy matching.
        fuzzy_semantic_threshold: Similarity threshold for semantic matching (0.0 - 1.0).
            Only used by SemanticEmbeddingProvider.
        fuzzy_match_provider: Provider class name for fuzzy matching logic.
            Options: "TokenBlockMatch" (token-level) or "SemanticEmbedding" (embedding-based).
        cache_fuzzy_results: Whether to cache fuzzy match results for future reuse.
        fuzzy_eviction_policy: Eviction policy for fuzzy radix tree.
            Supported: "lru", "lfu", "fifo", "mru", "filo", "priority", "slru".
        fuzzy_non_prefix_max_entries: Maximum entries in non-prefix store.
        fuzzy_block_size: Block size for TokenBlockMatchProvider (tokens per block).
        embedding_model_name: Embedding model name for SemanticEmbeddingProvider.
    """
    
    # Enable fuzzy prefix matching
    enable_fuzzy_match: bool = False
    
    # Minimum number of tokens that must be matched for fuzzy to trigger
    fuzzy_min_match_length: int = 16
    
    # Cosine-similarity threshold for SemanticEmbedding matches.
    # Range [0.0, 1.0]; ignored by TokenBlockMatch.
    # Higher = stricter (fewer matches, higher precision); lower =
    # more permissive. Below ~0.50 alignment quality drops quickly.
    fuzzy_semantic_threshold: float = 0.60
    
    # Provider class for fuzzy matching logic
    # Options: "SemanticEmbedding" (embedding-based) or "TokenBlockMatch" (token-level)
    fuzzy_match_provider: str = "TokenBlockMatch"
    
    # Cache fuzzy match results for future reuse
    cache_fuzzy_results: bool = True
    
    # Eviction policy for fuzzy radix tree
    fuzzy_eviction_policy: str = "LRU"
    
    # Maximum entries in non-prefix store
    fuzzy_non_prefix_max_entries: int = 10000
    
    # Block size for TokenBlockMatchProvider (tokens per block)
    fuzzy_block_size: int = 16
    
    # Embedding model name for SemanticEmbeddingProvider
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # ----------------------------------------------------------------
    # SemanticEmbeddingProvider-specific fields (ignored by other providers)
    # ----------------------------------------------------------------
    #
    # SemanticEmbedding is process-local: in-process MiniLM embedding and
    # numpy donor store. There is no remote backend / service mode.

    # Whether the in-process embedder may use GPU (auto-detected).
    embedding_use_gpu: bool = True

    # Model architecture tag for bathtub-curve preset selection
    # ("llama" | "qwen2.5-7b" | "qwen2.5-1.5b" | None).
    model_arch: Optional[str] = None

    # Whether to populate ``layer_recompute_mask`` from the bathtub curve.
    enable_bathtub: bool = True

    # Top-K candidates pulled from the ANN/donor store before alignment.
    fuzzy_top_k: int = 5

    # Minimum reuse ratio (matched / prompt tokens) required for the
    # provider to surface a hit. Below this ratio the projected wall-
    # clock benefit of reusing donor KV typically does not exceed the
    # cost of the layer-recompute mask the bathtub model emits.
    fuzzy_min_reuse_ratio: float = 0.50

    # Informational PPL guardrail (telemetry; not enforced in-flight).
    quality_gate_ppl_threshold: float = 1.065

    # Discovery-only mode: provider runs the full pipeline (embed +
    # search + align) and emits hit metrics, but does NOT inject donor
    # KV indices into match_prefix's device_indices. Useful for
    # measuring semantic-discovery effectiveness in isolation from the
    # realization path.
    discovery_only: bool = False

    def __post_init__(self):
        """Validate configuration values."""
        if self.fuzzy_min_match_length < 1:
            raise ValueError(
                f"fuzzy_min_match_length must be >= 1, got {self.fuzzy_min_match_length}"
            )
        
        if not (0.0 <= self.fuzzy_semantic_threshold <= 1.0):
            raise ValueError(
                f"fuzzy_semantic_threshold must be in [0.0, 1.0], "
                f"got {self.fuzzy_semantic_threshold}"
            )
        
        if self.fuzzy_match_provider not in ("TokenBlockMatch", "SemanticEmbedding"):
            raise ValueError(
                f"fuzzy_match_provider must be 'TokenBlockMatch' or 'SemanticEmbedding', "
                f"got {self.fuzzy_match_provider}"
            )
        
        if self.fuzzy_block_size < 1:
            raise ValueError(
                f"fuzzy_block_size must be >= 1, got {self.fuzzy_block_size}"
            )
        
        if self.fuzzy_non_prefix_max_entries < 1:
            raise ValueError(
                f"fuzzy_non_prefix_max_entries must be >= 1, "
                f"got {self.fuzzy_non_prefix_max_entries}"
            )

        if not (0.0 < self.fuzzy_min_reuse_ratio <= 1.0):
            raise ValueError(
                f"fuzzy_min_reuse_ratio must be in (0.0, 1.0], "
                f"got {self.fuzzy_min_reuse_ratio}"
            )
    
    @classmethod
    def from_server_args(cls, server_args):
        """Create FuzzyMatchConfig from ServerArgs.
        
        Args:
            server_args: ServerArgs instance containing fuzzy match parameters.
            
        Returns:
            FuzzyMatchConfig instance.
        """
        return cls(
            enable_fuzzy_match=getattr(server_args, 'enable_fuzzy_match', False),
            fuzzy_min_match_length=getattr(server_args, 'fuzzy_min_match_length', 16),
            fuzzy_semantic_threshold=getattr(server_args, 'fuzzy_semantic_threshold', 0.60),
            fuzzy_match_provider=getattr(server_args, 'fuzzy_match_provider', 'TokenBlockMatch'),
            cache_fuzzy_results=getattr(server_args, 'cache_fuzzy_results', True),
            fuzzy_eviction_policy=getattr(server_args, 'fuzzy_eviction_policy', 'LRU'),
            fuzzy_non_prefix_max_entries=getattr(server_args, 'fuzzy_non_prefix_max_entries', 10000),
            fuzzy_block_size=getattr(server_args, 'fuzzy_block_size', 16),
            embedding_model_name=getattr(server_args, 'embedding_model_name', 'all-MiniLM-L6-v2'),
            embedding_use_gpu=getattr(server_args, 'embedding_use_gpu', True),
            model_arch=getattr(server_args, 'fuzzy_model_arch', None),
            enable_bathtub=getattr(server_args, 'enable_bathtub', True),
            fuzzy_top_k=getattr(server_args, 'fuzzy_top_k', 5),
            fuzzy_min_reuse_ratio=getattr(server_args, 'fuzzy_min_reuse_ratio', 0.50),
            quality_gate_ppl_threshold=getattr(
                server_args, 'quality_gate_ppl_threshold', 1.065,
            ),
            discovery_only=getattr(server_args, 'fuzzy_discovery_only', False),
        )
