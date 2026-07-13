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

from typing import Optional

import msgspec


class FuzzyMatchConfig(msgspec.Struct):
    """Fuzzy matching config. All fuzzy matching is opt-in."""

    # Enable fuzzy prefix matching
    enable_fuzzy_match: bool = False

    # Minimum token span a provider may reuse. Partial exact anchors shorter
    # than this are skipped; zero exact-prefix matches remain eligible.
    fuzzy_min_match_length: int = 16

    # Cosine-similarity threshold for SemanticEmbedding matches.
    # Range [0.0, 1.0].
    # Higher = stricter (fewer matches, higher precision); lower =
    # more permissive. Below ~0.50 alignment quality drops quickly.
    fuzzy_semantic_threshold: float = 0.60

    # Provider class for fuzzy matching logic.
    fuzzy_match_provider: str = "SemanticEmbedding"

    # Cache fuzzy match results for future reuse
    cache_fuzzy_results: bool = True

    # Maximum donors kept by the semantic provider.
    semantic_max_entries: int = 10000

    # Provider-internal donor chunk size.
    fuzzy_block_size: int = 16

    # Embedding model name for SemanticEmbeddingProvider
    embedding_model_name: str = "all-MiniLM-L6-v2"

    # SemanticEmbedding-only: model architecture tag for SemBlend presets.
    model_arch: Optional[str] = None

    # SemanticEmbedding-only: minimum covered prompt fraction for a hit.
    fuzzy_min_reuse_ratio: float = 0.50

    # Skip the provider lookup entirely when the exact-miss suffix is
    # shorter than this. Short suffixes cannot amortize the semantic
    # lookup (embedding the prompt costs more than the prefill it could
    # save), so this bounds the no-hit overhead on workloads without
    # long reusable content.
    fuzzy_min_suffix_tokens: int = 256

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

        if self.fuzzy_match_provider not in ("SemanticEmbedding",):
            raise ValueError(
                f"fuzzy_match_provider must be 'SemanticEmbedding', "
                f"got {self.fuzzy_match_provider}"
            )

        if self.fuzzy_block_size < 1:
            raise ValueError(
                f"fuzzy_block_size must be >= 1, got {self.fuzzy_block_size}"
            )

        if self.semantic_max_entries < 1:
            raise ValueError(
                f"semantic_max_entries must be >= 1, "
                f"got {self.semantic_max_entries}"
            )

        if not (0.0 < self.fuzzy_min_reuse_ratio <= 1.0):
            raise ValueError(
                f"fuzzy_min_reuse_ratio must be in (0.0, 1.0], "
                f"got {self.fuzzy_min_reuse_ratio}"
            )

        if self.fuzzy_min_suffix_tokens < 0:
            raise ValueError(
                f"fuzzy_min_suffix_tokens must be >= 0, "
                f"got {self.fuzzy_min_suffix_tokens}"
            )

    @classmethod
    def from_server_args(cls, server_args) -> "FuzzyMatchConfig":
        """Create a config from ServerArgs.

        Only called by the ``fuzzy_match`` radix-cache backend factory, so
        selecting the backend is what enables fuzzy matching; there is no
        separate enable flag.
        """
        return cls(
            enable_fuzzy_match=True,
            fuzzy_min_match_length=server_args.fuzzy_min_match_length,
            fuzzy_semantic_threshold=server_args.fuzzy_semantic_threshold,
            fuzzy_match_provider=server_args.fuzzy_match_provider,
            model_arch=server_args.fuzzy_model_arch,
            fuzzy_min_reuse_ratio=server_args.fuzzy_min_reuse_ratio,
        )
