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

"""
Fuzzy prefix matching support for RadixCache.

This module provides framework for fuzzy/semantic prefix matching,
allowing KV cache reuse for semantically similar but not exactly
matching token sequences.

Usage Example:
    # 1. Create fuzzy match config
    from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
    from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import create_fuzzy_match_provider

    config = FuzzyMatchConfig(
        enable_fuzzy_match=True,
        fuzzy_min_match_length=32,
        fuzzy_match_provider="TokenBlockMatch",
    )

    # 2. Create provider
    provider = create_fuzzy_match_provider(config)

    # 3. Initialize on RadixCache
    radix_cache.init_fuzzy_match(config, provider)

    # 4. Use in matching flow
    exact_result = radix_cache.match_prefix(params)
    if len(exact_result.device_indices) < len(params.key.token_ids) - 1:
        fuzzy_result = radix_cache.match_prefix_fuzzy(params, len(exact_result.device_indices))
        if fuzzy_result is not None:
            # Use fuzzy_result.kv_cache_indices
            pass
"""

from sglang.srt.mem_cache.fuzzy_match.config import FuzzyMatchConfig
from sglang.srt.mem_cache.fuzzy_match.fuzzy_match_provider import (
    FuzzyMatchProvider,
    FuzzyMatchResult,
    create_fuzzy_match_provider,
)
from sglang.srt.mem_cache.fuzzy_match.non_prefix_store import (
    NodeRef,
    NonPrefixEntry,
    NonPrefixKVStore,
)
from sglang.srt.mem_cache.fuzzy_match.rope_correction import (
    as_long_tensor,
    copy_kv_with_rope_correction,
)
from sglang.srt.mem_cache.fuzzy_match.token_block_match import TokenBlockMatchProvider

__all__ = [
    "FuzzyMatchConfig",
    "FuzzyMatchProvider",
    "FuzzyMatchResult",
    "NodeRef",
    "NonPrefixEntry",
    "NonPrefixKVStore",
    "TokenBlockMatchProvider",
    "as_long_tensor",
    "copy_kv_with_rope_correction",
    "create_fuzzy_match_provider",
]
