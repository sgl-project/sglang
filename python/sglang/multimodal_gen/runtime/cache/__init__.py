# SPDX-License-Identifier: Apache-2.0
"""
Cache acceleration module for SGLang-diffusion

This module provides various caching strategies to accelerate
diffusion transformer (DiT) inference:

- TeaCache: Temporal similarity-based caching for diffusion models
- cache-dit integration: Block-level caching with DBCache and TaylorSeer

"""
from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    CacheDitConfig,
    enable_cache_on_dual_transformer,
    enable_cache_on_transformer,
    get_scm_mask,
)
from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheContext, TeaCacheMixin

__all__ = [
    # TeaCache (always available)
    "TeaCacheContext",
    "TeaCacheMixin",
    # cache-dit integration (lazy-loaded, requires cache-dit package)
    "CacheDitConfig",
    "enable_cache_on_transformer",
    "enable_cache_on_dual_transformer",
    "get_scm_mask",
]
