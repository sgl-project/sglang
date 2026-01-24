# SPDX-License-Identifier: Apache-2.0
"""
Cache acceleration module for SGLang multimodal generation.

This module provides various caching strategies to accelerate
diffusion transformer (DiT) inference:

- TeaCache: Temporal similarity-based caching for diffusion models
- cache-dit integration: Block-level caching with DBCache and TaylorSeer

Usage:
    # TeaCache (built into DiT models via TeaCacheMixin)
    from sglang.multimodal_gen.runtime.cache import TeaCacheMixin, TeaCacheContext

    # cache-dit integration (requires cache-dit package installed)
    # These are lazy-loaded and will raise ImportError if cache-dit is not available
    from sglang.multimodal_gen.runtime.cache import (
        CacheDitConfig,
        enable_cache_on_transformer,
        enable_cache_on_dual_transformer,
        get_scm_mask,
    )

    # Or import directly from the submodule:
    from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
        CacheDitConfig,
        enable_cache_on_transformer,
    )
"""

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


# Lazy imports for cache-dit integration (requires cache-dit package)
def __getattr__(name):
    if name in (
        "CacheDitConfig",
        "enable_cache_on_transformer",
        "enable_cache_on_dual_transformer",
        "get_scm_mask",
    ):
        from sglang.multimodal_gen.runtime.cache import cache_dit_integration

        return getattr(cache_dit_integration, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
