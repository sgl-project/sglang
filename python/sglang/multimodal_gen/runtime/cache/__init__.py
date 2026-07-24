# SPDX-License-Identifier: Apache-2.0
"""
Cache acceleration module for SGLang-diffusion

This module provides various caching strategies to accelerate
diffusion transformer (DiT) inference:

- TeaCache: Temporal similarity-based caching for diffusion models
- cache-dit integration: Block-level caching with DBCache and TaylorSeer
- EasyCache: request-scoped block-residual reuse driven by online rate est.
- LTX-2 acceleration suite: stage-1 SCSP core, TeaCache residual replay,
  Pyramid Attention Broadcast, and the cache-dit BlockAdapter registration.

"""

from sglang.multimodal_gen.runtime.cache.cache_dit_integration import (
    CacheDitConfig,
    enable_cache_on_dual_transformer,
    enable_cache_on_transformer,
    get_scm_mask,
)
from sglang.multimodal_gen.runtime.cache.easycache import (
    EasyCacheController,
    EasyCacheDecision,
    EasyCacheState,
)
from sglang.multimodal_gen.runtime.cache.ltx2_pab import (
    LTX2PABAttentionKind,
    LTX2PABConfig,
    LTX2PABCoordinator,
    LTX2PABMixin,
    install_ltx2_pab_hooks,
)
from sglang.multimodal_gen.runtime.cache.ltx2_stage1_cache_core import (
    LTX2Stage1CacheCoreController,
    make_ltx2_stage1_cache_core_from_env,
)
from sglang.multimodal_gen.runtime.cache.ltx2_teacache import (
    LTX2TeaCacheConfig,
    LTX2TeaCacheCoordinator,
    LTX2TeaCacheDecision,
    get_ltx2_teacache_coordinator,
    make_ltx2_teacache_config_from_env,
)
from sglang.multimodal_gen.runtime.cache.teacache import TeaCacheContext, TeaCacheMixin

__all__ = [
    # cache-dit integration (lazy-loaded, requires cache-dit package)
    "CacheDitConfig",
    # EasyCache
    "EasyCacheController",
    "EasyCacheDecision",
    "EasyCacheState",
    # LTX-2 PAB
    "LTX2PABAttentionKind",
    "LTX2PABConfig",
    "LTX2PABCoordinator",
    "LTX2PABMixin",
    # LTX-2 stage-1 SCSP cache core
    "LTX2Stage1CacheCoreController",
    # LTX-2 model-level TeaCache
    "LTX2TeaCacheConfig",
    "LTX2TeaCacheCoordinator",
    "LTX2TeaCacheDecision",
    # TeaCache (always available)
    "TeaCacheContext",
    "TeaCacheMixin",
    "enable_cache_on_dual_transformer",
    "enable_cache_on_transformer",
    "get_scm_mask",
    "get_ltx2_teacache_coordinator",
    "install_ltx2_pab_hooks",
    "make_ltx2_stage1_cache_core_from_env",
    "make_ltx2_teacache_config_from_env",
]
