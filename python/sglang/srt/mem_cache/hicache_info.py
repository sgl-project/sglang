# Copyright 2026 SGLang Team
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
# ==============================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HiCacheInfo:
    """Realized per-DP-rank HiCache capacity in logical tokens."""

    host_total_tokens: int


def get_hicache_host_pool(tree_cache: Any) -> Any:
    """Return the host pool that owns the radix cache's logical token space."""
    host_pool = getattr(tree_cache, "token_to_kv_pool_host", None)
    if host_pool is None:
        host_pool = getattr(tree_cache, "full_kv_pool_host", None)
    if host_pool is None:
        raise RuntimeError("HiCache is enabled but no host pool was found")
    return host_pool


def get_hicache_info(tree_cache: Any) -> HiCacheInfo:
    """Build the public HiCache snapshot from the allocated host pool."""
    return build_hicache_info(get_hicache_host_pool(tree_cache))


def build_hicache_info(host_pool: Any) -> HiCacheInfo:
    """Build the public HiCache snapshot from a resolved host pool."""
    return HiCacheInfo(host_total_tokens=int(host_pool.logical_size))
