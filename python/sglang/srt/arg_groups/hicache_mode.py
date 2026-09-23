# SPDX-License-Identifier: Apache-2.0
"""Predicates over the HiCache mode (cache, buffer_only, linker)."""

from __future__ import annotations

from typing import Any

HICACHE_LINKER_BACKENDS = ("mooncake", "mori")


def hicache_uses_linker(cfg: Any) -> bool:
    """The unified radix cache links device pools straight to the storage backend."""
    return (
        bool(cfg.enable_hierarchical_cache) and cfg.hicache_host_memory_mode == "linker"
    )


def hicache_has_host_tier(cfg: Any) -> bool:
    """HiCache runs a host memory pool (cache or buffer_only mode)."""
    return (
        bool(cfg.enable_hierarchical_cache) and cfg.hicache_host_memory_mode != "linker"
    )
