# SPDX-License-Identifier: Apache-2.0

import torch

from sglang.multimodal_gen.runtime.vla.prefix_cache import (
    PrefixContext,
    VLAPrefixCacheManager,
)


def _context() -> PrefixContext:
    return PrefixContext(
        past_key_values=("kv",),
        prefix_pad_masks=torch.ones(1, 3, dtype=torch.bool),
        prefix_len=3,
    )


def test_pi05_prefix_cache_full_hit_returns_context():
    manager = VLAPrefixCacheManager(max_entries=4)
    key = "a"
    context = _context()

    manager.put(key, context)
    cached = manager.get(key)

    assert cached is context
    assert context.cache_key_digest == key


def test_pi05_prefix_cache_different_key_misses():
    manager = VLAPrefixCacheManager(max_entries=4)
    manager.put("a", _context())

    assert manager.get("b") is None


def test_pi05_prefix_cache_zero_capacity_does_not_retain_context():
    manager = VLAPrefixCacheManager(max_entries=0)
    context = _context()

    manager.put("a", context)

    assert manager.get("a") is None
    assert context.cache_key_digest is None


def test_pi05_prefix_cache_evicts_least_recently_used_entry():
    manager = VLAPrefixCacheManager(max_entries=2)
    context_a = _context()
    context_b = _context()
    context_c = _context()
    manager.put("a", context_a)
    manager.put("b", context_b)
    assert manager.get("a") is context_a

    manager.put("c", context_c)

    assert manager.get("a") is context_a
    assert manager.get("b") is None
    assert manager.get("c") is context_c
