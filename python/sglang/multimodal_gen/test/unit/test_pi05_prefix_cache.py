# SPDX-License-Identifier: Apache-2.0

from array import array

import torch

from sglang.multimodal_gen.runtime.cache.vla_prefix_cache import (
    PrefixContext,
    VLAPrefixCacheKey,
    VLAPrefixCacheManager,
)
from sglang.srt.mem_cache.radix_cache import RadixKey


def _key(values: list[int], digest: str) -> VLAPrefixCacheKey:
    radix_key = RadixKey(array("q", values), extra_key="pi05-test")
    return VLAPrefixCacheKey(
        digest=digest,
        radix_key=radix_key,
        full_prefix_len=len(radix_key),
    )


def _context() -> PrefixContext:
    return PrefixContext(
        past_key_values=("kv",),
        prefix_pad_masks=torch.ones(1, 3, dtype=torch.bool),
        prefix_position_ids=torch.arange(3).unsqueeze(0),
        prefix_len=3,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )


def test_pi05_prefix_cache_full_hit_returns_context():
    manager = VLAPrefixCacheManager(max_entries=4)
    key = _key([1, 2, 3], "a")
    context = _context()

    manager.put(key, context)
    lookup = manager.get(key)

    assert lookup.hit
    assert lookup.context is context
    assert lookup.match_len == key.full_prefix_len


def test_pi05_prefix_cache_rejects_partial_hit():
    manager = VLAPrefixCacheManager(max_entries=4)
    manager.put(_key([1, 2, 3], "a"), _context())

    lookup = manager.get(_key([1, 2, 4], "b"))

    assert not lookup.hit
    assert lookup.context is None
    assert lookup.match_len == 2
    assert lookup.partial_rejected


def test_pi05_prefix_cache_different_key_does_not_collide():
    manager = VLAPrefixCacheManager(max_entries=4)
    manager.put(_key([1, 2, 3], "a"), _context())

    lookup = manager.get(_key([9, 9, 9], "b"))

    assert not lookup.hit
    assert lookup.match_len == 0
    assert not lookup.partial_rejected
