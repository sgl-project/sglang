import sys
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


# lmcache and flexkv raise at import time when their package is missing, so the
# import is done under stand-in modules. Only the third-party surface is faked;
# the load-back arithmetic under test is the real one.
_LMCACHE_MODULES = [
    "lmcache",
    "lmcache.integration",
    "lmcache.integration.sglang",
    "lmcache.integration.sglang.multi_process_adapter",
    "lmcache.integration.sglang.sglang_adapter",
    "lmcache.integration.sglang.utils",
]

_FLEXKV_MODULES = [
    "flexkv",
    "flexkv.common",
    "flexkv.common.request",
    "flexkv.common.storage",
    "flexkv.integration",
    "flexkv.integration.config",
    "flexkv.kvmanager",
    "flexkv.server",
    "flexkv.server.client",
    "flexkv.transfer",
    "flexkv.transfer.layerwise",
    "flexkv.transfer_manager",
]


def _import_lmc_radix_cache():
    with patch.dict(sys.modules, {name: MagicMock() for name in _LMCACHE_MODULES}):
        from sglang.srt.mem_cache.storage.lmcache.lmc_radix_cache import LMCRadixCache

        return LMCRadixCache


def _import_flexkv_radix_cache():
    with patch.dict(sys.modules, {name: MagicMock() for name in _FLEXKV_MODULES}):
        from sglang.srt.mem_cache.storage.flexkv.flexkv_radix_cache import (
            FlexKVRadixCache,
        )

        return FlexKVRadixCache


class _FakeAllocator:
    def __init__(self, page_size: int, slots: torch.Tensor):
        self.page_size = page_size
        self.slots = slots
        self.freed = []

    def available_size(self):
        return self.slots.numel()

    def alloc(self, need_size: int):
        return self.slots[:need_size]

    def free(self, free_index: torch.Tensor):
        self.freed.append(free_index.tolist())


def _make_lmc_cache(
    *, alloc_page_size: int, cache_page_size: int, chunk_size: int, slots: torch.Tensor
):
    cls = _import_lmc_radix_cache()
    cache = object.__new__(cls)
    cache.token_to_kv_pool_allocator = _FakeAllocator(alloc_page_size, slots)
    cache.lmcache_connector = SimpleNamespace(chunk_size=lambda: chunk_size)
    cache.device = "cpu"
    cache.page_size = cache_page_size
    cache.evictable_size_ = 0
    cache.evictable_leaves = set()
    cache.enable_kv_cache_events = False
    return cache


def _load_back(cache, *, value_numel: int, uncached_len: int, num_retrieved: int):
    key = RadixKey(array("q", list(range(value_numel + uncached_len))))
    return cache._load_back(
        key=key,
        value_numel=value_numel,
        uncached_len=uncached_len,
        last_node=TreeNode(),
        load_fn=lambda slot_mapping, prefix_pad: num_retrieved,
    )


def test_lmc_load_back_is_unchanged_when_the_fetch_is_already_aligned():
    """An aligned fetch must floor to itself: same node, same free, same return."""
    slots = torch.arange(100, 116, dtype=torch.int64)
    cache = _make_lmc_cache(
        alloc_page_size=4, cache_page_size=4, chunk_size=4, slots=slots
    )

    # value_numel=8 -> prefix_pad = 8 % 4 = 0, so fetched = num_retrieved = 12.
    result = _load_back(cache, value_numel=8, uncached_len=16, num_retrieved=12)

    slot_values, node = result
    assert slot_values.tolist() == list(range(100, 112))
    assert node.value.tolist() == list(range(100, 112))
    assert cache.evictable_size_ == 12
    assert cache.token_to_kv_pool_allocator.freed == [list(range(112, 116))]


def test_lmc_load_back_accepts_a_chunk_size_that_does_not_divide_the_page():
    """page=6 with chunk=4 still lands an aligned fetch=6; it must not be rejected."""
    slots = torch.arange(100, 118, dtype=torch.int64)
    cache = _make_lmc_cache(
        alloc_page_size=6, cache_page_size=6, chunk_size=4, slots=slots
    )

    # prefix_pad = 6 % 4 = 2, so fetched = 8 - 2 = 6, an exact page.
    result = _load_back(cache, value_numel=6, uncached_len=12, num_retrieved=8)

    slot_values, node = result
    assert slot_values.tolist() == list(range(100, 106))
    assert node.value.tolist() == list(range(100, 106))
    assert cache.evictable_size_ == 6


def test_lmc_load_back_floors_an_unaligned_fetch_to_the_page():
    """fetched=6 with page=4 must keep only [0, 4) and free from 4 on, so no page is half-owned."""
    slots = torch.arange(100, 116, dtype=torch.int64)
    cache = _make_lmc_cache(
        alloc_page_size=4, cache_page_size=4, chunk_size=4, slots=slots
    )

    # prefix_pad = 0, num_retrieved = 6 -> fetched floors 6 down to 4.
    result = _load_back(cache, value_numel=8, uncached_len=16, num_retrieved=6)

    slot_values, node = result
    assert slot_values.tolist() == list(range(100, 104))
    assert node.value.tolist() == list(range(100, 104))
    assert len(node.key) == 4
    assert cache.evictable_size_ == 4
    assert cache.token_to_kv_pool_allocator.freed == [list(range(104, 116))]


def test_lmc_load_back_gives_up_when_the_fetch_floors_to_zero():
    """A fetch shorter than one page owns nothing; everything is freed and None returned."""
    slots = torch.arange(100, 116, dtype=torch.int64)
    cache = _make_lmc_cache(
        alloc_page_size=8, cache_page_size=8, chunk_size=8, slots=slots
    )

    result = _load_back(cache, value_numel=8, uncached_len=16, num_retrieved=6)

    assert result is None
    assert cache.token_to_kv_pool_allocator.freed == [list(range(100, 116))]


def test_lmc_load_back_handles_a_fetch_shorter_than_the_prefix_pad():
    """num_retrieved below prefix_pad must not slice with a negative index."""
    slots = torch.arange(100, 116, dtype=torch.int64)
    cache = _make_lmc_cache(
        alloc_page_size=4, cache_page_size=4, chunk_size=8, slots=slots
    )

    # prefix_pad = 6 % 8 = 6, num_retrieved = 2 -> max(2 - 6, 0) = 0.
    result = _load_back(cache, value_numel=6, uncached_len=16, num_retrieved=2)

    assert result is None
    assert cache.token_to_kv_pool_allocator.freed == [list(range(100, 116))]


def test_lmc_load_back_floors_on_the_allocator_page_not_the_cache_page():
    """The cache's page (1) must not win over the allocator's page (4) as the floor."""
    slots = torch.arange(100, 116, dtype=torch.int64)
    cache = _make_lmc_cache(
        alloc_page_size=4, cache_page_size=1, chunk_size=4, slots=slots
    )

    result = _load_back(cache, value_numel=8, uncached_len=16, num_retrieved=6)

    slot_values, _ = result
    assert slot_values.tolist() == list(range(100, 104))


def _make_flexkv_cache(*, alloc_page_size: int, slots: torch.Tensor):
    cls = _import_flexkv_radix_cache()
    cache = object.__new__(cls)
    cache.token_to_kv_pool_allocator = _FakeAllocator(alloc_page_size, slots)
    cache.device = "cpu"
    cache.page_size = alloc_page_size
    cache.evictable_size_ = 0
    cache.evictable_leaves = set()
    cache.enable_kv_cache_events = False
    return cache


def _allocate_and_load(
    cache, *, value_numel: int, uncached_len: int, num_retrieved: int
):
    key = RadixKey(array("q", list(range(value_numel + uncached_len))))
    return cache._allocate_and_load(
        key=key,
        value_numel=value_numel,
        uncached_len=uncached_len,
        last_node=TreeNode(),
        load_fn=lambda token_slots: num_retrieved,
    )


def test_flexkv_load_leaves_an_aligned_retrieval_unchanged():
    """FlexKV blocks are sglang pages, so an aligned retrieval passes through untouched."""
    slots = torch.arange(100, 116, dtype=torch.int64)
    cache = _make_flexkv_cache(alloc_page_size=4, slots=slots)

    result = _allocate_and_load(cache, value_numel=8, uncached_len=16, num_retrieved=12)

    slot_values, node = result
    assert slot_values.tolist() == list(range(100, 112))
    assert node.value.tolist() == list(range(100, 112))
    assert cache.token_to_kv_pool_allocator.freed == [list(range(112, 116))]


def test_flexkv_load_rejects_an_unaligned_retrieval():
    """FlexKV registers its KV layout with tokens_per_block=page_size, so this cannot happen."""
    slots = torch.arange(100, 116, dtype=torch.int64)
    cache = _make_flexkv_cache(alloc_page_size=4, slots=slots)

    with pytest.raises(AssertionError):
        _allocate_and_load(cache, value_numel=8, uncached_len=16, num_retrieved=6)


if __name__ == "__main__":
    import sys as _sys

    _sys.exit(pytest.main([__file__, "-v"]))
