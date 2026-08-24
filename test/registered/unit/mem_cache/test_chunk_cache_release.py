"""Unit tests for the split, sync-free KV release in the chunk caches.

``SWAChunkCache.cache_finished_req`` used to free one range through
``free()``, which rediscovers pages with ``torch.unique`` and reads the SWA
mapping back -- both data-dependent, so both sync the device. It now names each
pool's range instead: the full pool owns ``[cache_protected_len, kv_len)`` while
the SWA pool only still holds ``[swa_evicted_seqlen, kv_len)``, because window
eviction already released what came before.

Covered here:
  - ``ChunkCache.cache_finished_req`` releases the same pages as ``free()``;
  - the SWA split frees each pool exactly once, and never re-frees the range the
    window eviction already took;
  - an aliased-pool allocator (``PureSWATokenToKVPoolAllocator`` sets
    ``full_attn_allocator = swa_attn_allocator``) must NOT take the split path,
    or the same slots land on the free list twice;
  - ``PureSWAChunkCache`` skips the window-evicted range via ``free_segments``.

    python -m pytest test/registered/unit/mem_cache/test_chunk_cache_release.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator.swa import (
    PureSWATokenToKVPoolAllocator,
    SWATokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import (
    ChunkCache,
    PureSWAChunkCache,
    SWAChunkCache,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

_DTYPE = torch.bfloat16
_HEAD_NUM = 1
_HEAD_DIM = 8
_DEVICE = "cpu"
_WINDOW = 128
_NUM_REQS = 2
_ROW_PAGES = 8


def _build(page_size: int, *, pure: bool = False):
    """Pools plus a filled ``req_to_token``, as a running batch would have."""
    row_len = _ROW_PAGES * page_size
    kv_size = (_NUM_REQS * _ROW_PAGES + 4) * page_size
    pool = ReqToTokenPool(
        size=_NUM_REQS,
        max_context_len=row_len,
        device=_DEVICE,
        enable_memory_saver=False,
    )
    kv_pool = SWAKVPool(
        size=kv_size,
        size_swa=kv_size,
        page_size=page_size,
        dtype=_DTYPE,
        head_num=_HEAD_NUM,
        head_dim=_HEAD_DIM,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        device=_DEVICE,
    )
    if pure:
        allocator = PureSWATokenToKVPoolAllocator(
            size_swa=kv_size,
            page_size=page_size,
            dtype=_DTYPE,
            device=_DEVICE,
            kvcache=kv_pool,
            need_sort=False,
        )
        indices = allocator.alloc(_NUM_REQS * row_len)
    else:
        allocator = SWATokenToKVPoolAllocator(
            size=kv_size,
            size_swa=kv_size,
            page_size=page_size,
            dtype=_DTYPE,
            device=_DEVICE,
            kvcache=kv_pool,
            need_sort=False,
        )
        need = _NUM_REQS * row_len
        indices = allocator.full_attn_allocator.alloc(need)
        swa_indices = allocator.swa_attn_allocator.alloc(need)
        assert indices is not None and swa_indices is not None, "kv pool too small"
        allocator.full_to_swa_index_mapping[indices] = swa_indices
    assert indices is not None, "kv pool too small"
    for i in range(_NUM_REQS):
        pool.write((i, slice(0, row_len)), indices[i * row_len : (i + 1) * row_len])
    return pool, allocator, row_len


def _cache(cls, pool, allocator, page_size):
    return cls(
        CacheInitParams(
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            disable=False,
            is_eagle=False,
            sliding_window_size=_WINDOW,
        )
    )


def _make_req(req_pool_idx, *, protected_len=0, swa_evicted=0, evict_floor=0):
    return SimpleNamespace(
        req_pool_idx=req_pool_idx,
        cache_protected_len=protected_len,
        swa_evict_floor=evict_floor,
        kv=SimpleNamespace(swa_evicted_seqlen=swa_evicted),
    )


def _all_free_pages(sub_allocator) -> torch.Tensor:
    return torch.cat((sub_allocator.free_pages, sub_allocator.release_pages))


def _assert_no_duplicates(test, sub_allocator, msg=""):
    pages = _all_free_pages(sub_allocator)
    test.assertEqual(
        pages.numel(), torch.unique(pages).numel(), f"duplicate free pages {msg}"
    )


class TestChunkCacheRelease(CustomTestCase):
    PAGE = 64

    def test_chunk_cache_frees_same_pages_as_free(self):
        for protected_len in (0, 2 * self.PAGE):
            ref_pool, ref_alloc, row_len = _build(self.PAGE)
            new_pool, new_alloc, _ = _build(self.PAGE)
            req = _make_req(0, protected_len=protected_len)

            row = ref_pool.req_to_token[0]
            ref_alloc.free(row[protected_len:row_len])
            _cache(ChunkCache, new_pool, new_alloc, self.PAGE).cache_finished_req(
                req, kv_len_to_handle=row_len
            )

            self.assertEqual(
                ref_alloc.full_available_size(), new_alloc.full_available_size()
            )
            self.assertEqual(
                ref_alloc.swa_available_size(), new_alloc.swa_available_size()
            )
            _assert_no_duplicates(self, new_alloc.full_attn_allocator)


class TestSWAChunkCacheRelease(CustomTestCase):
    PAGE = 64

    def test_split_release_matches_free_when_nothing_was_evicted(self):
        ref_pool, ref_alloc, row_len = _build(self.PAGE)
        new_pool, new_alloc, _ = _build(self.PAGE)
        req = _make_req(0)

        ref_alloc.free(ref_pool.req_to_token[0][:row_len])
        _cache(SWAChunkCache, new_pool, new_alloc, self.PAGE).cache_finished_req(
            req, kv_len_to_handle=row_len
        )

        self.assertEqual(
            ref_alloc.full_available_size(), new_alloc.full_available_size()
        )
        self.assertEqual(ref_alloc.swa_available_size(), new_alloc.swa_available_size())
        self.assertTrue(
            torch.equal(
                ref_alloc.full_to_swa_index_mapping, new_alloc.full_to_swa_index_mapping
            )
        )

    def test_window_evicted_swa_range_is_not_freed_twice(self):
        """The SWA lower bound is swa_evicted_seqlen, not cache_protected_len."""
        pool, allocator, row_len = _build(self.PAGE)
        evicted = 3 * self.PAGE
        row = pool.req_to_token[0]

        # what maybe_evict_swa already did for this request
        allocator.free_swa_segment(row[:evicted], start_pos=0)
        swa_after_evict = allocator.swa_available_size()
        full_before = allocator.full_available_size()

        req = _make_req(0, swa_evicted=evicted)
        _cache(SWAChunkCache, pool, allocator, self.PAGE).cache_finished_req(
            req, kv_len_to_handle=row_len
        )

        self.assertEqual(full_before + row_len, allocator.full_available_size())
        self.assertEqual(
            swa_after_evict + (row_len - evicted), allocator.swa_available_size()
        )
        _assert_no_duplicates(self, allocator.swa_attn_allocator, "(swa pool)")
        _assert_no_duplicates(self, allocator.full_attn_allocator, "(full pool)")

    def test_protected_prefix_is_not_freed(self):
        pool, allocator, row_len = _build(self.PAGE)
        protected = 2 * self.PAGE
        full_before = allocator.full_available_size()

        req = _make_req(0, protected_len=protected)
        _cache(SWAChunkCache, pool, allocator, self.PAGE).cache_finished_req(
            req, kv_len_to_handle=row_len
        )

        self.assertEqual(
            full_before + (row_len - protected), allocator.full_available_size()
        )

    def test_aliased_pools_do_not_double_free(self):
        """PureSWA aliases the two pools: the split path would free twice."""
        pool, allocator, row_len = _build(1, pure=True)
        self.assertIs(allocator.full_attn_allocator, allocator.swa_attn_allocator)
        avail_before = allocator.available_size()

        req = _make_req(0)
        _cache(SWAChunkCache, pool, allocator, 1).cache_finished_req(
            req, kv_len_to_handle=row_len
        )

        self.assertEqual(avail_before + row_len, allocator.available_size())
        _assert_no_duplicates(self, allocator.swa_attn_allocator, "(aliased pool)")


class TestPureSWAChunkCacheRelease(CustomTestCase):
    def test_skips_the_window_evicted_range(self):
        pool, allocator, row_len = _build(1, pure=True)
        evict_floor, evicted = 2, 5
        row = pool.req_to_token[0]

        allocator.free_swa(row[evict_floor:evicted])
        avail_after_evict = allocator.available_size()

        req = _make_req(0, swa_evicted=evicted, evict_floor=evict_floor)
        _cache(PureSWAChunkCache, pool, allocator, 1).cache_finished_req(
            req, kv_len_to_handle=row_len
        )

        # [0, evict_floor) plus [evicted, row_len): everything except the range
        # the window eviction already released.
        expected = avail_after_evict + evict_floor + (row_len - evicted)
        self.assertEqual(expected, allocator.available_size())
        _assert_no_duplicates(self, allocator.swa_attn_allocator)

    def test_frees_whole_row_when_nothing_was_evicted(self):
        pool, allocator, row_len = _build(1, pure=True)
        avail_before = allocator.available_size()

        req = _make_req(0)
        _cache(PureSWAChunkCache, pool, allocator, 1).cache_finished_req(
            req, kv_len_to_handle=row_len
        )

        self.assertEqual(avail_before + row_len, allocator.available_size())
        _assert_no_duplicates(self, allocator.swa_attn_allocator)


if __name__ == "__main__":
    unittest.main()
