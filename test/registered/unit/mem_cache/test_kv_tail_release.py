"""Unit tests for the tail release of over-allocated KV and the window eviction.

Both paths used to end in ``free()`` / ``free_swa()``, which rediscover pages
with ``torch.unique`` and (for SWA) read the mapping back -- data-dependent
shapes, so a device sync each, and behind the WAR-fenced schedule stream a sync
costs a whole forward. They now name the range instead.

Covered here:
  - ``_release_overallocated_kv_indices``: the SWA split path (chunk cache,
    ``page_size > 1``) frees both pools exactly once, and the fallback path
    (``page_size == 1``, or a radix cache, or a plain allocator) still frees
    exactly what ``free()`` did;
  - ``free_swa_out_of_window_slots``: the segment release advances the frontier
    and frees the same SWA pages as the pre-existing ``free_swa()`` path.

    python -m pytest test/registered/unit/mem_cache/test_kv_tail_release.py -v
"""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

import sglang.srt.mem_cache.common as common_mod
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache, SWAChunkCache
from sglang.srt.mem_cache.common import (
    _release_overallocated_kv_indices,
    free_swa_out_of_window_slots,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=40, suite="base-a-test-cpu")

_DTYPE = torch.bfloat16
_HEAD_NUM = 1
_HEAD_DIM = 8
_DEVICE = "cpu"
_WINDOW = 128
_NUM_REQS = 2
_ROW_PAGES = 12


def _build(page_size: int):
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
    full = allocator.full_attn_allocator.alloc(need)
    swa = allocator.swa_attn_allocator.alloc(need)
    assert full is not None and swa is not None, "kv pool too small"
    allocator.full_to_swa_index_mapping[full] = swa
    for i in range(_NUM_REQS):
        pool.write((i, slice(0, row_len)), full[i * row_len : (i + 1) * row_len])
    return pool, allocator, row_len


def _cache(cls, pool, allocator, page_size, **extra):
    return cls(
        CacheInitParams(
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=page_size,
            disable=False,
            is_eagle=False,
            sliding_window_size=_WINDOW,
            **extra,
        )
    )


def _make_req(req_pool_idx=0, *, protected_len=0, swa_evicted=0, seq_len=0):
    return SimpleNamespace(
        req_pool_idx=req_pool_idx,
        cache_protected_len=protected_len,
        swa_evict_floor=0,
        seqlen=seq_len,
        kv_committed_len=0,
        kv=SimpleNamespace(swa_evicted_seqlen=swa_evicted, kv_allocated_len=seq_len),
    )


@contextmanager
def _spec_decode_enabled():
    """An over-allocated tail only exists under speculative decoding (or
    strip_thinking_cache); without one, the function asserts start == end."""
    with mock.patch.object(
        common_mod, "get_spec", lambda: SimpleNamespace(speculative_algorithm="EAGLE")
    ):
        yield


def _assert_no_duplicates(test, sub_allocator, msg=""):
    pages = torch.cat((sub_allocator.free_pages, sub_allocator.release_pages))
    test.assertEqual(
        pages.numel(), torch.unique(pages).numel(), f"duplicate free pages {msg}"
    )


class TestReleaseOverallocatedKvIndices(CustomTestCase):
    PAGE = 64

    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_chunk_cache_split_release_frees_both_pools_once(self):
        ref_pool, ref_alloc, _ = _build(self.PAGE)
        new_pool, new_alloc, _ = _build(self.PAGE)
        start_p, end_p = 4 * self.PAGE, 7 * self.PAGE

        ref_alloc.free(ref_pool.req_to_token[0][start_p:end_p])
        with _spec_decode_enabled():
            _release_overallocated_kv_indices(
                _make_req(),
                start_p,
                end_p,
                _cache(SWAChunkCache, new_pool, new_alloc, self.PAGE),
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
        _assert_no_duplicates(self, new_alloc.swa_attn_allocator, "(swa)")
        _assert_no_duplicates(self, new_alloc.full_attn_allocator, "(full)")

    def test_unaligned_start_is_ceil_aligned_to_a_page(self):
        """A tail that starts mid-page must not release that shared page."""
        pool, allocator, _ = _build(self.PAGE)
        start_p, end_p = 4 * self.PAGE + 1, 7 * self.PAGE
        full_before = allocator.full_available_size()

        with _spec_decode_enabled():
            _release_overallocated_kv_indices(
                _make_req(),
                start_p,
                end_p,
                _cache(SWAChunkCache, pool, allocator, self.PAGE),
            )

        # ceil_align(start) == 5 pages in, so pages 5 and 6 are freed, not 4.
        self.assertEqual(full_before + 2 * self.PAGE, allocator.full_available_size())
        _assert_no_duplicates(self, allocator.full_attn_allocator)

    def test_empty_tail_frees_nothing(self):
        pool, allocator, _ = _build(self.PAGE)
        full_before = allocator.full_available_size()
        swa_before = allocator.swa_available_size()
        start_p = end_p = 4 * self.PAGE

        with _spec_decode_enabled():
            _release_overallocated_kv_indices(
                _make_req(),
                start_p,
                end_p,
                _cache(SWAChunkCache, pool, allocator, self.PAGE),
            )

        self.assertEqual(full_before, allocator.full_available_size())
        self.assertEqual(swa_before, allocator.swa_available_size())

    def test_page_size_one_takes_the_fallback_path(self):
        ref_pool, ref_alloc, _ = _build(1)
        new_pool, new_alloc, _ = _build(1)
        start_p, end_p = 4, 9

        ref_alloc.free(ref_pool.req_to_token[0][start_p:end_p])
        with _spec_decode_enabled():
            _release_overallocated_kv_indices(
                _make_req(),
                start_p,
                end_p,
                _cache(SWAChunkCache, new_pool, new_alloc, 1),
            )

        self.assertEqual(
            ref_alloc.full_available_size(), new_alloc.full_available_size()
        )
        self.assertEqual(ref_alloc.swa_available_size(), new_alloc.swa_available_size())
        _assert_no_duplicates(self, new_alloc.swa_attn_allocator)

    def test_plain_chunk_cache_allocator_takes_the_fallback_path(self):
        """No free_swa_segment on the allocator -> plain free_segment."""
        ref_pool, ref_alloc, _ = _build(self.PAGE)
        new_pool, new_alloc, _ = _build(self.PAGE)
        start_p, end_p = 4 * self.PAGE, 6 * self.PAGE
        sub = new_alloc.full_attn_allocator
        self.assertFalse(hasattr(sub, "free_swa_segment"))

        ref_alloc.full_attn_allocator.free(ref_pool.req_to_token[0][start_p:end_p])
        with _spec_decode_enabled():
            _release_overallocated_kv_indices(
                _make_req(),
                start_p,
                end_p,
                _cache(ChunkCache, new_pool, sub, self.PAGE),
            )

        self.assertEqual(
            ref_alloc.full_attn_allocator.available_size(), sub.available_size()
        )
        _assert_no_duplicates(self, sub)


class TestFreeSwaOutOfWindowSlots(CustomTestCase):
    PAGE = 64

    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _run(self, page_size, seq_len):
        pool, allocator, _ = _build(page_size)
        req = _make_req(seq_len=seq_len)
        free_swa_out_of_window_slots(
            req,
            seq_len,
            sliding_window_size=_WINDOW,
            page_size=page_size,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            is_chunk_cache=True,
        )
        return allocator, req

    def test_segment_release_matches_free_swa(self):
        for page_size in (1, self.PAGE):
            seq_len = _WINDOW + 4 * page_size
            new_alloc, new_req = self._run(page_size, seq_len)

            # reference: same frontier, released through free_swa()
            ref_pool, ref_alloc, _ = _build(page_size)
            frontier = new_req.kv.swa_evicted_seqlen
            self.assertGreater(frontier, 0, f"nothing evicted for {page_size=}")
            ref_alloc.free_swa(ref_pool.req_to_token[0][:frontier])

            self.assertEqual(
                ref_alloc.swa_available_size(), new_alloc.swa_available_size()
            )
            self.assertEqual(
                ref_alloc.full_available_size(), new_alloc.full_available_size()
            )
            self.assertTrue(
                torch.equal(
                    ref_alloc.full_to_swa_index_mapping,
                    new_alloc.full_to_swa_index_mapping,
                )
            )
            _assert_no_duplicates(self, new_alloc.swa_attn_allocator)

    def test_frontier_is_page_aligned(self):
        seq_len = _WINDOW + 3 * self.PAGE + 7
        _, req = self._run(self.PAGE, seq_len)

        self.assertEqual(0, req.kv.swa_evicted_seqlen % self.PAGE)

    def test_inside_window_frees_nothing(self):
        pool, allocator, _ = _build(self.PAGE)
        swa_before = allocator.swa_available_size()
        req = _make_req(seq_len=_WINDOW // 2)

        free_swa_out_of_window_slots(
            req,
            _WINDOW // 2,
            sliding_window_size=_WINDOW,
            page_size=self.PAGE,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            is_chunk_cache=True,
        )

        self.assertEqual(0, req.kv.swa_evicted_seqlen)
        self.assertEqual(swa_before, allocator.swa_available_size())

    def test_second_call_starts_from_the_previous_frontier(self):
        page_size = self.PAGE
        pool, allocator, _ = _build(page_size)
        req = _make_req()

        def evict(pre_len):
            free_swa_out_of_window_slots(
                req,
                pre_len,
                sliding_window_size=_WINDOW,
                page_size=page_size,
                req_to_token_pool=pool,
                token_to_kv_pool_allocator=allocator,
                is_chunk_cache=True,
            )

        evict(_WINDOW + 2 * page_size)
        first = req.kv.swa_evicted_seqlen
        swa_after_first = allocator.swa_available_size()

        evict(_WINDOW + 5 * page_size)
        second = req.kv.swa_evicted_seqlen

        self.assertGreater(second, first)
        self.assertEqual(
            swa_after_first + (second - first), allocator.swa_available_size()
        )
        _assert_no_duplicates(self, allocator.swa_attn_allocator)


if __name__ == "__main__":
    unittest.main()
