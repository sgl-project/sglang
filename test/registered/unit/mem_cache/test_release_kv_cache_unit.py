"""Unit tests for release_kv_cache's page-aligned free contract."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from array import array
from types import SimpleNamespace
from typing import Optional

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import CacheFinishedReqResult
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase

_PAGE_SIZE = 4
_KV_BASE = 100


class _FakeAllocator:
    def __init__(self, page_size: int):
        self.page_size = page_size
        self.freed: list[torch.Tensor] = []

    def free(self, indices: torch.Tensor) -> None:
        self.freed.append(indices.detach().cpu().clone())


class _FakeReqToTokenPool:
    def __init__(self, req_to_token: torch.Tensor):
        self.req_to_token = req_to_token
        self.freed_reqs: list[object] = []

    def free(self, req: object) -> None:
        self.freed_reqs.append(req)


class _FakeTreeCache:
    def __init__(
        self,
        *,
        page_size: int,
        result: Optional[CacheFinishedReqResult],
        take_over_kv: bool = False,
    ):
        self.token_to_kv_pool_allocator = _FakeAllocator(page_size)
        self.req_to_token_pool = _FakeReqToTokenPool(
            torch.arange(_KV_BASE, _KV_BASE + 32, dtype=torch.int64).unsqueeze(0)
        )
        self.result = result
        self.take_over_kv = take_over_kv
        self.handled_kv_lens: list[int] = []

    def cache_finished_req(
        self, req, is_insert: bool = True, *, kv_len_to_handle: int
    ) -> Optional[CacheFinishedReqResult]:
        self.handled_kv_lens.append(kv_len_to_handle)
        if self.take_over_kv:
            req.req_pool_idx = None
            req.kv = None
        return self.result


class _FakeReq:
    def __init__(self, *, committed: int, allocated: int, protected: int = 0):
        self.req_pool_idx = 0
        self.cache_protected_len = protected
        self.kv_committed_len = committed
        self.kv = SimpleNamespace(kv_allocated_len=allocated)
        self.mamba_pool_idx = None
        self._committed = committed

    def effective_kv_committed_len(self) -> int:
        return self._committed


def _freed_indices(tree_cache: _FakeTreeCache) -> list[int]:
    if not tree_cache.token_to_kv_pool_allocator.freed:
        return []
    return torch.cat(tree_cache.token_to_kv_pool_allocator.freed).tolist()


class TestReleaseKvCache(CustomTestCase):
    def setUp(self):
        self._override = get_context().override_server_args(page_size=_PAGE_SIZE)
        self._override.install()
        self.addCleanup(self._override.restore)

    def _release(self, *, tree_cache: _FakeTreeCache, req: _FakeReq) -> None:
        release_kv_cache(req, tree_cache, is_insert=True)

    def test_release_frees_exactly_from_unhandled_kv_start_to_allocated_len(self):
        """The caller frees [unhandled_kv_start, kv_allocated_len), not the ceil of committed."""
        tree_cache = _FakeTreeCache(
            page_size=_PAGE_SIZE, result=CacheFinishedReqResult(unhandled_kv_start=8)
        )
        req = _FakeReq(committed=10, allocated=12)

        self._release(tree_cache=tree_cache, req=req)

        self.assertEqual(tree_cache.handled_kv_lens, [10])
        self.assertEqual(_freed_indices(tree_cache), [108, 109, 110, 111])

    def test_release_asserts_unhandled_kv_start_is_page_aligned(self):
        """A backend reporting a mid-page boundary is rejected instead of being ceil'd."""
        tree_cache = _FakeTreeCache(
            page_size=_PAGE_SIZE, result=CacheFinishedReqResult(unhandled_kv_start=9)
        )
        req = _FakeReq(committed=10, allocated=12)

        with self.assertRaises(AssertionError):
            self._release(tree_cache=tree_cache, req=req)

    def test_release_asserts_unhandled_kv_start_is_at_or_above_protected_len(self):
        """A boundary below cache_protected_len would free pages the tree still owns."""
        tree_cache = _FakeTreeCache(
            page_size=_PAGE_SIZE, result=CacheFinishedReqResult(unhandled_kv_start=4)
        )
        req = _FakeReq(committed=10, allocated=12, protected=8)

        with self.assertRaises(AssertionError):
            self._release(tree_cache=tree_cache, req=req)

    def test_release_asserts_unhandled_kv_start_is_at_or_below_committed_len(self):
        """A boundary above the committed length would leave committed KV unfreed."""
        tree_cache = _FakeTreeCache(
            page_size=_PAGE_SIZE, result=CacheFinishedReqResult(unhandled_kv_start=12)
        )
        req = _FakeReq(committed=10, allocated=12)

        with self.assertRaises(AssertionError):
            self._release(tree_cache=tree_cache, req=req)

    def test_release_allows_overallocated_tail_under_speculative_decoding(self):
        """Speculative decoding legitimately over-allocates, so the exactness check is gated off."""
        override = get_context().override_server_args(
            page_size=_PAGE_SIZE, speculative_algorithm="EAGLE"
        )
        override.install()
        self.addCleanup(override.restore)
        tree_cache = _FakeTreeCache(
            page_size=_PAGE_SIZE, result=CacheFinishedReqResult(unhandled_kv_start=8)
        )
        req = _FakeReq(committed=10, allocated=20)

        self._release(tree_cache=tree_cache, req=req)

        self.assertEqual(_freed_indices(tree_cache), list(range(108, 120)))

    def test_release_skips_free_when_the_cache_takes_over_the_kv(self):
        """A streaming session that adopts the KV nulls req.kv, and the caller must not free it."""
        tree_cache = _FakeTreeCache(
            page_size=_PAGE_SIZE,
            result=CacheFinishedReqResult(unhandled_kv_start=0),
            take_over_kv=True,
        )
        req = _FakeReq(committed=10, allocated=12)

        self._release(tree_cache=tree_cache, req=req)

        self.assertEqual(_freed_indices(tree_cache), [])
        self.assertEqual(tree_cache.req_to_token_pool.freed_reqs, [])

    def test_release_rejects_overallocation_of_a_whole_page_without_spec(self):
        """Without spec or strip_thinking_cache, a spare page means a bookkeeping bug."""
        tree_cache = _FakeTreeCache(
            page_size=_PAGE_SIZE, result=CacheFinishedReqResult(unhandled_kv_start=8)
        )
        req = _FakeReq(committed=10, allocated=16)

        with self.assertRaises(AssertionError):
            self._release(tree_cache=tree_cache, req=req)

    def test_release_rejects_sub_page_overallocation_without_spec(self):
        """A sub-page excess is a bug too: allocation is either exact or the page ceiling."""
        tree_cache = _FakeTreeCache(
            page_size=_PAGE_SIZE, result=CacheFinishedReqResult(unhandled_kv_start=8)
        )
        req = _FakeReq(committed=10, allocated=11)

        with self.assertRaises(AssertionError):
            self._release(tree_cache=tree_cache, req=req)

    def test_release_allows_allocated_len_equal_to_committed_len(self):
        """Allocators reporting real lengths rather than whole pages need no capability gate."""
        tree_cache = _FakeTreeCache(
            page_size=_PAGE_SIZE, result=CacheFinishedReqResult(unhandled_kv_start=8)
        )
        req = _FakeReq(committed=10, allocated=10)

        self._release(tree_cache=tree_cache, req=req)

        self.assertEqual(_freed_indices(tree_cache), [108, 109])

    def test_release_uses_the_allocator_page_size_not_the_server_args_page_size(self):
        """Under DCP the allocator page is page_size * dcp_size; server_args' value is too small."""
        tree_cache = _FakeTreeCache(
            page_size=8, result=CacheFinishedReqResult(unhandled_kv_start=8)
        )
        req = _FakeReq(committed=12, allocated=16)

        self._release(tree_cache=tree_cache, req=req)

        self.assertEqual(_freed_indices(tree_cache), list(range(108, 116)))

    def test_release_falls_back_to_ceil_for_legacy_backends_returning_none(self):
        """External backends on the deprecated contract keep their previous ceil behavior."""
        override = get_context().override_server_args(
            page_size=_PAGE_SIZE, speculative_algorithm="EAGLE"
        )
        override.install()
        self.addCleanup(override.restore)
        tree_cache = _FakeTreeCache(page_size=_PAGE_SIZE, result=None)
        req = _FakeReq(committed=10, allocated=20)

        self._release(tree_cache=tree_cache, req=req)

        self.assertEqual(_freed_indices(tree_cache), list(range(112, 120)))


class TestReleaseKvCacheFreesEachPageExactlyOnce(CustomTestCase):
    """Every page is either still live or freed exactly once, over the real release path."""

    def _run(self, *, need_sort: bool) -> None:
        override = get_context().override_server_args(page_size=_PAGE_SIZE)
        override.install()
        self.addCleanup(override.restore)

        num_pages = 10
        allocator = PagedTokenToKVPoolAllocator(
            size=num_pages * _PAGE_SIZE,
            page_size=_PAGE_SIZE,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=need_sort,
        )
        all_pages = set(allocator.free_pages.tolist())

        cache = RadixCache.create_simulated(
            mock_allocator=allocator, page_size=_PAGE_SIZE
        )
        req_to_token = torch.zeros(1, 32, dtype=torch.int64)
        cache.req_to_token_pool = _FakeReqToTokenPool(req_to_token)

        committed, allocated = 18, 20
        kv_indices = allocator.alloc(allocated)
        req_to_token[0, :allocated] = kv_indices

        req = _FakeReq(committed=committed, allocated=allocated)
        req.origin_input_ids = array("q", range(1, committed + 1))
        req.output_ids = array("q")
        req.extra_key = None
        req.last_node = None
        req.priority = 0

        release_kv_cache(req, cache, is_insert=True)

        live_pages = set((kv_indices[:16] // _PAGE_SIZE).tolist())
        freed_pages = allocator.free_pages.tolist() + allocator.release_pages.tolist()

        self.assertEqual(
            len(freed_pages),
            len(set(freed_pages)),
            f"page freed more than once ({need_sort=}): {sorted(freed_pages)}",
        )
        self.assertEqual(
            set(freed_pages),
            all_pages - live_pages,
            f"freed pages are not the complement of the live pages ({need_sort=})",
        )

    def test_each_page_is_freed_exactly_once_without_sorting(self):
        """Free-list pages stay unique and complementary when free() appends to free_pages."""
        self._run(need_sort=False)

    def test_each_page_is_freed_exactly_once_with_sorting(self):
        """Free-list pages stay unique and complementary when free() appends to release_pages."""
        self._run(need_sort=True)


if __name__ == "__main__":
    unittest.main()
