"""free_kv_row must not hand the allocator two frees that meet inside one page.

Under decode context parallelism the allocator page is page_size * dcp_size,
while cache-length caps (e.g. the Mamba track boundary) stay on the unwidened
grid, so cache_finished_req can free a request's tail as two adjacent ranges
split mid-page. free_segments rejects that as a double free.

    python -m pytest test/registered/unit/mem_cache/test_free_kv_row_coalesce.py -v
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# A 64-token page widened by dcp_size 8 is 512; scaled down 64x: page 8, and
# a Mamba track boundary at 576 tokens lands at position 9.
PAGE_SIZE = 8
NUM_PAGES = 16


class _RowCache(BasePrefixCache):
    """Just enough of a prefix cache to exercise free_kv_row."""

    def __init__(self, allocator, req_to_token):
        self.token_to_kv_pool_allocator = allocator
        self.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)

    def reset(self):
        raise NotImplementedError

    def match_prefix(self, params):
        raise NotImplementedError

    def cache_finished_req(self, req, is_insert=True, **kwargs):
        raise NotImplementedError

    def cache_unfinished_req(self, req, **kwargs):
        raise NotImplementedError

    def evict(self, params):
        raise NotImplementedError

    def inc_lock_ref(self, node):
        raise NotImplementedError

    def dec_lock_ref(self, node):
        raise NotImplementedError


def _make_cache_with_row(num_tokens):
    alloc = PagedTokenToKVPoolAllocator(
        size=NUM_PAGES * PAGE_SIZE,
        page_size=PAGE_SIZE,
        dtype=torch.float16,
        device="cpu",
        kvcache=None,
        need_sort=False,
    )
    row = alloc.alloc(num_tokens)
    req_to_token = torch.zeros((1, num_tokens), dtype=row.dtype)
    req_to_token[0] = row
    kv = SimpleNamespace(req_pool_idx=0, swa_evicted_seqlen=0)
    return _RowCache(alloc, req_to_token), alloc, row, kv


class TestFreeKvRowCoalesce(unittest.TestCase):
    def test_adjacent_ranges_split_mid_page_free_every_page_once(self):
        cache, alloc, row, kv = _make_cache_with_row(3 * PAGE_SIZE)
        before = len(alloc.free_pages)
        # Kept prefix [0, 8); the unaligned tail [8, 9) and the deferred
        # truncation tail [9, 24) meet at 9, inside page 1.
        cache.free_kv_row(kv, [(8, 9), (9, 3 * PAGE_SIZE)])
        freed = alloc.free_pages[: len(alloc.free_pages) - before]
        reference = torch.unique(row[8:] // PAGE_SIZE)
        self.assertTrue(torch.equal(torch.sort(freed)[0], reference))

    def test_non_adjacent_ranges_sharing_a_page_still_rejected(self):
        cache, _, _, kv = _make_cache_with_row(3 * PAGE_SIZE)
        # A gap between the ranges means the shared page really is freed twice.
        with self.assertRaises(AssertionError):
            cache.free_kv_row(kv, [(8, 9), (12, 3 * PAGE_SIZE)])


if __name__ == "__main__":
    unittest.main()
