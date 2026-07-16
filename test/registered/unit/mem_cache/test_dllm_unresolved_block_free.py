import types
import unittest
from typing import Any
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

maybe_stub_sgl_kernel()

from sglang.srt.dllm.mixin.scheduler import free_unresolved_dllm_block_kv  # noqa: E402
from sglang.srt.mem_cache.allocation import _plan_extend_alloc  # noqa: E402
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool  # noqa: E402

_DEV = "cpu"
_SIZE = 64
_ROW_WIDTH = 64
_POOL_IDX = 0


def _plan_extend(
    *, reqs: list[Any], prefix_lens: list[int], seq_lens: list[int], page_size: int
) -> Any:
    return _plan_extend_alloc(
        reqs=reqs,
        prefix_lens_cpu=torch.tensor(prefix_lens, dtype=torch.int64),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int64),
        page_size=page_size,
    )


def _make_allocator(page_size: int) -> PagedTokenToKVPoolAllocator:
    return PagedTokenToKVPoolAllocator(
        _SIZE, page_size, torch.float16, _DEV, MagicMock(), False
    )


def _make_req_to_token_pool() -> Any:
    return types.SimpleNamespace(
        req_to_token=torch.zeros((1, _ROW_WIDTH), dtype=torch.int64, device=_DEV)
    )


def _make_req(*, prefix_len: int, allocated_len: int) -> Any:
    return types.SimpleNamespace(
        prefix_indices=torch.zeros(prefix_len, dtype=torch.int64, device=_DEV),
        req_pool_idx=_POOL_IDX,
        kv_committed_len=allocated_len,
        cache_protected_len=0,
        kv=types.SimpleNamespace(kv_allocated_len=allocated_len, swa_evicted_seqlen=0),
    )


class TestFreeUnresolvedDllmBlockKv(CustomTestCase):
    def _setup(self, *, page_size: int, prefix_len: int, seq_len: int):
        allocator = _make_allocator(page_size)
        req_to_token_pool = _make_req_to_token_pool()

        allocated = allocator.alloc(seq_len)
        self.assertIsNotNone(allocated)
        req_to_token_pool.req_to_token[_POOL_IDX, :seq_len] = allocated

        req = _make_req(prefix_len=prefix_len, allocated_len=seq_len)
        return allocator, req_to_token_pool, req

    def test_freed_block_is_reallocated_by_the_next_extend_round(self):
        """A stale kv_allocated_len makes the next round plan need_size=0, leaving the req on freed pages."""
        allocator, req_to_token_pool, req = self._setup(
            page_size=1, prefix_len=8, seq_len=24
        )

        free_unresolved_dllm_block_kv(
            req, req_to_token_pool=req_to_token_pool, allocator=allocator
        )

        plan = _plan_extend(reqs=[req], prefix_lens=[8], seq_lens=[24], page_size=1)
        self.assertEqual(plan.alloc_starts_cpu.tolist(), [8])
        self.assertEqual(plan.alloc_ends_cpu.tolist(), [24])
        self.assertEqual(plan.need_size, 16)

    def _run_extend_round(self, allocator, req_to_token_pool, req, *, seq_len: int):
        plan = _plan_extend(
            reqs=[req],
            prefix_lens=[len(req.prefix_indices)],
            seq_lens=[seq_len],
            page_size=allocator.page_size,
        )
        alloc_start = plan.alloc_starts_cpu.tolist()[0]
        alloc_end = plan.alloc_ends_cpu.tolist()[0]

        if plan.need_size > 0:
            new_pages = allocator.alloc(plan.need_size)
            self.assertIsNotNone(new_pages, "allocator ran out of pages")
            req_to_token_pool.req_to_token[_POOL_IDX, alloc_start:alloc_end] = new_pages

        req.kv.kv_allocated_len = alloc_end
        req.kv_committed_len = seq_len

    def test_unresolved_then_resolved_round_never_frees_a_page_twice(self):
        """End-to-end FDFO shape: free an unresolved block, re-denoise it, finish."""
        page_size = 1
        allocator, req_to_token_pool, req = self._setup(
            page_size=page_size, prefix_len=8, seq_len=24
        )
        total_pages = allocator.num_pages

        free_unresolved_dllm_block_kv(
            req, req_to_token_pool=req_to_token_pool, allocator=allocator
        )
        self._run_extend_round(allocator, req_to_token_pool, req, seq_len=24)
        live_pages = {
            int(x) // page_size
            for x in req_to_token_pool.req_to_token[_POOL_IDX, :24].tolist()
        }
        self.assertEqual(
            live_pages & set(allocator.free_pages.tolist()),
            set(),
            "req is pointing at pages that are in the allocator's free list",
        )

        allocator.free(
            req_to_token_pool.req_to_token[_POOL_IDX, : req.kv_committed_len]
        )

        free_pages = allocator.free_pages.tolist()
        self.assertEqual(
            len(free_pages),
            len(set(free_pages)),
            "a page was returned to the free list twice (double free)",
        )
        self.assertEqual(
            len(free_pages), total_pages, "every page must be back exactly once"
        )

    def test_bookkeeping_rolls_back_to_the_prefix(self):
        """kv_allocated_len/kv_committed_len must not stay above the surviving prefix."""
        allocator, req_to_token_pool, req = self._setup(
            page_size=1, prefix_len=8, seq_len=24
        )

        free_unresolved_dllm_block_kv(
            req, req_to_token_pool=req_to_token_pool, allocator=allocator
        )

        self.assertEqual(req.kv.kv_allocated_len, 8)
        self.assertEqual(req.kv_committed_len, 8)

    def test_free_range_starts_on_a_page_boundary_when_paged(self):
        """free() folds by stride, so an unaligned start would free the prefix's own page."""
        page_size = 4
        allocator, req_to_token_pool, req = self._setup(
            page_size=page_size, prefix_len=6, seq_len=24
        )
        prefix_pages = {
            int(x) // page_size
            for x in req_to_token_pool.req_to_token[_POOL_IDX, :6].tolist()
        }

        free_unresolved_dllm_block_kv(
            req, req_to_token_pool=req_to_token_pool, allocator=allocator
        )

        self.assertEqual(req.kv.kv_allocated_len, 8)
        freed = set(allocator.free_pages.tolist())
        self.assertEqual(
            freed & prefix_pages, set(), "freed a page still holding prefix tokens"
        )

    def test_noop_when_nothing_is_allocated_above_the_prefix(self):
        """A block that never got KV above the prefix must not free anything."""
        allocator, req_to_token_pool, req = self._setup(
            page_size=1, prefix_len=24, seq_len=24
        )
        before = allocator.free_pages.tolist()

        free_unresolved_dllm_block_kv(
            req, req_to_token_pool=req_to_token_pool, allocator=allocator
        )

        self.assertEqual(allocator.free_pages.tolist(), before)
        self.assertEqual(req.kv.kv_allocated_len, 24)


class TestFdfoKeepsTheKvRowAcrossRounds(CustomTestCase):
    def test_unresolved_round_reuses_the_same_row_and_extends_from_the_watermark(self):
        """A parked FDFO req keeps its row: the pool must hand the same slot back, the planner must extend from ceil(prefix) on that row, and the final free must conserve every page exactly once."""
        page_size = 4
        allocator = _make_allocator(page_size)
        pool = ReqToTokenPool(
            size=4, max_context_len=_ROW_WIDTH, device=_DEV, enable_memory_saver=False
        )
        req = _make_req(prefix_len=6, allocated_len=0)
        req.req_pool_idx = None
        req.inflight_middle_chunks = 1

        [row] = pool.alloc([req])
        allocated = allocator.alloc(16)
        self.assertIsNotNone(allocated)
        pool.req_to_token[row, :16] = allocated
        req.kv.kv_allocated_len = 16
        req.kv_committed_len = 16

        free_unresolved_dllm_block_kv(req, req_to_token_pool=pool, allocator=allocator)
        self.assertEqual(req.kv.kv_allocated_len, 8)

        free_slots_before = list(pool.free_slots)
        [row_again] = pool.alloc([req])
        self.assertEqual(row_again, row)
        self.assertEqual(pool.free_slots, free_slots_before)

        plan = _plan_extend(
            reqs=[req], prefix_lens=[6], seq_lens=[16], page_size=page_size
        )
        self.assertEqual(plan.alloc_starts_cpu.tolist(), [8])
        new_pages = allocator.alloc(plan.need_size)
        self.assertIsNotNone(new_pages)
        pool.req_to_token[row, 8:16] = new_pages
        req.kv.kv_allocated_len = 16

        live_pages = {int(x) // page_size for x in pool.req_to_token[row, :16].tolist()}
        self.assertEqual(
            live_pages & set(allocator.free_pages.tolist()),
            set(),
            "req is pointing at pages that are in the allocator's free list",
        )

        allocator.free(pool.req_to_token[row, :16])
        free_pages = allocator.free_pages.tolist()
        self.assertEqual(
            len(free_pages),
            len(set(free_pages)),
            "a page was returned to the free list twice (double free)",
        )
        self.assertEqual(len(free_pages), allocator.num_pages)


if __name__ == "__main__":
    unittest.main()
