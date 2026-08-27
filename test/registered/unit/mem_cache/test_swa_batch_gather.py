"""Unit tests for the batched, sync-free SWA window eviction.

``free_swa_out_of_window_slots_batch`` gathers the whole batch's out-of-window
ranges in one indexed read (``_gather_slot_ranges``) and releases them in one
call. When the pages can be named -- page-aligned ranges, ``page_size > 1`` --
it gathers one token per page and goes through ``free_swa_page_reps``, so the
release costs one op and no device sync; otherwise it falls back to ``free_swa``.

Covered here:
  - ``_gather_slot_ranges`` at ``step=1`` still returns exactly
    ``cat(req_to_token[row, start:end])`` (single, uniform and ragged branches),
    and at ``step=page_size`` returns one token per page in range order;
  - the batched release frees the same SWA pages as the per-request loop, for
    both the reps path and the ``free_swa`` fallback;
  - the per-request path falls back to ``free_swa`` on an allocator without the
    segment API;
  - the debug-mode contract checks fire on a violated contract.

    python -m pytest test/registered/unit/mem_cache/test_swa_batch_gather.py -v
"""

import itertools
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.common import (
    _gather_slot_ranges,
    free_swa_out_of_window_slots,
    free_swa_out_of_window_slots_batch,
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
_NUM_REQS = 4
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


def _make_req(req_pool_idx):
    return SimpleNamespace(
        req_pool_idx=req_pool_idx,
        cache_protected_len=0,
        swa_evict_floor=0,
        kv=SimpleNamespace(swa_evicted_seqlen=0),
    )


def _evict_kwargs(pool, allocator, page_size):
    return dict(
        sliding_window_size=_WINDOW,
        page_size=page_size,
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=allocator,
        is_chunk_cache=True,
    )


class TestGatherSlotRanges(CustomTestCase):
    ROWS, COLS = 6, 512

    def setUp(self):
        self.req_to_token = torch.arange(
            self.ROWS * self.COLS, dtype=torch.int64
        ).reshape(self.ROWS, self.COLS)

    def _reference(self, ranges, step):
        return torch.cat(
            [self.req_to_token[row, start:end:step] for row, start, end in ranges]
        )

    def _check(self, ranges, step):
        got = _gather_slot_ranges(self.req_to_token, ranges, step=step)
        self.assertTrue(
            torch.equal(self._reference(ranges, step), got),
            f"{ranges=} {step=}",
        )

    def test_step_one_matches_plain_slicing(self):
        cases = (
            [(0, 0, 8)],  # single range: plain-slice branch
            [(0, 0, 8), (1, 8, 16)],  # uniform lengths
            [(0, 0, 8), (2, 4, 20), (3, 1, 2)],  # ragged lengths
            [(r, r, r + 4) for r in range(6)],
        )
        for ranges in cases:
            self._check(ranges, 1)

    def test_strided_gather_returns_one_token_per_page(self):
        page = 64
        cases = (
            [(0, 0, page)],
            [(0, 0, 4 * page)],
            [(0, 0, 2 * page), (1, page, 4 * page)],
            [(0, 0, 4 * page), (2, page, 2 * page), (3, 2 * page, 5 * page)],
        )
        for ranges in cases:
            self._check(ranges, page)
            got = _gather_slot_ranges(self.req_to_token, ranges, step=page)
            expected_pages = sum((end - start) // page for _, start, end in ranges)
            self.assertEqual(expected_pages, got.numel())

    def test_reps_name_exactly_the_pages_unique_would_name(self):
        page = 64
        ranges = [(0, 0, 3 * page), (1, page, 4 * page)]

        reps = _gather_slot_ranges(self.req_to_token, ranges, step=page)
        every_token = _gather_slot_ranges(self.req_to_token, ranges, step=1)

        self.assertEqual(
            set(torch.unique(every_token // page).tolist()),
            set((reps // page).tolist()),
        )

    def test_ragged_and_uniform_branches_agree_with_reference(self):
        page = 8
        for lengths in itertools.product((1, 2, 3), repeat=3):
            ranges = [(i, i * page, i * page + n * page) for i, n in enumerate(lengths)]
            self._check(ranges, page)
            self._check(ranges, 1)


class TestBatchedWindowEviction(CustomTestCase):
    PAGE = 64

    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def _run_batch(self, page_size, seq_lens):
        pool, allocator, _ = _build(page_size)
        reqs = [_make_req(i) for i in range(len(seq_lens))]
        free_swa_out_of_window_slots_batch(
            reqs, list(seq_lens), **_evict_kwargs(pool, allocator, page_size)
        )
        return allocator, reqs

    def _run_loop(self, page_size, seq_lens):
        pool, allocator, _ = _build(page_size)
        reqs = [_make_req(i) for i in range(len(seq_lens))]
        for req, pre_len in zip(reqs, seq_lens):
            free_swa_out_of_window_slots(
                req, pre_len, **_evict_kwargs(pool, allocator, page_size)
            )
        return allocator, reqs

    def _frontiers(self, reqs):
        return [r.kv.swa_evicted_seqlen for r in reqs]

    def test_uniform_batch_matches_per_request_loop(self):
        for page_size in (1, self.PAGE):
            seq_lens = [_WINDOW + 4 * page_size] * _NUM_REQS
            batched, batched_reqs = self._run_batch(page_size, seq_lens)
            looped, looped_reqs = self._run_loop(page_size, seq_lens)

            self.assertEqual(
                self._frontiers(looped_reqs), self._frontiers(batched_reqs)
            )
            self.assertEqual(looped.swa_available_size(), batched.swa_available_size())
            self.assertEqual(
                looped.full_available_size(), batched.full_available_size()
            )
            self.assertTrue(
                torch.equal(
                    looped.full_to_swa_index_mapping,
                    batched.full_to_swa_index_mapping,
                )
            )

    def test_ragged_batch_matches_per_request_loop(self):
        for page_size in (1, self.PAGE):
            seq_lens = [_WINDOW + n * page_size for n in (1, 4, 2, 6)[:_NUM_REQS]]
            batched, batched_reqs = self._run_batch(page_size, seq_lens)
            looped, looped_reqs = self._run_loop(page_size, seq_lens)

            self.assertEqual(
                self._frontiers(looped_reqs), self._frontiers(batched_reqs)
            )
            self.assertEqual(looped.swa_available_size(), batched.swa_available_size())

    def test_single_request_batch(self):
        seq_lens = [_WINDOW + 3 * self.PAGE]
        batched, batched_reqs = self._run_batch(self.PAGE, seq_lens)
        looped, looped_reqs = self._run_loop(self.PAGE, seq_lens)

        self.assertEqual(self._frontiers(looped_reqs), self._frontiers(batched_reqs))
        self.assertEqual(looped.swa_available_size(), batched.swa_available_size())

    def test_requests_inside_the_window_are_skipped(self):
        pool, allocator, _ = _build(self.PAGE)
        swa_before = allocator.swa_available_size()
        reqs = [_make_req(i) for i in range(2)]

        free_swa_out_of_window_slots_batch(
            reqs,
            [_WINDOW // 2, _WINDOW // 4],
            **_evict_kwargs(pool, allocator, self.PAGE),
        )

        self.assertEqual([0, 0], self._frontiers(reqs))
        self.assertEqual(swa_before, allocator.swa_available_size())

    def test_request_without_kv_is_skipped(self):
        pool, allocator, _ = _build(self.PAGE)
        req = _make_req(0)
        req.kv = None

        free_swa_out_of_window_slots_batch(
            [req],
            [_WINDOW + 4 * self.PAGE],
            **_evict_kwargs(pool, allocator, self.PAGE),
        )

        self.assertIsNone(req.kv)

    def test_page_size_one_uses_the_free_swa_fallback(self):
        """No page structure to name: the batch must still free every slot."""
        pool, allocator, _ = _build(1)
        reqs = [_make_req(i) for i in range(2)]
        swa_before = allocator.swa_available_size()

        free_swa_out_of_window_slots_batch(
            reqs, [_WINDOW + 3, _WINDOW + 5], **_evict_kwargs(pool, allocator, 1)
        )

        freed = sum(self._frontiers(reqs))
        self.assertEqual(swa_before + freed, allocator.swa_available_size())


class TestPerRequestFallback(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_allocator_without_segment_api_uses_free_swa(self):
        pool, _, _ = _build(1)
        calls = []
        allocator = SimpleNamespace(
            page_size=1, free_swa=lambda idx: calls.append(idx.tolist())
        )
        req = _make_req(0)

        free_swa_out_of_window_slots(
            req,
            _WINDOW + 5,
            sliding_window_size=_WINDOW,
            page_size=1,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            is_chunk_cache=True,
        )

        self.assertEqual(1, len(calls))
        self.assertEqual(5, len(calls[0]))
        self.assertEqual(5, req.kv.swa_evicted_seqlen)


class TestDebugModeContracts(CustomTestCase):
    PAGE = 64

    def test_reps_release_passes_the_debug_checks(self):
        pool, allocator, _ = _build(self.PAGE)
        allocator.swa_attn_allocator.debug_mode = True
        row = pool.req_to_token[0]
        swa_before = allocator.swa_available_size()

        allocator.free_swa_page_reps(row[0 : 3 * self.PAGE : self.PAGE])

        self.assertEqual(swa_before + 3 * self.PAGE, allocator.swa_available_size())

    def test_unmapped_range_is_rejected_in_debug_mode(self):
        """The mapping must cover the range, or page 0 would be released."""
        pool, allocator, _ = _build(self.PAGE)
        allocator.swa_attn_allocator.debug_mode = True
        row = pool.req_to_token[0]
        allocator.free_swa_page_reps(row[0 : self.PAGE : self.PAGE])

        with self.assertRaises(AssertionError):
            allocator.free_swa_page_reps(row[0 : self.PAGE : self.PAGE])

    def test_double_free_is_rejected_in_debug_mode(self):
        pool, allocator, _ = _build(self.PAGE)
        sub = allocator.full_attn_allocator
        sub.debug_mode = True
        row = pool.req_to_token[0]

        sub.free_page_reps(row[0 : 2 * self.PAGE : self.PAGE])

        with self.assertRaises(AssertionError):
            sub.free_page_reps(row[0 : 2 * self.PAGE : self.PAGE])


if __name__ == "__main__":
    unittest.main()
