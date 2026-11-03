"""Unit tests for srt/mem_cache/elastic (allocator shrink/grow bookkeeping)."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import signal
import unittest
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.elastic import elasticmem_orchestrator
from sglang.srt.mem_cache.elastic.elastic_allocator import (
    ElasticSWATokenToKVPoolAllocator,
    ElasticTokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.elastic.elasticmem_orchestrator import (
    ElasticMempoolOrchestrator,
    cu_page_size,
    install_shutdown_handlers,
)
from sglang.test.test_utils import CustomTestCase


class _FakeElasticPool:
    """Stands in for the kvcached-backed pool: reduce/expand hit the size exactly
    and one cu_page holds one token slot."""

    def __init__(self, size: int, events: List[str]):
        self.size = size
        self.allocation_label = "Full"
        self.state_memsize = cu_page_size()
        self.events = events

    def reduce(self, new_size: int) -> Tuple[int, int]:
        self.events.append("unmap")
        unmapped, self.size = self.size - new_size, new_size
        return unmapped, self.size

    def expand(self, new_size: int) -> Tuple[int, int]:
        mapped, self.size = new_size - self.size, new_size
        return mapped, self.size

    def cu_page_to_token(self, cu_page_num: int) -> int:
        return cu_page_num


def _make_allocator(size: int, need_sort: bool, events=None):
    pool = _FakeElasticPool(size, events if events is not None else [])
    return ElasticTokenToKVPoolAllocator(
        size, torch.int64, "cpu", pool, need_sort=need_sort
    )


def _free_slots(allocator) -> List[int]:
    return sorted(allocator.get_all_free_pages().tolist())


class TestElasticTokenToKVPoolAllocator(CustomTestCase):
    def test_shrink_keeps_every_slot_accounted_for(self):
        """Withheld tail slots stay available, are lent to an alloc that needs them, and
        come back on free; no slot is lost or handed out twice.
        """
        for need_sort in (False, True):
            with self.subTest(need_sort=need_sort):
                allocator = _make_allocator(16, need_sort)
                held = allocator.alloc(4)
                self.assertIs(allocator.mark_unmap_candidate(True), allocator)
                self.assertEqual(allocator.candidate_size, 10)
                self.assertEqual(allocator.available_size(), 12)

                borrowed = allocator.alloc(8)
                self.assertEqual(int((borrowed > 10).sum()), 2)
                self.assertFalse(allocator.can_do_unmap())

                allocator.free(borrowed)
                self.assertTrue(allocator.can_do_unmap())
                self.assertEqual(allocator.reduce(), 6)

                self.assertEqual(allocator.size, 10)
                expected = sorted(set(range(1, 11)) - set(held.tolist()))
                self.assertEqual(_free_slots(allocator), expected)

    def test_cancel_returns_withheld_slots(self):
        allocator = _make_allocator(16, need_sort=True)
        allocator.alloc(4)
        before = _free_slots(allocator)
        allocator.mark_unmap_candidate(True)
        allocator.mark_unmap_candidate(False)
        self.assertEqual(allocator.candidate_size, allocator.size)
        self.assertEqual(_free_slots(allocator), before)

    def test_shrink_never_goes_below_min_size(self):
        """A pool shrunk below the longest request would leave that request unschedulable
        forever.
        """
        allocator = _make_allocator(64, need_sort=False)
        allocator.set_min_size(48)
        allocator.mark_unmap_candidate(True)
        self.assertEqual(allocator.candidate_size, 48)

        allocator.mark_unmap_candidate(False)
        allocator.set_min_size(62)
        self.assertIsNone(allocator.mark_unmap_candidate(True))
        self.assertEqual(allocator.candidate_size, 64)

    def test_device_synchronizes_before_unmap(self):
        """Unmap is not stream-ordered; the in-flight forward must finish first."""
        events: List[str] = []
        allocator = _make_allocator(16, need_sort=False, events=events)
        allocator.mark_unmap_candidate(True)
        device_module = MagicMock()
        device_module.synchronize.side_effect = lambda: events.append("sync")
        with patch.object(torch, "get_device_module", return_value=device_module):
            allocator.reduce()
        self.assertEqual(events, ["sync", "unmap"])

    def test_drain_check_skips_tree_walk_until_state_changes(self):
        allocator = _make_allocator(16, need_sort=False)
        allocator.alloc(12)
        evictable_size = [8]
        indices_fn = MagicMock(return_value=torch.empty((0,), dtype=torch.int64))
        allocator.bind_tree_cache_fns(
            evictable_size_fn=lambda: evictable_size[0],
            evict_fn=lambda num_tokens: None,
            evictable_indices_fn=indices_fn,
        )
        allocator.mark_unmap_candidate(True)
        self.assertFalse(allocator.can_do_unmap())
        self.assertFalse(allocator.can_do_unmap())
        self.assertEqual(indices_fn.call_count, 1)

        evictable_size[0] = 9
        self.assertFalse(allocator.can_do_unmap())
        self.assertEqual(indices_fn.call_count, 2)


class TestElasticSWATokenToKVPoolAllocator(CustomTestCase):
    def test_swa_floor_is_window_plus_chunk(self):
        """A max_req_len floor on the SWA pool would stop it from ever giving
        pages to the full pool at long context lengths; with chunked prefill a
        request holds at most one chunk plus the window there.
        """
        cases = ((4096, 128 + 4096), (None, 65535))
        for chunked_prefill_size, swa_min_size in cases:
            with self.subTest(chunked_prefill_size=chunked_prefill_size):
                allocator = MagicMock()
                ElasticSWATokenToKVPoolAllocator.attach(
                    allocator,
                    tree_cache=object(),
                    max_req_len=65535,
                    sliding_window_size=128,
                    chunked_prefill_size=chunked_prefill_size,
                )
                allocator.full_attn_allocator.set_min_size.assert_called_once_with(
                    65535
                )
                allocator.swa_attn_allocator.set_min_size.assert_called_once_with(
                    swa_min_size
                )


class TestElasticMempoolOrchestrator(CustomTestCase):
    def test_leftover_pages_return_to_cold_pool(self):
        """Pages the hot pool cannot use whole must not stay unmapped."""
        cold, hot = MagicMock(), MagicMock()
        cold.reduce.return_value = 10
        hot.cu_page_to_token.return_value = 1
        hot.expand.return_value = 8
        cold.cu_page_to_token.return_value = 2
        cold.expand.return_value = 2
        with envs.SGLANG_ENABLE_ELASTIC_MEM_POOL.override(True):
            orchestrator = ElasticMempoolOrchestrator()
        orchestrator._do_resize(hot, cold)
        cold.cu_page_to_token.assert_called_once_with(2)
        cold.expand.assert_called_once_with(2)
        self.assertEqual(orchestrator.remaining_page, 0)


class TestInstallShutdownHandlers(CustomTestCase):
    def test_only_default_dispositions_are_replaced(self):
        """An ignored signal leaves the process running, so it must not shut kvcached down;
        a Python handler is followed by atexit instead.
        """
        sigs = (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT)
        saved = {sig: signal.getsignal(sig) for sig in sigs}
        python_handler = lambda signum, frame: None  # noqa: E731
        try:
            signal.signal(signal.SIGINT, python_handler)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGQUIT, signal.SIG_DFL)
            with patch.object(elasticmem_orchestrator.atexit, "register"):
                install_shutdown_handlers(MagicMock())
            self.assertIs(signal.getsignal(signal.SIGINT), python_handler)
            self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_IGN)
            self.assertNotEqual(signal.getsignal(signal.SIGQUIT), signal.SIG_DFL)
        finally:
            for sig, handler in saved.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    unittest.main()
