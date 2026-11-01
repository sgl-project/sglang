"""A chunked-prefill continuation never asks the single-pool allocator for more pages than it holds.

Regression: with the running decodes' reserved growth past the free pool, the budget returned the full chunk
("must make progress"), so a 16,384-token continuation met 13,824 free tokens, the allocator raised
"Prefill out of memory" and the scheduler died (MiniMax-M3.1 DEP2, AgentX at c48 on GB300).
"""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.prefill_budget import PrefillBudget
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PAGE = 128
CHUNK = 16384


def _budget(*, free, evictable=0, reserved_growth=0, pending=0):
    allocator = SimpleNamespace(page_size=PAGE, available_size=lambda: free)
    tree_cache = SimpleNamespace(
        supports_mamba=lambda: False, evictable_size=lambda: evictable
    )
    budget = PrefillBudget(allocator, tree_cache)
    # reserve() books immediate pages in both offsets and decode headroom in the total one
    budget.current_offset += pending
    budget.total_offset += pending + reserved_growth
    return budget


class TestChunkedContinuationBudget(CustomTestCase):
    def test_within_budget_is_unchanged(self):
        self.assertEqual(_budget(free=100_000).available_chunk_tokens(CHUNK), CHUNK)
        self.assertEqual(_budget(free=9_000).available_chunk_tokens(CHUNK), 9_000)

    def test_reserved_growth_past_the_pool_takes_the_free_pages(self):
        # the crash: 13,824 free, decode headroom booked beyond it
        budget = _budget(free=13_824, reserved_growth=40_000)
        for can_wait in (False, True):
            with self.subTest(can_wait=can_wait):
                got = budget.available_chunk_tokens(CHUNK, can_wait=can_wait)
                self.assertEqual(got, 13_824 - PAGE)
                self.assertEqual(got % PAGE, 0)

    def test_evictable_and_pending_pages_count(self):
        budget = _budget(
            free=4_000, evictable=6_000, reserved_growth=50_000, pending=2_000
        )
        # (4,000 + 6,000 - 2,000 already handed out this pass) less one page, in whole pages
        self.assertEqual(budget.available_chunk_tokens(CHUNK), 7_808)

    def test_nothing_allocatable_waits_only_while_a_decode_can_free_pages(self):
        budget = _budget(free=100, reserved_growth=10_000)
        self.assertIsNone(budget.available_chunk_tokens(CHUNK, can_wait=True))
        # no decode to free pages: attempt the chunk, so a real OOM is reported instead of a hang
        self.assertEqual(budget.available_chunk_tokens(CHUNK, can_wait=False), CHUNK)


if __name__ == "__main__":
    unittest.main()
