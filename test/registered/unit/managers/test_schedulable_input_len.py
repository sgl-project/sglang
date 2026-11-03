"""A prompt at max_req_input_len (where --allow-auto-truncate cuts) must be
admissible on an empty KV pool with at least one new token. Otherwise its
max_new_tokens clips to 0, PrefillAdder never admits it, and it blocks the
head of the waiting queue forever.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.utils import max_schedulable_input_len
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _admission(input_len, kv_capacity, page_size):
    """Clip max_new_tokens like the scheduler and apply PrefillAdder's check."""
    max_req_len = kv_capacity - 1
    max_new_tokens = max(0, min(1 << 30, max_req_len - input_len - 1))
    max_new_tokens = BaseTokenToKVPoolAllocator.max_new_tokens_for_memory(
        SimpleNamespace(page_size=page_size),
        input_len,
        max_new_tokens,
        token_capacity=kv_capacity,
        sliding_window_size=None,
        chunk_size=None,
    )
    admitted = input_len + max_new_tokens + page_size < kv_capacity
    return admitted, max_new_tokens


class TestSchedulableInputLen(CustomTestCase):
    def test_capped_prompt_is_admitted_with_new_tokens(self):
        for page_size in (1, 16, 64, 128):
            for kv_capacity in (4096, 4100, 12345, 1 << 20):
                with self.subTest(page_size=page_size, kv_capacity=kv_capacity):
                    input_len = max_schedulable_input_len(
                        kv_capacity - 1, kv_capacity, page_size
                    )
                    admitted, max_new_tokens = _admission(
                        input_len, kv_capacity, page_size
                    )
                    self.assertTrue(admitted)
                    self.assertGreaterEqual(max_new_tokens, 1)

    def test_old_cap_is_unschedulable_with_pages(self):
        admitted, max_new_tokens = _admission(4096 - 6, 4096, 16)
        self.assertFalse(admitted)
        self.assertEqual(max_new_tokens, 0)

    def test_context_length_bound_unchanged(self):
        self.assertEqual(max_schedulable_input_len(8191, 1 << 20, 64), 8186)
        self.assertEqual(max_schedulable_input_len(4095, 4096, 1), 4090)


if __name__ == "__main__":
    unittest.main()
