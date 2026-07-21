"""Unit tests for one-logical-request beam row ownership."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.beam_search.coordinator import (
    release_vectorized_beam_resources,
)
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _RecordingAllocator:
    def __init__(self):
        self.freed = []

    def free(self, slots):
        self.freed.extend(slots.tolist())


class TestVectorizedBeamRows(CustomTestCase):
    def test_raw_request_row_lifecycle(self):
        pool = ReqToTokenPool(
            size=4,
            max_context_len=8,
            device="cpu",
            enable_memory_saver=False,
        )

        rows = pool.alloc_by_count(2)
        self.assertEqual(rows, [1, 2])
        self.assertEqual(pool.available_size(), 2)

        pool.free_by_indices(rows)
        self.assertEqual(pool.available_size(), 4)

    def test_release_frees_unique_decode_slots_only(self):
        pool = ReqToTokenPool(
            size=4,
            max_context_len=8,
            device="cpu",
            enable_memory_saver=False,
        )
        rows = pool.alloc_by_count(2)
        row_tensor = torch.tensor(rows, dtype=torch.int64)
        pool.req_to_token[row_tensor[0], :4] = torch.tensor([10, 11, 20, 21])
        pool.req_to_token[row_tensor[1], :4] = torch.tensor([10, 11, 20, 31])

        group = SimpleNamespace(
            beam_req_pool_indices=row_tensor,
            prompt_len=2,
            beam_seq_len=4,
            resources_released=False,
        )
        allocator = _RecordingAllocator()

        release_vectorized_beam_resources(group, pool, allocator)

        self.assertEqual(sorted(allocator.freed), [20, 21, 31])
        self.assertIsNone(group.beam_req_pool_indices)
        self.assertTrue(group.resources_released)
        self.assertEqual(pool.available_size(), 4)


if __name__ == "__main__":
    unittest.main()
