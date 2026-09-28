import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from test_unified_byte_budget_sizing import _swa_factory

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

ROWS = 4


def _pool(chunk_size: int = ROWS) -> SWAKVPool:
    pool = object.__new__(SWAKVPool)
    pool.swa_kv_pool = SimpleNamespace(cpu_offloading_chunk_size=chunk_size)
    return pool


def _rows(t: int):
    return torch.arange(ROWS * 2).reshape(ROWS, 2) + t * 100


def _chunk(num_tensors: int):
    """swa_kv_cpu is layers -> chunks -> tensors; one layer, one chunk here."""
    return [[[_rows(t) for t in range(num_tensors)]]]


class TestSWACpuCopyFilter(unittest.TestCase):
    def test_keeps_every_tensor_a_chunk_carries(self):
        """A quantized sub-pool puts the block scales in the same chunk as K/V.
        Trimming rows must not drop them: the load side unpacks whatever the get
        side produced, and a short chunk resumes against the wrong exponents."""
        row_mask = torch.tensor([True, False, True, False])

        filtered = _pool()._filter_swa_cpu_copy(_chunk(4), row_mask)

        self.assertEqual(len(filtered[0][0]), 4)
        for t, tensor in enumerate(filtered[0][0]):
            expected = _rows(t)[row_mask]
            self.assertTrue(torch.equal(tensor, expected))

    def test_unquantized_chunk_is_unchanged(self):
        row_mask = torch.tensor([False, True, True, False])

        filtered = _pool()._filter_swa_cpu_copy(_chunk(2), row_mask)

        self.assertEqual(len(filtered[0][0]), 2)
        self.assertEqual(filtered[0][0][0].shape[0], 2)

    def test_all_rows_kept_returns_the_input(self):
        original = _chunk(4)

        self.assertIs(
            _pool()._filter_swa_cpu_copy(original, torch.tensor([True] * ROWS)),
            original,
        )


class TestUnifiedSWATransfers(CustomTestCase):
    def test_cpu_copy_round_trip_with_request_index_and_swa_tombstone(self):
        bundle = _swa_factory(page_size=4)
        allocator = bundle.token_to_kv_pool_allocator
        pool = bundle.token_to_kv_pool
        indices = allocator.alloc(12)
        allocator.free_swa(indices[:4])
        raw = bundle.unified_memory_pool._raw
        raw.copy_(torch.arange(raw.numel()).remainder(251).to(torch.uint8))
        full_pages = allocator.translate_kv_indices_for_transfer(indices)[::4] // 4
        swa_pages = allocator.translate_swa_indices_for_transfer(indices[4:])[::4] // 4
        full_buffer, swa_buffer = (
            raw[: side.num_pages * side.entry_bytes_per_page].view(
                side.num_pages, side.entry_bytes_per_page
            )
            for side in (allocator.full_attn_allocator, allocator.swa_attn_allocator)
        )
        expected_full = full_buffer[full_pages].clone()
        expected_swa = swa_buffer[swa_pages].clone()

        # The buffers are CPU tensors; no device synchronization is needed.
        with mock.patch(
            "sglang.srt.mem_cache.memory_pool.current_platform.synchronize"
        ):
            saved = pool.get_cpu_copy(indices, req_pool_index=0)
            raw.zero_()
            pool.load_cpu_copy(saved, indices, req_pool_index=0)

        self.assertTrue(torch.equal(full_buffer[full_pages], expected_full))
        self.assertTrue(torch.equal(swa_buffer[swa_pages], expected_swa))
        self.assertEqual(saved["swa_mask"].tolist(), [False] * 4 + [True] * 8)


if __name__ == "__main__":
    unittest.main()
