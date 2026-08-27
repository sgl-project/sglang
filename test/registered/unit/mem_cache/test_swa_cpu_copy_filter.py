import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

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


if __name__ == "__main__":
    unittest.main()
