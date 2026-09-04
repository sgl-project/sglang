import unittest

import torch

from sglang.srt.managers.utils import batch_convert_2d_tensors_to_lists
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestBatchConvert2DTensorsToLists(CustomTestCase):
    """The batched path must stay bit-exact with the per-tensor ``.tolist()`` it replaces."""

    def _assert_matches_per_tensor(self, tensors):
        self.assertEqual(
            batch_convert_2d_tensors_to_lists(tensors),
            [t.tolist() for t in tensors],
        )

    def test_uniform_cols_varying_rows(self):
        # The fast path: one torch.cat + one .tolist() must re-split by row count.
        self._assert_matches_per_tensor([torch.randn(rows, 4) for rows in (3, 1, 5, 0)])

    def test_ragged_cols_fall_back_to_per_tensor(self):
        # Per-request matryoshka truncation makes torch.cat invalid; the fallback
        # must still produce the same nested lists rather than raising.
        self._assert_matches_per_tensor([torch.randn(2, 4), torch.randn(2, 8)])

    def test_empty_batch(self):
        self.assertEqual(batch_convert_2d_tensors_to_lists([]), [])


if __name__ == "__main__":
    unittest.main()
