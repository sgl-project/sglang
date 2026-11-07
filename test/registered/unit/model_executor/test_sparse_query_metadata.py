# SPDX-License-Identifier: Apache-2.0
import unittest

import torch

from sglang.srt.model_executor.sparse_query import SparseQueryMetadata
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestSparseQueryMetadata(CustomTestCase):
    def test_original_positions_and_request_boundaries(self):
        metadata = SparseQueryMetadata((2, 1), torch.tensor([0, 3, 1]))
        self.assertEqual(
            metadata.causal_mask([5, 3]).tolist(),
            [
                True,
                False,
                False,
                False,
                False,
                True,
                True,
                True,
                True,
                False,
                True,
                True,
                False,
            ],
        )

    def test_queries_retain_keys_before_the_selected_range(self):
        metadata = SparseQueryMetadata((2,), torch.tensor([2, 4]))
        self.assertEqual(
            metadata.causal_mask([5]).reshape(2, 5).tolist(),
            [[True, True, True, False, False], [True] * 5],
        )

    def test_rejects_inconsistent_layouts(self):
        for counts, positions, kv_lens in (
            ((2,), [0], [3]),
            ((1, 1), [0, 0], [3]),
            ((0,), [], [3]),
            ((2,), [0, 1], [1]),
        ):
            with self.subTest(counts=counts, kv_lens=kv_lens):
                with self.assertRaises(ValueError):
                    SparseQueryMetadata(counts, torch.tensor(positions)).causal_mask(
                        kv_lens
                    )


if __name__ == "__main__":
    unittest.main()
