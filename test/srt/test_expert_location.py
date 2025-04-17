import unittest

import torch
from sglang.srt.managers.expert_location import compute_logical_to_rank_dispatch_physical_map
from sglang.test.test_utils import CustomTestCase


class TestExpertLocation(CustomTestCase):
    def test_compute_logical_to_rank_dispatch_physical_map(self):
        # 8 logical expert
        cases = [
            # Identity map
            (
                [[[0], [1], [2], [3], [4], [5], [6], [7]]],
                [
                    [[0, 1, 2, 3, 4, 5, 6, 7]],
                    [[0, 1, 2, 3, 4, 5, 6, 7]],
                    [[0, 1, 2, 3, 4, 5, 6, 7]],
                    [[0, 1, 2, 3, 4, 5, 6, 7]],
                ],
            ),
            # Identity map + consider redundant experts
            (
                [[[0, 8], [1, 9], [2, 10], [3, 11], [4, -1], [5, -1], [6, -1], [7, -1]]],
                [[[0, 1, 2, 11, 4, 5, 6, 7]], [[8, 9, 2, 3, 4, 5, 6, 7]],
                 [[8, 1, 10, 3, 4, 5, 6, 7]], [[0, 9, 10, 11, 4, 5, 6, 7]]],
            ),
            # One logical expert is put on ALL gpus
            (
                [[[0, 3, 6, 9], [1, -1, -1, -1], [2, -1, -1, -1], [4, -1, -1, -1], [5, -1, -1, -1], [7, -1, -1, -1],
                  [8, -1, -1, -1], [10, -1, -1, -1]]],
                [[[0, 1, 2, 4, 5, 7, 8, 10]], [[3, 1, 2, 4, 5, 7, 8, 10]], [[6, 1, 2, 4, 5, 7, 8, 10]],
                 [[9, 1, 2, 4, 5, 7, 8, 10]]],
            ),
            # One logical expert is put multiple times on ONE gpu
            (
                [[[0, 1, 2], [3, -1, -1], [4, -1, -1], [5, -1, -1], [6, -1, -1], [7, -1, -1], [8, -1, -1],
                  [9, -1, -1]]],
                [[[0, 3, 4, 5, 6, 7, 8, 9]], [[1, 3, 4, 5, 6, 7, 8, 9]], [[0, 3, 4, 5, 6, 7, 8, 9]],
                 [[2, 3, 4, 5, 6, 7, 8, 9]]],
            ),
            # Random
            (
                [[[4, 11, -1], [5, 9, 0], [6, -1, -1], [8, -1, -1], [1, -1, -1], [10, -1, -1], [2, 3, -1],
                  [7, -1, -1]]],
                [[[11, 0, 6, 8, 1, 10, 2, 7]], [[4, 5, 6, 8, 1, 10, 3, 7]], [[4, 5, 6, 8, 1, 10, 2, 7]],
                 [[11, 9, 6, 8, 1, 10, 3, 7]]],
            ),
        ]

        actual_outputs = []

        for logical_to_all_physical_map, expect_output in cases:
            actual_output = compute_logical_to_rank_dispatch_physical_map(
                logical_to_all_physical_map=torch.tensor(logical_to_all_physical_map),
                num_gpus=4,
                num_physical_experts=12,
            ).tolist()
            actual_outputs.append(actual_output)
            print(f"{actual_output=} {expect_output=}")

        for (logical_to_all_physical_map, expect_output), actual_output in zip(cases, actual_outputs):
            self.assertEqual(actual_output, expect_output)


if __name__ == "__main__":
    unittest.main()
