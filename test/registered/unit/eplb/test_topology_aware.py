"""CPU tests for the topology-aware EPLB placement primitive."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.eplb.eplb_algorithms.topology_aware import (
    rebalance_experts_topology_aware,
)


class TestTopologyAwarePlacement(unittest.TestCase):
    def test_prefers_the_cheapest_destination(self):
        # Four experts, two source/destination ranks, two slots per rank.
        # Experts 0 and 1 originate on rank 0; experts 2 and 3 on rank 1.
        counts = torch.zeros((1, 2, 4), dtype=torch.int64)
        counts[0, 0, :2] = 10
        counts[0, 1, 2:] = 10
        costs = torch.tensor([[0.0, 5.0], [5.0, 0.0]])

        physical_to_logical, logical_to_physical, expert_count = (
            rebalance_experts_topology_aware(counts, costs)
        )

        self.assertEqual(physical_to_logical[0].tolist(), [0, 1, 2, 3])
        self.assertEqual(logical_to_physical[0, :, 0].tolist(), [0, 1, 2, 3])
        self.assertEqual(expert_count[0].tolist(), [1, 1, 1, 1])

    def test_maps_are_permutations_with_equal_capacity(self):
        counts = torch.tensor(
            [
                [
                    [4, 0, 2, 1, 0, 0],
                    [0, 3, 0, 0, 5, 1],
                    [2, 1, 0, 0, 1, 4],
                ],
                [
                    [0, 1, 6, 0, 2, 0],
                    [3, 0, 0, 4, 0, 1],
                    [1, 0, 0, 2, 5, 0],
                ],
            ],
            dtype=torch.int64,
        )
        costs = torch.tensor(
            [[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]]
        )

        physical_to_logical, logical_to_physical, _ = (
            rebalance_experts_topology_aware(counts, costs)
        )
        for layer in range(counts.shape[0]):
            self.assertEqual(
                sorted(physical_to_logical[layer].tolist()), list(range(6))
            )
            self.assertEqual(
                sorted(logical_to_physical[layer, :, 0].tolist()), list(range(6))
            )
            # Two physical experts per rank.
            for rank in range(3):
                start = rank * 2
                end = start + 2
                self.assertEqual(
                    sum(
                        int(logical_to_physical[layer, expert, 0]) in range(start, end)
                        for expert in range(6)
                    ),
                    2,
                )

    def test_is_deterministic(self):
        counts = torch.tensor([[[3, 3, 1, 1], [1, 1, 3, 3]]], dtype=torch.int64)
        costs = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
        first = rebalance_experts_topology_aware(counts, costs)
        second = rebalance_experts_topology_aware(counts, costs)
        for first_value, second_value in zip(first, second):
            self.assertTrue(torch.equal(first_value, second_value))

    def test_rejects_replication_until_supported(self):
        counts = torch.ones((1, 2, 4), dtype=torch.int64)
        costs = torch.zeros((2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(NotImplementedError, "no redundant experts"):
            rebalance_experts_topology_aware(
                counts, costs, num_physical_experts=6
            )

    def test_rejects_nonzero_self_cost(self):
        counts = torch.ones((1, 2, 4), dtype=torch.int64)
        costs = torch.ones((2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "diagonal"):
            rebalance_experts_topology_aware(counts, costs)


if __name__ == "__main__":
    unittest.main()
