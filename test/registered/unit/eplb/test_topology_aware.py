"""CPU tests for the topology-aware EPLB placement primitive."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

import unittest
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import torch

from sglang.srt.arg_groups.parallel_hook import handle_eplb_and_dispatch
from sglang.srt.eplb.topology import load_rank_cost_matrix
from sglang.srt.eplb.eplb_algorithms.topology_aware import (
    rebalance_experts_topology_aware,
)
from sglang.srt.eplb.eplb_algorithms import EplbAlgorithm, rebalance_experts
from sglang.test.test_utils import CustomTestCase


class TestTopologyAwarePlacement(CustomTestCase):
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

    def test_public_algorithm_dispatch(self):
        counts = torch.tensor([[[2, 0], [0, 2]]], dtype=torch.int64)
        costs = torch.tensor([[0.0, 3.0], [3.0, 0.0]])
        direct = rebalance_experts_topology_aware(counts, costs)
        dispatched = rebalance_experts(
            tokens_per_expert=counts.sum(dim=1),
            num_physical_experts=2,
            num_local_physical_experts=1,
            num_groups=None,
            num_nodes=1,
            algorithm=EplbAlgorithm.topology_aware,
            tokens_per_source_expert=counts,
            rank_cost_matrix=costs,
        )
        for direct_value, dispatched_value in zip(direct, dispatched):
            self.assertTrue(torch.equal(direct_value, dispatched_value))

    def test_public_algorithm_requires_topology_inputs(self):
        counts = torch.ones((1, 2), dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "source-rank expert counts"):
            rebalance_experts(
                tokens_per_expert=counts,
                num_physical_experts=2,
                num_local_physical_experts=1,
                num_groups=None,
                num_nodes=1,
                algorithm=EplbAlgorithm.topology_aware,
            )

    def test_default_algorithm_remains_available_without_topology(self):
        # The public legacy API receives [DP, layers, experts] and reduces the
        # DP dimension before invoking the original EPLB implementation.
        tokens_per_expert = torch.tensor([[[4, 2, 1, 0]]], dtype=torch.int64)

        physical_to_logical, logical_to_physical, expert_count = rebalance_experts(
            tokens_per_expert=tokens_per_expert,
            num_physical_experts=4,
            num_local_physical_experts=2,
            num_groups=None,
            num_nodes=1,
            algorithm=EplbAlgorithm.deepseek,
        )

        self.assertEqual(physical_to_logical.shape, (1, 4))
        self.assertEqual(logical_to_physical.shape, (1, 4, 1))
        self.assertEqual(expert_count.shape, (1, 4))
        self.assertTrue(torch.all(expert_count == 1))

    def test_topology_swaps_without_worsening_load_balance(self):
        # The load-balanced seed assigns [0, 1] to rank 0 and [2, 3] to rank
        # 1. The traffic is reversed, so two swaps should make the placement
        # local while keeping both ranks at the same total load.
        counts = torch.zeros((1, 2, 4), dtype=torch.int64)
        counts[0, 0, 2:] = 10
        counts[0, 1, :2] = 10
        costs = torch.tensor([[0.0, 5.0], [5.0, 0.0]])

        physical_to_logical, _, _ = rebalance_experts_topology_aware(counts, costs)

        self.assertEqual(physical_to_logical[0].tolist(), [2, 3, 0, 1])

    def test_server_guard_requires_eplb(self):
        with self.assertRaisesRegex(ValueError, "requires --enable-eplb"):
            handle_eplb_and_dispatch(
                SimpleNamespace(eplb_algorithm="topology_aware", enable_eplb=False)
            )

    def test_server_guard_rejects_elastic_ep(self):
        with self.assertRaisesRegex(ValueError, "does not support elastic EP"):
            handle_eplb_and_dispatch(
                SimpleNamespace(
                    eplb_algorithm="topology_aware",
                    enable_eplb=True,
                    eplb_topology="topology.json",
                    ep_num_redundant_experts=0,
                    elastic_ep_backend="mooncake",
                    expert_distribution_recorder_mode=None,
                )
            )

    def test_server_guard_requires_a2a_backend(self):
        with self.assertRaisesRegex(ValueError, "requires an MoE A2A backend"):
            handle_eplb_and_dispatch(
                SimpleNamespace(
                    eplb_algorithm="topology_aware",
                    enable_eplb=True,
                    eplb_topology="topology.json",
                    ep_num_redundant_experts=0,
                    elastic_ep_backend=None,
                    moe_dp_size=1,
                    expert_distribution_recorder_mode="stat",
                    ep_size=2,
                    tp_size=2,
                    moe_a2a_backend="none",
                )
            )

    def test_server_guard_rejects_deepep_low_latency(self):
        with self.assertRaisesRegex(ValueError, "source-aware A2A path"):
            handle_eplb_and_dispatch(
                SimpleNamespace(
                    eplb_algorithm="topology_aware",
                    enable_eplb=True,
                    eplb_topology="topology.json",
                    ep_num_redundant_experts=0,
                    elastic_ep_backend=None,
                    moe_dp_size=1,
                    expert_distribution_recorder_mode="stat",
                    ep_size=2,
                    tp_size=2,
                    moe_a2a_backend="deepep",
                    deepep_mode="low_latency",
                )
            )

    def test_server_guard_rejects_approximate_non_deepep_stats(self):
        with self.assertRaisesRegex(ValueError, "stat_approx requires DeepEP"):
            handle_eplb_and_dispatch(
                SimpleNamespace(
                    eplb_algorithm="topology_aware",
                    enable_eplb=True,
                    eplb_topology="topology.json",
                    ep_num_redundant_experts=0,
                    elastic_ep_backend=None,
                    moe_dp_size=1,
                    expert_distribution_recorder_mode="stat_approx",
                    ep_size=2,
                    tp_size=2,
                    moe_a2a_backend="flashinfer",
                    deepep_mode="normal",
                )
            )

    def test_rejects_replication_until_supported(self):
        counts = torch.ones((1, 2, 4), dtype=torch.int64)
        costs = torch.zeros((2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(NotImplementedError, "no redundant experts"):
            rebalance_experts_topology_aware(
                counts, costs, num_physical_experts=6
            )

    def test_rejects_inconsistent_physical_expert_dimension(self):
        counts = torch.ones((1, 2, 4), dtype=torch.int64)
        costs = torch.zeros((2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "must match"):
            rebalance_experts_topology_aware(
                counts, costs, num_physical_experts=2
            )

    def test_rejects_nonzero_self_cost(self):
        counts = torch.ones((1, 2, 4), dtype=torch.int64)
        costs = torch.ones((2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "diagonal"):
            rebalance_experts_topology_aware(counts, costs)

    def test_rejects_rank_cost_matrix_shape_mismatch(self):
        counts = torch.ones((1, 2, 4), dtype=torch.int64)
        costs = torch.zeros((3, 3), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "does not match"):
            rebalance_experts_topology_aware(counts, costs)

    def test_loads_and_validates_json_topology(self):
        topology = {"rank_cost_matrix": [[0, 1], [1, 0]]}
        matrix = load_rank_cost_matrix(topology, expected_num_ranks=2)
        self.assertEqual(matrix.dtype, torch.float64)
        self.assertTrue(
            torch.equal(matrix, torch.tensor(topology["rank_cost_matrix"]).double())
        )

        with NamedTemporaryFile(mode="w+", suffix=".json") as file:
            file.write('{"rank_cost_matrix": [[0, 2], [2, 0]]}')
            file.flush()
            loaded = load_rank_cost_matrix(file.name, expected_num_ranks=2)
        self.assertEqual(loaded.tolist(), [[0.0, 2.0], [2.0, 0.0]])

    def test_preserves_ep_rank_order_in_asymmetric_matrix(self):
        topology = {
            "rank_cost_matrix": [
                [0, 1, 4],
                [2, 0, 3],
                [5, 6, 0],
            ]
        }
        matrix = load_rank_cost_matrix(topology, expected_num_ranks=3)
        self.assertEqual(matrix.tolist(), topology["rank_cost_matrix"])


if __name__ == "__main__":
    unittest.main()
