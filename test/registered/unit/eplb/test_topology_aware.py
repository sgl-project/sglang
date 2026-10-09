"""CPU tests for the topology-aware EPLB placement primitive."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=7, suite="base-a-test-cpu")

import unittest
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import torch

from sglang.srt.arg_groups.parallel_hook import handle_eplb_and_dispatch
from sglang.srt.eplb.eplb_algorithms import EplbAlgorithm, rebalance_experts
from sglang.srt.eplb.eplb_algorithms.topology_aware import (
    _improve_topology,
    _node_uniform_seed,
    _rank_loads,
    rebalance_experts_topology_aware,
)
from sglang.srt.eplb.topology import load_rank_cost_matrix
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
        costs = torch.tensor([[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]])

        physical_to_logical, logical_to_physical, _ = rebalance_experts_topology_aware(
            counts, costs
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
        # The node-uniform seed assigns [0, 1] to rank 0 and [2, 3] to rank
        # 1. The traffic is reversed, so two swaps should make the placement
        # local while keeping both ranks at the same total load.
        counts = torch.zeros((1, 2, 4), dtype=torch.int64)
        counts[0, 0, 2:] = 10
        counts[0, 1, :2] = 10
        costs = torch.tensor([[0.0, 5.0], [5.0, 0.0]])

        physical_to_logical, _, _ = rebalance_experts_topology_aware(counts, costs)

        self.assertEqual(physical_to_logical[0].tolist(), [2, 3, 0, 1])

    def test_topology_does_not_regress_node_uniform_communication(self):
        # A topology-only swap can move a slightly hotter expert away
        # from its source rank.  The topology pass may use the node-uniform
        # layout's load envelope, but must never make its communication cost
        # worse.
        counts = torch.zeros((1, 2, 4), dtype=torch.int64)
        counts[0, 0] = torch.tensor([10, 9, 8, 7])
        costs = torch.tensor([[0.0, 10.0], [10.0, 0.0]])

        physical_to_logical, _, _ = rebalance_experts_topology_aware(counts, costs)
        destination = torch.empty(4, dtype=torch.long)
        for rank in range(2):
            start = rank * 2
            destination[physical_to_logical[0, start : start + 2]] = rank

        node_uniform_destination = torch.tensor([0, 0, 1, 1])
        communication = counts[0].to(torch.float64) * costs[:, destination]
        node_uniform_communication = (
            counts[0].to(torch.float64) * costs[:, node_uniform_destination]
        )
        self.assertLessEqual(
            communication.sum().item(), node_uniform_communication.sum().item()
        )

        total_load = counts[0].sum(dim=0)
        candidate_load = torch.zeros(2, dtype=torch.int64)
        candidate_load.scatter_add_(0, destination, total_load)
        node_uniform_load = torch.tensor([total_load[:2].sum(), total_load[2:].sum()])
        self.assertLessEqual(candidate_load.max(), node_uniform_load.max())

    def test_load_phase_spends_only_explicit_communication_budget(self):
        # A hot expert pair leaves an opportunity to lower the critical rank
        # after the communication-only phase.  The private helper is exercised
        # directly so the test can use a deliberately large budget; the
        # public planner keeps its much smaller production budget.
        counts = torch.tensor(
            [
                [80, 20, 1, 0],
                [1, 0, 10, 10],
            ],
            dtype=torch.float64,
        )
        costs = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]],
            dtype=torch.float64,
        )
        total_load = counts.sum(dim=0)
        communication = counts.transpose(0, 1).matmul(costs)
        seed = _node_uniform_seed(4, 2)
        seed_load = _rank_loads(seed, total_load, 2)
        assignment = _improve_topology(
            seed.clone(),
            total_load,
            communication,
            2,
            max_allowed_load=seed_load.max().item(),
            max_swaps=10,
            max_comm_regression_ratio=20.0,
        )
        candidate_load = _rank_loads(assignment, total_load, 2)
        self.assertLess(candidate_load.max(), seed_load.max())
        candidate_cost = communication[torch.arange(4), assignment].sum().item()
        seed_cost = communication[torch.arange(4), seed].sum().item()
        self.assertLessEqual(candidate_cost, seed_cost * 21.0 + 1e-12)

    def test_system_phase_reserves_search_budget_for_critical_rank(self):
        counts = torch.tensor([[28, 8, 23, 7], [0, 8, 21, 10]], dtype=torch.float64)
        costs = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
        total_load = counts.sum(dim=0)
        communication = counts.transpose(0, 1).matmul(costs)
        seed = _node_uniform_seed(4, 2)
        seed_load = _rank_loads(seed, total_load, 2)

        communication_only = _improve_topology(
            seed.clone(),
            total_load,
            communication,
            2,
            max_allowed_load=seed_load.max().item(),
            max_swaps=4,
        )
        system_aware = _improve_topology(
            seed.clone(),
            total_load,
            communication,
            2,
            max_allowed_load=seed_load.max().item(),
            max_swaps=4,
            max_comm_regression_ratio=1.0,
        )
        self.assertLess(
            _rank_loads(system_aware, total_load, 2).max(),
            _rank_loads(communication_only, total_load, 2).max(),
        )

    def test_system_phase_breaks_equal_peak_load_ties(self):
        # With three ranks, a first swap can leave the busiest rank unchanged
        # while reducing the other ranks' variance.  Accepting that tie-break
        # gives the next swap a better starting point.
        total_load = torch.tensor([12, 8, 6, 4, 12, 8], dtype=torch.float64)
        communication = torch.zeros((6, 3), dtype=torch.float64)
        seed = _node_uniform_seed(6, 3)
        seed_load = _rank_loads(seed, total_load, 3)
        assignment = _improve_topology(
            seed.clone(),
            total_load,
            communication,
            3,
            max_allowed_load=seed_load.max().item(),
            max_swaps=1,
            max_comm_regression_ratio=1.0,
        )
        candidate_load = _rank_loads(assignment, total_load, 3)
        self.assertLess(candidate_load.square().sum(), seed_load.square().sum())
        self.assertEqual(candidate_load.max(), seed_load.max())

    def test_rejects_negative_communication_budget(self):
        counts = torch.ones((1, 2, 4), dtype=torch.int64)
        costs = torch.tensor(
            [[0.0, 1.0], [1.0, 0.0]],
            dtype=torch.float64,
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _improve_topology(
                _node_uniform_seed(4, 2),
                counts[0].sum(dim=0),
                counts[0].to(torch.float64).transpose(0, 1).matmul(costs),
                2,
                max_comm_regression_ratio=-1.0,
            )

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

    def test_server_guard_requires_static_dispatch(self):
        with self.assertRaisesRegex(
            ValueError, "requires.*ep-dispatch-algorithm static"
        ):
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
                    ep_dispatch_algorithm="dynamic",
                    ep_join_mode=None,
                    moe_a2a_backend="flashinfer",
                    deepep_mode="auto",
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
            rebalance_experts_topology_aware(counts, costs, num_physical_experts=6)

    def test_rejects_inconsistent_physical_expert_dimension(self):
        counts = torch.ones((1, 2, 4), dtype=torch.int64)
        costs = torch.zeros((2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "must match"):
            rebalance_experts_topology_aware(counts, costs, num_physical_experts=2)

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
