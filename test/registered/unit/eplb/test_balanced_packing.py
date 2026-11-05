"""Unit tests for balanced_packing — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import unittest
from unittest.mock import patch

import torch

from sglang.srt.eplb.eplb_algorithms import deepseek
from sglang.srt.eplb.eplb_algorithms.deepseek import balanced_packing
from sglang.test.test_utils import CustomTestCase


class TestBalancedPacking(CustomTestCase):
    """Tests for balanced_packing(weight, num_packs).

    Invariants:
    - Output shapes match input: both [X, n].
    - pack_index values are in [0, num_packs).
    - Each pack receives exactly n // num_packs items per layer.
    - rank_in_pack values are in [0, groups_per_pack).
    - Each (pack, rank) slot is used exactly once per layer.
    - Packs are as weight-balanced as possible (greedy optimality).
    """

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _rank_scan(weight, num_packs):
        """Greedy policy with the lowest rank winning equal-load ties."""
        capacity = weight.shape[1] // num_packs
        if capacity == 1:
            return (
                torch.arange(weight.shape[1]).expand(weight.shape),
                torch.zeros_like(weight, dtype=torch.int64),
            )
        assignments = torch.empty(weight.shape, dtype=torch.int64)
        positions = torch.empty_like(assignments)
        for layer, order in enumerate(weight.float().argsort(descending=True).tolist()):
            loads = [0] * num_packs
            counts = [0] * num_packs
            values = weight[layer].tolist()
            for item in order:
                rank = min(
                    (rank for rank in range(num_packs) if counts[rank] < capacity),
                    key=loads.__getitem__,
                )
                assignments[layer, item] = rank
                positions[layer, item] = counts[rank]
                counts[rank] += 1
                loads[rank] += values[item]
        return assignments, positions

    def test_matches_rank_scan_with_ties_and_random_loads(self):
        generator = torch.Generator().manual_seed(42)
        for packs in (1, 2, 4, 8, 32):
            for capacity in (1, 2, 5, 14):
                shape = (3, packs * capacity)
                cases = (
                    torch.zeros(shape),
                    torch.ones(shape),
                    torch.randint(0, 4, shape, generator=generator),
                    torch.rand(shape, generator=generator),
                    torch.randint(0, 100000, shape, generator=generator).double(),
                )
                for weight in cases:
                    with self.subTest(
                        packs=packs, capacity=capacity, dtype=weight.dtype
                    ):
                        expected = self._rank_scan(weight, packs)
                        actual = balanced_packing(weight, packs)
                        for output, reference in zip(actual, expected):
                            torch.testing.assert_close(
                                output, reference, rtol=0, atol=0
                            )

    def test_complete_placement_matches_rank_scan(self):
        generator = torch.Generator().manual_seed(17)
        for logical, physical, groups, nodes, ranks in (
            (12, 16, 4, 2, 4),
            (384, 448, 8, 8, 32),
        ):
            for hierarchical in (False, True):
                loads = torch.randint(0, 64, (3, logical), generator=generator).float()
                kwargs = dict(
                    weight=loads,
                    num_replicas=physical,
                    num_groups=groups,
                    num_nodes=nodes,
                    num_gpus=ranks,
                    enable_hierarchical=hierarchical,
                )
                with self.subTest(logical=logical, hierarchical=hierarchical):
                    actual = deepseek.rebalance_experts(**kwargs)
                    with patch.object(deepseek, "balanced_packing", self._rank_scan):
                        expected = deepseek.rebalance_experts(**kwargs)
                    for output, reference in zip(actual, expected):
                        torch.testing.assert_close(output, reference, rtol=0, atol=0)

    def _check_shapes(self, weight, pack_index, rank_in_pack):
        self.assertEqual(pack_index.shape, weight.shape)
        self.assertEqual(rank_in_pack.shape, weight.shape)

    def _check_pack_index_range(self, pack_index, num_packs):
        self.assertTrue(torch.all(pack_index >= 0))
        self.assertTrue(torch.all(pack_index < num_packs))

    def _check_items_per_pack(self, pack_index, num_packs, groups_per_pack):
        """Every pack must hold exactly groups_per_pack items in every layer."""
        for layer in range(pack_index.shape[0]):
            counts = torch.bincount(pack_index[layer], minlength=num_packs)
            self.assertTrue(
                torch.all(counts == groups_per_pack),
                f"layer {layer}: pack counts {counts.tolist()} != {groups_per_pack}",
            )

    def _check_rank_in_pack_range(self, rank_in_pack, groups_per_pack):
        self.assertTrue(torch.all(rank_in_pack >= 0))
        self.assertTrue(torch.all(rank_in_pack < groups_per_pack))

    def _check_unique_slots(self, pack_index, rank_in_pack, num_packs, groups_per_pack):
        """Each (pack, rank) slot is occupied exactly once per layer."""
        num_layers = pack_index.shape[0]
        for layer in range(num_layers):
            slots = set(zip(pack_index[layer].tolist(), rank_in_pack[layer].tolist()))
            self.assertEqual(len(slots), num_packs * groups_per_pack)

    # ------------------------------------------------------------------ tests

    def test_pack_index_range(self):
        """All pack indices are in [0, num_packs)."""
        weight = torch.rand(2, 6)
        pack_index, _ = balanced_packing(weight, num_packs=3)
        self._check_pack_index_range(pack_index, num_packs=3)

    def test_each_pack_receives_equal_items(self):
        """Each pack receives exactly n // num_packs items per layer."""
        weight = torch.rand(4, 8)
        num_packs = 4
        pack_index, _ = balanced_packing(weight, num_packs=num_packs)
        self._check_items_per_pack(pack_index, num_packs, groups_per_pack=2)

    def test_rank_in_pack_range(self):
        """rank_in_pack values are in [0, groups_per_pack)."""
        weight = torch.rand(2, 8)
        num_packs = 4
        groups_per_pack = 8 // num_packs
        _, rank_in_pack = balanced_packing(weight, num_packs=num_packs)
        self._check_rank_in_pack_range(rank_in_pack, groups_per_pack)

    def test_unique_pack_rank_slots(self):
        """Each (pack, rank) slot is used exactly once per layer."""
        weight = torch.rand(3, 8)
        num_packs = 4
        pack_index, rank_in_pack = balanced_packing(weight, num_packs=num_packs)
        self._check_unique_slots(pack_index, rank_in_pack, num_packs, groups_per_pack=2)

    def test_groups_per_pack_one_special_case(self):
        """When groups_per_pack == 1 (num_packs == n), each item gets its own pack."""
        n = 6
        weight = torch.rand(2, n)
        pack_index, rank_in_pack = balanced_packing(weight, num_packs=n)
        # pack_index[layer] should be a permutation of [0, n)
        for layer in range(weight.shape[0]):
            self.assertEqual(sorted(pack_index[layer].tolist()), list(range(n)))
        # rank_in_pack is all zeros
        self.assertTrue(torch.all(rank_in_pack == 0))

    def test_single_layer(self):
        """Works correctly with a single layer."""
        weight = torch.tensor([[3.0, 1.0, 4.0, 1.0]])
        pack_index, rank_in_pack = balanced_packing(weight, num_packs=2)
        self._check_shapes(weight, pack_index, rank_in_pack)
        self._check_items_per_pack(pack_index, num_packs=2, groups_per_pack=2)

    def test_uniform_weights_all_invariants(self):
        """Uniform weights: all invariants hold regardless of assignment."""
        weight = torch.ones(3, 8)
        num_packs = 4
        pack_index, rank_in_pack = balanced_packing(weight, num_packs=num_packs)
        self._check_shapes(weight, pack_index, rank_in_pack)
        self._check_pack_index_range(pack_index, num_packs)
        self._check_items_per_pack(pack_index, num_packs, groups_per_pack=2)
        self._check_rank_in_pack_range(rank_in_pack, groups_per_pack=2)
        self._check_unique_slots(pack_index, rank_in_pack, num_packs, groups_per_pack=2)

    def test_balance_property(self):
        """Heavier items are spread across packs to minimize max pack weight."""
        # Weights: [9, 1, 1, 1] with 2 packs → optimal: {9,1} and {1,1}, not {9,1,1} and {1}
        weight = torch.tensor([[9.0, 1.0, 1.0, 1.0]])
        pack_index, _ = balanced_packing(weight, num_packs=2)
        pack_weights = torch.zeros(2)
        for i, p in enumerate(pack_index[0].tolist()):
            pack_weights[p] += weight[0, i]
        # Max pack weight should be 10 (9+1), not 11 (9+1+1)
        self.assertEqual(pack_weights.max().item(), 10.0)

    def test_deterministic(self):
        """Same input always produces the same output."""
        weight = torch.rand(3, 8)
        result1 = balanced_packing(weight.clone(), num_packs=4)
        result2 = balanced_packing(weight.clone(), num_packs=4)
        self.assertTrue(torch.equal(result1[0], result2[0]))
        self.assertTrue(torch.equal(result1[1], result2[1]))

    def test_many_layers(self):
        """All invariants hold across many layers."""
        weight = torch.rand(16, 8)
        num_packs = 4
        pack_index, rank_in_pack = balanced_packing(weight, num_packs=num_packs)
        self._check_shapes(weight, pack_index, rank_in_pack)
        self._check_pack_index_range(pack_index, num_packs)
        self._check_items_per_pack(pack_index, num_packs, groups_per_pack=2)
        self._check_unique_slots(pack_index, rank_in_pack, num_packs, groups_per_pack=2)


if __name__ == "__main__":
    unittest.main()
