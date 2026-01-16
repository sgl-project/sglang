"""
Comprehensive unit tests for DeepEP Waterfill modules.

Tests each module independently:
1. Expert ID Remapping
2. Shared Expert Weight Calculation
3. Waterfill Load Balancing
4. Token Count Aggregation
5. Local Shared Expert Identification

Run: python test_waterfill_modules.py
"""

import os
import sys

# Add sglang path
module_path = os.path.join(os.path.dirname(__file__), "python/sglang/srt/layers/moe")
sys.path.insert(0, module_path)

import unittest
from typing import Tuple

import torch

# Import functions to test
from deepep_waterfill import (
    LOCAL_SHARED_MARKER,
    DeepEPWaterfillBalancer,
    assign_shared_destination_pytorch,
    compute_local_shared_expert,
    count_routed_per_rank_pytorch,
    expand_topk_with_shared_expert,
    identify_shared_expert_tokens,
)


class TestExpertIDRemapping(unittest.TestCase):
    """Test expert ID remapping logic.

    Old layout: 256 experts, 32 per rank (ranks 0-7)
    New layout: 264 experts, 33 per rank (32 routed + 1 shared)

    Remapping: old_id -> old_id + (old_id // old_experts_per_rank)
    """

    def setUp(self):
        self.num_routed_experts = 256
        self.world_size = 8
        self.old_experts_per_rank = 32
        self.new_experts_per_rank = 33

    def test_rank0_expert_remapping(self):
        """Rank 0 experts [0-31] should stay [0-31]."""
        for old_id in range(32):
            old_rank = old_id // self.old_experts_per_rank  # 0
            new_id = old_id + old_rank  # old_id + 0
            self.assertEqual(
                new_id, old_id, f"Rank 0 expert {old_id} should not change"
            )

    def test_rank1_expert_remapping(self):
        """Rank 1 experts [32-63] should become [33-64]."""
        for local_id in range(32):
            old_id = 32 + local_id
            old_rank = old_id // self.old_experts_per_rank  # 1
            new_id = old_id + old_rank  # old_id + 1
            expected = 33 + local_id
            self.assertEqual(
                new_id, expected, f"Expert {old_id} -> {new_id}, expected {expected}"
            )

    def test_rank7_expert_remapping(self):
        """Rank 7 experts [224-255] should become [231-262]."""
        for local_id in range(32):
            old_id = 224 + local_id
            old_rank = old_id // self.old_experts_per_rank  # 7
            new_id = old_id + old_rank  # old_id + 7
            expected = 231 + local_id
            self.assertEqual(
                new_id, expected, f"Expert {old_id} -> {new_id}, expected {expected}"
            )

    def test_shared_expert_ids(self):
        """Shared expert IDs should be at end of each rank's range."""
        for rank in range(self.world_size):
            shared_id = rank * self.new_experts_per_rank + self.old_experts_per_rank
            expected = rank * 33 + 32
            self.assertEqual(shared_id, expected, f"Rank {rank} shared expert ID")

        # Verify shared expert IDs
        expected_shared_ids = [32, 65, 98, 131, 164, 197, 230, 263]
        for rank, expected in enumerate(expected_shared_ids):
            actual = rank * self.new_experts_per_rank + self.old_experts_per_rank
            self.assertEqual(actual, expected, f"Rank {rank} shared ID")

    def test_expand_topk_remapping(self):
        """Test that expand_topk_with_shared_expert correctly remaps IDs."""
        topk_ids = torch.tensor(
            [
                [0, 32, 64, 96, 128, 160, 192, 224],  # One expert from each rank
            ],
            dtype=torch.int64,
        )
        topk_weights = torch.ones(1, 8, dtype=torch.float32) * 0.125
        shared_destination = torch.tensor([0], dtype=torch.int64)

        expanded_ids, expanded_weights, local_mask = expand_topk_with_shared_expert(
            topk_ids,
            topk_weights,
            shared_destination,
            self.num_routed_experts,
            self.world_size,
            0,
            0.4,
        )

        # Expected remapped IDs: 0+0, 32+1, 64+2, 96+3, 128+4, 160+5, 192+6, 224+7
        expected_remapped = [0, 33, 66, 99, 132, 165, 198, 231]
        for i, expected in enumerate(expected_remapped):
            self.assertEqual(
                expanded_ids[0, i].item(),
                expected,
                f"Column {i}: expected {expected}, got {expanded_ids[0, i].item()}",
            )

        # 9th column should be shared expert ID for rank 0: 0 * 33 + 32 = 32
        self.assertEqual(expanded_ids[0, 8].item(), 32)


class TestSharedExpertWeight(unittest.TestCase):
    """Test shared expert weight calculation.

    shared_weight = 1.0 / routed_scaling_factor
    """

    def test_rsf_2_5(self):
        """rsf=2.5 -> shared_weight=0.4"""
        balancer = DeepEPWaterfillBalancer(256, 8, 0, 2.5)
        self.assertAlmostEqual(balancer.shared_weight, 0.4, places=6)

    def test_rsf_1_0(self):
        """rsf=1.0 -> shared_weight=1.0"""
        balancer = DeepEPWaterfillBalancer(256, 8, 0, 1.0)
        self.assertAlmostEqual(balancer.shared_weight, 1.0, places=6)

    def test_rsf_4_0(self):
        """rsf=4.0 -> shared_weight=0.25"""
        balancer = DeepEPWaterfillBalancer(256, 8, 0, 4.0)
        self.assertAlmostEqual(balancer.shared_weight, 0.25, places=6)

    def test_weight_in_expanded_topk(self):
        """Test that 9th column weight equals shared_weight."""
        balancer = DeepEPWaterfillBalancer(256, 8, 0, 2.5)

        topk_ids = torch.randint(0, 256, (10, 8), dtype=torch.int64)
        topk_weights = torch.rand(10, 8, dtype=torch.float32)
        routed_counts = torch.randint(100, 200, (8,), dtype=torch.int64)

        expanded_ids, expanded_weights, _ = balancer.prepare_dispatch(
            topk_ids, topk_weights, routed_counts
        )

        # All 9th column weights should be 0.4
        expected_weight = 0.4
        for i in range(10):
            self.assertAlmostEqual(
                expanded_weights[i, 8].item(),
                expected_weight,
                places=5,
                msg=f"Token {i} shared weight",
            )


class TestWaterfillLoadBalancing(unittest.TestCase):
    """Test waterfill load balancing algorithm."""

    def setUp(self):
        self.num_experts = 256
        self.world_size = 8

    def test_selects_lowest_load_candidate(self):
        """Waterfill should select the lowest-load candidate rank."""
        # Token routes to ranks 0, 1, 2 (experts 0, 32, 64)
        topk_ids = torch.tensor(
            [
                [0, 32, 64, -1, -1, -1, -1, -1],
            ],
            dtype=torch.int64,
        )

        # Rank 2 has lowest load among candidates
        routed_counts = torch.tensor(
            [100, 90, 20, 80, 70, 60, 50, 40], dtype=torch.int64
        )

        dest = assign_shared_destination_pytorch(
            topk_ids, routed_counts, self.num_experts, self.world_size, source_rank=0
        )

        self.assertEqual(dest[0].item(), 2, "Should select rank 2 (lowest load)")

    def test_source_rank_can_be_selected(self):
        """Source rank should be selected if it has lowest load."""
        topk_ids = torch.tensor(
            [
                [32, 64, 96, -1, -1, -1, -1, -1],  # routes to ranks 1, 2, 3
            ],
            dtype=torch.int64,
        )

        # Source rank 0 has lowest load
        routed_counts = torch.tensor(
            [5, 100, 100, 100, 100, 100, 100, 100], dtype=torch.int64
        )

        dest = assign_shared_destination_pytorch(
            topk_ids, routed_counts, self.num_experts, self.world_size, source_rank=0
        )

        self.assertEqual(dest[0].item(), 0, "Should select source rank 0")

    def test_waterfill_distribution(self):
        """Test that waterfill distributes load to low-load ranks."""
        num_tokens = 1000

        # Tokens route to multiple ranks
        topk_ids = torch.zeros(num_tokens, 8, dtype=torch.int64)
        for t in range(num_tokens):
            topk_ids[t, 0] = t % 32  # rank 0
            topk_ids[t, 1] = 64 + (t % 32)  # rank 2
            topk_ids[t, 2] = 224 + (t % 32)  # rank 7
            topk_ids[t, 3:] = -1

        # High load on rank 0, low load on ranks 2, 7
        routed_counts = torch.tensor(
            [1000, 500, 50, 500, 500, 500, 500, 50], dtype=torch.int64
        )

        dest = assign_shared_destination_pytorch(
            topk_ids, routed_counts, self.num_experts, self.world_size, source_rank=0
        )

        dest_counts = torch.bincount(dest, minlength=self.world_size)

        # Low load ranks (2, 7) should get more tokens
        low_load_total = dest_counts[2].item() + dest_counts[7].item()
        high_load = dest_counts[0].item()

        self.assertGreater(
            low_load_total,
            high_load,
            f"Low load ranks should get more tokens: {low_load_total} vs {high_load}",
        )


class TestTokenCountAggregation(unittest.TestCase):
    """Test token counting per rank."""

    def test_basic_count(self):
        """Test basic token counting."""
        topk_ids = torch.tensor(
            [
                [0, 32, 64],  # ranks 0, 1, 2 -> 1 each
                [0, 1, 2],  # rank 0 only -> 3
                [224, 225, 226],  # rank 7 only -> 3
            ],
            dtype=torch.int64,
        )

        counts = count_routed_per_rank_pytorch(topk_ids, 256, 8)

        # Expected: rank 0 has 4 (1+3), rank 1 has 1, rank 2 has 1, rank 7 has 3
        expected = torch.tensor([4, 1, 1, 0, 0, 0, 0, 3], dtype=torch.int64)
        self.assertTrue(
            torch.equal(counts, expected), f"Expected {expected}, got {counts}"
        )

    def test_invalid_ids_ignored(self):
        """Test that -1 IDs are ignored."""
        topk_ids = torch.tensor(
            [
                [0, -1, -1],
                [-1, -1, -1],
                [32, 64, -1],
            ],
            dtype=torch.int64,
        )

        counts = count_routed_per_rank_pytorch(topk_ids, 256, 8)

        expected = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.int64)
        self.assertTrue(torch.equal(counts, expected))

    def test_empty_input(self):
        """Test empty input handling."""
        topk_ids = torch.empty(0, 8, dtype=torch.int64)
        counts = count_routed_per_rank_pytorch(topk_ids, 256, 8)

        expected = torch.zeros(8, dtype=torch.int64)
        self.assertTrue(torch.equal(counts, expected))


class TestLocalSharedExpertIdentification(unittest.TestCase):
    """Test identification of tokens for local shared expert computation."""

    def test_identify_remote_shared_tokens(self):
        """Test identification of remote shared expert tokens.

        NOTE: identify_shared_expert_tokens uses num_experts (original routed experts)
        and computes target_rank = virtual_id // experts_per_rank.

        With num_experts=256, experts_per_rank=32:
        - virtual_id 64 -> rank 64//32 = 2
        - virtual_id 32 -> rank 32//32 = 1
        - virtual_id 96 -> rank 96//32 = 3
        """
        # Using old virtual ID scheme (expert_id // 32 = target_rank)
        recv_topk_ids = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 64],  # 9th col = 64, rank = 64//32 = 2
                [0, 1, 2, 3, 4, 5, 6, 7, 32],  # 9th col = 32, rank = 32//32 = 1
                [0, 1, 2, 3, 4, 5, 6, 7, 0],  # 9th col = 0, rank = 0//32 = 0
                [0, 1, 2, 3, 4, 5, 6, 7, 65],  # 9th col = 65, rank = 65//32 = 2
            ],
            dtype=torch.int64,
        )

        # Current rank = 2, should identify tokens 0 and 3 (virtual IDs 64 and 65)
        indices = identify_shared_expert_tokens(recv_topk_ids, 256, 8, current_rank=2)

        expected = torch.tensor([0, 3])
        self.assertTrue(
            torch.equal(indices, expected), f"Expected {expected}, got {indices}"
        )

    def test_local_mask_from_balancer(self):
        """Test local_shared_mask from prepare_dispatch."""
        balancer = DeepEPWaterfillBalancer(256, 8, 0, 2.5)

        # Create tokens where some should be local (routed to source rank 0)
        topk_ids = torch.tensor(
            [
                [0, 32, 64, -1, -1, -1, -1, -1],  # routes to 0, 1, 2
                [32, 64, 96, -1, -1, -1, -1, -1],  # routes to 1, 2, 3 (not 0)
                [0, 1, 2, -1, -1, -1, -1, -1],  # routes to 0 only
            ],
            dtype=torch.int64,
        )
        topk_weights = torch.ones(3, 8) * 0.125

        # Source rank 0 has lowest load for tokens 0 and 2
        routed_counts = torch.tensor(
            [10, 100, 100, 100, 100, 100, 100, 100], dtype=torch.int64
        )

        _, _, local_mask = balancer.prepare_dispatch(
            topk_ids, topk_weights, routed_counts
        )

        # Tokens 0 and 2 should be local (source rank 0 is candidate and has lowest load)
        # Token 1 routes to 1,2,3 so source rank 0 is still a candidate, but rank 1 might have lower load
        # Actually all tokens can include source rank as candidate
        self.assertEqual(
            local_mask.sum().item(),
            3,
            "All tokens should be local when source rank has lowest load",
        )


class TestComputeLocalSharedExpert(unittest.TestCase):
    """Test local shared expert computation helper."""

    def test_extracts_correct_tokens(self):
        """Test that correct tokens are extracted for local computation."""
        hidden_states = torch.arange(10 * 4).reshape(10, 4).float()
        local_mask = torch.tensor(
            [False, True, False, True, True, False, False, True, False, False]
        )

        def mock_expert_fn(x):
            return x * 2

        output, indices = compute_local_shared_expert(
            hidden_states, local_mask, mock_expert_fn
        )

        expected_indices = torch.tensor([1, 3, 4, 7])
        self.assertTrue(torch.equal(indices, expected_indices))

        # Output should be 2x the selected hidden states
        expected_output = hidden_states[expected_indices] * 2
        self.assertTrue(torch.allclose(output, expected_output))

    def test_empty_mask(self):
        """Test when no tokens are local."""
        hidden_states = torch.randn(10, 4)
        local_mask = torch.zeros(10, dtype=torch.bool)

        output, indices = compute_local_shared_expert(
            hidden_states, local_mask, lambda x: x
        )

        self.assertIsNone(output)
        self.assertIsNone(indices)

    def test_all_local(self):
        """Test when all tokens are local."""
        hidden_states = torch.randn(5, 4)
        local_mask = torch.ones(5, dtype=torch.bool)

        output, indices = compute_local_shared_expert(
            hidden_states, local_mask, lambda x: x * 3
        )

        self.assertEqual(len(indices), 5)
        self.assertTrue(torch.allclose(output, hidden_states * 3))


class TestBalancerConfiguration(unittest.TestCase):
    """Test DeepEPWaterfillBalancer configuration."""

    def test_expert_counts(self):
        """Test expert count configuration."""
        balancer = DeepEPWaterfillBalancer(256, 8, 0, 2.5)

        self.assertEqual(balancer.num_routed_experts, 256)
        self.assertEqual(balancer.old_experts_per_rank, 32)
        self.assertEqual(balancer.new_experts_per_rank, 33)
        self.assertEqual(balancer.num_experts, 264)  # 33 * 8

    def test_my_shared_expert_id(self):
        """Test per-rank shared expert ID."""
        for rank in range(8):
            balancer = DeepEPWaterfillBalancer(256, 8, rank, 2.5)
            expected = rank * 33 + 32
            self.assertEqual(
                balancer.my_shared_expert_id, expected, f"Rank {rank} shared expert ID"
            )

    def test_min_batch_optimization(self):
        """Test that small batches are all local."""
        balancer = DeepEPWaterfillBalancer(256, 8, 0, 2.5)

        # Batch smaller than MIN_BATCH_FOR_BALANCE
        small_batch = 32
        topk_ids = torch.randint(0, 256, (small_batch, 8), dtype=torch.int64)
        topk_weights = torch.rand(small_batch, 8)
        routed_counts = torch.randint(100, 200, (8,), dtype=torch.int64)

        _, _, local_mask = balancer.prepare_dispatch(
            topk_ids, topk_weights, routed_counts
        )

        # All should be local for small batches
        self.assertTrue(local_mask.all(), "Small batches should be all local")


class TestEndToEndFlow(unittest.TestCase):
    """Test end-to-end waterfill flow."""

    def test_prepare_dispatch_shapes(self):
        """Test that prepare_dispatch returns correct shapes."""
        balancer = DeepEPWaterfillBalancer(256, 8, 0, 2.5)

        batch_size = 100
        topk_ids = torch.randint(0, 256, (batch_size, 8), dtype=torch.int64)
        topk_weights = torch.rand(batch_size, 8)
        routed_counts = torch.randint(100, 200, (8,), dtype=torch.int64)

        expanded_ids, expanded_weights, local_mask = balancer.prepare_dispatch(
            topk_ids, topk_weights, routed_counts
        )

        # Check shapes
        self.assertEqual(expanded_ids.shape, (batch_size, 9))
        self.assertEqual(expanded_weights.shape, (batch_size, 9))
        self.assertEqual(local_mask.shape, (batch_size,))

    def test_weights_preservation(self):
        """Test that routed weights are preserved."""
        balancer = DeepEPWaterfillBalancer(256, 8, 0, 2.5)

        topk_ids = torch.randint(0, 256, (50, 8), dtype=torch.int64)
        topk_weights = torch.rand(50, 8)
        routed_counts = torch.randint(100, 200, (8,), dtype=torch.int64)

        _, expanded_weights, _ = balancer.prepare_dispatch(
            topk_ids, topk_weights, routed_counts
        )

        # First 8 columns should match original weights
        self.assertTrue(torch.allclose(expanded_weights[:, :8], topk_weights))

    def test_empty_batch(self):
        """Test empty batch handling."""
        balancer = DeepEPWaterfillBalancer(256, 8, 0, 2.5)

        topk_ids = torch.empty(0, 8, dtype=torch.int64)
        topk_weights = torch.empty(0, 8)
        routed_counts = torch.zeros(8, dtype=torch.int64)

        expanded_ids, expanded_weights, local_mask = balancer.prepare_dispatch(
            topk_ids, topk_weights, routed_counts
        )

        self.assertEqual(expanded_ids.shape, (0, 9))
        self.assertEqual(expanded_weights.shape, (0, 9))
        self.assertEqual(local_mask.shape, (0,))


def run_tests():
    """Run all tests and print summary."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestExpertIDRemapping,
        TestSharedExpertWeight,
        TestWaterfillLoadBalancing,
        TestTokenCountAggregation,
        TestLocalSharedExpertIdentification,
        TestComputeLocalSharedExpert,
        TestBalancerConfiguration,
        TestEndToEndFlow,
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    # Run with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✓ ALL TESTS PASSED")
    else:
        print("\n✗ SOME TESTS FAILED")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
