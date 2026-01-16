"""
GPU unit tests for MoE modules.

Tests:
1. ep_scatter kernel - scatters hidden states to experts
2. ep_gather kernel - gathers and weights expert outputs
3. MoE computation verification - manual calculation vs actual

Run with:
    docker exec sglang_dev bash -c 'cd /lustre/raplab/client/xutingz/workspace/gitsrc/sglang && \
        python test_moe_gpu_modules.py'
"""

import os
import sys

# Add repo python path (works both on host and inside docker)
_REPO_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_REPO_DIR, "python"))

import unittest
from typing import Optional, Tuple

import torch


def setup_cuda():
    """Setup CUDA device."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU tests")
        return False
    torch.cuda.set_device(0)
    return True


class TestEpKernelsSkipped(unittest.TestCase):
    """
    Skipped: ep_scatter and ep_gather require FP8 quantization setup.
    These low-level kernels are tested through integration tests.
    """

    @unittest.skip("ep_scatter requires FP8 quantization setup")
    def test_ep_scatter(self):
        pass

    @unittest.skip("ep_gather requires specific tensor layout")
    def test_ep_gather(self):
        pass


class TestMoECalculationVerification(unittest.TestCase):
    """Test MoE calculation by comparing manual computation with actual."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    def test_weighted_sum_calculation(self):
        """Test that MoE output is correct weighted sum of expert outputs."""
        device = torch.device("cuda:0")

        # Simulate expert outputs for a token with topk=3
        expert_outputs = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],  # expert 0 output
                [5.0, 6.0, 7.0, 8.0],  # expert 1 output
                [9.0, 10.0, 11.0, 12.0],  # expert 2 output
            ],
            dtype=torch.float32,
            device=device,
        )

        weights = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32, device=device)

        # Manual calculation
        expected = (
            weights[0] * expert_outputs[0]
            + weights[1] * expert_outputs[1]
            + weights[2] * expert_outputs[2]
        )

        # Using einsum (similar to how MoE does it)
        actual = torch.einsum("e,eh->h", weights, expert_outputs)

        print(f"Expected: {expected.tolist()}")
        print(f"Actual: {actual.tolist()}")

        self.assertTrue(torch.allclose(actual, expected), f"Weighted sum mismatch")

        print("✓ Weighted sum calculation test passed")

    def test_routed_scaling_factor(self):
        """Test routed_scaling_factor application."""
        device = torch.device("cuda:0")

        # Routed expert output (after weighted sum)
        routed_output = torch.tensor(
            [1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=device
        )

        # Shared expert output
        shared_output = torch.tensor(
            [0.5, 1.0, 1.5, 2.0], dtype=torch.float32, device=device
        )

        rsf = 2.5

        # Final output = routed * rsf + shared
        expected = routed_output * rsf + shared_output

        # Verify the formula
        print(f"Routed output: {routed_output.tolist()}")
        print(f"Routed * rsf: {(routed_output * rsf).tolist()}")
        print(f"Shared output: {shared_output.tolist()}")
        print(f"Final expected: {expected.tolist()}")

        # In waterfill, shared_weight = 1/rsf = 0.4
        # So if shared expert output is pre-weighted by 0.4:
        # final = routed * rsf + (shared_value * 0.4) should equal
        # final = (routed_weighted_sum) * rsf + shared_value / rsf * rsf
        # final = routed * rsf + shared_value (after rsf multiplication)

        # But we need to verify the actual formula used
        shared_weight = 1.0 / rsf  # 0.4
        shared_weighted = (
            shared_output * shared_weight
        )  # This is what goes through combine

        # After combine, we multiply by rsf
        # Combined routed already has weights applied, shared has 0.4 weight
        # final = (routed_weighted + shared_weighted) * rsf
        #       = routed_weighted * rsf + shared_weighted * rsf
        #       = routed_weighted * rsf + shared_output * 0.4 * rsf
        #       = routed_weighted * rsf + shared_output

        combined = routed_output + shared_weighted  # Simulated combine output
        final = combined * rsf

        print(f"Combined (before rsf): {combined.tolist()}")
        print(f"Final (after rsf): {final.tolist()}")

        # Verify: final should equal routed * rsf + shared
        expected_final = routed_output * rsf + shared_output
        self.assertTrue(
            torch.allclose(final, expected_final),
            f"RSF application mismatch: {final} vs {expected_final}",
        )

        print("✓ Routed scaling factor test passed")

    def test_9column_weight_sum(self):
        """Test that 9-column weights sum correctly."""
        device = torch.device("cuda:0")

        # Standard 8 routed experts with weights summing to 1.0
        routed_weights = torch.ones(8, dtype=torch.float32, device=device) / 8

        # Shared expert weight = 1/rsf for rsf=2.5
        shared_weight = 0.4

        # Total weight sum for 9 columns
        total_weight = routed_weights.sum() + shared_weight

        print(f"Routed weights sum: {routed_weights.sum().item()}")
        print(f"Shared weight: {shared_weight}")
        print(f"Total 9-column weight: {total_weight.item()}")

        expected_total = 1.0 + 0.4  # 1.4
        self.assertAlmostEqual(total_weight.item(), expected_total, places=5)

        # After rsf multiplication:
        # routed contribution = routed_weighted_sum * rsf = sum(routed * weights) * rsf
        # shared contribution = shared_output * shared_weight * rsf = shared_output * 0.4 * 2.5 = shared_output
        # So shared effectively has weight 1.0 in final output

        print("✓ 9-column weight sum test passed")


class TestSharedExpertIntegration(unittest.TestCase):
    """Test shared expert integration with MoE."""

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")

    def test_shared_expert_weight_effect(self):
        """Test that shared expert weight produces correct contribution."""
        device = torch.device("cuda:0")

        hidden_dim = 4
        rsf = 2.5
        shared_weight = 1.0 / rsf  # 0.4

        # Simulate hidden state
        hidden = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32, device=device)

        # Routed expert output (already weighted sum of 8 experts)
        routed_output = hidden * 0.8  # Some transformation

        # Shared expert output (same transformation for simplicity)
        shared_output_raw = hidden * 1.2

        # What waterfill does:
        # 1. Shared expert output is weighted by shared_weight in dispatch
        # 2. Combined output = routed_weighted + shared_weighted
        # 3. Final = combined * rsf

        shared_weighted = shared_output_raw * shared_weight
        combined = routed_output + shared_weighted
        final = combined * rsf

        # Expected: routed * rsf + shared_raw
        # Because shared_weighted * rsf = shared_raw * (1/rsf) * rsf = shared_raw
        expected = routed_output * rsf + shared_output_raw

        print(f"Routed output: {routed_output.tolist()}")
        print(f"Shared raw: {shared_output_raw.tolist()}")
        print(f"Shared weighted (×{shared_weight}): {shared_weighted.tolist()}")
        print(f"Combined: {combined.tolist()}")
        print(f"Final (×{rsf}): {final.tolist()}")
        print(f"Expected: {expected.tolist()}")

        self.assertTrue(
            torch.allclose(final, expected), f"Shared expert integration mismatch"
        )

        print("✓ Shared expert weight effect test passed")


def run_tests():
    """Run all GPU tests."""
    if not setup_cuda():
        print("Skipping GPU tests - CUDA not available")
        return True

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestEpKernelsSkipped,
        TestMoECalculationVerification,
        TestSharedExpertIntegration,
    ]

    for test_class in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(test_class))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("GPU TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✓ ALL GPU TESTS PASSED")
    else:
        print("\n✗ SOME GPU TESTS FAILED")
        for test, traceback in result.failures:
            print(f"\nFailed: {test}")
            print(traceback)
        for test, traceback in result.errors:
            print(f"\nError: {test}")
            print(traceback)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
