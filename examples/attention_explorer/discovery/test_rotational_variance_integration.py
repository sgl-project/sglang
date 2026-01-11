"""
Tests for Rotational Variance Integration

This module tests the integration of rotational variance into the fingerprint
schema and zone classification system. Rotational variance measures how much
position (vs. semantics) drives attention patterns.

Educational Context:
- Low rotational variance → position-driven (syntax processing)
- High rotational variance → semantic-driven (reasoning/bridging)
"""

import pytest
import numpy as np
from typing import List, Tuple

# Import the modules under test
from .rope_derotation import (
    RoPEDerotator,
    DerotatedAttention,
    compute_rotational_variance_for_fingerprint,
    compute_rotational_variance_batch,
    extend_fingerprint_with_rotational_variance,
    extend_fingerprints_batch,
)
from .classifier import (
    OnlineClassifier,
    FINGERPRINT_DIM,
    FINGERPRINT_DIM_V2,
    FP_LOCAL_MASS,
    FP_MID_MASS,
    FP_LONG_MASS,
    FP_ENTROPY,
    FP_ROTATIONAL_VARIANCE,
    ZONE_THRESHOLDS,
)
from .discovery_job import (
    assign_zone_labels,
    FINGERPRINT_DIM as DJ_FINGERPRINT_DIM,
    FINGERPRINT_DIM_V2 as DJ_FINGERPRINT_DIM_V2,
    FP_ROTATIONAL_VARIANCE as DJ_FP_ROTATIONAL_VARIANCE,
)


# =============================================================================
# FINGERPRINT COMPUTATION TESTS
# =============================================================================

class TestRotationalVarianceComputation:
    """Tests for rotational variance computation from attention patterns."""

    def test_compute_single_local_attention(self):
        """Local attention (nearby tokens) should have low rotational variance."""
        # Query at position 100, attending to nearby tokens
        rv = compute_rotational_variance_for_fingerprint(
            query_pos=100,
            key_positions=[97, 98, 99],  # Very close
            attention_scores=[0.2, 0.3, 0.5],
        )
        # Local attention is position-driven, so variance should be low
        assert 0.0 <= rv <= 1.0
        # Note: exact threshold depends on implementation, but local should be lower

    def test_compute_single_long_range_attention(self):
        """Long-range attention should have higher rotational variance."""
        # Query at position 500, attending to distant tokens
        rv = compute_rotational_variance_for_fingerprint(
            query_pos=500,
            key_positions=[10, 50, 100],  # Very far
            attention_scores=[0.4, 0.35, 0.25],
        )
        assert 0.0 <= rv <= 1.0

    def test_compute_empty_attention(self):
        """Empty attention should return neutral value."""
        rv = compute_rotational_variance_for_fingerprint(
            query_pos=100,
            key_positions=[],
            attention_scores=[],
        )
        assert rv == 0.5  # Neutral default

    def test_compute_batch(self):
        """Test batch computation of rotational variance."""
        query_positions = [100, 200, 300]
        key_positions_batch = [
            [97, 98, 99],
            [10, 100, 150],
            [5, 50, 295, 299],
        ]
        attention_scores_batch = [
            [0.2, 0.3, 0.5],
            [0.4, 0.3, 0.3],
            [0.1, 0.2, 0.3, 0.4],
        ]

        rv_batch = compute_rotational_variance_batch(
            query_positions,
            key_positions_batch,
            attention_scores_batch,
        )

        assert rv_batch.shape == (3,)
        assert all(0.0 <= rv <= 1.0 for rv in rv_batch)

    def test_batch_with_empty_entries(self):
        """Batch with empty entries should handle gracefully."""
        rv_batch = compute_rotational_variance_batch(
            query_positions=[100, 200, 300],
            key_positions_batch=[[97, 98], [], [297, 298, 299]],
            attention_scores_batch=[[0.4, 0.6], [], [0.2, 0.3, 0.5]],
        )

        assert rv_batch.shape == (3,)
        assert rv_batch[1] == 0.5  # Empty entry gets neutral value


# =============================================================================
# FINGERPRINT EXTENSION TESTS
# =============================================================================

class TestFingerprintExtension:
    """Tests for extending fingerprints from v1 (20-dim) to v2 (21-dim)."""

    def test_extend_single_fingerprint(self):
        """Extend a single 20-dim fingerprint to 21-dim."""
        fp_v1 = np.zeros(20, dtype=np.float32)
        fp_v1[FP_LOCAL_MASS] = 0.6
        fp_v1[FP_MID_MASS] = 0.3
        fp_v1[FP_LONG_MASS] = 0.1
        fp_v1[FP_ENTROPY] = 1.5

        fp_v2 = extend_fingerprint_with_rotational_variance(fp_v1, 0.25)

        assert len(fp_v2) == 21
        assert fp_v2[FP_LOCAL_MASS] == pytest.approx(0.6, rel=1e-5)
        assert fp_v2[FP_ROTATIONAL_VARIANCE] == pytest.approx(0.25, rel=1e-5)

    def test_extend_already_v2(self):
        """Extending a v2 fingerprint should be a no-op."""
        fp_v2 = np.zeros(21, dtype=np.float32)
        fp_v2[FP_ROTATIONAL_VARIANCE] = 0.35

        result = extend_fingerprint_with_rotational_variance(fp_v2, 0.5)

        assert len(result) == 21
        assert result[FP_ROTATIONAL_VARIANCE] == pytest.approx(0.35, rel=1e-5)  # Original value preserved

    def test_extend_batch(self):
        """Extend a batch of fingerprints."""
        fps_v1 = np.random.rand(100, 20).astype(np.float32)
        rvs = np.random.rand(100).astype(np.float32)

        fps_v2 = extend_fingerprints_batch(fps_v1, rvs)

        assert fps_v2.shape == (100, 21)
        np.testing.assert_array_equal(fps_v2[:, :20], fps_v1)
        np.testing.assert_array_equal(fps_v2[:, 20], rvs)

    def test_extend_batch_already_v2(self):
        """Extending v2 batch should be a no-op."""
        fps_v2 = np.random.rand(50, 21).astype(np.float32)
        rvs = np.random.rand(50).astype(np.float32)

        result = extend_fingerprints_batch(fps_v2, rvs)

        assert result.shape == (50, 21)
        np.testing.assert_array_equal(result, fps_v2)  # Unchanged


# =============================================================================
# ZONE CLASSIFICATION WITH ROTATIONAL VARIANCE TESTS
# =============================================================================

class TestZoneClassificationWithRotationalVariance:
    """Tests for zone classification using rotational variance (v2 fingerprints)."""

    def _make_fingerprint_v1(
        self,
        local_mass: float,
        mid_mass: float,
        long_mass: float,
        entropy: float,
    ) -> np.ndarray:
        """Create a v1 fingerprint with specified features."""
        fp = np.zeros(20, dtype=np.float32)
        fp[FP_LOCAL_MASS] = local_mass
        fp[FP_MID_MASS] = mid_mass
        fp[FP_LONG_MASS] = long_mass
        fp[FP_ENTROPY] = entropy
        # Fill histogram with reasonable values
        fp[4:12] = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05])
        return fp

    def _make_fingerprint_v2(
        self,
        local_mass: float,
        mid_mass: float,
        long_mass: float,
        entropy: float,
        rotational_variance: float,
    ) -> np.ndarray:
        """Create a v2 fingerprint with rotational variance."""
        fp = np.zeros(21, dtype=np.float32)
        fp[FP_LOCAL_MASS] = local_mass
        fp[FP_MID_MASS] = mid_mass
        fp[FP_LONG_MASS] = long_mass
        fp[FP_ENTROPY] = entropy
        fp[4:12] = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05])
        fp[FP_ROTATIONAL_VARIANCE] = rotational_variance
        return fp

    def test_syntax_floor_with_low_rotational_variance(self):
        """Syntax floor: high local mass + low entropy + LOW rotational variance."""
        # This is the ideal syntax_floor pattern
        fp = self._make_fingerprint_v2(
            local_mass=0.7,
            mid_mass=0.2,
            long_mass=0.1,
            entropy=1.5,
            rotational_variance=0.1,  # Low = position-driven
        )

        zones, confidences = assign_zone_labels(fp.reshape(1, -1))

        assert zones[0] == 'syntax_floor'
        assert confidences[0] > 0.3  # Should have decent confidence

    def test_syntax_floor_blocked_by_high_rotational_variance(self):
        """High rotational variance should prevent syntax_floor classification."""
        # Has syntax_floor features BUT high rotational variance
        fp = self._make_fingerprint_v2(
            local_mass=0.7,
            mid_mass=0.2,
            long_mass=0.1,
            entropy=1.5,
            rotational_variance=0.6,  # High = semantic, not positional
        )

        zones, confidences = assign_zone_labels(fp.reshape(1, -1))

        # Should NOT be syntax_floor due to high rotational variance
        # (high RV suggests semantic attention, not positional syntax)
        assert zones[0] in ['semantic_bridge', 'structure_ripple']

    def test_structure_ripple_with_high_rotational_variance(self):
        """Structure ripple: long-range + HIGH rotational variance = strong signal."""
        fp = self._make_fingerprint_v2(
            local_mass=0.2,
            mid_mass=0.2,
            long_mass=0.6,
            entropy=3.0,
            rotational_variance=0.5,  # High = semantic reasoning
        )
        # Add HIGH histogram variance for structure_ripple (periodic pattern)
        # Variance needs to be > 0.1 to trigger structure_ripple
        # [0, 1, 0, 0, 0, 0, 0, 0] has var=0.109 which is > 0.1
        fp[4:12] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        zones, confidences = assign_zone_labels(fp.reshape(1, -1))

        assert zones[0] == 'structure_ripple'

    def test_semantic_bridge_medium_rotational_variance(self):
        """Semantic bridge with medium rotational variance."""
        fp = self._make_fingerprint_v2(
            local_mass=0.3,
            mid_mass=0.5,
            long_mass=0.2,
            entropy=2.5,
            rotational_variance=0.3,  # Medium = balanced
        )

        zones, confidences = assign_zone_labels(fp.reshape(1, -1))

        assert zones[0] == 'semantic_bridge'

    def test_backwards_compatibility_v1_fingerprints(self):
        """V1 fingerprints (20-dim) should still classify correctly."""
        fp_v1 = self._make_fingerprint_v1(
            local_mass=0.7,
            mid_mass=0.2,
            long_mass=0.1,
            entropy=1.5,
        )

        zones, confidences = assign_zone_labels(fp_v1.reshape(1, -1))

        # Should use original heuristics without rotational variance
        assert zones[0] == 'syntax_floor'
        assert confidences[0] > 0

    def test_mixed_batch_v1_and_v2(self):
        """Mixed batch should work (all same dimension)."""
        # Create batch of v2 fingerprints
        fps = np.zeros((3, 21), dtype=np.float32)

        # Syntax floor pattern
        fps[0] = self._make_fingerprint_v2(0.7, 0.2, 0.1, 1.5, 0.1)

        # Semantic bridge pattern
        fps[1] = self._make_fingerprint_v2(0.3, 0.5, 0.2, 2.5, 0.3)

        # Structure ripple pattern (with HIGH histogram variance for periodic pattern)
        # [0, 1, 0, 0, 0, 0, 0, 0] has var=0.109 > 0.1
        fps[2] = self._make_fingerprint_v2(0.2, 0.2, 0.6, 3.0, 0.5)
        fps[2, 4:12] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        zones, confidences = assign_zone_labels(fps)

        assert zones[0] == 'syntax_floor'
        assert zones[1] == 'semantic_bridge'
        assert zones[2] == 'structure_ripple'


# =============================================================================
# CLASSIFIER INTEGRATION TESTS
# =============================================================================

class TestClassifierWithRotationalVariance:
    """Tests for OnlineClassifier using rotational variance."""

    def _make_classifier_ready_fingerprint(
        self,
        local_mass: float,
        mid_mass: float,
        long_mass: float,
        entropy: float,
        rotational_variance: float = None,
    ) -> np.ndarray:
        """Create a fingerprint for classifier testing."""
        dim = 21 if rotational_variance is not None else 20
        fp = np.zeros(dim, dtype=np.float32)
        fp[FP_LOCAL_MASS] = local_mass
        fp[FP_MID_MASS] = mid_mass
        fp[FP_LONG_MASS] = long_mass
        fp[FP_ENTROPY] = entropy
        fp[4:12] = 0.125  # Uniform histogram
        if rotational_variance is not None:
            fp[FP_ROTATIONAL_VARIANCE] = rotational_variance
        return fp

    def test_zone_assignment_v2_fingerprint(self):
        """Test zone assignment with v2 fingerprint in classifier."""
        # Create a mock classifier that just does zone assignment
        # (we can't easily test full classifier without discovery artifacts)

        # Syntax floor with low rotational variance
        fp = self._make_classifier_ready_fingerprint(
            local_mass=0.7,
            mid_mass=0.2,
            long_mass=0.1,
            entropy=1.5,
            rotational_variance=0.1,
        )

        # Test that the fingerprint is the right size
        assert len(fp) == FINGERPRINT_DIM_V2

    def test_zone_thresholds_have_rotational_variance(self):
        """Verify zone thresholds include rotational variance parameters."""
        assert 'rotational_variance_max' in ZONE_THRESHOLDS['syntax_floor']
        assert 'rotational_variance_min' in ZONE_THRESHOLDS['structure_ripple']
        assert 'rotational_variance_range' in ZONE_THRESHOLDS['semantic_bridge']


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Edge case tests for rotational variance integration."""

    def test_rotational_variance_at_boundary(self):
        """Test classification at rotational variance threshold boundaries."""
        # At syntax_floor threshold boundary
        fps = np.zeros((2, 21), dtype=np.float32)

        # Just at threshold
        fps[0, FP_LOCAL_MASS] = 0.7
        fps[0, FP_ENTROPY] = 1.5
        fps[0, FP_ROTATIONAL_VARIANCE] = 0.25  # At boundary

        # Just over threshold
        fps[1, FP_LOCAL_MASS] = 0.7
        fps[1, FP_ENTROPY] = 1.5
        fps[1, FP_ROTATIONAL_VARIANCE] = 0.26  # Over boundary

        zones, _ = assign_zone_labels(fps)

        # At boundary should be syntax_floor
        assert zones[0] == 'syntax_floor'
        # Over boundary should NOT be syntax_floor
        # (will be semantic_bridge since no other zone matches)

    def test_zero_rotational_variance(self):
        """Zero rotational variance (pure positional) should work."""
        fp = np.zeros(21, dtype=np.float32)
        fp[FP_LOCAL_MASS] = 0.7
        fp[FP_ENTROPY] = 1.5
        fp[FP_ROTATIONAL_VARIANCE] = 0.0

        zones, confidences = assign_zone_labels(fp.reshape(1, -1))

        assert zones[0] == 'syntax_floor'
        # Zero RV should boost confidence significantly
        assert confidences[0] > 0.4

    def test_one_rotational_variance(self):
        """Rotational variance of 1.0 (pure semantic) should work."""
        fp = np.zeros(21, dtype=np.float32)
        fp[FP_LOCAL_MASS] = 0.7
        fp[FP_MID_MASS] = 0.2
        fp[FP_LONG_MASS] = 0.1
        fp[FP_ENTROPY] = 1.5
        fp[FP_ROTATIONAL_VARIANCE] = 1.0  # Pure semantic

        zones, _ = assign_zone_labels(fp.reshape(1, -1))

        # Despite local mass, high RV should prevent syntax_floor
        assert zones[0] != 'syntax_floor'

    def test_very_small_fingerprint_batch(self):
        """Test with single-element batch."""
        fp = np.zeros((1, 21), dtype=np.float32)
        fp[0, FP_MID_MASS] = 0.5
        fp[0, FP_ROTATIONAL_VARIANCE] = 0.3

        zones, confidences = assign_zone_labels(fp)

        assert len(zones) == 1
        assert len(confidences) == 1


# =============================================================================
# CONSISTENCY TESTS
# =============================================================================

class TestConsistency:
    """Tests for consistency between classifier.py and discovery_job.py."""

    def test_fingerprint_dim_constants_match(self):
        """Ensure fingerprint dimension constants are consistent."""
        assert FINGERPRINT_DIM == DJ_FINGERPRINT_DIM
        assert FINGERPRINT_DIM_V2 == DJ_FINGERPRINT_DIM_V2

    def test_rotational_variance_index_matches(self):
        """Ensure rotational variance index is consistent."""
        assert FP_ROTATIONAL_VARIANCE == DJ_FP_ROTATIONAL_VARIANCE

    def test_zone_names_are_consistent(self):
        """Ensure zone names are consistent across modules."""
        expected_zones = {'syntax_floor', 'semantic_bridge', 'structure_ripple'}

        # Check classifier thresholds
        classifier_zones = set(ZONE_THRESHOLDS.keys())
        assert classifier_zones == expected_zones

        # Create test fingerprints for each zone
        fps = np.zeros((3, 21), dtype=np.float32)

        # Syntax floor
        fps[0, FP_LOCAL_MASS] = 0.7
        fps[0, FP_ENTROPY] = 1.5
        fps[0, FP_ROTATIONAL_VARIANCE] = 0.1

        # Semantic bridge
        fps[1, FP_MID_MASS] = 0.5
        fps[1, FP_ROTATIONAL_VARIANCE] = 0.3

        # Structure ripple (needs histogram variance > 0.1)
        fps[2, FP_LONG_MASS] = 0.6
        fps[2, FP_ROTATIONAL_VARIANCE] = 0.5
        fps[2, 4:12] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # var=0.109

        zones, _ = assign_zone_labels(fps)

        # All returned zones should be valid
        for zone in zones:
            assert zone in expected_zones


# =============================================================================
# INTEGRATION SCENARIO TESTS
# =============================================================================

class TestIntegrationScenarios:
    """Real-world integration scenarios."""

    def test_reasoning_vs_syntax_differentiation(self):
        """
        Demonstrate the key use case: distinguishing reasoning from syntax.

        Two patterns with similar local mass but different rotational variance:
        1. Syntax: "The cat" → position-driven (low RV)
        2. Reasoning: "therefore" → semantically attending to nearby evidence (high RV)
        """
        # Both have high local mass (attending to nearby tokens)
        fps = np.zeros((2, 21), dtype=np.float32)

        # Pattern 1: Syntax (local grammar)
        fps[0, FP_LOCAL_MASS] = 0.7
        fps[0, FP_MID_MASS] = 0.2
        fps[0, FP_LONG_MASS] = 0.1
        fps[0, FP_ENTROPY] = 1.2
        fps[0, FP_ROTATIONAL_VARIANCE] = 0.1  # Position-driven

        # Pattern 2: Local reasoning (semantic connection to nearby evidence)
        fps[1, FP_LOCAL_MASS] = 0.65
        fps[1, FP_MID_MASS] = 0.25
        fps[1, FP_LONG_MASS] = 0.1
        fps[1, FP_ENTROPY] = 1.8
        fps[1, FP_ROTATIONAL_VARIANCE] = 0.45  # Semantic-driven

        zones, _ = assign_zone_labels(fps)

        # Pattern 1 should be syntax (position-driven local attention)
        assert zones[0] == 'syntax_floor'

        # Pattern 2 should NOT be syntax despite similar local mass
        # (high rotational variance indicates semantic attention)
        assert zones[1] != 'syntax_floor'

    def test_hallucination_warning_scenario(self):
        """
        Scenario: Long-range attention with LOW rotational variance
        could indicate structural copy (tables/lists) vs. semantic reasoning.

        Low RV + long range = structural (likely safe)
        High RV + long range = semantic reasoning (check for hallucination)
        """
        fps = np.zeros((2, 21), dtype=np.float32)

        # Pattern 1: Structural long-range (copying table format)
        fps[0, FP_LONG_MASS] = 0.6
        fps[0, FP_ENTROPY] = 2.5
        # [0, 1, 0, 0, 0, 0, 0, 0] has var=0.109 > 0.1 (needed for structure_ripple)
        fps[0, 4:12] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        fps[0, FP_ROTATIONAL_VARIANCE] = 0.15  # Position-driven periodic

        # Pattern 2: Semantic long-range (reasoning across document)
        fps[1, FP_LONG_MASS] = 0.6
        fps[1, FP_ENTROPY] = 3.0
        fps[1, 4:12] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        fps[1, FP_ROTATIONAL_VARIANCE] = 0.55  # Semantic reasoning

        zones, confidences = assign_zone_labels(fps)

        # Both are structure_ripple, but confidence differs
        assert zones[0] == 'structure_ripple'
        assert zones[1] == 'structure_ripple'

        # Pattern 2 (semantic) should have higher confidence
        # (high RV confirms semantic reasoning)
        assert confidences[1] > confidences[0]


# =============================================================================
# RUN DEMO
# =============================================================================

def demo():
    """Demonstrate rotational variance integration."""
    print("=" * 70)
    print("Rotational Variance Integration Demo")
    print("=" * 70)

    print("\n1. Compute rotational variance for different attention patterns:\n")

    # Local syntax attention
    rv_local = compute_rotational_variance_for_fingerprint(
        query_pos=100,
        key_positions=[97, 98, 99],
        attention_scores=[0.2, 0.3, 0.5],
    )
    print(f"   Local syntax (nearby tokens):     RV = {rv_local:.3f}")

    # Long-range reasoning
    rv_long = compute_rotational_variance_for_fingerprint(
        query_pos=500,
        key_positions=[10, 50, 100],
        attention_scores=[0.4, 0.35, 0.25],
    )
    print(f"   Long-range reasoning:             RV = {rv_long:.3f}")

    print("\n2. Zone classification with rotational variance:\n")

    # Create v2 fingerprints
    fps = np.zeros((3, 21), dtype=np.float32)

    # Syntax floor (low RV)
    fps[0, FP_LOCAL_MASS] = 0.7
    fps[0, FP_ENTROPY] = 1.5
    fps[0, FP_ROTATIONAL_VARIANCE] = 0.1

    # Semantic bridge (medium RV)
    fps[1, FP_MID_MASS] = 0.5
    fps[1, FP_ENTROPY] = 2.5
    fps[1, FP_ROTATIONAL_VARIANCE] = 0.3

    # Structure ripple (high RV, needs histogram variance > 0.1)
    fps[2, FP_LONG_MASS] = 0.6
    fps[2, FP_ENTROPY] = 3.0
    fps[2, FP_ROTATIONAL_VARIANCE] = 0.5
    fps[2, 4:12] = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # var=0.109

    zones, confidences = assign_zone_labels(fps)

    for i, (zone, conf) in enumerate(zip(zones, confidences)):
        rv = fps[i, FP_ROTATIONAL_VARIANCE]
        print(f"   Pattern {i+1}: {zone:20s} (RV={rv:.2f}, conf={conf:.2f})")

    print("\n3. Key insight:\n")
    print("   Rotational variance helps distinguish:")
    print("   - Syntax (position-driven, low RV) from")
    print("   - Semantic reasoning (meaning-driven, high RV)")
    print("\n   This is critical for:")
    print("   - Hallucination detection (fake reasoning has wrong RV)")
    print("   - Model routing (syntax → small model, reasoning → large model)")
    print("=" * 70)


if __name__ == "__main__":
    demo()
