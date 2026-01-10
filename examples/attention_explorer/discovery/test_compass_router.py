"""
Tests for Compass Router

Tests the angular analysis and routing logic for the compass-based
query router that uses Sinq anchoring.
"""

import pytest
import numpy as np
from typing import List

from .compass_router import (
    SinqAnchor,
    CompassReading,
    CompassRoutingDecision,
    CompassHeading,
    RoutingTier,
    CompassAnalyzer,
    CompassRouter,
    CompassRouterConfig,
    COMPASS_GLOSSARY,
    create_compass_router,
    analyze_attention_compass,
)


# =============================================================================
# SINQ ANCHOR TESTS
# =============================================================================

class TestSinqAnchor:
    """Test SinqAnchor extraction."""

    def test_from_attention_basic(self):
        """Test basic anchor extraction."""
        anchor = SinqAnchor.from_attention(
            positions=[0, 1, 2, 10, 20],
            attention_scores=[0.2, 0.1, 0.1, 0.3, 0.3],
            sink_threshold=5,
        )

        assert anchor.sink_positions == [0, 1, 2]
        assert anchor.sink_total == pytest.approx(0.4, abs=0.01)
        assert anchor.is_sink_dominated  # 40% > 30% threshold

    def test_sink_dominated_detection(self):
        """Test detection of sink-dominated attention."""
        anchor = SinqAnchor.from_attention(
            positions=[0, 1, 2, 3, 10],
            attention_scores=[0.3, 0.2, 0.2, 0.1, 0.2],
            sink_threshold=5,
        )

        assert anchor.is_sink_dominated  # 80% > 30%

    def test_no_sink_tokens(self):
        """Test when no sink tokens in attention."""
        anchor = SinqAnchor.from_attention(
            positions=[10, 20, 30],
            attention_scores=[0.3, 0.4, 0.3],
            sink_threshold=5,
        )

        assert anchor.sink_positions == []
        assert anchor.sink_total == 0.0
        assert not anchor.is_sink_dominated

    def test_sink_entropy(self):
        """Test entropy calculation within sink."""
        # Uniform distribution has max entropy
        anchor_uniform = SinqAnchor.from_attention(
            positions=[0, 1, 2, 3, 4],
            attention_scores=[0.2, 0.2, 0.2, 0.2, 0.2],
            sink_threshold=5,
        )

        # Single token has zero entropy
        anchor_single = SinqAnchor.from_attention(
            positions=[0, 10, 20],
            attention_scores=[1.0, 0.0, 0.0],
            sink_threshold=5,
        )

        assert anchor_uniform.sink_entropy > anchor_single.sink_entropy

    def test_to_dict(self):
        """Test serialization."""
        anchor = SinqAnchor.from_attention(
            positions=[0, 1, 10],
            attention_scores=[0.3, 0.2, 0.5],
            sink_threshold=5,
        )

        d = anchor.to_dict()
        assert "sink_positions" in d
        assert "sink_total" in d
        assert "is_sink_dominated" in d


# =============================================================================
# COMPASS ANALYZER TESTS
# =============================================================================

class TestCompassAnalyzer:
    """Test CompassAnalyzer."""

    def test_analyze_local_pattern(self):
        """Local attention should have low variance."""
        analyzer = CompassAnalyzer()

        reading = analyzer.analyze(
            query_pos=100,
            key_positions=[95, 96, 97, 98, 99],
            attention_scores=[0.1, 0.15, 0.2, 0.25, 0.3]
        )

        assert reading.angular_variance < 0.3
        assert reading.heading == CompassHeading.EAST
        assert reading.pattern_type == "focused"

    def test_analyze_distributed_pattern(self):
        """Distributed attention should have medium variance."""
        analyzer = CompassAnalyzer()

        reading = analyzer.analyze(
            query_pos=500,
            key_positions=[10, 100, 200, 400, 499],
            attention_scores=[0.2, 0.2, 0.2, 0.2, 0.2]
        )

        assert 0.2 < reading.angular_variance < 0.8
        assert reading.pattern_type in ["distributed", "bimodal"]

    def test_analyze_scattered_pattern(self):
        """Very spread attention should have high variance."""
        analyzer = CompassAnalyzer()

        # Attention to many widely spread positions
        reading = analyzer.analyze(
            query_pos=1000,
            key_positions=[10, 100, 300, 500, 700, 999],
            attention_scores=[0.16, 0.17, 0.17, 0.17, 0.17, 0.16]
        )

        # Should have relatively high variance
        assert reading.angular_variance > 0.3

    def test_analyze_sink_dominated(self):
        """Sink-dominated attention should be classified correctly."""
        analyzer = CompassAnalyzer()

        reading = analyzer.analyze(
            query_pos=100,
            key_positions=[0, 1, 2, 3, 50],
            attention_scores=[0.4, 0.2, 0.15, 0.15, 0.1]
        )

        assert reading.anchor.is_sink_dominated
        assert reading.pattern_type == "sink_dominated"

    def test_directional_mass_local(self):
        """Local attention should have consistent heading."""
        analyzer = CompassAnalyzer()

        reading = analyzer.analyze(
            query_pos=100,
            key_positions=[90, 95, 99],
            attention_scores=[0.2, 0.3, 0.5]
        )

        # Local tokens should have low variance and clear heading
        assert reading.angular_variance < 0.5
        assert reading.heading in [CompassHeading.EAST, CompassHeading.NORTH]

    def test_directional_mass_distant(self):
        """Distant attention should have high north_mass."""
        analyzer = CompassAnalyzer()

        reading = analyzer.analyze(
            query_pos=1000,
            key_positions=[10, 20, 30],
            attention_scores=[0.4, 0.3, 0.3]
        )

        # Distant tokens should map to NORTH or WEST
        assert reading.north_mass + reading.west_mass > 0.5

    def test_angular_concentration(self):
        """Concentration should be inverse of variance."""
        analyzer = CompassAnalyzer()

        reading = analyzer.analyze(
            query_pos=100,
            key_positions=[95, 96, 97, 98, 99],
            attention_scores=[0.2, 0.2, 0.2, 0.2, 0.2]
        )

        assert reading.angular_concentration == pytest.approx(
            1.0 - reading.angular_variance, abs=0.01
        )


# =============================================================================
# COMPASS ROUTER TESTS
# =============================================================================

class TestCompassRouter:
    """Test CompassRouter routing logic."""

    def test_route_local_to_small(self):
        """Local syntactic pattern should route to SMALL."""
        router = CompassRouter()

        decision = router.route(
            query_pos=100,
            key_positions=[95, 96, 97, 98, 99],
            attention_scores=[0.1, 0.15, 0.2, 0.25, 0.3]
        )

        assert decision.tier == RoutingTier.SMALL
        assert not decision.use_chain_of_thought
        assert "syntactic" in decision.reason.lower() or "focused" in decision.reason.lower()

    def test_route_sink_dominated_to_large(self):
        """Sink-dominated attention should route to LARGE."""
        router = CompassRouter()

        decision = router.route(
            query_pos=100,
            key_positions=[0, 1, 2, 3, 50],
            attention_scores=[0.4, 0.2, 0.15, 0.15, 0.1]
        )

        assert decision.tier == RoutingTier.LARGE
        assert decision.use_chain_of_thought
        assert "sink" in decision.reason.lower() or "uncertainty" in decision.reason.lower()

    def test_route_returns_recommendations(self):
        """Route should return model recommendations."""
        router = CompassRouter()

        decision = router.route(
            query_pos=100,
            key_positions=[90, 95, 99],
            attention_scores=[0.3, 0.3, 0.4]
        )

        assert decision.recommended_model is not None
        assert decision.recommended_temperature > 0
        assert decision.recommended_max_tokens > 0

    def test_cot_for_high_variance(self):
        """High variance should enable CoT."""
        config = CompassRouterConfig(
            cot_variance_threshold=0.3  # Lower threshold for testing
        )
        router = CompassRouter(config)

        # Create moderately spread attention
        decision = router.route(
            query_pos=500,
            key_positions=[10, 100, 200, 400, 499],
            attention_scores=[0.2, 0.2, 0.2, 0.2, 0.2]
        )

        # Should use CoT if variance > 0.3
        if decision.compass.angular_variance > 0.3:
            assert decision.use_chain_of_thought

    def test_confidence_higher_for_clear_patterns(self):
        """Focused patterns should have higher confidence."""
        router = CompassRouter()

        # Very focused pattern
        decision_focused = router.route(
            query_pos=100,
            key_positions=[98, 99],
            attention_scores=[0.3, 0.7]
        )

        # Distributed pattern
        decision_distributed = router.route(
            query_pos=500,
            key_positions=[10, 100, 200, 300, 400, 499],
            attention_scores=[0.16, 0.17, 0.17, 0.17, 0.17, 0.16]
        )

        assert decision_focused.confidence >= decision_distributed.confidence

    def test_statistics_tracking(self):
        """Router should track routing statistics."""
        router = CompassRouter()

        # Make several routing decisions
        for _ in range(5):
            router.route(
                query_pos=100,
                key_positions=[95, 96, 97, 98, 99],
                attention_scores=[0.2, 0.2, 0.2, 0.2, 0.2]
            )

        stats = router.get_statistics()
        assert stats["total_routes"] == 5
        assert "distribution" in stats

    def test_reset_statistics(self):
        """Reset should clear statistics."""
        router = CompassRouter()

        router.route(
            query_pos=100,
            key_positions=[95, 96, 97, 98, 99],
            attention_scores=[0.2, 0.2, 0.2, 0.2, 0.2]
        )

        router.reset_statistics()
        stats = router.get_statistics()
        assert stats["total_routes"] == 0


# =============================================================================
# FINGERPRINT ROUTING TESTS
# =============================================================================

class TestFingerprintRouting:
    """Test routing from fingerprint vectors."""

    def test_route_fingerprint_local(self):
        """Fingerprint with high local_ratio should route to small."""
        router = CompassRouter()

        # Create fingerprint with high local ratio
        # Structure: [entropy, entropy_std, local_ratio, mid_ratio, long_ratio, ...]
        fingerprint = np.zeros(20)
        fingerprint[0] = 1.5   # Low entropy
        fingerprint[2] = 0.7   # High local ratio
        fingerprint[4] = 0.1   # Low long ratio

        decision = router.route_fingerprint(fingerprint)

        assert decision.tier in [RoutingTier.SMALL, RoutingTier.MEDIUM]
        assert decision.compass.pattern_type == "focused"

    def test_route_fingerprint_long_range(self):
        """Fingerprint with high long_ratio should indicate reasoning."""
        router = CompassRouter()

        fingerprint = np.zeros(20)
        fingerprint[0] = 3.5   # High entropy
        fingerprint[2] = 0.2   # Low local ratio
        fingerprint[4] = 0.5   # High long ratio

        decision = router.route_fingerprint(fingerprint)

        assert decision.compass.north_mass > 0.3
        # Confidence should be reduced for fingerprint-based routing
        assert decision.confidence < 0.9


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestCompassRouterConfig:
    """Test configuration options."""

    def test_custom_thresholds(self):
        """Custom thresholds should affect routing."""
        # Very low threshold - everything is "high variance"
        config = CompassRouterConfig(
            low_variance_threshold=0.01,
            high_variance_threshold=0.1,
        )
        router = CompassRouter(config)

        decision = router.route(
            query_pos=100,
            key_positions=[95, 96, 97, 98, 99],
            attention_scores=[0.2, 0.2, 0.2, 0.2, 0.2]
        )

        # Should route to LARGE due to low thresholds
        # (even focused attention exceeds 0.1)
        if decision.compass.angular_variance > 0.1:
            assert decision.tier == RoutingTier.LARGE

    def test_custom_models(self):
        """Custom model names should be returned."""
        config = CompassRouterConfig(
            small_model="my-small-model",
            medium_model="my-medium-model",
            large_model="my-large-model",
        )
        router = CompassRouter(config)

        decision = router.route(
            query_pos=100,
            key_positions=[95, 96, 97, 98, 99],
            attention_scores=[0.2, 0.2, 0.2, 0.2, 0.2]
        )

        assert decision.recommended_model in [
            "my-small-model", "my-medium-model", "my-large-model"
        ]


# =============================================================================
# GLOSSARY TESTS
# =============================================================================

class TestCompassGlossary:
    """Test educational glossary."""

    def test_glossary_has_required_terms(self):
        """Glossary should have key terms."""
        required_terms = ["sinq_anchor", "compass_heading", "angular_variance", "routing_tier"]

        for term in required_terms:
            assert term in COMPASS_GLOSSARY
            assert "simple" in COMPASS_GLOSSARY[term]
            assert "detailed" in COMPASS_GLOSSARY[term]

    def test_glossary_entry_structure(self):
        """Each entry should have required fields."""
        for term, entry in COMPASS_GLOSSARY.items():
            assert "term" in entry
            assert "simple" in entry
            assert "why_it_matters" in entry


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunctions:
    """Test factory/convenience functions."""

    def test_create_compass_router(self):
        """create_compass_router should create router."""
        router = create_compass_router()
        assert isinstance(router, CompassRouter)

        config = CompassRouterConfig(sink_threshold=3)
        router_custom = create_compass_router(config)
        assert router_custom.config.sink_threshold == 3

    def test_analyze_attention_compass(self):
        """analyze_attention_compass should return reading."""
        reading = analyze_attention_compass(
            query_pos=100,
            key_positions=[90, 95, 99],
            attention_scores=[0.3, 0.3, 0.4]
        )

        assert isinstance(reading, CompassReading)
        assert reading.heading is not None


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================

class TestSerialization:
    """Test to_dict serialization."""

    def test_compass_reading_to_dict(self):
        """CompassReading should serialize to dict."""
        analyzer = CompassAnalyzer()
        reading = analyzer.analyze(
            query_pos=100,
            key_positions=[90, 95, 99],
            attention_scores=[0.3, 0.3, 0.4]
        )

        d = reading.to_dict()

        assert isinstance(d, dict)
        assert d["heading"] in ["north", "east", "south", "west", "scattered"]
        assert "angular_variance" in d
        assert "anchor" in d

    def test_routing_decision_to_dict(self):
        """CompassRoutingDecision should serialize to dict."""
        router = CompassRouter()
        decision = router.route(
            query_pos=100,
            key_positions=[90, 95, 99],
            attention_scores=[0.3, 0.3, 0.4]
        )

        d = decision.to_dict()

        assert isinstance(d, dict)
        assert d["tier"] in ["small", "medium", "large", "context"]
        assert "compass" in d
        assert "recommended_model" in d


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token_attention(self):
        """Should handle attention to single token."""
        analyzer = CompassAnalyzer()

        reading = analyzer.analyze(
            query_pos=100,
            key_positions=[50],
            attention_scores=[1.0]
        )

        # Single token should have approximately zero variance (floating point)
        assert reading.angular_variance == pytest.approx(0.0, abs=1e-10)
        assert reading.pattern_type == "focused"

    def test_all_sink_attention(self):
        """Should handle attention only to sink tokens."""
        analyzer = CompassAnalyzer()

        reading = analyzer.analyze(
            query_pos=100,
            key_positions=[0, 1, 2, 3, 4],
            attention_scores=[0.3, 0.25, 0.2, 0.15, 0.1]
        )

        assert reading.pattern_type == "sink_dominated"
        assert reading.heading == CompassHeading.NORTH

    def test_zero_query_position(self):
        """Should handle query at position 0."""
        analyzer = CompassAnalyzer()

        # At position 0, can only attend to sink (which is position 0)
        reading = analyzer.analyze(
            query_pos=0,
            key_positions=[0],
            attention_scores=[1.0]
        )

        assert reading is not None

    def test_very_long_context(self):
        """Should handle very long context positions."""
        analyzer = CompassAnalyzer()

        reading = analyzer.analyze(
            query_pos=100000,
            key_positions=[100, 1000, 10000, 50000, 99999],
            attention_scores=[0.2, 0.2, 0.2, 0.2, 0.2]
        )

        assert reading.angular_variance >= 0
        assert reading.angular_variance <= 1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests."""

    def test_demo_runs_without_error(self):
        """Demo should complete without error."""
        from .compass_router import demo
        demo()  # Should not raise

    def test_full_routing_pipeline(self):
        """Test complete routing pipeline."""
        # Create router
        config = CompassRouterConfig(
            small_model="test-small",
            medium_model="test-medium",
            large_model="test-large",
        )
        router = CompassRouter(config)

        # Route several patterns
        patterns = [
            # Local syntactic
            {
                "query_pos": 100,
                "key_positions": [95, 96, 97, 98, 99],
                "attention_scores": [0.1, 0.15, 0.2, 0.25, 0.3],
            },
            # Distributed
            {
                "query_pos": 500,
                "key_positions": [10, 100, 200, 400, 499],
                "attention_scores": [0.2, 0.2, 0.2, 0.2, 0.2],
            },
            # Sink-dominated
            {
                "query_pos": 100,
                "key_positions": [0, 1, 2, 50, 99],
                "attention_scores": [0.4, 0.25, 0.15, 0.1, 0.1],
            },
        ]

        decisions = []
        for pattern in patterns:
            decision = router.route(**pattern)
            decisions.append(decision)
            assert decision.tier is not None
            assert decision.recommended_model in ["test-small", "test-medium", "test-large"]

        # Verify statistics
        stats = router.get_statistics()
        assert stats["total_routes"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
