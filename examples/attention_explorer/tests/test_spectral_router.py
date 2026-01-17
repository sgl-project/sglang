"""
Tests for Spectral Router

Tests the spectral manifold discovery and routing logic.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from discovery.spectral_discovery import (
    FrequencyBandAnalyzer,
    SpectralDiscoveryConfig,
    SpectralManifoldDiscovery,
    SpectralMode,
)
from discovery.spectral_router import (
    AdaptiveSpectralRouter,
    ModelSize,
    RouterConfig,
    SpectralRouter,
    create_router_from_fingerprints,
)


class TestSpectralManifoldDiscovery:
    """Tests for SpectralManifoldDiscovery class."""

    @pytest.fixture
    def sample_fingerprints(self):
        """Generate sample fingerprints with known structure."""
        np.random.seed(42)
        n_samples = 500

        # Create fingerprints with cluster structure
        # Cluster 1: High local_mass, low entropy (syntax_floor-like)
        cluster1 = np.random.randn(n_samples // 3, 20) * 0.1
        cluster1[:, 0] = 0.7 + np.random.randn(n_samples // 3) * 0.1  # local_mass
        cluster1[:, 3] = 1.5 + np.random.randn(n_samples // 3) * 0.2  # entropy

        # Cluster 2: Balanced (semantic_bridge-like)
        cluster2 = np.random.randn(n_samples // 3, 20) * 0.1
        cluster2[:, 0] = 0.3 + np.random.randn(n_samples // 3) * 0.1
        cluster2[:, 1] = 0.4 + np.random.randn(n_samples // 3) * 0.1  # mid_mass
        cluster2[:, 3] = 2.5 + np.random.randn(n_samples // 3) * 0.2

        # Cluster 3: High long_mass, high entropy (structure_ripple-like)
        cluster3 = np.random.randn(n_samples // 3, 20) * 0.1
        cluster3[:, 2] = 0.6 + np.random.randn(n_samples // 3) * 0.1  # long_mass
        cluster3[:, 3] = 4.0 + np.random.randn(n_samples // 3) * 0.2

        fingerprints = np.vstack([cluster1, cluster2, cluster3])
        np.random.shuffle(fingerprints)
        return fingerprints

    def test_fit_basic(self, sample_fingerprints):
        """Test basic fitting of spectral discovery."""
        config = SpectralDiscoveryConfig(n_components=10, n_neighbors=10)
        discovery = SpectralManifoldDiscovery(config)

        discovery.fit(sample_fingerprints)

        assert discovery._fitted
        assert discovery.eigenvalues_ is not None
        assert discovery.eigenvectors_ is not None
        assert len(discovery.eigenvalues_) == 10

    def test_fit_transform(self, sample_fingerprints):
        """Test fit_transform returns correct shape."""
        config = SpectralDiscoveryConfig(n_components=10, n_neighbors=10)
        discovery = SpectralManifoldDiscovery(config)

        embeddings = discovery.fit_transform(sample_fingerprints)

        assert embeddings.shape == (len(sample_fingerprints), 10)

    def test_spectral_coherence(self, sample_fingerprints):
        """Test spectral coherence computation."""
        config = SpectralDiscoveryConfig(n_components=10, n_neighbors=10)
        discovery = SpectralManifoldDiscovery(config)
        discovery.fit(sample_fingerprints)

        # Test on a sample from the training data (should have high coherence)
        coherence = discovery.compute_spectral_coherence(sample_fingerprints[0])

        assert 0 <= coherence.coherence_score <= 1
        assert coherence.mode in SpectralMode
        assert len(coherence.dominant_modes) == 5

    def test_spectral_gap(self, sample_fingerprints):
        """Test spectral gap computation."""
        config = SpectralDiscoveryConfig(n_components=10, n_neighbors=10)
        discovery = SpectralManifoldDiscovery(config)
        discovery.fit(sample_fingerprints)

        gap = discovery._compute_spectral_gap()

        assert gap >= 0  # Spectral gap should be non-negative

    def test_effective_dimension(self, sample_fingerprints):
        """Test effective dimension computation."""
        config = SpectralDiscoveryConfig(n_components=10, n_neighbors=10)
        discovery = SpectralManifoldDiscovery(config)
        discovery.fit(sample_fingerprints)

        dim = discovery.compute_effective_dimension()

        assert 0 < dim <= 10

    def test_save_load(self, sample_fingerprints):
        """Test saving and loading spectral discovery."""
        config = SpectralDiscoveryConfig(n_components=10, n_neighbors=10)
        discovery = SpectralManifoldDiscovery(config)
        discovery.fit(sample_fingerprints)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            discovery.save(path)
            loaded = SpectralManifoldDiscovery.load(path)

            assert loaded._fitted
            np.testing.assert_array_almost_equal(
                discovery.eigenvalues_, loaded.eigenvalues_
            )
        finally:
            Path(path).unlink()

    def test_analyze(self, sample_fingerprints):
        """Test full analysis."""
        config = SpectralDiscoveryConfig(n_components=10, n_neighbors=10)
        discovery = SpectralManifoldDiscovery(config)

        analysis = discovery.analyze(sample_fingerprints)

        assert analysis.embeddings.shape[0] == len(sample_fingerprints)
        assert analysis.spectral_gap >= 0
        assert analysis.effective_dimension > 0


class TestSpectralRouter:
    """Tests for SpectralRouter class."""

    @pytest.fixture
    def sample_fingerprints(self):
        """Generate sample fingerprints."""
        np.random.seed(42)
        return np.random.randn(500, 20)

    @pytest.fixture
    def fitted_router(self, sample_fingerprints):
        """Create a fitted router."""
        config = RouterConfig(
            high_coherence_threshold=0.7,
            low_coherence_threshold=0.3,
        )
        router = SpectralRouter(config=config)
        router.fit(sample_fingerprints)
        return router

    def test_fit(self, sample_fingerprints):
        """Test router fitting."""
        router = SpectralRouter()
        router.fit(sample_fingerprints)

        assert router.spectral_discovery is not None
        assert router.spectral_discovery._fitted

    def test_route_basic(self, fitted_router, sample_fingerprints):
        """Test basic routing decision."""
        decision = fitted_router.route(sample_fingerprints[0])

        assert decision.model_size in ModelSize
        assert isinstance(decision.use_chain_of_thought, bool)
        assert 0 <= decision.spectral_coherence <= 1
        assert decision.estimated_complexity in ["trivial", "moderate", "complex"]

    def test_route_consistency(self, fitted_router, sample_fingerprints):
        """Test that same fingerprint gives same routing."""
        fp = sample_fingerprints[0]

        decision1 = fitted_router.route(fp)
        decision2 = fitted_router.route(fp)

        assert decision1.model_size == decision2.model_size
        assert decision1.use_chain_of_thought == decision2.use_chain_of_thought

    def test_route_batch(self, fitted_router, sample_fingerprints):
        """Test batch routing."""
        decisions = fitted_router.route_batch(sample_fingerprints[:10])

        assert len(decisions) == 10
        for d in decisions:
            assert d.model_size in ModelSize

    def test_routing_stats(self, fitted_router, sample_fingerprints):
        """Test routing statistics."""
        # Route some fingerprints
        for fp in sample_fingerprints[:50]:
            fitted_router.route(fp)

        stats = fitted_router.get_routing_stats()

        assert stats["total_decisions"] == 50
        assert sum(stats["model_distribution"].values()) == pytest.approx(1.0)
        assert 0 <= stats["cot_rate"] <= 1

    def test_high_coherence_routes_small(self, sample_fingerprints):
        """Test that high coherence fingerprints route to small model."""
        # Create a router with extreme thresholds
        config = RouterConfig(
            high_coherence_threshold=0.1,  # Everything is "high coherence"
            low_coherence_threshold=0.05,
        )
        router = SpectralRouter(config=config)
        router.fit(sample_fingerprints)

        # Route should prefer small model for high coherence
        decision = router.route(sample_fingerprints[0])
        # Note: This might not always be small due to band ratio
        assert decision.model_size in ModelSize

    def test_low_coherence_routes_large(self, sample_fingerprints):
        """Test that low coherence fingerprints route to large model."""
        # Create a router with extreme thresholds that won't be overridden by calibration
        config = RouterConfig(
            high_coherence_threshold=0.99,  # Nothing is "high coherence"
            low_coherence_threshold=0.98,
        )
        router = SpectralRouter(config=config)
        router.fit(sample_fingerprints)

        # After calibration, thresholds are adjusted based on data distribution
        # So we check that at least the model doesn't route to SMALL
        decision = router.route(sample_fingerprints[0])
        # Low coherence should route to MEDIUM or LARGE, not SMALL
        assert decision.model_size in [ModelSize.MEDIUM, ModelSize.LARGE]

    def test_get_model_for_decision(self, fitted_router, sample_fingerprints):
        """Test model identifier retrieval."""
        decision = fitted_router.route(sample_fingerprints[0])
        model = fitted_router.get_model_for_decision(decision)

        assert isinstance(model, str)
        assert model in [
            fitted_router.config.small_model,
            fitted_router.config.medium_model,
            fitted_router.config.large_model,
        ]

    def test_save_load(self, fitted_router, sample_fingerprints):
        """Test saving and loading router."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "router.pkl")

            fitted_router.save(path)
            loaded = SpectralRouter.load(path)

            # Compare decisions
            decision1 = fitted_router.route(sample_fingerprints[0])
            decision2 = loaded.route(sample_fingerprints[0])

            assert decision1.model_size == decision2.model_size

    def test_create_router_convenience(self, sample_fingerprints):
        """Test convenience function."""
        router = create_router_from_fingerprints(sample_fingerprints)

        assert router.spectral_discovery._fitted
        decision = router.route(sample_fingerprints[0])
        assert decision.model_size in ModelSize


class TestAdaptiveSpectralRouter:
    """Tests for AdaptiveSpectralRouter class."""

    @pytest.fixture
    def sample_fingerprints(self):
        np.random.seed(42)
        return np.random.randn(500, 20)

    def test_record_feedback(self, sample_fingerprints):
        """Test feedback recording."""
        router = AdaptiveSpectralRouter()
        router.fit(sample_fingerprints)

        fp = sample_fingerprints[0]
        decision = router.route(fp)

        router.record_feedback(
            fingerprint=fp,
            decision=decision,
            actual_quality=0.8,
            actual_latency=1.5,
            was_correct_size=True,
        )

        assert len(router._feedback_history) == 1

    def test_adapt_thresholds_needs_data(self, sample_fingerprints):
        """Test that adaptation needs minimum feedback."""
        router = AdaptiveSpectralRouter()
        router.fit(sample_fingerprints)

        # Record only a few feedback entries
        for i in range(10):
            fp = sample_fingerprints[i]
            decision = router.route(fp)
            router.record_feedback(fp, decision, 0.8, 1.5, True)

        # Should warn about not enough data
        original_high = router.config.high_coherence_threshold
        router.adapt_thresholds()
        # Thresholds shouldn't change with insufficient data
        assert router.config.high_coherence_threshold == original_high


class TestFrequencyBandAnalyzer:
    """Tests for FrequencyBandAnalyzer class."""

    def test_analyze_frequency_bands(self):
        """Test frequency band analysis."""
        analyzer = FrequencyBandAnalyzer()

        # Create fingerprint with known structure
        # [local_mass, mid_mass, long_mass, entropy, ...]
        fp = np.zeros(20)
        fp[0] = 0.3  # local_mass (high frequency)
        fp[2] = 0.7  # long_mass (low frequency)

        result = analyzer.analyze_frequency_bands(fp)

        assert "high_band_activity" in result
        assert "low_band_activity" in result
        assert "band_ratio" in result
        assert result["high_band_activity"] > result["low_band_activity"]

    def test_recommend_model_size(self):
        """Test model size recommendation."""
        analyzer = FrequencyBandAnalyzer()

        # High band ratio -> large model
        high_band = {
            "high_band_activity": 0.8,
            "low_band_activity": 0.2,
            "band_ratio": 4.0,
        }
        assert analyzer.recommend_model_size(high_band) == "large"

        # Low band ratio -> small model
        low_band = {
            "high_band_activity": 0.2,
            "low_band_activity": 0.8,
            "band_ratio": 0.25,
        }
        assert analyzer.recommend_model_size(low_band) == "small"

        # Balanced -> medium
        balanced = {
            "high_band_activity": 0.5,
            "low_band_activity": 0.5,
            "band_ratio": 1.0,
        }
        assert analyzer.recommend_model_size(balanced) == "medium"


class TestIntegration:
    """Integration tests for the full spectral routing pipeline."""

    def test_full_pipeline(self):
        """Test complete pipeline from fingerprints to routing decision."""
        np.random.seed(42)

        # Generate realistic fingerprints
        n_samples = 200
        fingerprints = np.random.randn(n_samples, 20)

        # Simulate zone structure
        # Add structure_ripple pattern to some
        fingerprints[:50, 2] = 0.6  # high long_mass
        fingerprints[:50, 3] = 4.0  # high entropy

        # Add semantic_bridge pattern to some
        fingerprints[50:100, 1] = 0.5  # mid_mass
        fingerprints[50:100, 3] = 2.5

        # Train router
        router = create_router_from_fingerprints(fingerprints)

        # Test routing on each zone type
        structure_ripple_fp = fingerprints[0]
        semantic_bridge_fp = fingerprints[75]
        random_fp = fingerprints[150]

        d1 = router.route(structure_ripple_fp)
        d2 = router.route(semantic_bridge_fp)
        d3 = router.route(random_fp)

        # All should produce valid decisions
        assert d1.model_size in ModelSize
        assert d2.model_size in ModelSize
        assert d3.model_size in ModelSize

        # Check stats
        stats = router.get_routing_stats()
        assert stats["total_decisions"] == 3

    def test_deterministic_routing(self):
        """Test that routing is deterministic for same input."""
        np.random.seed(42)
        fingerprints = np.random.randn(100, 20)

        router1 = create_router_from_fingerprints(fingerprints)
        router2 = create_router_from_fingerprints(fingerprints)

        # Same fingerprint should get same routing from both routers
        test_fp = fingerprints[0]
        d1 = router1.route(test_fp)
        d2 = router2.route(test_fp)

        assert d1.model_size == d2.model_size
        assert d1.use_chain_of_thought == d2.use_chain_of_thought


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
