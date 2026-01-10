"""
Tests for Manifold Firewall

Tests the hallucination detection system that monitors attention pattern
trajectories for suspicious drift.
"""

import pytest
import numpy as np
import time
from collections import deque

from .manifold_firewall import (
    ManifoldFirewall,
    ManifoldBatchAnalyzer,
    FirewallConfig,
    FirewallState,
    FirewallCheckResult,
    ManifoldPoint,
    DriftEvent,
    DriftType,
    AlertSeverity,
    SuddenJumpDetector,
    ZoneViolationDetector,
    EntropySpikeDetector,
    SinkCollapseDetector,
    OscillationDetector,
    FIREWALL_GLOSSARY,
    ZONE_TRANSITIONS,
    create_firewall,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def firewall():
    """Create a firewall with default config."""
    return ManifoldFirewall()


@pytest.fixture
def config():
    """Create default config."""
    return FirewallConfig()


@pytest.fixture
def sample_fingerprint():
    """Create a sample fingerprint."""
    return np.random.randn(20) * 0.1


def make_point(
    zone: str = "semantic_bridge",
    entropy: float = 2.0,
    sink_ratio: float = 0.1,
    token_position: int = 0,
    fingerprint: np.ndarray = None,
) -> ManifoldPoint:
    """Helper to create ManifoldPoint."""
    if fingerprint is None:
        fingerprint = np.random.randn(20) * 0.1
    return ManifoldPoint(
        fingerprint=fingerprint,
        zone=zone,
        cluster_id=None,
        embedding=None,
        entropy=entropy,
        sink_ratio=sink_ratio,
        timestamp=time.time(),
        token_position=token_position,
    )


# =============================================================================
# MANIFOLD POINT TESTS
# =============================================================================

class TestManifoldPoint:
    """Test ManifoldPoint class."""

    def test_distance_to_same(self):
        """Distance to same point should be zero."""
        fp = np.array([1.0, 2.0, 3.0] + [0.0] * 17)
        point = make_point(fingerprint=fp)
        assert point.distance_to(point) == pytest.approx(0.0)

    def test_distance_to_different(self):
        """Distance between different points should be positive."""
        fp1 = np.zeros(20)
        fp2 = np.ones(20)
        point1 = make_point(fingerprint=fp1)
        point2 = make_point(fingerprint=fp2)

        distance = point1.distance_to(point2)
        expected = np.sqrt(20)  # sqrt(20 * 1^2)
        assert distance == pytest.approx(expected, abs=0.01)


# =============================================================================
# ALERT SEVERITY TESTS
# =============================================================================

class TestAlertSeverity:
    """Test AlertSeverity enum."""

    def test_all_severities_defined(self):
        """All expected severities should be defined."""
        assert AlertSeverity.SAFE.value == "safe"
        assert AlertSeverity.WATCH.value == "watch"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ALERT.value == "alert"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_severity_ordering(self):
        """Severities should have defined ordering."""
        severities = list(AlertSeverity)
        assert severities.index(AlertSeverity.SAFE) < severities.index(AlertSeverity.CRITICAL)


# =============================================================================
# DRIFT TYPE TESTS
# =============================================================================

class TestDriftType:
    """Test DriftType enum."""

    def test_all_drift_types_defined(self):
        """All expected drift types should be defined."""
        expected = ["none", "gradual", "sudden_jump", "zone_violation",
                    "entropy_spike", "sink_collapse", "reversal", "oscillation"]
        for value in expected:
            assert any(dt.value == value for dt in DriftType)


# =============================================================================
# DETECTOR TESTS
# =============================================================================

class TestSuddenJumpDetector:
    """Test SuddenJumpDetector."""

    def test_no_detection_with_short_trajectory(self):
        """Should not detect with < 2 points."""
        detector = SuddenJumpDetector()
        config = FirewallConfig()
        trajectory = deque([make_point()])
        current = make_point()

        result = detector.detect(current, trajectory, config)
        assert result is None

    def test_detects_large_jump(self):
        """Should detect large fingerprint jump."""
        detector = SuddenJumpDetector()
        config = FirewallConfig()

        # Build trajectory with small steps
        trajectory = deque()
        for i in range(5):
            trajectory.append(make_point(fingerprint=np.ones(20) * 0.1 * i))

        # Large jump
        current = make_point(fingerprint=np.ones(20) * 10.0)

        result = detector.detect(current, trajectory, config)
        assert result is not None
        assert result.drift_type == DriftType.SUDDEN_JUMP

    def test_no_detection_for_small_step(self):
        """Should not detect normal progression."""
        detector = SuddenJumpDetector()
        config = FirewallConfig()

        trajectory = deque()
        for i in range(5):
            trajectory.append(make_point(fingerprint=np.ones(20) * 0.1 * i))

        # Small step
        current = make_point(fingerprint=np.ones(20) * 0.5)

        result = detector.detect(current, trajectory, config)
        assert result is None


class TestZoneViolationDetector:
    """Test ZoneViolationDetector."""

    def test_detects_suspicious_transition(self):
        """Should detect suspicious zone transitions."""
        detector = ZoneViolationDetector()
        config = FirewallConfig()

        # structure_ripple → syntax_floor is suspicious
        trajectory = deque([make_point(zone="structure_ripple")])
        current = make_point(zone="syntax_floor")

        result = detector.detect(current, trajectory, config)
        assert result is not None
        assert result.drift_type == DriftType.ZONE_VIOLATION

    def test_allows_normal_transition(self):
        """Should allow normal zone transitions."""
        detector = ZoneViolationDetector()
        config = FirewallConfig()

        # syntax_floor → semantic_bridge is normal
        trajectory = deque([make_point(zone="syntax_floor")])
        current = make_point(zone="semantic_bridge")

        result = detector.detect(current, trajectory, config)
        assert result is None


class TestEntropySpikeDetector:
    """Test EntropySpikeDetector."""

    def test_detects_entropy_spike(self):
        """Should detect large entropy increase."""
        detector = EntropySpikeDetector()
        config = FirewallConfig(entropy_spike_threshold=1.5)

        # Build trajectory with low entropy
        trajectory = deque()
        for i in range(5):
            trajectory.append(make_point(entropy=1.5))

        # High entropy spike
        current = make_point(entropy=4.0)

        result = detector.detect(current, trajectory, config)
        assert result is not None
        assert result.drift_type == DriftType.ENTROPY_SPIKE

    def test_no_detection_for_stable_entropy(self):
        """Should not detect stable entropy."""
        detector = EntropySpikeDetector()
        config = FirewallConfig()

        trajectory = deque()
        for i in range(5):
            trajectory.append(make_point(entropy=2.0))

        current = make_point(entropy=2.1)

        result = detector.detect(current, trajectory, config)
        assert result is None


class TestSinkCollapseDetector:
    """Test SinkCollapseDetector."""

    def test_detects_sink_collapse(self):
        """Should detect high sink after low sink."""
        detector = SinkCollapseDetector()
        config = FirewallConfig(sink_collapse_threshold=0.4)

        # Build trajectory with low sink
        trajectory = deque()
        for i in range(5):
            trajectory.append(make_point(sink_ratio=0.1))

        # High sink
        current = make_point(sink_ratio=0.6)

        result = detector.detect(current, trajectory, config)
        assert result is not None
        assert result.drift_type == DriftType.SINK_COLLAPSE

    def test_no_detection_for_stable_sink(self):
        """Should not detect stable sink ratio."""
        detector = SinkCollapseDetector()
        config = FirewallConfig()

        trajectory = deque()
        for i in range(5):
            trajectory.append(make_point(sink_ratio=0.15))

        current = make_point(sink_ratio=0.18)

        result = detector.detect(current, trajectory, config)
        assert result is None


class TestOscillationDetector:
    """Test OscillationDetector."""

    def test_detects_oscillation(self):
        """Should detect rapid zone changes."""
        detector = OscillationDetector()
        config = FirewallConfig(oscillation_window=5, oscillation_threshold=3)

        # Build oscillating trajectory (need at least oscillation_window points)
        zones = ["syntax_floor", "semantic_bridge", "syntax_floor", "semantic_bridge", "syntax_floor"]
        trajectory = deque()
        for zone in zones:
            trajectory.append(make_point(zone=zone))

        # Another change - this triggers oscillation (4 changes in 5 tokens)
        current = make_point(zone="semantic_bridge")

        result = detector.detect(current, trajectory, config)
        assert result is not None
        assert result.drift_type == DriftType.OSCILLATION

    def test_no_detection_for_stable_zone(self):
        """Should not detect stable zone."""
        detector = OscillationDetector()
        config = FirewallConfig()

        # All same zone
        trajectory = deque()
        for _ in range(5):
            trajectory.append(make_point(zone="semantic_bridge"))

        current = make_point(zone="semantic_bridge")

        result = detector.detect(current, trajectory, config)
        assert result is None


# =============================================================================
# FIREWALL TESTS
# =============================================================================

class TestManifoldFirewall:
    """Test ManifoldFirewall class."""

    def test_start_session(self, firewall):
        """Should create new session."""
        session_id = firewall.start_session("test-session")
        assert session_id == "test-session"

        state = firewall.get_session_state("test-session")
        assert state is not None
        assert state.session_id == "test-session"

    def test_check_returns_result(self, firewall, sample_fingerprint):
        """Check should return FirewallCheckResult."""
        session = firewall.start_session("test")

        result = firewall.check(
            session_id=session,
            fingerprint=sample_fingerprint,
            zone="semantic_bridge",
            entropy=2.0,
            sink_ratio=0.1,
            token_position=0,
        )

        assert isinstance(result, FirewallCheckResult)
        assert result.session_id == session

    def test_check_starts_safe(self, firewall, sample_fingerprint):
        """Initial checks should be SAFE."""
        session = firewall.start_session("test")

        for i in range(3):
            result = firewall.check(
                session_id=session,
                fingerprint=sample_fingerprint,
                zone="semantic_bridge",
                entropy=2.0,
                sink_ratio=0.1,
                token_position=i,
            )
            assert result.severity == AlertSeverity.SAFE

    def test_check_detects_anomaly(self, firewall):
        """Should detect anomaly after normal tokens."""
        session = firewall.start_session("test")

        # Normal tokens
        for i in range(5):
            firewall.check(
                session_id=session,
                fingerprint=np.ones(20) * 0.1,
                zone="semantic_bridge",
                entropy=2.0,
                sink_ratio=0.1,
                token_position=i,
            )

        # Anomaly: large jump + high sink
        result = firewall.check(
            session_id=session,
            fingerprint=np.ones(20) * 10.0,
            zone="syntax_floor",
            entropy=4.0,
            sink_ratio=0.7,
            token_position=5,
        )

        assert result.severity != AlertSeverity.SAFE
        assert len(result.new_events) > 0

    def test_end_session(self, firewall, sample_fingerprint):
        """End session should return state and remove session."""
        session = firewall.start_session("test")

        firewall.check(
            session_id=session,
            fingerprint=sample_fingerprint,
            zone="semantic_bridge",
            entropy=2.0,
            sink_ratio=0.1,
            token_position=0,
        )

        final = firewall.end_session(session)

        assert final is not None
        assert "session_id" in final
        assert firewall.get_session_state(session) is None

    def test_statistics_tracking(self, firewall, sample_fingerprint):
        """Should track statistics."""
        session = firewall.start_session("test")

        for i in range(5):
            firewall.check(
                session_id=session,
                fingerprint=sample_fingerprint,
                zone="semantic_bridge",
                entropy=2.0,
                sink_ratio=0.1,
                token_position=i,
            )

        stats = firewall.get_statistics()
        assert stats["total_checks"] == 5
        assert stats["active_sessions"] == 1

    def test_reset_statistics(self, firewall, sample_fingerprint):
        """Reset should clear statistics."""
        session = firewall.start_session("test")

        firewall.check(
            session_id=session,
            fingerprint=sample_fingerprint,
            zone="semantic_bridge",
            entropy=2.0,
            sink_ratio=0.1,
            token_position=0,
        )

        firewall.reset_statistics()
        stats = firewall.get_statistics()
        assert stats["total_checks"] == 0


# =============================================================================
# FIREWALL CHECK RESULT TESTS
# =============================================================================

class TestFirewallCheckResult:
    """Test FirewallCheckResult class."""

    def test_is_safe_property(self):
        """is_safe should return True for SAFE severity."""
        result = FirewallCheckResult(
            severity=AlertSeverity.SAFE,
            cumulative_drift=0.0,
            new_events=[],
            trajectory_length=5,
            session_id="test",
        )
        assert result.is_safe is True

    def test_needs_attention_property(self):
        """needs_attention should return True for WARNING+."""
        for severity in [AlertSeverity.WARNING, AlertSeverity.ALERT, AlertSeverity.CRITICAL]:
            result = FirewallCheckResult(
                severity=severity,
                cumulative_drift=1.0,
                new_events=[],
                trajectory_length=5,
                session_id="test",
            )
            assert result.needs_attention is True

    def test_should_stop_property(self):
        """should_stop should return True only for CRITICAL."""
        result_critical = FirewallCheckResult(
            severity=AlertSeverity.CRITICAL,
            cumulative_drift=5.0,
            new_events=[],
            trajectory_length=5,
            session_id="test",
        )
        assert result_critical.should_stop is True

        result_alert = FirewallCheckResult(
            severity=AlertSeverity.ALERT,
            cumulative_drift=2.0,
            new_events=[],
            trajectory_length=5,
            session_id="test",
        )
        assert result_alert.should_stop is False

    def test_to_dict(self):
        """to_dict should serialize correctly."""
        result = FirewallCheckResult(
            severity=AlertSeverity.WARNING,
            cumulative_drift=1.5,
            new_events=[],
            trajectory_length=10,
            session_id="test-123",
        )

        d = result.to_dict()
        assert d["severity"] == "warning"
        assert d["cumulative_drift"] == 1.5
        assert d["session_id"] == "test-123"


# =============================================================================
# BATCH ANALYZER TESTS
# =============================================================================

class TestManifoldBatchAnalyzer:
    """Test ManifoldBatchAnalyzer class."""

    def test_analyze_returns_dict(self):
        """analyze should return analysis dict."""
        analyzer = ManifoldBatchAnalyzer()

        fingerprints = [np.random.randn(20) * 0.1 for _ in range(10)]
        zones = ["semantic_bridge"] * 10
        entropies = [2.0] * 10
        sink_ratios = [0.1] * 10

        result = analyzer.analyze(fingerprints, zones, entropies, sink_ratios)

        assert isinstance(result, dict)
        assert "total_tokens" in result
        assert "max_severity" in result
        assert "recommendations" in result

    def test_analyze_detects_suspicious_regions(self):
        """Should detect suspicious regions in batch."""
        analyzer = ManifoldBatchAnalyzer()

        # Normal tokens, then anomaly
        fingerprints = [np.ones(20) * 0.1 for _ in range(5)]
        fingerprints.append(np.ones(20) * 10.0)  # Jump
        fingerprints.extend([np.ones(20) * 0.1 for _ in range(4)])

        zones = ["semantic_bridge"] * 10
        entropies = [2.0] * 5 + [4.0] + [2.0] * 4
        sink_ratios = [0.1] * 5 + [0.6] + [0.1] * 4

        result = analyzer.analyze(fingerprints, zones, entropies, sink_ratios)

        assert result["total_events"] > 0


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================

class TestFirewallConfig:
    """Test FirewallConfig class."""

    def test_default_values(self):
        """Default config should have sensible values."""
        config = FirewallConfig()

        assert config.window_size == 20
        assert config.sudden_jump_threshold > 0
        assert config.critical_threshold > config.alert_threshold

    def test_custom_config(self):
        """Custom config should be respected."""
        config = FirewallConfig(
            window_size=50,
            sudden_jump_threshold=1.0,
            critical_threshold=5.0,
        )

        assert config.window_size == 50
        assert config.sudden_jump_threshold == 1.0
        assert config.critical_threshold == 5.0


# =============================================================================
# GLOSSARY TESTS
# =============================================================================

class TestFirewallGlossary:
    """Test educational glossary."""

    def test_glossary_has_required_terms(self):
        """Glossary should have key terms."""
        required = ["manifold_drift", "sudden_jump", "sink_collapse"]

        for term in required:
            assert term in FIREWALL_GLOSSARY
            assert "simple" in FIREWALL_GLOSSARY[term]
            assert "why_it_matters" in FIREWALL_GLOSSARY[term]


# =============================================================================
# ZONE TRANSITIONS TESTS
# =============================================================================

class TestZoneTransitions:
    """Test zone transition matrix."""

    def test_all_zones_covered(self):
        """All zone pairs should have transitions defined."""
        zones = ["syntax_floor", "semantic_bridge", "structure_ripple"]

        for from_zone in zones:
            for to_zone in zones:
                assert (from_zone, to_zone) in ZONE_TRANSITIONS

    def test_same_zone_not_suspicious(self):
        """Same zone should never be suspicious."""
        zones = ["syntax_floor", "semantic_bridge", "structure_ripple"]

        for zone in zones:
            is_suspicious, _ = ZONE_TRANSITIONS[(zone, zone)]
            assert is_suspicious is False


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_firewall(self):
        """create_firewall should create firewall."""
        firewall = create_firewall()
        assert isinstance(firewall, ManifoldFirewall)

    def test_create_firewall_with_config(self):
        """create_firewall should accept config."""
        config = FirewallConfig(window_size=100)
        firewall = create_firewall(config)
        assert firewall.config.window_size == 100


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests."""

    def test_demo_runs_without_error(self):
        """Demo should complete without error."""
        from .manifold_firewall import demo
        demo()  # Should not raise

    def test_full_session_lifecycle(self):
        """Test complete session from start to end."""
        firewall = ManifoldFirewall()

        # Start session
        session = firewall.start_session("integration-test")

        # Send normal tokens
        for i in range(10):
            result = firewall.check(
                session_id=session,
                fingerprint=np.random.randn(20) * 0.1,
                zone="semantic_bridge",
                entropy=2.0 + np.random.randn() * 0.1,
                sink_ratio=0.1 + np.random.randn() * 0.02,
                token_position=i,
            )
            assert result.session_id == session

        # Send anomaly
        result = firewall.check(
            session_id=session,
            fingerprint=np.random.randn(20) * 5.0,
            zone="structure_ripple",
            entropy=5.0,
            sink_ratio=0.5,
            token_position=10,
        )

        # Check severity increased
        assert result.severity != AlertSeverity.SAFE

        # End session
        final = firewall.end_session(session)
        assert final is not None
        assert final["event_count"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
