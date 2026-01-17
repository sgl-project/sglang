"""
Manifold Firewall: Hallucination Detection via Attention Drift

Detects potential hallucinations by monitoring attention pattern trajectories
on the manifold. Flags when patterns deviate from expected regions or exhibit
suspicious behaviors.

=== CORE CONCEPT ===

During normal generation, attention patterns follow predictable trajectories
on the manifold:
- Factual retrieval: Stays in consistent cluster regions
- Reasoning: Gradual drift through semantic zones
- Syntax: Quick, local oscillations in syntax_floor

Hallucination indicators:
1. SUDDEN JUMP: Pattern teleports to distant manifold region
2. ZONE VIOLATION: Unexpected zone transition (e.g., factual → creative)
3. ENTROPY SPIKE: Attention suddenly becomes scattered
4. SINK COLLAPSE: High sink attention after confident pattern
5. TRAJECTORY REVERSAL: Pattern retraces unexpected path

=== DETECTION ALGORITHM ===

1. Maintain sliding window of recent fingerprints (trajectory)
2. Compute expected next region based on trajectory momentum
3. Compare actual fingerprint to expected region
4. Score anomaly based on multiple drift metrics
5. Alert if cumulative anomaly exceeds threshold

=== SEVERITY LEVELS ===

- SAFE: Normal pattern, no concerns
- WATCH: Minor drift, continue monitoring
- WARNING: Suspicious pattern, may need verification
- ALERT: Likely hallucination, recommend re-generation
- CRITICAL: Strong hallucination signal, stop generation

=== INTEGRATION ===

The firewall can run:
1. Real-time during generation (streaming mode)
2. Post-hoc on complete response (batch mode)
3. As a sidecar service receiving fingerprints via API
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class AlertSeverity(Enum):
    """Severity levels for firewall alerts."""

    SAFE = "safe"  # Normal, no concerns
    WATCH = "watch"  # Minor drift, monitoring
    WARNING = "warning"  # Suspicious, may need verification
    ALERT = "alert"  # Likely hallucination
    CRITICAL = "critical"  # Strong signal, stop generation


class DriftType(Enum):
    """Types of manifold drift."""

    NONE = "none"
    GRADUAL = "gradual"  # Normal progression
    SUDDEN_JUMP = "sudden_jump"  # Teleport to distant region
    ZONE_VIOLATION = "zone_violation"  # Unexpected zone change
    ENTROPY_SPIKE = "entropy_spike"  # Scattered attention
    SINK_COLLAPSE = "sink_collapse"  # High sink after confidence
    TRAJECTORY_REVERSAL = "reversal"  # Unexpected backtrack
    OSCILLATION = "oscillation"  # Rapid zone switching


# Zone transition matrix: expected vs suspicious
ZONE_TRANSITIONS = {
    # (from_zone, to_zone): (is_suspicious, reason)
    ("syntax_floor", "syntax_floor"): (False, "Normal syntax processing"),
    ("syntax_floor", "semantic_bridge"): (False, "Transitioning to semantics"),
    ("syntax_floor", "structure_ripple"): (True, "Jump from local to long-range"),
    ("semantic_bridge", "syntax_floor"): (False, "Returning to syntax"),
    ("semantic_bridge", "semantic_bridge"): (False, "Normal bridging"),
    ("semantic_bridge", "structure_ripple"): (False, "Expanding to structure"),
    ("structure_ripple", "syntax_floor"): (True, "Sudden drop from structure"),
    ("structure_ripple", "semantic_bridge"): (False, "Narrowing focus"),
    ("structure_ripple", "structure_ripple"): (False, "Continued reasoning"),
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class ManifoldPoint:
    """A point on the attention manifold."""

    fingerprint: np.ndarray  # 20-dim fingerprint
    zone: str  # syntax_floor, semantic_bridge, etc.
    cluster_id: Optional[int]  # Cluster assignment
    embedding: Optional[Tuple[float, float]]  # (x, y) if computed
    entropy: float  # Attention entropy
    sink_ratio: float  # Attention to sink tokens
    timestamp: float  # When captured
    token_position: int  # Position in generation

    def distance_to(self, other: "ManifoldPoint") -> float:
        """Euclidean distance in fingerprint space."""
        return float(np.linalg.norm(self.fingerprint - other.fingerprint))


@dataclass
class DriftEvent:
    """A detected drift event."""

    drift_type: DriftType
    severity: AlertSeverity
    magnitude: float  # How severe (0-1)
    from_point: ManifoldPoint
    to_point: ManifoldPoint
    reason: str
    recommendations: List[str]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drift_type": self.drift_type.value,
            "severity": self.severity.value,
            "magnitude": self.magnitude,
            "from_zone": self.from_point.zone,
            "to_zone": self.to_point.zone,
            "from_position": self.from_point.token_position,
            "to_position": self.to_point.token_position,
            "reason": self.reason,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }


@dataclass
class FirewallState:
    """Current state of the firewall for a generation session."""

    session_id: str
    trajectory: Deque[ManifoldPoint]
    events: List[DriftEvent]
    cumulative_drift: float
    current_severity: AlertSeverity
    zone_history: List[str]
    started_at: float
    last_update: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "trajectory_length": len(self.trajectory),
            "event_count": len(self.events),
            "cumulative_drift": self.cumulative_drift,
            "current_severity": self.current_severity.value,
            "zone_history": self.zone_history[-10:],  # Last 10
            "duration_seconds": self.last_update - self.started_at,
            "events": [e.to_dict() for e in self.events[-5:]],  # Last 5
        }


@dataclass
class FirewallConfig:
    """Configuration for the manifold firewall."""

    # Trajectory settings
    window_size: int = 20  # Number of points to track
    min_points_for_detection: int = 3  # Min points before alerting

    # Drift thresholds
    gradual_drift_threshold: float = 0.3  # Normal drift
    sudden_jump_threshold: float = 0.8  # Teleport detection
    entropy_spike_threshold: float = 1.5  # Entropy increase ratio
    sink_collapse_threshold: float = 0.4  # Sink ratio jump

    # Severity thresholds (cumulative drift)
    watch_threshold: float = 0.5
    warning_threshold: float = 1.0
    alert_threshold: float = 2.0
    critical_threshold: float = 3.0

    # Zone transition settings
    zone_violation_weight: float = 0.5  # Weight for zone violations
    oscillation_window: int = 5  # Window for oscillation detection
    oscillation_threshold: int = 3  # Zone changes for oscillation

    # Decay settings
    drift_decay: float = 0.9  # Decay factor per step
    event_retention: int = 100  # Max events to keep


# =============================================================================
# DRIFT DETECTORS
# =============================================================================


class DriftDetector:
    """Base class for drift detection strategies."""

    def detect(
        self,
        current: ManifoldPoint,
        trajectory: Deque[ManifoldPoint],
        config: FirewallConfig,
    ) -> Optional[DriftEvent]:
        """Detect drift and return event if found."""
        raise NotImplementedError


class SuddenJumpDetector(DriftDetector):
    """Detects sudden jumps to distant manifold regions."""

    def detect(
        self,
        current: ManifoldPoint,
        trajectory: Deque[ManifoldPoint],
        config: FirewallConfig,
    ) -> Optional[DriftEvent]:
        if len(trajectory) < 2:
            return None

        previous = trajectory[-1]
        distance = current.distance_to(previous)

        # Compute average step size for comparison
        if len(trajectory) >= 3:
            recent_distances = [
                trajectory[i].distance_to(trajectory[i - 1])
                for i in range(1, min(5, len(trajectory)))
            ]
            avg_step = np.mean(recent_distances) if recent_distances else 0.3
        else:
            avg_step = 0.3

        # Normalize by average step
        normalized_jump = distance / (avg_step + 0.01)

        if normalized_jump > config.sudden_jump_threshold / 0.3:  # Scale threshold
            magnitude = min(1.0, normalized_jump / 5.0)
            severity = self._compute_severity(magnitude, config)

            return DriftEvent(
                drift_type=DriftType.SUDDEN_JUMP,
                severity=severity,
                magnitude=magnitude,
                from_point=previous,
                to_point=current,
                reason=f"Sudden jump: {normalized_jump:.1f}x normal step size",
                recommendations=[
                    "Verify factual claims in recent tokens",
                    "Consider re-generating from last stable point",
                ],
                timestamp=time.time(),
            )

        return None

    def _compute_severity(
        self, magnitude: float, config: FirewallConfig
    ) -> AlertSeverity:
        if magnitude < 0.3:
            return AlertSeverity.WATCH
        elif magnitude < 0.5:
            return AlertSeverity.WARNING
        elif magnitude < 0.8:
            return AlertSeverity.ALERT
        else:
            return AlertSeverity.CRITICAL


class ZoneViolationDetector(DriftDetector):
    """Detects suspicious zone transitions."""

    def detect(
        self,
        current: ManifoldPoint,
        trajectory: Deque[ManifoldPoint],
        config: FirewallConfig,
    ) -> Optional[DriftEvent]:
        if len(trajectory) < 1:
            return None

        previous = trajectory[-1]
        transition = (previous.zone, current.zone)

        is_suspicious, reason = ZONE_TRANSITIONS.get(
            transition, (False, "Unknown transition")
        )

        if is_suspicious:
            return DriftEvent(
                drift_type=DriftType.ZONE_VIOLATION,
                severity=AlertSeverity.WARNING,
                magnitude=config.zone_violation_weight,
                from_point=previous,
                to_point=current,
                reason=f"Suspicious zone transition: {previous.zone} → {current.zone}. {reason}",
                recommendations=[
                    "Check if context supports this transition",
                    "May indicate topic drift or hallucination",
                ],
                timestamp=time.time(),
            )

        return None


class EntropySpikeDetector(DriftDetector):
    """Detects sudden increases in attention entropy."""

    def detect(
        self,
        current: ManifoldPoint,
        trajectory: Deque[ManifoldPoint],
        config: FirewallConfig,
    ) -> Optional[DriftEvent]:
        if len(trajectory) < 2:
            return None

        # Compute recent average entropy
        recent_entropies = [p.entropy for p in list(trajectory)[-5:]]
        avg_entropy = np.mean(recent_entropies)

        if avg_entropy > 0.1:  # Avoid division issues
            entropy_ratio = current.entropy / avg_entropy

            if entropy_ratio > config.entropy_spike_threshold:
                magnitude = min(1.0, (entropy_ratio - 1.0) / 2.0)

                return DriftEvent(
                    drift_type=DriftType.ENTROPY_SPIKE,
                    severity=(
                        AlertSeverity.WARNING
                        if magnitude < 0.5
                        else AlertSeverity.ALERT
                    ),
                    magnitude=magnitude,
                    from_point=trajectory[-1],
                    to_point=current,
                    reason=f"Entropy spike: {entropy_ratio:.1f}x average ({current.entropy:.2f} vs {avg_entropy:.2f})",
                    recommendations=[
                        "Model may be uncertain about next token",
                        "Consider if response is becoming incoherent",
                    ],
                    timestamp=time.time(),
                )

        return None


class SinkCollapseDetector(DriftDetector):
    """Detects high sink attention after confident patterns."""

    def detect(
        self,
        current: ManifoldPoint,
        trajectory: Deque[ManifoldPoint],
        config: FirewallConfig,
    ) -> Optional[DriftEvent]:
        if len(trajectory) < 3:
            return None

        # Check if recent patterns had low sink ratio
        recent_sink_ratios = [p.sink_ratio for p in list(trajectory)[-5:]]
        avg_sink = np.mean(recent_sink_ratios)

        # Detect jump to high sink
        sink_increase = current.sink_ratio - avg_sink

        if sink_increase > config.sink_collapse_threshold and current.sink_ratio > 0.3:
            magnitude = min(1.0, sink_increase / 0.5)

            return DriftEvent(
                drift_type=DriftType.SINK_COLLAPSE,
                severity=AlertSeverity.ALERT,
                magnitude=magnitude,
                from_point=trajectory[-1],
                to_point=current,
                reason=f"Sink collapse: {current.sink_ratio:.0%} sink attention (was {avg_sink:.0%})",
                recommendations=[
                    "Model lost confident retrieval target",
                    "May be confabulating or uncertain",
                    "Check if preceding content is factually grounded",
                ],
                timestamp=time.time(),
            )

        return None


class OscillationDetector(DriftDetector):
    """Detects rapid oscillation between zones."""

    def detect(
        self,
        current: ManifoldPoint,
        trajectory: Deque[ManifoldPoint],
        config: FirewallConfig,
    ) -> Optional[DriftEvent]:
        if len(trajectory) < config.oscillation_window:
            return None

        # Count zone changes in recent window
        recent_zones = [p.zone for p in list(trajectory)[-config.oscillation_window :]]
        recent_zones.append(current.zone)

        zone_changes = sum(
            1
            for i in range(1, len(recent_zones))
            if recent_zones[i] != recent_zones[i - 1]
        )

        if zone_changes >= config.oscillation_threshold:
            magnitude = min(1.0, zone_changes / config.oscillation_window)

            return DriftEvent(
                drift_type=DriftType.OSCILLATION,
                severity=AlertSeverity.WARNING,
                magnitude=magnitude,
                from_point=trajectory[-1],
                to_point=current,
                reason=f"Zone oscillation: {zone_changes} changes in {config.oscillation_window} tokens",
                recommendations=[
                    "Model may be unstable or confused",
                    "Pattern suggests competing interpretations",
                ],
                timestamp=time.time(),
            )

        return None


# =============================================================================
# MANIFOLD FIREWALL
# =============================================================================


class ManifoldFirewall:
    """
    Monitors attention patterns for hallucination indicators.

    Usage:
        firewall = ManifoldFirewall()
        session_id = firewall.start_session("request-123")

        for token_idx, fingerprint in enumerate(generation):
            result = firewall.check(
                session_id=session_id,
                fingerprint=fingerprint,
                zone="semantic_bridge",
                entropy=2.5,
                sink_ratio=0.15,
                token_position=token_idx,
            )

            if result.severity >= AlertSeverity.ALERT:
                print(f"Hallucination warning: {result.reason}")

        final_state = firewall.end_session(session_id)
    """

    def __init__(self, config: Optional[FirewallConfig] = None):
        self.config = config or FirewallConfig()

        # Active sessions
        self._sessions: Dict[str, FirewallState] = {}

        # Drift detectors (order matters - more specific first)
        self._detectors: List[DriftDetector] = [
            SinkCollapseDetector(),
            SuddenJumpDetector(),
            ZoneViolationDetector(),
            EntropySpikeDetector(),
            OscillationDetector(),
        ]

        # Statistics
        self._total_checks = 0
        self._total_events = 0
        self._severity_counts = {s: 0 for s in AlertSeverity}

    def start_session(self, session_id: str) -> str:
        """Start monitoring a new generation session."""
        now = time.time()
        self._sessions[session_id] = FirewallState(
            session_id=session_id,
            trajectory=deque(maxlen=self.config.window_size),
            events=[],
            cumulative_drift=0.0,
            current_severity=AlertSeverity.SAFE,
            zone_history=[],
            started_at=now,
            last_update=now,
        )
        return session_id

    def check(
        self,
        session_id: str,
        fingerprint: np.ndarray,
        zone: str,
        entropy: float,
        sink_ratio: float,
        token_position: int,
        cluster_id: Optional[int] = None,
        embedding: Optional[Tuple[float, float]] = None,
    ) -> "FirewallCheckResult":
        """
        Check a new fingerprint for drift.

        Returns:
            FirewallCheckResult with current severity and any new events
        """
        self._total_checks += 1

        if session_id not in self._sessions:
            self.start_session(session_id)

        state = self._sessions[session_id]
        now = time.time()

        # Create manifold point
        point = ManifoldPoint(
            fingerprint=np.array(fingerprint),
            zone=zone,
            cluster_id=cluster_id,
            embedding=embedding,
            entropy=entropy,
            sink_ratio=sink_ratio,
            timestamp=now,
            token_position=token_position,
        )

        # Run drift detectors
        new_events = []
        if len(state.trajectory) >= self.config.min_points_for_detection:
            for detector in self._detectors:
                event = detector.detect(point, state.trajectory, self.config)
                if event:
                    new_events.append(event)
                    state.events.append(event)
                    self._total_events += 1

                    # Limit stored events
                    if len(state.events) > self.config.event_retention:
                        state.events = state.events[-self.config.event_retention :]

        # Update cumulative drift
        if new_events:
            drift_increase = sum(e.magnitude for e in new_events)
            state.cumulative_drift = (
                state.cumulative_drift * self.config.drift_decay + drift_increase
            )
        else:
            # Decay drift over time
            state.cumulative_drift *= self.config.drift_decay

        # Update severity based on cumulative drift
        state.current_severity = self._compute_severity(state.cumulative_drift)
        self._severity_counts[state.current_severity] += 1

        # Update trajectory and history
        state.trajectory.append(point)
        state.zone_history.append(zone)
        state.last_update = now

        return FirewallCheckResult(
            severity=state.current_severity,
            cumulative_drift=state.cumulative_drift,
            new_events=new_events,
            trajectory_length=len(state.trajectory),
            session_id=session_id,
        )

    def _compute_severity(self, cumulative_drift: float) -> AlertSeverity:
        """Compute severity from cumulative drift."""
        if cumulative_drift >= self.config.critical_threshold:
            return AlertSeverity.CRITICAL
        elif cumulative_drift >= self.config.alert_threshold:
            return AlertSeverity.ALERT
        elif cumulative_drift >= self.config.warning_threshold:
            return AlertSeverity.WARNING
        elif cumulative_drift >= self.config.watch_threshold:
            return AlertSeverity.WATCH
        else:
            return AlertSeverity.SAFE

    def get_session_state(self, session_id: str) -> Optional[FirewallState]:
        """Get current state of a session."""
        return self._sessions.get(session_id)

    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """End a session and return final state."""
        state = self._sessions.pop(session_id, None)
        if state:
            return state.to_dict()
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get firewall statistics."""
        return {
            "total_checks": self._total_checks,
            "total_events": self._total_events,
            "active_sessions": len(self._sessions),
            "severity_distribution": {
                s.value: count for s, count in self._severity_counts.items()
            },
        }

    def reset_statistics(self):
        """Reset statistics counters."""
        self._total_checks = 0
        self._total_events = 0
        self._severity_counts = {s: 0 for s in AlertSeverity}


@dataclass
class FirewallCheckResult:
    """Result of a firewall check."""

    severity: AlertSeverity
    cumulative_drift: float
    new_events: List[DriftEvent]
    trajectory_length: int
    session_id: str

    @property
    def is_safe(self) -> bool:
        return self.severity == AlertSeverity.SAFE

    @property
    def needs_attention(self) -> bool:
        return self.severity in [
            AlertSeverity.WARNING,
            AlertSeverity.ALERT,
            AlertSeverity.CRITICAL,
        ]

    @property
    def should_stop(self) -> bool:
        return self.severity == AlertSeverity.CRITICAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity.value,
            "cumulative_drift": self.cumulative_drift,
            "new_events": [e.to_dict() for e in self.new_events],
            "trajectory_length": self.trajectory_length,
            "session_id": self.session_id,
            "is_safe": self.is_safe,
            "needs_attention": self.needs_attention,
            "should_stop": self.should_stop,
        }


# =============================================================================
# BATCH ANALYZER
# =============================================================================


class ManifoldBatchAnalyzer:
    """
    Analyzes complete generation for hallucination patterns.

    For post-hoc analysis of responses.
    """

    def __init__(self, config: Optional[FirewallConfig] = None):
        self.config = config or FirewallConfig()
        self.firewall = ManifoldFirewall(config)

    def analyze(
        self,
        fingerprints: List[np.ndarray],
        zones: List[str],
        entropies: List[float],
        sink_ratios: List[float],
    ) -> Dict[str, Any]:
        """
        Analyze a complete sequence of fingerprints.

        Returns:
            Analysis results with events, severity timeline, and recommendations
        """
        session_id = f"batch-{time.time()}"
        self.firewall.start_session(session_id)

        results = []
        for i, (fp, zone, entropy, sink) in enumerate(
            zip(fingerprints, zones, entropies, sink_ratios)
        ):
            result = self.firewall.check(
                session_id=session_id,
                fingerprint=fp,
                zone=zone,
                entropy=entropy,
                sink_ratio=sink,
                token_position=i,
            )
            results.append(result)

        final_state = self.firewall.end_session(session_id)

        # Compute summary
        severity_timeline = [r.severity.value for r in results]
        max_severity = max(results, key=lambda r: list(AlertSeverity).index(r.severity))
        total_events = sum(len(r.new_events) for r in results)

        # Find suspicious regions
        suspicious_regions = []
        in_region = False
        region_start = 0

        for i, result in enumerate(results):
            if result.needs_attention and not in_region:
                in_region = True
                region_start = i
            elif not result.needs_attention and in_region:
                in_region = False
                suspicious_regions.append(
                    {
                        "start": region_start,
                        "end": i - 1,
                        "length": i - region_start,
                    }
                )

        if in_region:
            suspicious_regions.append(
                {
                    "start": region_start,
                    "end": len(results) - 1,
                    "length": len(results) - region_start,
                }
            )

        # Generate recommendations
        recommendations = self._generate_recommendations(results, suspicious_regions)

        return {
            "total_tokens": len(fingerprints),
            "total_events": total_events,
            "max_severity": max_severity.severity.value,
            "final_cumulative_drift": results[-1].cumulative_drift if results else 0,
            "severity_timeline": severity_timeline,
            "suspicious_regions": suspicious_regions,
            "recommendations": recommendations,
            "events": final_state.get("events", []) if final_state else [],
        }

    def _generate_recommendations(
        self,
        results: List[FirewallCheckResult],
        suspicious_regions: List[Dict],
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if not suspicious_regions:
            recommendations.append(
                "Generation appears stable with no major drift events."
            )
            return recommendations

        # Count event types
        event_types = {}
        for result in results:
            for event in result.new_events:
                event_types[event.drift_type] = event_types.get(event.drift_type, 0) + 1

        # Generate type-specific recommendations
        if event_types.get(DriftType.SUDDEN_JUMP, 0) > 0:
            recommendations.append(
                f"Detected {event_types[DriftType.SUDDEN_JUMP]} sudden jumps. "
                "Verify factual claims near these positions."
            )

        if event_types.get(DriftType.SINK_COLLAPSE, 0) > 0:
            recommendations.append(
                f"Detected {event_types[DriftType.SINK_COLLAPSE]} sink collapse events. "
                "Model may have lost confident retrieval - check for confabulation."
            )

        if event_types.get(DriftType.ENTROPY_SPIKE, 0) > 0:
            recommendations.append(
                f"Detected {event_types[DriftType.ENTROPY_SPIKE]} entropy spikes. "
                "Model showed uncertainty - verify coherence in affected regions."
            )

        if event_types.get(DriftType.OSCILLATION, 0) > 0:
            recommendations.append(
                f"Detected zone oscillation. "
                "Model may be unstable or responding to ambiguous context."
            )

        # Region-specific
        if len(suspicious_regions) > 2:
            recommendations.append(
                f"Multiple suspicious regions detected ({len(suspicious_regions)}). "
                "Consider re-generating with different prompt or temperature."
            )

        return recommendations


# =============================================================================
# EDUCATIONAL GLOSSARY
# =============================================================================

FIREWALL_GLOSSARY = {
    "manifold_drift": {
        "term": "Manifold Drift",
        "simple": "How much the attention pattern moves from its expected path",
        "detailed": "Measures the distance traveled on the attention manifold relative to the expected trajectory. Normal generation follows predictable paths; hallucination often involves unexpected jumps.",
        "analogy": "Like a GPS tracking a car - normal driving follows roads, but teleporting to a random location is suspicious.",
        "why_it_matters": "High drift correlates with factual errors and confabulation.",
    },
    "sudden_jump": {
        "term": "Sudden Jump",
        "simple": "Attention pattern teleports to a distant region",
        "detailed": "When the attention pattern moves much further than typical between consecutive tokens. This often indicates the model has 'lost its place' and may be generating without proper grounding.",
        "analogy": "Like a conversation suddenly changing topic mid-sentence without transition.",
        "why_it_matters": "Strong indicator of potential hallucination or topic drift.",
    },
    "sink_collapse": {
        "term": "Sink Collapse",
        "simple": "Attention suddenly flows to the beginning (sink tokens)",
        "detailed": "After generating confidently (low sink attention), the model suddenly redirects attention to the first few tokens. This 'garbage collection' behavior often indicates lost context or uncertainty.",
        "analogy": "Like someone confidently giving directions, then suddenly saying 'wait, let me start over.'",
        "why_it_matters": "Often precedes fabricated information as the model 'resets.'",
    },
    "zone_oscillation": {
        "term": "Zone Oscillation",
        "simple": "Rapid switching between different attention patterns",
        "detailed": "When the model rapidly alternates between syntax_floor, semantic_bridge, and structure_ripple zones instead of progressing smoothly. Indicates unstable or competing interpretations.",
        "analogy": "Like a driver constantly changing lanes without clear purpose.",
        "why_it_matters": "Suggests the model is uncertain or the prompt is ambiguous.",
    },
    "cumulative_drift": {
        "term": "Cumulative Drift Score",
        "simple": "Running total of all drift events, with decay",
        "detailed": "Accumulates drift magnitude over time but decays between tokens. A single small event is fine, but repeated small events or large events push the score up, triggering alerts.",
        "analogy": "Like a 'suspicion meter' that fills up with each anomaly but slowly drains.",
        "why_it_matters": "Prevents false positives from single events while catching sustained issues.",
    },
}


# =============================================================================
# FACTORY AND DEMO
# =============================================================================


def create_firewall(config: Optional[FirewallConfig] = None) -> ManifoldFirewall:
    """Create a manifold firewall with default or custom config."""
    return ManifoldFirewall(config)


def demo():
    """Demonstrate the manifold firewall."""
    print("=" * 60)
    print("Manifold Firewall Demo: Hallucination Detection")
    print("=" * 60)

    firewall = ManifoldFirewall()
    session = firewall.start_session("demo-session")

    # Simulate normal generation
    print("\n--- Phase 1: Normal Generation ---")
    normal_fps = [
        (np.random.randn(20) * 0.1, "syntax_floor", 1.5, 0.1),
        (np.random.randn(20) * 0.1, "syntax_floor", 1.6, 0.12),
        (np.random.randn(20) * 0.1, "semantic_bridge", 2.0, 0.15),
        (np.random.randn(20) * 0.1, "semantic_bridge", 2.1, 0.14),
        (np.random.randn(20) * 0.1, "semantic_bridge", 2.2, 0.13),
    ]

    for i, (fp, zone, entropy, sink) in enumerate(normal_fps):
        result = firewall.check(
            session_id=session,
            fingerprint=fp,
            zone=zone,
            entropy=entropy,
            sink_ratio=sink,
            token_position=i,
        )
        print(
            f"Token {i}: {result.severity.value} (drift={result.cumulative_drift:.2f})"
        )

    # Simulate sudden jump (hallucination indicator)
    print("\n--- Phase 2: Sudden Jump (Potential Hallucination) ---")
    jump_fp = np.random.randn(20) * 2.0  # Much larger magnitude
    result = firewall.check(
        session_id=session,
        fingerprint=jump_fp,
        zone="structure_ripple",
        entropy=3.5,
        sink_ratio=0.1,
        token_position=5,
    )
    print(f"Token 5: {result.severity.value} (drift={result.cumulative_drift:.2f})")
    for event in result.new_events:
        print(f"  EVENT: {event.drift_type.value} - {event.reason}")

    # Simulate sink collapse
    print("\n--- Phase 3: Sink Collapse ---")
    result = firewall.check(
        session_id=session,
        fingerprint=np.random.randn(20) * 0.1,
        zone="syntax_floor",
        entropy=2.0,
        sink_ratio=0.6,  # High sink
        token_position=6,
    )
    print(f"Token 6: {result.severity.value} (drift={result.cumulative_drift:.2f})")
    for event in result.new_events:
        print(f"  EVENT: {event.drift_type.value} - {event.reason}")

    # Final state
    print("\n--- Session Summary ---")
    final = firewall.end_session(session)
    print(f"Total events: {final['event_count']}")
    print(f"Final severity: {final['current_severity']}")
    print(f"Duration: {final['duration_seconds']:.2f}s")

    # Statistics
    print("\n--- Firewall Statistics ---")
    stats = firewall.get_statistics()
    print(f"Total checks: {stats['total_checks']}")
    print(f"Severity distribution: {stats['severity_distribution']}")


if __name__ == "__main__":
    demo()
