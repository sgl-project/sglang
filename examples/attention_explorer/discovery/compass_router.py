"""
Compass Router: Angle-Based Query Routing with Sinq Anchoring

Routes queries based on the angular variance of attention patterns,
using the attention sink as a coordinate origin ("true north").

=== CORE CONCEPT ===

Traditional routing looks at WHERE attention goes.
Compass routing looks at the DIRECTION of attention relative to a fixed reference.

The "attention sink" (first few tokens like <BOS>) acts as true north.
All other attention is measured as an angle relative to this anchor.

Routing Logic:
- Low angle variance → Syntactic patterns → Small model
- High angle variance → Reasoning patterns → Large model
- Oscillating variance → Creative patterns → Needs context

=== MATHEMATICAL FOUNDATION ===

1. Sinq Anchor: Extract attention vector to sink tokens (positions 0-4)
   sink_attention = [a_0, a_1, a_2, a_3, a_4]  # Normalized

2. Content Attention: Extract attention to non-sink tokens
   content_attention = [a_5, a_6, ..., a_n]  # Normalized separately

3. Angular Decomposition: For each token i, compute:
   - Radial component: How much attention (magnitude)
   - Angular component: Direction relative to sink (using 2D projection)

4. Compass Heading: The dominant angular direction of attention
   heading = atan2(sum(a_i * sin(θ_i)), sum(a_i * cos(θ_i)))

5. Angular Variance: How spread out the angular distribution is
   variance = 1 - |mean(e^(iθ))|  # Circular variance

=== ROUTING INTERPRETATION ===

Low Angular Variance (< 0.3):
- Attention is focused in one direction
- Likely syntactic or retrieval pattern
- Route to: Small model, no CoT

Medium Angular Variance (0.3 - 0.7):
- Attention has some spread but clear center
- Moderate reasoning or bridging
- Route to: Medium model, optional CoT

High Angular Variance (> 0.7):
- Attention is scattered in many directions
- Complex reasoning or uncertainty
- Route to: Large model, enable CoT

Oscillating Pattern:
- Variance changes significantly across layers/heads
- Creative or exploratory thinking
- Route to: Large model with full context
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================


class CompassHeading(Enum):
    """Cardinal directions for attention patterns."""

    NORTH = "north"  # Toward beginning (retrieval)
    EAST = "east"  # Toward recent local context
    SOUTH = "south"  # Away from sink (creative)
    WEST = "west"  # Toward middle context
    SCATTERED = "scattered"  # No clear direction


class RoutingTier(Enum):
    """Model tier for routing decision."""

    SMALL = "small"  # 4B-8B, fast syntactic
    MEDIUM = "medium"  # 13B-30B, balanced
    LARGE = "large"  # 70B+, complex reasoning
    CONTEXT = "context"  # Needs full context window


@dataclass
class SinqAnchor:
    """
    Attention sink anchor for compass orientation.

    The sink tokens (typically positions 0-4) act as "true north"
    for measuring attention direction.
    """

    sink_positions: List[int]  # Which positions are sink
    sink_attention: np.ndarray  # Attention mass to sink
    sink_total: float  # Total sink attention
    is_sink_dominated: bool  # >30% attention to sink
    sink_entropy: float  # Entropy within sink

    @classmethod
    def from_attention(
        cls,
        positions: List[int],
        attention_scores: List[float],
        sink_threshold: int = 5,
    ) -> "SinqAnchor":
        """
        Extract sink anchor from attention pattern.

        Args:
            positions: Positions of attended tokens
            attention_scores: Attention weights
            sink_threshold: Positions below this are sink
        """
        positions = np.array(positions)
        scores = np.array(attention_scores)

        # Identify sink tokens
        sink_mask = positions < sink_threshold
        sink_positions = positions[sink_mask].tolist()
        sink_attention = scores[sink_mask]
        sink_total = float(sink_attention.sum())

        # Check if sink dominated
        is_sink_dominated = sink_total > 0.3

        # Compute entropy within sink
        if sink_attention.sum() > 0:
            sink_probs = sink_attention / sink_attention.sum()
            sink_probs = sink_probs[sink_probs > 0]
            sink_entropy = -float(np.sum(sink_probs * np.log2(sink_probs + 1e-10)))
        else:
            sink_entropy = 0.0

        return cls(
            sink_positions=sink_positions,
            sink_attention=sink_attention,
            sink_total=sink_total,
            is_sink_dominated=is_sink_dominated,
            sink_entropy=sink_entropy,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sink_positions": self.sink_positions,
            "sink_attention": (
                self.sink_attention.tolist()
                if isinstance(self.sink_attention, np.ndarray)
                else self.sink_attention
            ),
            "sink_total": self.sink_total,
            "is_sink_dominated": self.is_sink_dominated,
            "sink_entropy": self.sink_entropy,
        }


@dataclass
class CompassReading:
    """
    Angular analysis of attention pattern relative to sink anchor.
    """

    # Angular metrics
    heading: CompassHeading  # Dominant direction
    heading_angle: float  # Angle in radians [0, 2π]
    angular_variance: float  # How spread out (0=focused, 1=scattered)
    angular_concentration: float  # Inverse variance (0=scattered, 1=focused)

    # Directional breakdown
    north_mass: float  # Attention toward beginning
    east_mass: float  # Attention toward recent
    south_mass: float  # Attention away from sink
    west_mass: float  # Attention toward middle

    # Pattern classification
    pattern_type: str  # "focused", "bimodal", "scattered", "oscillating"
    rotational_variance: float  # From RoPE de-rotation

    # Sink analysis
    anchor: SinqAnchor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "heading": self.heading.value,
            "heading_angle": self.heading_angle,
            "angular_variance": self.angular_variance,
            "angular_concentration": self.angular_concentration,
            "north_mass": self.north_mass,
            "east_mass": self.east_mass,
            "south_mass": self.south_mass,
            "west_mass": self.west_mass,
            "pattern_type": self.pattern_type,
            "rotational_variance": self.rotational_variance,
            "anchor": self.anchor.to_dict(),
        }


@dataclass
class CompassRoutingDecision:
    """
    Routing decision based on compass analysis.
    """

    tier: RoutingTier
    use_chain_of_thought: bool
    reason: str
    confidence: float

    # Compass metrics
    compass: CompassReading

    # Recommendations
    recommended_model: str
    recommended_temperature: float
    recommended_max_tokens: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "use_chain_of_thought": self.use_chain_of_thought,
            "reason": self.reason,
            "confidence": self.confidence,
            "compass": self.compass.to_dict(),
            "recommended_model": self.recommended_model,
            "recommended_temperature": self.recommended_temperature,
            "recommended_max_tokens": self.recommended_max_tokens,
        }


@dataclass
class CompassRouterConfig:
    """Configuration for the compass router."""

    # Sink configuration
    sink_threshold: int = 5  # Positions 0-4 are sink

    # Angular variance thresholds
    low_variance_threshold: float = 0.3  # Below = syntactic
    high_variance_threshold: float = 0.7  # Above = reasoning

    # Model mapping
    small_model: str = "Qwen/Qwen3-4B"
    medium_model: str = "Qwen/Qwen3-14B"
    large_model: str = "Qwen/Qwen3-72B"

    # Temperature recommendations
    syntactic_temperature: float = 0.3  # Low for precision
    reasoning_temperature: float = 0.7  # Medium for exploration
    creative_temperature: float = 0.9  # High for creativity

    # Token limits
    syntactic_max_tokens: int = 256
    reasoning_max_tokens: int = 1024
    creative_max_tokens: int = 2048

    # CoT thresholds
    cot_variance_threshold: float = 0.5  # Above = enable CoT
    cot_sink_threshold: float = 0.4  # High sink = uncertain, enable CoT


# =============================================================================
# COMPASS ANALYZER
# =============================================================================


class CompassAnalyzer:
    """
    Analyzes attention patterns using angular decomposition.

    Uses the attention sink as a reference point and measures
    the angular distribution of attention to other tokens.
    """

    def __init__(self, config: Optional[CompassRouterConfig] = None):
        self.config = config or CompassRouterConfig()

    def analyze(
        self,
        query_pos: int,
        key_positions: List[int],
        attention_scores: List[float],
        rotational_variance: Optional[float] = None,
    ) -> CompassReading:
        """
        Analyze attention pattern and compute compass reading.

        Args:
            query_pos: Position of the query token
            key_positions: Positions of attended tokens
            attention_scores: Attention weights (should sum to 1)
            rotational_variance: Pre-computed RoPE variance (optional)

        Returns:
            CompassReading with angular analysis
        """
        positions = np.array(key_positions)
        scores = np.array(attention_scores)

        # Normalize scores
        if scores.sum() > 0:
            scores = scores / scores.sum()

        # Extract sink anchor
        anchor = SinqAnchor.from_attention(
            positions.tolist(), scores.tolist(), self.config.sink_threshold
        )

        # Get non-sink attention
        non_sink_mask = positions >= self.config.sink_threshold
        content_positions = positions[non_sink_mask]
        content_scores = scores[non_sink_mask]

        if len(content_positions) == 0:
            # All attention to sink - return sink-dominated reading
            return self._sink_dominated_reading(anchor, rotational_variance or 0.0)

        # Normalize content attention
        if content_scores.sum() > 0:
            content_scores = content_scores / content_scores.sum()

        # Compute angular positions relative to query
        # Map position differences to angles [0, 2π]
        angles = self._positions_to_angles(query_pos, content_positions)

        # Compute weighted circular mean (heading)
        heading_angle, angular_variance = self._circular_statistics(
            angles, content_scores
        )

        # Classify heading into cardinal direction
        heading = self._angle_to_heading(heading_angle, angular_variance)

        # Compute directional mass
        north_mass, east_mass, south_mass, west_mass = self._directional_mass(
            angles, content_scores
        )

        # Classify pattern type
        pattern_type = self._classify_pattern(
            angular_variance,
            anchor.is_sink_dominated,
            north_mass,
            east_mass,
            south_mass,
            west_mass,
        )

        return CompassReading(
            heading=heading,
            heading_angle=float(heading_angle),
            angular_variance=float(angular_variance),
            angular_concentration=float(1.0 - angular_variance),
            north_mass=float(north_mass),
            east_mass=float(east_mass),
            south_mass=float(south_mass),
            west_mass=float(west_mass),
            pattern_type=pattern_type,
            rotational_variance=rotational_variance or 0.0,
            anchor=anchor,
        )

    def _positions_to_angles(
        self,
        query_pos: int,
        content_positions: np.ndarray,
    ) -> np.ndarray:
        """
        Convert positions to angular coordinates based on relative position.

        Mapping (using log-distance for scale):
        - Very close (within 16) → EAST quadrant [7π/4 to π/4]
        - Close (16-64) → partial EAST/NORTH
        - Medium (64-256) → WEST quadrant [5π/4 to 7π/4]
        - Far (256+) → NORTH quadrant [3π/4 to 5π/4]

        This groups nearby tokens together and distant tokens together,
        enabling variance to distinguish local vs. long-range patterns.
        """
        # Compute distance from query (all should be negative for causal)
        distances = query_pos - content_positions  # Positive = tokens before query

        # Use log-scale distance for better spread
        # Add 1 to avoid log(0), then normalize
        log_dist = np.log1p(np.abs(distances))
        max_log = np.log1p(query_pos) if query_pos > 0 else 1.0

        # Normalize to [0, 1] where 0 = at query, 1 = at beginning
        normalized = log_dist / max_log

        # Map to angles:
        # 0 (local) → 0 (EAST direction)
        # 0.5 (mid) → π (WEST direction)
        # 1.0 (far) → 3π/2 (NORTH direction, wrapping)
        angles = normalized * 1.5 * np.pi  # Maps [0,1] to [0, 1.5π]

        return angles

    def _circular_statistics(
        self,
        angles: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute circular mean and variance.

        Returns:
            (mean_angle, variance) where variance is in [0, 1]
        """
        # Weighted circular mean using complex representation
        z = np.sum(weights * np.exp(1j * angles))

        # Mean angle
        mean_angle = np.angle(z)
        if mean_angle < 0:
            mean_angle += 2 * np.pi

        # Circular variance: 1 - |mean resultant length|
        R = np.abs(z) / weights.sum() if weights.sum() > 0 else 0
        variance = 1 - R

        return mean_angle, variance

    def _angle_to_heading(
        self,
        angle: float,
        variance: float,
    ) -> CompassHeading:
        """Convert angle to cardinal heading."""
        if variance > 0.8:
            return CompassHeading.SCATTERED

        # Divide circle into quadrants
        # North: 7π/4 to π/4 (toward beginning)
        # East: π/4 to 3π/4 (toward recent)
        # South: 3π/4 to 5π/4 (away from sink)
        # West: 5π/4 to 7π/4 (toward middle)

        if angle < np.pi / 4 or angle >= 7 * np.pi / 4:
            return CompassHeading.NORTH
        elif angle < 3 * np.pi / 4:
            return CompassHeading.EAST
        elif angle < 5 * np.pi / 4:
            return CompassHeading.SOUTH
        else:
            return CompassHeading.WEST

    def _directional_mass(
        self,
        angles: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Compute attention mass in each cardinal direction."""
        # Define direction masks
        north_mask = (angles < np.pi / 4) | (angles >= 7 * np.pi / 4)
        east_mask = (angles >= np.pi / 4) & (angles < 3 * np.pi / 4)
        south_mask = (angles >= 3 * np.pi / 4) & (angles < 5 * np.pi / 4)
        west_mask = (angles >= 5 * np.pi / 4) & (angles < 7 * np.pi / 4)

        north_mass = float(weights[north_mask].sum())
        east_mass = float(weights[east_mask].sum())
        south_mass = float(weights[south_mask].sum())
        west_mass = float(weights[west_mask].sum())

        return north_mass, east_mass, south_mass, west_mass

    def _classify_pattern(
        self,
        variance: float,
        sink_dominated: bool,
        north: float,
        east: float,
        south: float,
        west: float,
    ) -> str:
        """Classify the attention pattern type."""
        if sink_dominated:
            return "sink_dominated"

        if variance < 0.2:
            return "focused"

        if variance > 0.8:
            return "scattered"

        # Check for bimodal pattern (two dominant directions)
        masses = sorted([north, east, south, west], reverse=True)
        if masses[0] > 0.3 and masses[1] > 0.3 and masses[0] + masses[1] > 0.7:
            return "bimodal"

        return "distributed"

    def _sink_dominated_reading(
        self,
        anchor: SinqAnchor,
        rotational_variance: float,
    ) -> CompassReading:
        """Create reading for sink-dominated attention."""
        return CompassReading(
            heading=CompassHeading.NORTH,
            heading_angle=0.0,
            angular_variance=0.0,
            angular_concentration=1.0,
            north_mass=1.0,
            east_mass=0.0,
            south_mass=0.0,
            west_mass=0.0,
            pattern_type="sink_dominated",
            rotational_variance=rotational_variance,
            anchor=anchor,
        )


# =============================================================================
# COMPASS ROUTER
# =============================================================================


class CompassRouter:
    """
    Routes queries based on compass analysis of attention patterns.

    Uses angular variance and directional analysis to determine:
    1. Which model tier to use (small/medium/large)
    2. Whether to enable chain-of-thought
    3. Recommended generation parameters
    """

    def __init__(self, config: Optional[CompassRouterConfig] = None):
        self.config = config or CompassRouterConfig()
        self.analyzer = CompassAnalyzer(config)

        # Track routing statistics
        self._route_counts = {tier: 0 for tier in RoutingTier}
        self._total_routes = 0

    def route(
        self,
        query_pos: int,
        key_positions: List[int],
        attention_scores: List[float],
        rotational_variance: Optional[float] = None,
    ) -> CompassRoutingDecision:
        """
        Route a query based on its attention pattern.

        Args:
            query_pos: Position of query token
            key_positions: Positions of attended tokens
            attention_scores: Attention weights
            rotational_variance: Pre-computed RoPE variance

        Returns:
            CompassRoutingDecision with model recommendation
        """
        # Analyze attention pattern
        compass = self.analyzer.analyze(
            query_pos, key_positions, attention_scores, rotational_variance
        )

        # Determine routing tier
        tier, reason = self._determine_tier(compass)

        # Determine CoT
        use_cot = self._should_use_cot(compass, tier)

        # Compute confidence
        confidence = self._compute_confidence(compass, tier)

        # Get recommendations
        model, temperature, max_tokens = self._get_recommendations(tier, compass)

        # Update statistics
        self._route_counts[tier] += 1
        self._total_routes += 1

        return CompassRoutingDecision(
            tier=tier,
            use_chain_of_thought=use_cot,
            reason=reason,
            confidence=confidence,
            compass=compass,
            recommended_model=model,
            recommended_temperature=temperature,
            recommended_max_tokens=max_tokens,
        )

    def route_fingerprint(
        self,
        fingerprint: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CompassRoutingDecision:
        """
        Route based on pre-computed fingerprint.

        The fingerprint should contain:
        - entropy, local_ratio, long_range_ratio (for zone)
        - Optional: position/score arrays for full analysis

        Args:
            fingerprint: 20-dimensional fingerprint vector
            metadata: Optional dict with positions/scores

        Returns:
            CompassRoutingDecision
        """
        # Extract relevant features from fingerprint
        # Fingerprint structure (from discovery_job.py):
        # [0]: avg_entropy
        # [1]: entropy_std
        # [2]: local_ratio (within 16 tokens)
        # [3]: mid_ratio (16-256 tokens)
        # [4]: long_ratio (256+ tokens)
        # ... etc

        avg_entropy = fingerprint[0]
        local_ratio = fingerprint[2]
        long_ratio = fingerprint[4]

        # Approximate angular variance from fingerprint
        # High entropy + low local ratio = scattered (high variance)
        # Low entropy + high local ratio = focused (low variance)
        approx_variance = min(1.0, avg_entropy / 4.0 * (1 - local_ratio))

        # Create synthetic compass reading from fingerprint
        if local_ratio > 0.5:
            heading = CompassHeading.EAST
            pattern = "focused"
        elif long_ratio > 0.4:
            heading = CompassHeading.NORTH
            pattern = "distributed"
        else:
            heading = CompassHeading.SCATTERED
            pattern = "scattered"

        # Create minimal anchor
        anchor = SinqAnchor(
            sink_positions=[0, 1, 2, 3, 4],
            sink_attention=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            sink_total=0.2,  # Estimated
            is_sink_dominated=False,
            sink_entropy=2.3,  # Max entropy for 5 tokens
        )

        compass = CompassReading(
            heading=heading,
            heading_angle=0.0,
            angular_variance=approx_variance,
            angular_concentration=1.0 - approx_variance,
            north_mass=long_ratio,
            east_mass=local_ratio,
            south_mass=0.0,
            west_mass=1.0 - local_ratio - long_ratio,
            pattern_type=pattern,
            rotational_variance=0.0,
            anchor=anchor,
        )

        # Route using compass
        tier, reason = self._determine_tier(compass)
        use_cot = self._should_use_cot(compass, tier)
        confidence = self._compute_confidence(compass, tier)
        model, temperature, max_tokens = self._get_recommendations(tier, compass)

        return CompassRoutingDecision(
            tier=tier,
            use_chain_of_thought=use_cot,
            reason=f"From fingerprint: {reason}",
            confidence=confidence * 0.8,  # Lower confidence for fingerprint-based
            compass=compass,
            recommended_model=model,
            recommended_temperature=temperature,
            recommended_max_tokens=max_tokens,
        )

    def _determine_tier(
        self,
        compass: CompassReading,
    ) -> Tuple[RoutingTier, str]:
        """Determine routing tier from compass reading."""
        variance = compass.angular_variance
        pattern = compass.pattern_type

        # Sink-dominated = uncertain, needs large model
        if pattern == "sink_dominated":
            return RoutingTier.LARGE, "Sink-dominated attention indicates uncertainty"

        # Low variance = syntactic, small model
        if variance < self.config.low_variance_threshold:
            if compass.heading == CompassHeading.EAST:
                return RoutingTier.SMALL, "Focused local attention - syntactic pattern"
            else:
                return RoutingTier.SMALL, "Focused attention - retrieval pattern"

        # High variance = reasoning, large model
        if variance > self.config.high_variance_threshold:
            if pattern == "scattered":
                return (
                    RoutingTier.LARGE,
                    "Scattered attention - complex reasoning needed",
                )
            else:
                return RoutingTier.LARGE, "High variance - multi-hop reasoning"

        # Medium variance
        if pattern == "bimodal":
            return RoutingTier.MEDIUM, "Bimodal attention - bridging concepts"

        # Check directional balance
        if compass.north_mass > 0.4:
            return RoutingTier.MEDIUM, "Long-range retrieval - moderate complexity"

        return RoutingTier.MEDIUM, "Balanced attention - standard processing"

    def _should_use_cot(
        self,
        compass: CompassReading,
        tier: RoutingTier,
    ) -> bool:
        """Determine if chain-of-thought should be enabled."""
        # Always use CoT for large tier
        if tier == RoutingTier.LARGE:
            return True

        # Use CoT for high variance
        if compass.angular_variance > self.config.cot_variance_threshold:
            return True

        # Use CoT for sink-dominated (uncertainty)
        if compass.anchor.sink_total > self.config.cot_sink_threshold:
            return True

        # Use CoT for bimodal (bridging)
        if compass.pattern_type == "bimodal":
            return True

        return False

    def _compute_confidence(
        self,
        compass: CompassReading,
        tier: RoutingTier,
    ) -> float:
        """Compute confidence in routing decision."""
        # Base confidence from angular concentration
        confidence = compass.angular_concentration

        # Adjust for pattern clarity
        if compass.pattern_type == "focused":
            confidence = min(1.0, confidence + 0.2)
        elif compass.pattern_type == "scattered":
            confidence = max(0.3, confidence - 0.2)
        elif compass.pattern_type == "sink_dominated":
            confidence = 0.5  # Uncertain

        # Tier-specific adjustments
        if tier == RoutingTier.SMALL and compass.angular_variance < 0.2:
            confidence = min(1.0, confidence + 0.1)
        elif tier == RoutingTier.LARGE and compass.angular_variance > 0.8:
            confidence = min(1.0, confidence + 0.1)

        return float(np.clip(confidence, 0.3, 0.95))

    def _get_recommendations(
        self,
        tier: RoutingTier,
        compass: CompassReading,
    ) -> Tuple[str, float, int]:
        """Get model and generation recommendations."""
        if tier == RoutingTier.SMALL:
            model = self.config.small_model
            temperature = self.config.syntactic_temperature
            max_tokens = self.config.syntactic_max_tokens
        elif tier == RoutingTier.MEDIUM:
            model = self.config.medium_model
            temperature = self.config.reasoning_temperature
            max_tokens = self.config.reasoning_max_tokens
        else:  # LARGE or CONTEXT
            model = self.config.large_model
            temperature = self.config.creative_temperature
            max_tokens = self.config.creative_max_tokens

        # Adjust temperature based on pattern
        if compass.pattern_type == "focused":
            temperature = max(0.1, temperature - 0.2)
        elif compass.pattern_type == "scattered":
            temperature = min(1.0, temperature + 0.1)

        return model, temperature, max_tokens

    def get_statistics(self) -> Dict[str, Any]:
        """Get routing statistics."""
        if self._total_routes == 0:
            return {"total_routes": 0, "distribution": {}}

        return {
            "total_routes": self._total_routes,
            "distribution": {
                tier.value: {
                    "count": count,
                    "percentage": count / self._total_routes * 100,
                }
                for tier, count in self._route_counts.items()
            },
        }

    def reset_statistics(self):
        """Reset routing statistics."""
        self._route_counts = {tier: 0 for tier in RoutingTier}
        self._total_routes = 0


# =============================================================================
# EDUCATIONAL GLOSSARY ADDITIONS
# =============================================================================

COMPASS_GLOSSARY = {
    "sinq_anchor": {
        "term": "Sinq Anchor (Attention Sink Origin)",
        "simple": "Using the first few tokens as a 'true north' reference point",
        "detailed": "The attention sink (positions 0-4, typically <BOS> and system tokens) absorbs 'leftover' attention. Instead of filtering it out, we use it as a coordinate origin. All other attention is measured relative to this anchor, revealing the 'heading' of each thought.",
        "analogy": "Like using the North Star for navigation - it's always there and gives you a consistent reference point to measure direction.",
        "example": "If attention to sink is 30% and to 'hypothesis' at position 50 is 40%, the 'hypothesis' direction is measured relative to the sink reference.",
        "why_it_matters": "Provides stable, consistent measurements across different contexts and sequence lengths.",
    },
    "compass_heading": {
        "term": "Compass Heading",
        "simple": "The dominant direction of attention relative to the anchor",
        "detailed": "Maps attention positions to angular coordinates, then computes the weighted mean direction. Cardinal directions: NORTH (toward beginning/retrieval), EAST (toward recent/local), SOUTH (away from sink/creative), WEST (toward middle context).",
        "analogy": "Like a compass needle pointing in the direction the model is 'looking' most strongly.",
        "example": "Heading=NORTH with low variance means focused retrieval from early context. Heading=SCATTERED means attention is distributed everywhere.",
        "why_it_matters": "Different headings indicate different cognitive operations - retrieval vs. local processing vs. creative generation.",
    },
    "angular_variance": {
        "term": "Angular Variance",
        "simple": "How spread out the attention direction is (focused vs. scattered)",
        "detailed": "Uses circular statistics to measure the spread of attention around its mean direction. Low variance (< 0.3) means focused attention. High variance (> 0.7) means scattered attention in many directions.",
        "analogy": "Like measuring whether a flashlight beam is a tight spotlight (low variance) or a diffuse floodlight (high variance).",
        "example": "Variance=0.1: Attention tightly focused on one area. Variance=0.9: Attention scattered across the entire context.",
        "why_it_matters": "Low variance = syntactic/retrieval (use small model). High variance = reasoning (use large model).",
    },
    "routing_tier": {
        "term": "Routing Tier",
        "simple": "Which size model to use based on attention pattern",
        "detailed": "SMALL (4B-8B) for syntactic patterns with low variance. MEDIUM (13B-30B) for bridging concepts. LARGE (70B+) for complex reasoning with high variance or uncertainty.",
        "analogy": "Like choosing between a pocket calculator, scientific calculator, or supercomputer based on problem complexity.",
        "example": "Simple grammar completion → SMALL. Multi-step reasoning → LARGE. Connecting two concepts → MEDIUM.",
        "why_it_matters": "Routes simple queries to fast models and complex queries to capable models, optimizing cost and latency.",
    },
}


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_compass_router(
    config: Optional[CompassRouterConfig] = None,
) -> CompassRouter:
    """Create a compass router with default or custom config."""
    return CompassRouter(config)


def analyze_attention_compass(
    query_pos: int,
    key_positions: List[int],
    attention_scores: List[float],
    config: Optional[CompassRouterConfig] = None,
) -> CompassReading:
    """
    Quick analysis of attention pattern using compass.

    Convenience function for one-off analysis.
    """
    analyzer = CompassAnalyzer(config)
    return analyzer.analyze(query_pos, key_positions, attention_scores)


# =============================================================================
# DEMO
# =============================================================================


def demo():
    """Demonstrate the compass router."""
    print("=" * 60)
    print("Compass Router Demo: Angular Analysis for Query Routing")
    print("=" * 60)

    router = CompassRouter()

    # Example 1: Local syntactic pattern
    print("\n--- Example 1: Local Syntax ---")
    print("Query at position 100, attending to nearby tokens 95-99")

    result1 = router.route(
        query_pos=100,
        key_positions=[95, 96, 97, 98, 99],
        attention_scores=[0.1, 0.15, 0.2, 0.25, 0.3],
    )

    print(f"Tier: {result1.tier.value}")
    print(f"Heading: {result1.compass.heading.value}")
    print(f"Angular Variance: {result1.compass.angular_variance:.2f}")
    print(f"Pattern: {result1.compass.pattern_type}")
    print(f"Reason: {result1.reason}")
    print(f"Recommended: {result1.recommended_model}")

    # Example 2: Long-range reasoning
    print("\n--- Example 2: Long-Range Reasoning ---")
    print("Query at position 500, attending across full context")

    result2 = router.route(
        query_pos=500,
        key_positions=[10, 50, 100, 200, 450, 499],
        attention_scores=[0.2, 0.15, 0.15, 0.2, 0.15, 0.15],
    )

    print(f"Tier: {result2.tier.value}")
    print(f"Heading: {result2.compass.heading.value}")
    print(f"Angular Variance: {result2.compass.angular_variance:.2f}")
    print(f"Pattern: {result2.compass.pattern_type}")
    print(f"Reason: {result2.reason}")
    print(f"Use CoT: {result2.use_chain_of_thought}")

    # Example 3: Sink-dominated (uncertain)
    print("\n--- Example 3: Sink-Dominated (Uncertain) ---")
    print("Query at position 100, mostly attending to sink tokens")

    result3 = router.route(
        query_pos=100,
        key_positions=[0, 1, 2, 3, 50, 99],
        attention_scores=[0.3, 0.2, 0.15, 0.1, 0.15, 0.1],
    )

    print(f"Tier: {result3.tier.value}")
    print(f"Sink Total: {result3.compass.anchor.sink_total:.2f}")
    print(f"Pattern: {result3.compass.pattern_type}")
    print(f"Reason: {result3.reason}")
    print(f"Confidence: {result3.confidence:.2f}")

    # Statistics
    print("\n--- Routing Statistics ---")
    stats = router.get_statistics()
    print(f"Total routes: {stats['total_routes']}")
    for tier, data in stats["distribution"].items():
        print(f"  {tier}: {data['count']} ({data['percentage']:.0f}%)")


if __name__ == "__main__":
    demo()
