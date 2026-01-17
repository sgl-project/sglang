"""
Spectral Router

Routes queries based on spectral coherence with the model's internal
geometric memory. Based on the insight that models store facts as
geometric structures, not associative lookups.

Routing Logic:
- High spectral coherence: Query is "on the manifold" -> Small model, no CoT
- Low spectral coherence: Query is "in the void" -> Large model, enable CoT
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .spectral_discovery import (
    FrequencyBandAnalyzer,
    SpectralCoherence,
    SpectralDiscoveryConfig,
    SpectralManifoldDiscovery,
)

logger = logging.getLogger(__name__)


class ModelSize(Enum):
    """Available model sizes for routing."""

    SMALL = "small"  # 4B-8B parameters
    MEDIUM = "medium"  # 13B-30B parameters
    LARGE = "large"  # 70B+ parameters


@dataclass
class RoutingDecision:
    """Result of routing decision."""

    model_size: ModelSize
    use_chain_of_thought: bool
    reason: str
    confidence: float
    spectral_coherence: float
    dominant_modes: List[int]
    frequency_band_ratio: float
    estimated_complexity: str  # "trivial", "moderate", "complex"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_size": self.model_size.value,
            "use_chain_of_thought": self.use_chain_of_thought,
            "reason": self.reason,
            "confidence": self.confidence,
            "spectral_coherence": self.spectral_coherence,
            "dominant_modes": self.dominant_modes,
            "frequency_band_ratio": self.frequency_band_ratio,
            "estimated_complexity": self.estimated_complexity,
        }


@dataclass
class RouterConfig:
    """Configuration for the spectral router."""

    # Coherence thresholds
    high_coherence_threshold: float = 0.7
    low_coherence_threshold: float = 0.3

    # Frequency band thresholds
    high_band_ratio_threshold: float = 2.0  # Above = complex reasoning
    low_band_ratio_threshold: float = 0.5  # Below = local/grammar

    # Model mapping
    small_model: str = "Qwen/Qwen3-4B"
    medium_model: str = "Qwen/Qwen3-14B"
    large_model: str = "Qwen/Qwen3-72B"

    # CoT settings
    cot_coherence_threshold: float = 0.5  # Below = enable CoT
    cot_band_ratio_threshold: float = 1.5  # Above = enable CoT

    # Confidence calibration
    min_confidence: float = 0.3
    calibration_samples: int = 1000


class SpectralRouter:
    """
    Routes queries to appropriate models based on spectral coherence.

    The router analyzes how well a query's attention fingerprint
    projects onto the learned spectral skeleton of the model's
    geometric memory.
    """

    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        spectral_discovery: Optional[SpectralManifoldDiscovery] = None,
    ):
        self.config = config or RouterConfig()
        self.spectral_discovery = spectral_discovery
        self.frequency_analyzer = FrequencyBandAnalyzer()
        self._routing_history: List[RoutingDecision] = []
        self._calibrated = False

    def fit(self, fingerprints: np.ndarray) -> "SpectralRouter":
        """
        Fit the router on historical fingerprint data.

        Args:
            fingerprints: Training fingerprints of shape (n_samples, n_features)

        Returns:
            self for chaining
        """
        logger.info(f"Fitting spectral router on {len(fingerprints)} fingerprints")

        # Initialize spectral discovery if not provided
        if self.spectral_discovery is None:
            spectral_config = SpectralDiscoveryConfig(
                coherence_threshold_high=self.config.high_coherence_threshold,
                coherence_threshold_low=self.config.low_coherence_threshold,
            )
            self.spectral_discovery = SpectralManifoldDiscovery(spectral_config)

        # Fit spectral discovery
        self.spectral_discovery.fit(fingerprints)

        # Calibrate thresholds using the training data
        self._calibrate(fingerprints)

        logger.info("Spectral router fitted successfully")
        return self

    def _calibrate(self, fingerprints: np.ndarray) -> None:
        """
        Calibrate routing thresholds based on coherence distribution.
        """
        n_samples = min(self.config.calibration_samples, len(fingerprints))
        sample_indices = np.random.choice(len(fingerprints), n_samples, replace=False)

        coherence_scores = []
        for idx in sample_indices:
            coherence = self.spectral_discovery.compute_spectral_coherence(
                fingerprints[idx]
            )
            coherence_scores.append(coherence.coherence_score)

        coherence_scores = np.array(coherence_scores)

        # Set thresholds at percentiles
        self.config.high_coherence_threshold = float(
            np.percentile(coherence_scores, 75)
        )
        self.config.low_coherence_threshold = float(np.percentile(coherence_scores, 25))

        logger.info(
            f"Calibrated thresholds: high={self.config.high_coherence_threshold:.3f}, "
            f"low={self.config.low_coherence_threshold:.3f}"
        )
        self._calibrated = True

    def route(self, fingerprint: np.ndarray) -> RoutingDecision:
        """
        Make a routing decision for a single fingerprint.

        Args:
            fingerprint: Attention fingerprint vector

        Returns:
            RoutingDecision with model size, CoT setting, and reasoning
        """
        if self.spectral_discovery is None or not self.spectral_discovery._fitted:
            raise RuntimeError("Router must be fitted before routing")

        # Compute spectral coherence
        coherence = self.spectral_discovery.compute_spectral_coherence(fingerprint)

        # Analyze frequency bands
        band_analysis = self.frequency_analyzer.analyze_frequency_bands(fingerprint)
        band_ratio = band_analysis["band_ratio"]

        # Determine model size
        model_size, size_reason = self._determine_model_size(coherence, band_ratio)

        # Determine CoT usage
        use_cot, cot_reason = self._determine_cot(coherence, band_ratio)

        # Estimate complexity
        complexity = self._estimate_complexity(coherence, band_ratio)

        # Build reason string
        reason = f"{size_reason}; {cot_reason}"

        decision = RoutingDecision(
            model_size=model_size,
            use_chain_of_thought=use_cot,
            reason=reason,
            confidence=coherence.confidence,
            spectral_coherence=coherence.coherence_score,
            dominant_modes=coherence.dominant_modes,
            frequency_band_ratio=band_ratio,
            estimated_complexity=complexity,
        )

        self._routing_history.append(decision)
        return decision

    def _determine_model_size(
        self, coherence: SpectralCoherence, band_ratio: float
    ) -> Tuple[ModelSize, str]:
        """
        Determine optimal model size based on coherence and frequency bands.
        """
        score = coherence.coherence_score

        # High coherence = geometric lookup, use small model
        if score > self.config.high_coherence_threshold:
            if band_ratio < self.config.low_band_ratio_threshold:
                return (
                    ModelSize.SMALL,
                    "High coherence + low-band dominant: simple lookup",
                )
            else:
                return (
                    ModelSize.MEDIUM,
                    "High coherence but high-band activity: moderate",
                )

        # Low coherence = void computation, use large model
        elif score < self.config.low_coherence_threshold:
            return (
                ModelSize.LARGE,
                "Low coherence: query in spectral void, needs deep reasoning",
            )

        # Medium coherence = semantic reasoning
        else:
            if band_ratio > self.config.high_band_ratio_threshold:
                return (
                    ModelSize.LARGE,
                    "Medium coherence + high-band: complex reasoning",
                )
            else:
                return ModelSize.MEDIUM, "Medium coherence: balanced semantic reasoning"

    def _determine_cot(
        self, coherence: SpectralCoherence, band_ratio: float
    ) -> Tuple[bool, str]:
        """
        Determine whether to enable Chain-of-Thought.
        """
        # Low coherence = need explicit reasoning
        if coherence.coherence_score < self.config.cot_coherence_threshold:
            return (
                True,
                "CoT enabled: low spectral coherence requires explicit reasoning",
            )

        # High frequency band activity = complex computation
        if band_ratio > self.config.cot_band_ratio_threshold:
            return True, "CoT enabled: high-band activity indicates complex reasoning"

        # High coherence = geometric lookup, no CoT needed
        if coherence.coherence_score > self.config.high_coherence_threshold:
            return False, "CoT disabled: high coherence allows direct geometric lookup"

        # Default: medium cases use CoT for safety
        return True, "CoT enabled: medium coherence, safer with explicit reasoning"

    def _estimate_complexity(
        self, coherence: SpectralCoherence, band_ratio: float
    ) -> str:
        """
        Estimate query complexity level.
        """
        score = coherence.coherence_score

        if score > 0.8 and band_ratio < 0.5:
            return "trivial"
        elif score < 0.3 or band_ratio > 2.0:
            return "complex"
        else:
            return "moderate"

    def route_batch(self, fingerprints: np.ndarray) -> List[RoutingDecision]:
        """
        Route a batch of fingerprints.
        """
        return [self.route(fp) for fp in fingerprints]

    def get_model_for_decision(self, decision: RoutingDecision) -> str:
        """
        Get the model identifier for a routing decision.
        """
        if decision.model_size == ModelSize.SMALL:
            return self.config.small_model
        elif decision.model_size == ModelSize.MEDIUM:
            return self.config.medium_model
        else:
            return self.config.large_model

    def get_routing_stats(self) -> Dict[str, Any]:
        """
        Get statistics about routing decisions.
        """
        if not self._routing_history:
            return {}

        decisions = self._routing_history

        size_counts = {size: 0 for size in ModelSize}
        cot_count = 0
        coherence_scores = []
        complexity_counts = {"trivial": 0, "moderate": 0, "complex": 0}

        for d in decisions:
            size_counts[d.model_size] += 1
            if d.use_chain_of_thought:
                cot_count += 1
            coherence_scores.append(d.spectral_coherence)
            complexity_counts[d.estimated_complexity] += 1

        total = len(decisions)
        return {
            "total_decisions": total,
            "model_distribution": {
                size.value: count / total for size, count in size_counts.items()
            },
            "cot_rate": cot_count / total,
            "avg_coherence": float(np.mean(coherence_scores)),
            "coherence_std": float(np.std(coherence_scores)),
            "complexity_distribution": {
                k: v / total for k, v in complexity_counts.items()
            },
        }

    def save(self, path: str) -> None:
        """Save the router to disk."""
        import pickle

        # Create directory if needed
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        state = {"config": self.config, "calibrated": self._calibrated}

        # Save router state
        with open(path, "wb") as f:
            pickle.dump(state, f)

        # Save spectral discovery separately
        if self.spectral_discovery is not None:
            spectral_path = str(Path(path).with_suffix(".spectral.pkl"))
            self.spectral_discovery.save(spectral_path)

        logger.info(f"Saved spectral router to {path}")

    @classmethod
    def load(cls, path: str) -> "SpectralRouter":
        """Load a router from disk."""
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        router = cls(config=state["config"])
        router._calibrated = state["calibrated"]

        # Load spectral discovery
        spectral_path = str(Path(path).with_suffix(".spectral.pkl"))
        if Path(spectral_path).exists():
            router.spectral_discovery = SpectralManifoldDiscovery.load(spectral_path)

        logger.info(f"Loaded spectral router from {path}")
        return router


class AdaptiveSpectralRouter(SpectralRouter):
    """
    Adaptive router that learns from feedback to improve routing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._feedback_history: List[Dict[str, Any]] = []

    def record_feedback(
        self,
        fingerprint: np.ndarray,
        decision: RoutingDecision,
        actual_quality: float,  # 0-1, how good was the response
        actual_latency: float,  # seconds
        was_correct_size: bool,
    ) -> None:
        """
        Record feedback about a routing decision for future learning.
        """
        self._feedback_history.append(
            {
                "fingerprint": fingerprint,
                "decision": decision,
                "actual_quality": actual_quality,
                "actual_latency": actual_latency,
                "was_correct_size": was_correct_size,
            }
        )

    def adapt_thresholds(self) -> None:
        """
        Adapt routing thresholds based on accumulated feedback.
        """
        if len(self._feedback_history) < 100:
            logger.warning("Not enough feedback for adaptation (need 100+)")
            return

        # Analyze feedback to adjust thresholds
        correct = [f for f in self._feedback_history if f["was_correct_size"]]
        incorrect = [f for f in self._feedback_history if not f["was_correct_size"]]

        if correct:
            correct_coherence = np.mean(
                [f["decision"].spectral_coherence for f in correct]
            )
        else:
            correct_coherence = 0.5

        if incorrect:
            # Adjust thresholds away from incorrect decisions
            for f in incorrect:
                coherence = f["decision"].spectral_coherence
                decision = f["decision"]

                # If we went small but should have gone large
                if decision.model_size == ModelSize.SMALL:
                    self.config.high_coherence_threshold = min(
                        self.config.high_coherence_threshold + 0.05, 0.9
                    )
                # If we went large but should have gone small
                elif decision.model_size == ModelSize.LARGE:
                    self.config.low_coherence_threshold = max(
                        self.config.low_coherence_threshold - 0.05, 0.1
                    )

        logger.info(
            f"Adapted thresholds: high={self.config.high_coherence_threshold:.3f}, "
            f"low={self.config.low_coherence_threshold:.3f}"
        )


def create_router_from_fingerprints(
    fingerprints: np.ndarray, config: Optional[RouterConfig] = None
) -> SpectralRouter:
    """
    Convenience function to create and fit a router from fingerprints.
    """
    router = SpectralRouter(config=config)
    router.fit(fingerprints)
    return router
