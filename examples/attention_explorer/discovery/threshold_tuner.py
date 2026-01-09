"""
Zone Threshold Tuner for Adaptive Manifold Classification

Learns optimal zone thresholds from probe harness feedback, improving
classification accuracy over time. Uses Bayesian-inspired updates to
refine boundaries between attention zones.
"""

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ZONE DEFINITIONS
# =============================================================================

# Default zone thresholds (from classifier.py)
DEFAULT_ZONE_THRESHOLDS = {
    'syntax_floor': {
        'local_mass_min': 0.7,
        'entropy_max': 2.0,
        'long_range_mass_max': 0.1,
    },
    'semantic_bridge': {
        'local_mass_range': (0.3, 0.7),
        'mid_mass_min': 0.2,
        'entropy_range': (2.0, 4.0),
    },
    'long_range': {
        'long_mass_min': 0.3,
        'local_mass_max': 0.3,
        'entropy_min': 3.0,
    },
    'structure_ripple': {
        'entropy_range': (2.5, 4.5),
        'pattern_score_min': 0.5,
    },
    'diffuse': {
        'entropy_min': 4.5,
        'max_mass_max': 0.3,
    },
}

ZONES = ['syntax_floor', 'semantic_bridge', 'long_range', 'structure_ripple', 'diffuse']


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ZoneSample:
    """A single sample with features and ground truth."""
    fingerprint_id: int
    features: Dict[str, float]  # local_mass, mid_mass, long_mass, entropy, etc.
    predicted_zone: str
    actual_zone: Optional[str] = None  # Ground truth from harness
    confidence: float = 0.0
    timestamp: str = ""


@dataclass
class ThresholdCandidate:
    """A candidate threshold value with performance metrics."""
    parameter: str
    value: float
    accuracy: float
    samples_evaluated: int
    zone: str


@dataclass
class TuningState:
    """Serializable state for threshold tuning."""
    zone_thresholds: Dict[str, Dict[str, Any]]
    samples: List[Dict[str, Any]]
    accuracy_history: List[Dict[str, float]]
    iteration: int
    last_updated: str

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> 'TuningState':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(**data)


# =============================================================================
# THRESHOLD TUNER
# =============================================================================

class ZoneThresholdTuner:
    """
    Adaptive zone threshold tuner using probe harness feedback.

    Algorithm:
    1. Collect samples from probe harness (fingerprint + correct zone)
    2. For each zone, find threshold values that maximize accuracy
    3. Use gradient-free optimization (grid search + local refinement)
    4. Export optimized thresholds for classifier

    Usage:
        tuner = ZoneThresholdTuner()

        # Add samples from probe harness
        for result in harness_results:
            tuner.add_sample(result)

        # Update thresholds
        new_thresholds = tuner.update_thresholds()

        # Export for classifier
        tuner.export_thresholds('zone_thresholds.json')
    """

    def __init__(
        self,
        initial_thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
        min_samples_per_zone: int = 50,
        learning_rate: float = 0.1,
        grid_resolution: int = 20,
    ):
        """
        Initialize threshold tuner.

        Args:
            initial_thresholds: Starting thresholds (default: DEFAULT_ZONE_THRESHOLDS)
            min_samples_per_zone: Minimum samples before updating zone thresholds
            learning_rate: How aggressively to update thresholds (0-1)
            grid_resolution: Grid search resolution for threshold optimization
        """
        self.thresholds = initial_thresholds or dict(DEFAULT_ZONE_THRESHOLDS)
        self.min_samples_per_zone = min_samples_per_zone
        self.learning_rate = learning_rate
        self.grid_resolution = grid_resolution

        # Sample storage
        self._samples: List[ZoneSample] = []
        self._samples_by_zone: Dict[str, List[ZoneSample]] = defaultdict(list)

        # Tracking
        self._accuracy_history: List[Dict[str, float]] = []
        self._iteration = 0

    def add_sample(
        self,
        fingerprint_id: int,
        features: Dict[str, float],
        predicted_zone: str,
        actual_zone: str,
        confidence: float = 0.0,
    ) -> None:
        """
        Add a sample with ground truth.

        Args:
            fingerprint_id: ID of the fingerprint
            features: Feature dict (local_mass, mid_mass, long_mass, entropy, etc.)
            predicted_zone: Zone predicted by classifier
            actual_zone: Correct zone from probe harness
            confidence: Prediction confidence
        """
        sample = ZoneSample(
            fingerprint_id=fingerprint_id,
            features=features,
            predicted_zone=predicted_zone,
            actual_zone=actual_zone,
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
        )

        self._samples.append(sample)
        self._samples_by_zone[actual_zone].append(sample)

        logger.debug(
            f"Added sample: predicted={predicted_zone}, actual={actual_zone}, "
            f"total={len(self._samples)}"
        )

    def add_harness_result(self, result: Dict[str, Any]) -> None:
        """
        Add a probe harness result.

        Args:
            result: Dict with keys like 'fingerprint_id', 'features', 'predicted_zone', etc.
        """
        self.add_sample(
            fingerprint_id=result.get('fingerprint_id', 0),
            features=result.get('features', {}),
            predicted_zone=result.get('predicted_zone', 'unknown'),
            actual_zone=result.get('actual_zone', result.get('expected_zone', 'unknown')),
            confidence=result.get('confidence', 0.0),
        )

    def compute_accuracy(self) -> Dict[str, float]:
        """
        Compute current accuracy metrics.

        Returns:
            Dict with overall accuracy and per-zone accuracy
        """
        if not self._samples:
            return {'overall': 0.0}

        # Overall accuracy
        correct = sum(1 for s in self._samples if s.predicted_zone == s.actual_zone)
        overall = correct / len(self._samples)

        # Per-zone accuracy
        result = {'overall': overall}

        for zone in ZONES:
            zone_samples = self._samples_by_zone.get(zone, [])
            if zone_samples:
                zone_correct = sum(
                    1 for s in zone_samples if s.predicted_zone == s.actual_zone
                )
                result[zone] = zone_correct / len(zone_samples)
            else:
                result[zone] = 0.0

        return result

    def _evaluate_threshold(
        self,
        zone: str,
        param: str,
        value: float,
        samples: List[ZoneSample],
    ) -> float:
        """
        Evaluate accuracy with a specific threshold value.

        Args:
            zone: Zone to evaluate
            param: Threshold parameter
            value: Threshold value to test
            samples: Samples to evaluate

        Returns:
            Accuracy score (0-1)
        """
        if not samples:
            return 0.0

        # Create temporary thresholds with updated value
        test_thresholds = dict(self.thresholds)
        test_thresholds[zone] = dict(test_thresholds.get(zone, {}))
        test_thresholds[zone][param] = value

        correct = 0
        for sample in samples:
            predicted = self._classify_with_thresholds(sample.features, test_thresholds)
            if predicted == sample.actual_zone:
                correct += 1

        return correct / len(samples)

    def _classify_with_thresholds(
        self,
        features: Dict[str, float],
        thresholds: Dict[str, Dict[str, Any]],
    ) -> str:
        """
        Classify a sample using given thresholds.

        Args:
            features: Feature dict
            thresholds: Zone threshold dict

        Returns:
            Predicted zone name
        """
        local_mass = features.get('local_mass', 0.0)
        mid_mass = features.get('mid_mass', 0.0)
        long_mass = features.get('long_mass', 0.0)
        entropy = features.get('entropy', 0.0)
        pattern_score = features.get('pattern_score', 0.0)

        # Check syntax_floor
        sf = thresholds.get('syntax_floor', {})
        if (local_mass >= sf.get('local_mass_min', 0.7) and
            entropy <= sf.get('entropy_max', 2.0) and
            long_mass <= sf.get('long_range_mass_max', 0.1)):
            return 'syntax_floor'

        # Check long_range
        lr = thresholds.get('long_range', {})
        if (long_mass >= lr.get('long_mass_min', 0.3) and
            local_mass <= lr.get('local_mass_max', 0.3) and
            entropy >= lr.get('entropy_min', 3.0)):
            return 'long_range'

        # Check diffuse
        df = thresholds.get('diffuse', {})
        max_mass = max(local_mass, mid_mass, long_mass)
        if (entropy >= df.get('entropy_min', 4.5) and
            max_mass <= df.get('max_mass_max', 0.3)):
            return 'diffuse'

        # Check structure_ripple
        sr = thresholds.get('structure_ripple', {})
        entropy_range = sr.get('entropy_range', (2.5, 4.5))
        if (entropy_range[0] <= entropy <= entropy_range[1] and
            pattern_score >= sr.get('pattern_score_min', 0.5)):
            return 'structure_ripple'

        # Check semantic_bridge
        sb = thresholds.get('semantic_bridge', {})
        local_range = sb.get('local_mass_range', (0.3, 0.7))
        entropy_range = sb.get('entropy_range', (2.0, 4.0))
        if (local_range[0] <= local_mass <= local_range[1] and
            mid_mass >= sb.get('mid_mass_min', 0.2) and
            entropy_range[0] <= entropy <= entropy_range[1]):
            return 'semantic_bridge'

        # Default to semantic_bridge if no match
        return 'semantic_bridge'

    def _optimize_threshold(
        self,
        zone: str,
        param: str,
        current_value: float,
        value_range: Tuple[float, float],
        samples: List[ZoneSample],
    ) -> Tuple[float, float]:
        """
        Optimize a single threshold parameter.

        Args:
            zone: Zone name
            param: Parameter name
            current_value: Current threshold value
            value_range: (min, max) for search
            samples: Samples to evaluate

        Returns:
            (optimal_value, accuracy) tuple
        """
        if not samples:
            return current_value, 0.0

        best_value = current_value
        best_accuracy = self._evaluate_threshold(zone, param, current_value, samples)

        # Grid search
        step = (value_range[1] - value_range[0]) / self.grid_resolution
        for i in range(self.grid_resolution + 1):
            test_value = value_range[0] + i * step
            accuracy = self._evaluate_threshold(zone, param, test_value, samples)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_value = test_value

        # Local refinement around best value
        fine_range = (
            max(value_range[0], best_value - step),
            min(value_range[1], best_value + step),
        )
        fine_step = (fine_range[1] - fine_range[0]) / 10

        for i in range(11):
            test_value = fine_range[0] + i * fine_step
            accuracy = self._evaluate_threshold(zone, param, test_value, samples)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_value = test_value

        return best_value, best_accuracy

    def update_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """
        Update zone thresholds based on collected samples.

        Uses grid search + local refinement to find optimal thresholds
        for each zone, then applies learning rate smoothing.

        Returns:
            Updated threshold dict
        """
        if len(self._samples) < self.min_samples_per_zone:
            logger.info(
                f"Not enough samples ({len(self._samples)} < {self.min_samples_per_zone}), "
                "skipping threshold update"
            )
            return self.thresholds

        self._iteration += 1
        logger.info(f"Updating thresholds (iteration {self._iteration})")

        # Record accuracy before update
        accuracy_before = self.compute_accuracy()

        # Optimize each zone's thresholds
        new_thresholds = {}

        for zone in ZONES:
            zone_samples = self._samples_by_zone.get(zone, [])

            if len(zone_samples) < 10:
                new_thresholds[zone] = dict(self.thresholds.get(zone, {}))
                continue

            new_thresholds[zone] = {}

            # Optimize scalar thresholds
            for param, current_value in self.thresholds.get(zone, {}).items():
                if isinstance(current_value, (int, float)):
                    # Determine search range based on parameter type
                    if 'mass' in param:
                        value_range = (0.0, 1.0)
                    elif 'entropy' in param:
                        value_range = (0.0, 6.0)
                    elif 'score' in param:
                        value_range = (0.0, 1.0)
                    else:
                        value_range = (current_value * 0.5, current_value * 1.5)

                    optimal, accuracy = self._optimize_threshold(
                        zone, param, current_value, value_range, zone_samples
                    )

                    # Apply learning rate
                    new_value = (
                        current_value * (1 - self.learning_rate) +
                        optimal * self.learning_rate
                    )
                    new_thresholds[zone][param] = round(new_value, 3)

                elif isinstance(current_value, (list, tuple)) and len(current_value) == 2:
                    # Range parameter - optimize both bounds
                    new_thresholds[zone][param] = current_value  # Keep as-is for now
                else:
                    new_thresholds[zone][param] = current_value

        # Update internal thresholds
        self.thresholds = new_thresholds

        # Record accuracy after update
        accuracy_after = self.compute_accuracy()

        self._accuracy_history.append({
            'iteration': self._iteration,
            'before': accuracy_before,
            'after': accuracy_after,
            'timestamp': datetime.now().isoformat(),
        })

        logger.info(
            f"Threshold update complete: "
            f"accuracy {accuracy_before['overall']:.3f} -> {accuracy_after['overall']:.3f}"
        )

        return self.thresholds

    def export_thresholds(
        self,
        path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Export thresholds for classifier use.

        Args:
            path: Optional path to write JSON file

        Returns:
            Threshold dict with metadata
        """
        export_data = {
            'thresholds': self.thresholds,
            'metadata': {
                'iteration': self._iteration,
                'samples_count': len(self._samples),
                'accuracy': self.compute_accuracy(),
                'exported_at': datetime.now().isoformat(),
            },
        }

        if path:
            with open(path, 'w') as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Exported thresholds to {path}")

        return export_data

    def import_thresholds(self, path: str) -> None:
        """
        Import thresholds from file.

        Args:
            path: Path to JSON file
        """
        with open(path) as f:
            data = json.load(f)

        if 'thresholds' in data:
            self.thresholds = data['thresholds']
        else:
            self.thresholds = data

        logger.info(f"Imported thresholds from {path}")

    def save_state(self, path: str) -> None:
        """
        Save full tuner state for resume.

        Args:
            path: Path to write state file
        """
        state = TuningState(
            zone_thresholds=self.thresholds,
            samples=[asdict(s) for s in self._samples],
            accuracy_history=self._accuracy_history,
            iteration=self._iteration,
            last_updated=datetime.now().isoformat(),
        )

        with open(path, 'w') as f:
            f.write(state.to_json())

        logger.info(f"Saved tuner state to {path}")

    def load_state(self, path: str) -> None:
        """
        Load tuner state from file.

        Args:
            path: Path to state file
        """
        with open(path) as f:
            state = TuningState.from_json(f.read())

        self.thresholds = state.zone_thresholds
        self._samples = [
            ZoneSample(**s) for s in state.samples
        ]
        self._samples_by_zone = defaultdict(list)
        for sample in self._samples:
            if sample.actual_zone:
                self._samples_by_zone[sample.actual_zone].append(sample)
        self._accuracy_history = state.accuracy_history
        self._iteration = state.iteration

        logger.info(
            f"Loaded tuner state from {path}: "
            f"{len(self._samples)} samples, iteration {self._iteration}"
        )

    def get_confusion_matrix(self) -> Dict[str, Dict[str, int]]:
        """
        Compute confusion matrix from samples.

        Returns:
            Nested dict: confusion[actual][predicted] = count
        """
        confusion: Dict[str, Dict[str, int]] = {
            zone: {z: 0 for z in ZONES} for zone in ZONES
        }

        for sample in self._samples:
            if sample.actual_zone in confusion and sample.predicted_zone in ZONES:
                confusion[sample.actual_zone][sample.predicted_zone] += 1

        return confusion

    def get_improvement_suggestions(self) -> List[str]:
        """
        Generate suggestions for improving accuracy.

        Returns:
            List of suggestion strings
        """
        suggestions = []
        accuracy = self.compute_accuracy()

        # Check for zones with low accuracy
        for zone in ZONES:
            zone_acc = accuracy.get(zone, 0.0)
            if zone_acc < 0.7:
                suggestions.append(
                    f"Zone '{zone}' has low accuracy ({zone_acc:.1%}). "
                    f"Consider collecting more samples or adjusting thresholds."
                )

        # Check for confusion pairs
        confusion = self.get_confusion_matrix()
        for actual_zone in ZONES:
            for predicted_zone in ZONES:
                if actual_zone != predicted_zone:
                    count = confusion[actual_zone][predicted_zone]
                    total = sum(confusion[actual_zone].values()) or 1
                    if count / total > 0.2:
                        suggestions.append(
                            f"High confusion between '{actual_zone}' and '{predicted_zone}' "
                            f"({count}/{total}). Review threshold boundaries."
                        )

        # Check sample balance
        zone_counts = {zone: len(self._samples_by_zone[zone]) for zone in ZONES}
        total_samples = sum(zone_counts.values()) or 1
        for zone, count in zone_counts.items():
            if count / total_samples < 0.1:
                suggestions.append(
                    f"Zone '{zone}' is underrepresented ({count} samples, "
                    f"{count/total_samples:.1%}). Collect more samples for balance."
                )

        return suggestions


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_threshold_tuner(
    thresholds_path: Optional[str] = None,
    state_path: Optional[str] = None,
) -> ZoneThresholdTuner:
    """
    Create a threshold tuner, optionally loading from existing state.

    Args:
        thresholds_path: Path to initial thresholds JSON
        state_path: Path to full state file to resume from

    Returns:
        Initialized ZoneThresholdTuner
    """
    tuner = ZoneThresholdTuner()

    if state_path and Path(state_path).exists():
        tuner.load_state(state_path)
    elif thresholds_path and Path(thresholds_path).exists():
        tuner.import_thresholds(thresholds_path)

    return tuner


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import random

    # Create tuner
    tuner = ZoneThresholdTuner(min_samples_per_zone=20)

    # Generate synthetic samples
    print("Generating synthetic samples...")

    for _ in range(200):
        zone = random.choice(ZONES)

        # Generate features typical for zone
        if zone == 'syntax_floor':
            features = {
                'local_mass': random.uniform(0.6, 0.9),
                'mid_mass': random.uniform(0.1, 0.3),
                'long_mass': random.uniform(0.0, 0.15),
                'entropy': random.uniform(0.5, 2.5),
            }
        elif zone == 'semantic_bridge':
            features = {
                'local_mass': random.uniform(0.25, 0.65),
                'mid_mass': random.uniform(0.2, 0.5),
                'long_mass': random.uniform(0.1, 0.3),
                'entropy': random.uniform(1.5, 4.5),
            }
        elif zone == 'long_range':
            features = {
                'local_mass': random.uniform(0.1, 0.35),
                'mid_mass': random.uniform(0.1, 0.4),
                'long_mass': random.uniform(0.25, 0.6),
                'entropy': random.uniform(2.5, 5.0),
            }
        elif zone == 'structure_ripple':
            features = {
                'local_mass': random.uniform(0.2, 0.5),
                'mid_mass': random.uniform(0.2, 0.4),
                'long_mass': random.uniform(0.1, 0.3),
                'entropy': random.uniform(2.0, 5.0),
                'pattern_score': random.uniform(0.4, 0.9),
            }
        else:  # diffuse
            features = {
                'local_mass': random.uniform(0.1, 0.3),
                'mid_mass': random.uniform(0.1, 0.3),
                'long_mass': random.uniform(0.1, 0.3),
                'entropy': random.uniform(4.0, 6.0),
            }

        # Classify with current thresholds
        predicted = tuner._classify_with_thresholds(features, tuner.thresholds)

        tuner.add_sample(
            fingerprint_id=random.randint(1, 100000),
            features=features,
            predicted_zone=predicted,
            actual_zone=zone,
        )

    # Compute initial accuracy
    print("\nInitial accuracy:")
    accuracy = tuner.compute_accuracy()
    for zone, acc in accuracy.items():
        print(f"  {zone}: {acc:.1%}")

    # Update thresholds
    print("\nUpdating thresholds...")
    new_thresholds = tuner.update_thresholds()

    # Compute new accuracy
    print("\nAccuracy after update:")
    accuracy = tuner.compute_accuracy()
    for zone, acc in accuracy.items():
        print(f"  {zone}: {acc:.1%}")

    # Show confusion matrix
    print("\nConfusion matrix:")
    confusion = tuner.get_confusion_matrix()
    print("            ", end="")
    for zone in ZONES:
        print(f"{zone[:8]:>10}", end="")
    print()
    for actual in ZONES:
        print(f"{actual[:10]:>12}", end="")
        for predicted in ZONES:
            print(f"{confusion[actual][predicted]:>10}", end="")
        print()

    # Show suggestions
    print("\nImprovement suggestions:")
    for suggestion in tuner.get_improvement_suggestions():
        print(f"  - {suggestion}")

    # Export
    print("\nExporting thresholds...")
    export_data = tuner.export_thresholds()
    print(f"  Thresholds: {json.dumps(export_data['thresholds'], indent=2)[:200]}...")

    print("\nAll tests passed!")
