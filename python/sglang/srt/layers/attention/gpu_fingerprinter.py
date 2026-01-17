"""
GPU Fingerprinter for In-Kernel Attention Pattern Classification

Computes log-binned offset histograms directly on GPU to classify
attention manifolds (Syntax Floor, Semantic Bridge, Structure Ripple)
without exporting edge lists.

Reduces bandwidth from ~500KB (edge list) to ~64 bytes (16-float histogram).

Three Manifold Zones detected:
1. Syntax Floor (early/final layers): Local jitter, 95% mass in offset bins 0-2
2. Semantic Bridge (mid layers): Retrieval to anchors, high mass in bins 8-12
3. Structure Ripple (late-mid): Periodic patterns (4, 8, 12 token lookback)

Usage in model_runner:
    fingerprinter = GPUFingerprinter(n_bins=16)

    # During decode step:
    histogram = fingerprinter.process_step(
        topk_indices,  # [Batch, Layers, K]
        topk_weights,  # [Batch, Layers, K]
        current_pos,   # [Batch]
    )

    # After generation, classify:
    manifold = fingerprinter.classify_manifold(histogram)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

import torch


class ManifoldZone(Enum):
    """Attention manifold classification."""

    SYNTAX_FLOOR = "syntax_floor"  # Local jitter (early/final layers)
    SEMANTIC_BRIDGE = "semantic_bridge"  # Retrieval to anchors (mid layers)
    STRUCTURE_RIPPLE = "structure_ripple"  # Periodic patterns
    UNKNOWN = "unknown"


@dataclass
class FingerprintConfig:
    """Configuration for GPU fingerprinting."""

    n_bins: int = 16  # Log2 bins (covers 2^16 = 64K tokens)
    enable_early_exit: bool = True  # Stop fingerprinting after classification
    early_exit_steps: int = 256  # Steps to run before early exit
    classify_threshold: float = 0.7  # Confidence threshold for classification


class GPUFingerprinter:
    """
    GPU-accelerated attention fingerprinting via log-binned offset histograms.

    Computes a 16-dimensional vector representing the "cognitive state"
    of the model's attention pattern, suitable for fast clustering.
    """

    def __init__(
        self,
        n_bins: int = 16,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize GPU fingerprinter.

        Args:
            n_bins: Number of log2 bins (16 covers 64K tokens)
            device: Device for computation
            dtype: Data type for histograms
        """
        self.n_bins = n_bins
        self.device = device
        self.dtype = dtype

        # Pre-allocate bin indices for efficiency
        self._buckets = torch.arange(n_bins, device=device, dtype=torch.long)

        # Manifold classification thresholds (tuned empirically)
        # These represent the expected bin mass distribution for each manifold
        self._syntax_bins = torch.tensor(
            [0.4, 0.3, 0.15, 0.1, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            device=device,
            dtype=dtype,
        )  # 95% mass in bins 0-2 (offset < 4)

        self._semantic_bins = torch.tensor(
            [
                0.05,
                0.05,
                0.05,
                0.05,
                0.1,
                0.15,
                0.2,
                0.15,
                0.1,
                0.05,
                0.03,
                0.02,
                0,
                0,
                0,
                0,
            ],
            device=device,
            dtype=dtype,
        )  # High mass in bins 5-9 (offset 32-512)

        self._structure_bins = torch.tensor(
            [0.1, 0.05, 0.25, 0.05, 0.25, 0.05, 0.15, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0],
            device=device,
            dtype=dtype,
        )  # Comb-like pattern at bins 2, 4, 6 (periodic 4, 16, 64)

    def process_step(
        self,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        current_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Process a single decode step to update fingerprint histogram.

        Args:
            topk_indices: [Batch, Layers, K] or [Batch, K] - Global token IDs attended to
            topk_weights: [Batch, Layers, K] or [Batch, K] - Attention probabilities
            current_pos: [Batch] - Current decode position

        Returns:
            histogram: [Batch, n_bins] - Log-binned offset histogram for this step
        """
        batch_size = current_pos.shape[0]

        # Ensure 3D shape: [Batch, Layers, K]
        if topk_indices.dim() == 2:
            topk_indices = topk_indices.unsqueeze(1)
            topk_weights = topk_weights.unsqueeze(1)

        # 1. Compute offsets (how far back we're looking)
        # Shape: [Batch, Layers, K]
        dists = current_pos.view(-1, 1, 1) - topk_indices
        dists = torch.clamp(dists.float(), min=1)  # Minimum distance of 1

        # 2. Log2 binning - the manifold classifier
        # Bins: 0->1, 1->2-3, 2->4-7, 3->8-15, ... , 10->1024-2047, etc.
        log_dist = torch.floor(torch.log2(dists)).long()
        log_dist = torch.clamp(log_dist, min=0, max=self.n_bins - 1)

        # 3. Scatter-add to build histogram
        # Flatten across layers and K, then scatter
        flat_bins = log_dist.view(batch_size, -1)  # [Batch, Layers*K]
        flat_weights = topk_weights.view(batch_size, -1)  # [Batch, Layers*K]

        histogram = torch.zeros(
            (batch_size, self.n_bins), device=self.device, dtype=self.dtype
        )
        histogram.scatter_add_(1, flat_bins, flat_weights)

        return histogram

    def accumulate_histogram(
        self,
        running_histogram: torch.Tensor,
        step_histogram: torch.Tensor,
        step_count: int,
    ) -> torch.Tensor:
        """
        Accumulate step histogram into running total with exponential smoothing.

        Args:
            running_histogram: [Batch, n_bins] - Running histogram
            step_histogram: [Batch, n_bins] - This step's histogram
            step_count: Current step number (1-indexed)

        Returns:
            Updated running histogram
        """
        # Simple average for now; could use exponential smoothing
        alpha = 1.0 / step_count
        return running_histogram * (1 - alpha) + step_histogram * alpha

    def classify_manifold(
        self,
        histogram: torch.Tensor,
        threshold: float = 0.7,
    ) -> Tuple[ManifoldZone, float]:
        """
        Classify attention manifold from histogram.

        Args:
            histogram: [n_bins] or [Batch, n_bins] - Normalized histogram
            threshold: Minimum cosine similarity for classification

        Returns:
            (manifold_zone, confidence) tuple
        """
        if histogram.dim() == 1:
            histogram = histogram.unsqueeze(0)

        # Normalize histogram
        histogram = histogram / (histogram.sum(dim=-1, keepdim=True) + 1e-9)

        # Compute cosine similarity to reference patterns
        # Average across batch if present
        hist_mean = histogram.mean(dim=0)

        syntax_sim = torch.nn.functional.cosine_similarity(
            hist_mean.unsqueeze(0), self._syntax_bins.unsqueeze(0)
        ).item()

        semantic_sim = torch.nn.functional.cosine_similarity(
            hist_mean.unsqueeze(0), self._semantic_bins.unsqueeze(0)
        ).item()

        structure_sim = torch.nn.functional.cosine_similarity(
            hist_mean.unsqueeze(0), self._structure_bins.unsqueeze(0)
        ).item()

        # Find best match
        similarities = {
            ManifoldZone.SYNTAX_FLOOR: syntax_sim,
            ManifoldZone.SEMANTIC_BRIDGE: semantic_sim,
            ManifoldZone.STRUCTURE_RIPPLE: structure_sim,
        }

        best_zone = max(similarities, key=similarities.get)
        best_sim = similarities[best_zone]

        if best_sim < threshold:
            return ManifoldZone.UNKNOWN, best_sim

        return best_zone, best_sim

    def extract_features(self, histogram: torch.Tensor) -> Dict[str, float]:
        """
        Extract interpretable features from histogram.

        Args:
            histogram: [n_bins] or [Batch, n_bins]

        Returns:
            Dict with named features
        """
        if histogram.dim() == 1:
            histogram = histogram.unsqueeze(0)

        # Normalize
        hist = histogram / (histogram.sum(dim=-1, keepdim=True) + 1e-9)
        hist = hist.mean(dim=0)  # Average across batch

        # Feature extraction
        local_mass = hist[:3].sum().item()  # Bins 0-2: offset < 8
        mid_mass = hist[3:8].sum().item()  # Bins 3-7: offset 8-255
        long_mass = hist[8:].sum().item()  # Bins 8+: offset > 256

        # Entropy (concentration measure)
        entropy = -(hist * torch.log(hist + 1e-9)).sum().item()
        max_entropy = torch.log(torch.tensor(self.n_bins, dtype=torch.float32)).item()
        normalized_entropy = entropy / max_entropy

        # Peak detection for periodicity
        peaks = []
        for i in range(1, self.n_bins - 1):
            if hist[i] > hist[i - 1] and hist[i] > hist[i + 1]:
                peaks.append(i)

        # Periodicity score (how comb-like is the pattern?)
        periodicity = len(peaks) / (self.n_bins / 2) if peaks else 0

        return {
            "local_mass": local_mass,  # Syntax floor signal
            "mid_mass": mid_mass,  # Semantic bridge signal
            "long_mass": long_mass,  # Long-range retrieval
            "entropy": normalized_entropy,  # Concentration (low = focused)
            "periodicity": periodicity,  # Structure ripple signal
            "peak_bins": peaks,  # Which bins have peaks
        }

    def to_vector(self, histogram: torch.Tensor) -> torch.Tensor:
        """
        Convert histogram to feature vector for clustering.

        Returns 20D vector: [local_mass, mid_mass, long_mass, entropy, histogram...]
        """
        features = self.extract_features(histogram)

        if histogram.dim() == 1:
            hist = histogram
        else:
            hist = histogram.mean(dim=0)

        # Normalize histogram
        hist = hist / (hist.sum() + 1e-9)

        prefix = torch.tensor(
            [
                features["local_mass"],
                features["mid_mass"],
                features["long_mass"],
                features["entropy"],
            ],
            device=self.device,
            dtype=self.dtype,
        )

        return torch.cat([prefix, hist])


class StreamingFingerprinter:
    """
    Streaming fingerprinter for online manifold detection.

    Maintains running histogram and provides early-exit capability
    once manifold is classified with high confidence.
    """

    def __init__(
        self,
        config: FingerprintConfig,
        device: str = "cuda",
    ):
        self.config = config
        self.fingerprinter = GPUFingerprinter(
            n_bins=config.n_bins,
            device=device,
        )
        self.reset()

    def reset(self):
        """Reset state for new request."""
        self._running_histogram = None
        self._step_count = 0
        self._classified = False
        self._manifold = ManifoldZone.UNKNOWN
        self._confidence = 0.0

    def update(
        self,
        topk_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        current_pos: torch.Tensor,
    ) -> Optional[ManifoldZone]:
        """
        Update with new decode step.

        Returns manifold classification if confident, else None.
        """
        if self._classified and self.config.enable_early_exit:
            # Already classified, skip computation
            return self._manifold

        # Compute step histogram
        step_hist = self.fingerprinter.process_step(
            topk_indices, topk_weights, current_pos
        )

        self._step_count += 1

        # Update running histogram
        if self._running_histogram is None:
            self._running_histogram = step_hist
        else:
            self._running_histogram = self.fingerprinter.accumulate_histogram(
                self._running_histogram, step_hist, self._step_count
            )

        # Try classification after early_exit_steps
        if (
            self.config.enable_early_exit
            and self._step_count >= self.config.early_exit_steps
        ):

            manifold, confidence = self.fingerprinter.classify_manifold(
                self._running_histogram,
                threshold=self.config.classify_threshold,
            )

            if manifold != ManifoldZone.UNKNOWN:
                self._classified = True
                self._manifold = manifold
                self._confidence = confidence
                return manifold

        return None

    def finalize(self) -> Tuple[ManifoldZone, float, torch.Tensor]:
        """
        Finalize fingerprinting and return results.

        Returns:
            (manifold, confidence, histogram) tuple
        """
        if self._running_histogram is None:
            return ManifoldZone.UNKNOWN, 0.0, torch.zeros(self.config.n_bins)

        if not self._classified:
            manifold, confidence = self.fingerprinter.classify_manifold(
                self._running_histogram,
                threshold=self.config.classify_threshold,
            )
            self._manifold = manifold
            self._confidence = confidence

        return self._manifold, self._confidence, self._running_histogram.mean(dim=0)

    def get_features(self) -> Dict[str, float]:
        """Get interpretable features from current histogram."""
        if self._running_histogram is None:
            return {}
        return self.fingerprinter.extract_features(self._running_histogram)

    def get_vector(self) -> Optional[torch.Tensor]:
        """Get feature vector for clustering."""
        if self._running_histogram is None:
            return None
        return self.fingerprinter.to_vector(self._running_histogram)


# Convenience function for integration
def create_fingerprinter(
    n_bins: int = 16,
    enable_early_exit: bool = True,
    early_exit_steps: int = 256,
    device: str = "cuda",
) -> StreamingFingerprinter:
    """
    Create a streaming fingerprinter for request processing.

    Args:
        n_bins: Number of log2 histogram bins
        enable_early_exit: Stop fingerprinting after classification
        early_exit_steps: Steps before attempting early exit
        device: Computation device

    Returns:
        StreamingFingerprinter instance
    """
    config = FingerprintConfig(
        n_bins=n_bins,
        enable_early_exit=enable_early_exit,
        early_exit_steps=early_exit_steps,
    )
    return StreamingFingerprinter(config, device)
