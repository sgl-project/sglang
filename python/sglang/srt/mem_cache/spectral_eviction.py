"""
Spectral KV Cache Eviction Strategy

Implements spectral-based eviction that keeps geometrically important "skeleton"
tokens while evicting redundant tokens that lie on interpolated paths.

The key insight is that attention patterns form a low-rank geometric structure.
Tokens at the extremes of this structure (spectral skeleton) define the shape
of the context, while tokens in between can be reconstructed/interpolated.

Algorithm:
1. Collect attention fingerprints during generation
2. Compute spectral embedding of fingerprints
3. Identify skeleton tokens (extremes + cluster centroids)
4. Assign eviction priority based on spectral importance

Usage:
    # In server_args.py:
    --radix-eviction-policy spectral
    --spectral-retention-ratio 0.3
    --spectral-weight 0.7

Author: SGLang Team
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from sglang.srt.mem_cache.evict_policy import EvictionStrategy, LRUStrategy

if TYPE_CHECKING:
    from sglang.srt.mem_cache.radix_cache import TreeNode

logger = logging.getLogger(__name__)

# Try to import sklearn for spectral computation
try:
    from sklearn.cluster import KMeans
    from sklearn.manifold import SpectralEmbedding
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning(
        "sklearn not available. SpectralEvictionStrategy will fallback to LRU. "
        "Install with: pip install scikit-learn"
    )


# =============================================================================
# CONSTANTS
# =============================================================================

# Manifold zone importance scores (higher = more important to keep)
# Zones represent attention distance patterns (see fingerprint_schema.py for full docs):
#   - syntax_floor: local/short-range attention → less important (reconstructible)
#   - structure_ripple: long-range structural patterns → moderately important
#   - semantic_bridge: mid-range retrieval anchors → most important (bridging)
ZONE_IMPORTANCE = {
    "semantic_bridge": 0.95,    # Retrieval anchors - critical
    "long_range": 0.85,         # Long-range dependencies
    "structure_ripple": 0.70,   # Structural patterns
    "syntax_floor": 0.30,       # Local syntax - can be reconstructed
    "diffuse": 0.20,            # Uniform attention - least important
    "unknown": 0.50,            # Default
}

# Fingerprint feature indices
# Schema v1: 20 dimensions, Schema v2: 21 dimensions (adds rotational_variance at [20])
# See examples/attention_explorer/discovery/fingerprint_schema.py for full documentation
FP_LOCAL_MASS = 0
FP_MID_MASS = 1
FP_LONG_MASS = 2
FP_ENTROPY = 3
FP_HISTOGRAM_START = 4
FP_ROTATIONAL_VARIANCE = 20  # Schema v2 extension (if dim >= 21)


# =============================================================================
# SPECTRAL SKELETON COMPUTER
# =============================================================================

@dataclass
class SkeletonResult:
    """Result of skeleton computation."""
    skeleton_indices: List[int]     # Token indices to KEEP
    computation_time_ms: float      # Time to compute
    n_spectral_dims: int            # Spectral dimensions used
    n_clusters: int                 # K-means clusters used
    cache_key: str                  # For invalidation


class SpectralSkeletonComputer:
    """
    Identifies skeleton tokens that define the geometric structure of attention.

    The skeleton consists of:
    1. Tokens at extremes of spectral eigenmodes (landmarks)
    2. Tokens near K-means cluster centroids (representatives)

    Together, these define the "shape" of the attention manifold.
    """

    def __init__(
        self,
        n_components: int = 10,
        retention_ratio: float = 0.3,
        min_samples: int = 20,
        random_state: int = 42,
    ):
        """
        Args:
            n_components: Number of spectral dimensions for embedding
            retention_ratio: Fraction of tokens to keep (0.3 = 30%)
            min_samples: Minimum samples needed for spectral computation
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.retention_ratio = retention_ratio
        self.min_samples = min_samples
        self.random_state = random_state

        # Cache for computed skeletons
        self._skeleton_cache: Dict[str, SkeletonResult] = {}

    def compute_skeleton(
        self,
        fingerprints: np.ndarray,
        seq_len: int,
        cache_key: Optional[str] = None,
    ) -> SkeletonResult:
        """
        Compute skeleton token indices from fingerprints.

        Args:
            fingerprints: [n_tokens, feature_dim] attention fingerprints
            seq_len: Total sequence length (for retention calculation)
            cache_key: Optional key for caching result

        Returns:
            SkeletonResult with indices of tokens to keep
        """
        start_time = time.perf_counter()

        # Check cache
        if cache_key and cache_key in self._skeleton_cache:
            cached = self._skeleton_cache[cache_key]
            # Invalidate if sequence grew significantly (>20%)
            if len(fingerprints) <= len(cached.skeleton_indices) * 1.2:
                return cached

        n_tokens = len(fingerprints)
        n_keep = max(1, int(seq_len * self.retention_ratio))

        # If we have fewer tokens than we want to keep, keep all
        if n_tokens <= n_keep:
            result = SkeletonResult(
                skeleton_indices=list(range(n_tokens)),
                computation_time_ms=(time.perf_counter() - start_time) * 1000,
                n_spectral_dims=0,
                n_clusters=0,
                cache_key=cache_key or "",
            )
            if cache_key:
                self._skeleton_cache[cache_key] = result
            return result

        # Not enough samples for spectral embedding
        if n_tokens < self.min_samples:
            # Fallback: keep evenly spaced tokens + first/last
            indices = self._evenly_spaced_skeleton(n_tokens, n_keep)
            result = SkeletonResult(
                skeleton_indices=indices,
                computation_time_ms=(time.perf_counter() - start_time) * 1000,
                n_spectral_dims=0,
                n_clusters=0,
                cache_key=cache_key or "",
            )
            if cache_key:
                self._skeleton_cache[cache_key] = result
            return result

        # Full spectral computation
        skeleton_indices = self._compute_spectral_skeleton(fingerprints, n_keep)

        result = SkeletonResult(
            skeleton_indices=skeleton_indices,
            computation_time_ms=(time.perf_counter() - start_time) * 1000,
            n_spectral_dims=min(self.n_components, n_keep // 2, n_tokens - 1),
            n_clusters=max(1, n_keep - len(skeleton_indices)),
            cache_key=cache_key or "",
        )

        if cache_key:
            self._skeleton_cache[cache_key] = result

        logger.debug(
            f"Spectral skeleton computed: {len(skeleton_indices)} tokens from {n_tokens} "
            f"in {result.computation_time_ms:.1f}ms"
        )

        return result

    def _compute_spectral_skeleton(
        self,
        fingerprints: np.ndarray,
        n_keep: int,
    ) -> List[int]:
        """
        Full spectral skeleton computation using SpectralEmbedding + KMeans.
        """
        if not HAS_SKLEARN:
            return self._evenly_spaced_skeleton(len(fingerprints), n_keep)

        n_tokens = len(fingerprints)
        skeleton: Set[int] = set()

        try:
            # 1. Standardize features
            scaler = StandardScaler()
            scaled_fp = scaler.fit_transform(fingerprints)

            # 2. Spectral embedding
            n_comp = min(self.n_components, n_keep // 2, n_tokens - 1)
            if n_comp < 2:
                n_comp = 2

            embedding = SpectralEmbedding(
                n_components=n_comp,
                affinity='nearest_neighbors',
                n_neighbors=min(15, n_tokens - 1),
                random_state=self.random_state,
            )
            coords = embedding.fit_transform(scaled_fp)

            # 3. Find extremes of each spectral dimension (landmarks)
            for dim in range(coords.shape[1]):
                skeleton.add(int(np.argmin(coords[:, dim])))
                skeleton.add(int(np.argmax(coords[:, dim])))

            # 4. Fill remaining slots with K-means centroids
            remaining = n_keep - len(skeleton)
            if remaining > 0:
                n_clusters = min(remaining, n_tokens - len(skeleton))
                if n_clusters > 0:
                    kmeans = KMeans(
                        n_clusters=n_clusters,
                        random_state=self.random_state,
                        n_init=10,
                    )
                    kmeans.fit(scaled_fp)

                    # Find token closest to each centroid
                    for centroid in kmeans.cluster_centers_:
                        distances = np.linalg.norm(scaled_fp - centroid, axis=1)
                        closest = int(np.argmin(distances))
                        skeleton.add(closest)

        except Exception as e:
            logger.warning(f"Spectral computation failed: {e}. Using fallback.")
            return self._evenly_spaced_skeleton(n_tokens, n_keep)

        return sorted(skeleton)[:n_keep]

    def _evenly_spaced_skeleton(self, n_tokens: int, n_keep: int) -> List[int]:
        """Fallback: evenly spaced tokens."""
        if n_keep >= n_tokens:
            return list(range(n_tokens))

        # Always include first and last
        skeleton = {0, n_tokens - 1}

        # Add evenly spaced tokens
        step = n_tokens / (n_keep - 1) if n_keep > 1 else 1
        for i in range(n_keep - 2):
            idx = int((i + 1) * step)
            if idx < n_tokens:
                skeleton.add(idx)

        return sorted(skeleton)[:n_keep]

    def clear_cache(self):
        """Clear the skeleton cache."""
        self._skeleton_cache.clear()

    def invalidate_cache(self, cache_key: str):
        """Invalidate a specific cache entry."""
        self._skeleton_cache.pop(cache_key, None)


# =============================================================================
# SPECTRAL EVICTION STRATEGY
# =============================================================================

class SpectralEvictionStrategy(EvictionStrategy):
    """
    Eviction strategy based on spectral importance of tokens.

    Tokens are scored based on:
    1. Spectral coherence (how well they fit the learned manifold)
    2. Manifold zone (semantic_bridge > syntax_floor)
    3. Attention entropy (focused > diffuse)
    4. Recency (LRU as tiebreaker)

    The strategy combines spectral importance with LRU, weighted by
    `spectral_weight`. This allows graceful degradation when spectral
    data is unavailable.
    """

    def __init__(
        self,
        retention_ratio: float = 0.3,
        spectral_weight: float = 0.7,
        fallback_strategy: str = "lru",
    ):
        """
        Args:
            retention_ratio: Fraction of tokens to keep (0.3 = 30%)
            spectral_weight: Weight of spectral score vs LRU (0.7 = 70% spectral)
            fallback_strategy: Strategy when no spectral data ("lru", "lfu", "fifo")

        Raises:
            ImportError: If sklearn is not installed (required for spectral computation)
        """
        # Explicit startup error if sklearn is missing
        # User explicitly selected spectral eviction, so fail fast rather than silently degrade
        if not HAS_SKLEARN:
            raise ImportError(
                "SpectralEvictionStrategy requires scikit-learn but it is not installed. "
                "Install with: pip install scikit-learn\n"
                "Or use a different eviction policy: --radix-eviction-policy lru"
            )

        self.retention_ratio = retention_ratio
        self.spectral_weight = spectral_weight
        self.fallback = LRUStrategy()

        # Skeleton computer for identifying important tokens
        self.skeleton_computer = SpectralSkeletonComputer(
            retention_ratio=retention_ratio
        )

        # Track skeleton membership per sequence
        self._skeleton_sets: Dict[str, Set[int]] = {}

    def get_priority(self, node: "TreeNode") -> Union[float, Tuple[float, float]]:
        """
        Compute eviction priority for a node.

        Higher priority = keep longer, lower priority = evict first.

        Returns a tuple (spectral_score, lru_score) for proper ordering.
        """
        # Check if node has spectral metadata
        spectral_fp = getattr(node, 'spectral_fingerprint', None)
        manifold_zone = getattr(node, 'manifold_zone', None)
        spectral_coherence = getattr(node, 'spectral_coherence', None)

        if spectral_fp is None:
            # No spectral data - use pure LRU
            return self.fallback.get_priority(node)

        # Compute spectral importance score
        spectral_score = self._compute_spectral_importance(
            spectral_fp, manifold_zone, spectral_coherence
        )

        # LRU score for tiebreaking
        lru_score = node.last_access_time

        # Combine scores
        # Higher combined = more important = evicted later
        combined_score = (
            self.spectral_weight * spectral_score +
            (1 - self.spectral_weight) * self._normalize_time(lru_score)
        )

        return (combined_score, lru_score)

    def _compute_spectral_importance(
        self,
        fingerprint: np.ndarray,
        manifold_zone: Optional[str],
        coherence: Optional[float],
    ) -> float:
        """
        Compute spectral importance score from fingerprint and metadata.

        Returns score in [0, 1] where higher = more important.
        """
        scores = []

        # 1. Coherence score (how well token fits manifold)
        if coherence is not None:
            scores.append(coherence)
        else:
            scores.append(0.5)

        # 2. Zone importance
        zone = manifold_zone or "unknown"
        zone_score = ZONE_IMPORTANCE.get(zone, 0.5)
        scores.append(zone_score)

        # 3. Entropy-based score (low entropy = focused = important)
        if fingerprint is not None and len(fingerprint) > FP_ENTROPY:
            entropy = fingerprint[FP_ENTROPY]
            # Clamp entropy to [0, 1] and invert
            entropy_score = 1.0 - min(1.0, max(0.0, entropy))
            scores.append(entropy_score)
        else:
            scores.append(0.5)

        # 4. Long-range mass (high = important for retrieval)
        if fingerprint is not None and len(fingerprint) > FP_LONG_MASS:
            long_mass = fingerprint[FP_LONG_MASS]
            scores.append(min(1.0, max(0.0, long_mass)))
        else:
            scores.append(0.5)

        # Weighted average
        weights = [0.3, 0.35, 0.2, 0.15]
        importance = sum(s * w for s, w in zip(scores, weights))

        return importance

    def _normalize_time(self, timestamp: float) -> float:
        """Normalize timestamp to [0, 1] range for combining with spectral score."""
        # Use relative time from current time
        now = time.monotonic()
        age = now - timestamp
        # Decay: recent = 1.0, old = approaches 0
        # Half-life of ~60 seconds
        return 1.0 / (1.0 + age / 60.0)

    def register_skeleton(self, sequence_key: str, skeleton_indices: List[int]):
        """Register skeleton tokens for a sequence."""
        self._skeleton_sets[sequence_key] = set(skeleton_indices)

    def is_skeleton_token(self, sequence_key: str, token_idx: int) -> bool:
        """Check if a token is part of the skeleton."""
        skeleton = self._skeleton_sets.get(sequence_key)
        if skeleton is None:
            return False
        return token_idx in skeleton

    def clear_sequence(self, sequence_key: str):
        """Clear skeleton data for a sequence."""
        self._skeleton_sets.pop(sequence_key, None)
        self.skeleton_computer.invalidate_cache(sequence_key)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_sequence_skeleton(
    fingerprints: List[np.ndarray],
    retention_ratio: float = 0.3,
) -> List[int]:
    """
    Convenience function to compute skeleton for a sequence.

    Args:
        fingerprints: List of fingerprint arrays (one per token)
        retention_ratio: Fraction of tokens to keep

    Returns:
        List of token indices to keep
    """
    if not fingerprints:
        return []

    computer = SpectralSkeletonComputer(retention_ratio=retention_ratio)
    fp_array = np.array(fingerprints)
    result = computer.compute_skeleton(fp_array, len(fingerprints))
    return result.skeleton_indices


def score_token_importance(
    fingerprint: np.ndarray,
    manifold_zone: Optional[str] = None,
    coherence: Optional[float] = None,
) -> float:
    """
    Score a single token's importance for eviction decisions.

    Args:
        fingerprint: Token's attention fingerprint
        manifold_zone: Optional zone classification
        coherence: Optional coherence score

    Returns:
        Importance score in [0, 1]
    """
    strategy = SpectralEvictionStrategy()
    return strategy._compute_spectral_importance(fingerprint, manifold_zone, coherence)


# =============================================================================
# TESTING
# =============================================================================

def _test_spectral_eviction():
    """Quick test of spectral eviction components."""
    print("=" * 60)
    print("Testing SpectralSkeletonComputer")
    print("=" * 60)

    # Generate synthetic fingerprints
    np.random.seed(42)
    n_tokens = 100
    feature_dim = 20

    # Create fingerprints with cluster structure
    fingerprints = np.vstack([
        np.random.randn(30, feature_dim) + np.array([1, 0, 0] + [0] * 17),  # Cluster 1
        np.random.randn(40, feature_dim) + np.array([0, 1, 0] + [0] * 17),  # Cluster 2
        np.random.randn(30, feature_dim) + np.array([0, 0, 1] + [0] * 17),  # Cluster 3
    ])

    computer = SpectralSkeletonComputer(retention_ratio=0.3)
    result = computer.compute_skeleton(fingerprints, n_tokens)

    print(f"Input: {n_tokens} tokens")
    print(f"Output: {len(result.skeleton_indices)} skeleton tokens")
    print(f"Computation time: {result.computation_time_ms:.1f}ms")
    print(f"Spectral dims: {result.n_spectral_dims}")
    print(f"Skeleton indices: {result.skeleton_indices[:10]}...")

    # Verify skeleton covers all clusters
    cluster_1 = set(range(0, 30))
    cluster_2 = set(range(30, 70))
    cluster_3 = set(range(70, 100))
    skeleton_set = set(result.skeleton_indices)

    print(f"\nCluster coverage:")
    print(f"  Cluster 1 (0-29): {len(skeleton_set & cluster_1)} tokens")
    print(f"  Cluster 2 (30-69): {len(skeleton_set & cluster_2)} tokens")
    print(f"  Cluster 3 (70-99): {len(skeleton_set & cluster_3)} tokens")

    print("\n" + "=" * 60)
    print("Testing SpectralEvictionStrategy")
    print("=" * 60)

    # Create mock nodes
    class MockNode:
        def __init__(self, fp, zone, coherence, access_time):
            self.spectral_fingerprint = fp
            self.manifold_zone = zone
            self.spectral_coherence = coherence
            self.last_access_time = access_time

    nodes = [
        MockNode(fingerprints[0], "semantic_bridge", 0.9, time.monotonic()),
        MockNode(fingerprints[50], "syntax_floor", 0.3, time.monotonic() - 100),
        MockNode(fingerprints[80], "diffuse", 0.1, time.monotonic() - 50),
        MockNode(None, None, None, time.monotonic() - 200),  # No spectral data
    ]

    strategy = SpectralEvictionStrategy()

    print("Node priorities (higher = keep longer):")
    for i, node in enumerate(nodes):
        priority = strategy.get_priority(node)
        print(f"  Node {i}: zone={node.manifold_zone}, coherence={node.spectral_coherence}, priority={priority}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    _test_spectral_eviction()
