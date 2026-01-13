#!/usr/bin/env python3
"""
Online Fingerprint Classifier

Real-time classification of attention fingerprints using artifacts from
the discovery job. Designed for use in the sidecar service.

Features:
- Fast approximate cluster assignment (HDBSCAN approximate_predict or centroid matching)
- Zone assignment (syntax_floor / semantic_bridge / structure_ripple)
- Caching and lazy loading of models
- Thread-safe for concurrent requests

Usage:
    from classifier import OnlineClassifier

    classifier = OnlineClassifier('./discovery_outputs')
    result = classifier.classify(fingerprint_vector)
    # result = {'cluster_id': 3, 'zone': 'semantic_bridge', 'confidence': 0.85, ...}

Dependencies:
    pip install numpy joblib hdbscan

Author: SGLang Attention Explorer
"""

import logging
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

# Optional: HDBSCAN for approximate_predict
try:
    import hdbscan

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

logger = logging.getLogger(__name__)

# =============================================================================
# CONSTANTS - Import from central schema module
# =============================================================================

from .fingerprint_schema import (
    FP_ENTROPY,
    FP_LOCAL_MASS,
    FP_LONG_MASS,
    FP_MID_MASS,
    FP_ROTATIONAL_VARIANCE,
)
from .fingerprint_schema import V1_DIM as FINGERPRINT_DIM
from .fingerprint_schema import ZONE_THRESHOLDS

# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ClassificationResult:
    """Result of fingerprint classification."""

    cluster_id: int
    cluster_label: str
    cluster_probability: float
    zone: str
    zone_confidence: float
    embedding: Optional[Tuple[float, float]] = None  # (x, y) if computed


@dataclass
class ClusterInfo:
    """Cached cluster information."""

    cluster_id: int
    label: str
    zone: str
    centroid_fingerprint: np.ndarray
    centroid_xy: Tuple[float, float]
    size: int


# =============================================================================
# ONLINE CLASSIFIER
# =============================================================================


class OnlineClassifier:
    """
    Online classifier for attention fingerprints.

    Uses artifacts from the discovery job to classify new fingerprints
    in real-time. Supports two classification modes:

    1. HDBSCAN approximate_predict (if clusterer has prediction_data)
       - More accurate but slower
       - Uses the full HDBSCAN model

    2. Nearest centroid (fallback)
       - Faster and more stable
       - Uses precomputed cluster centroids

    Thread-safe for concurrent use in sidecar service.
    """

    def __init__(
        self,
        discovery_dir: str,
        use_approximate_predict: bool = True,
        precompute_embeddings: bool = False,
    ):
        """
        Initialize classifier.

        Args:
            discovery_dir: Directory containing discovery artifacts
                          (should have 'latest' symlink or run directories)
            use_approximate_predict: If True and available, use HDBSCAN's
                                    approximate_predict for better accuracy
            precompute_embeddings: If True, compute UMAP embeddings for each
                                  fingerprint (slower but enables visualization)
        """
        self.discovery_dir = Path(discovery_dir)
        self.use_approximate_predict = use_approximate_predict and HAS_HDBSCAN
        self.precompute_embeddings = precompute_embeddings

        # Lazy-loaded models
        self._clusterer: Optional[Any] = None
        self._embedding_models: Optional[Dict[str, Any]] = None
        self._clusters: Optional[Dict[int, ClusterInfo]] = None
        self._centroids: Optional[np.ndarray] = None
        self._centroid_ids: Optional[List[int]] = None

        # Thread safety
        self._lock = threading.RLock()
        self._loaded = False

        # Current run info
        self._run_id: Optional[str] = None

    def _get_latest_dir(self) -> Path:
        """Get path to latest discovery run."""
        latest = self.discovery_dir / "latest"
        if latest.is_symlink() or latest.exists():
            return latest.resolve() if latest.is_symlink() else latest

        # Fallback: find most recent directory
        runs = sorted(
            [
                d
                for d in self.discovery_dir.iterdir()
                if d.is_dir() and d.name != "latest"
            ],
            reverse=True,
        )

        if not runs:
            raise FileNotFoundError(f"No discovery runs found in {self.discovery_dir}")

        return runs[0]

    def _load_models(self):
        """Load models and cluster info from discovery artifacts."""
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            latest_dir = self._get_latest_dir()
            logger.info(f"Loading classifier models from {latest_dir}")

            # Load clusterer
            clusterer_path = latest_dir / "clusterer.joblib"
            if clusterer_path.exists():
                self._clusterer = joblib.load(clusterer_path)
                logger.info("  Loaded HDBSCAN clusterer")

                # Check if prediction_data is available
                if hasattr(self._clusterer, "_prediction_data"):
                    logger.info(
                        "  Clusterer has prediction_data (approximate_predict available)"
                    )
                else:
                    logger.info(
                        "  Clusterer lacks prediction_data, using centroid fallback"
                    )
                    self.use_approximate_predict = False

            # Load embedding models (for precompute_embeddings mode)
            if self.precompute_embeddings:
                models_path = latest_dir / "embedding_models.joblib"
                if models_path.exists():
                    self._embedding_models = joblib.load(models_path)
                    logger.info("  Loaded embedding models (scaler, PCA, UMAP)")

            # Load cluster info
            self._load_clusters(latest_dir)

            # Read run_id from manifest
            manifest_path = latest_dir / "manifest.json"
            if manifest_path.exists():
                import json

                with open(manifest_path) as f:
                    manifest = json.load(f)
                    self._run_id = manifest.get("run_id")

            self._loaded = True
            logger.info(f"  Classifier ready (run_id={self._run_id})")

    def _load_clusters(self, latest_dir: Path):
        """Load cluster info from Parquet file."""
        import pyarrow.parquet as pq

        clusters_path = latest_dir / "clusters.parquet"
        if not clusters_path.exists():
            logger.warning(f"Clusters file not found: {clusters_path}")
            self._clusters = {}
            return

        table = pq.read_table(clusters_path)
        df = table.to_pandas()

        self._clusters = {}
        centroids = []
        centroid_ids = []

        for _, row in df.iterrows():
            cluster_id = int(row["cluster_id"])

            # Unpack centroid fingerprint
            centroid_fp = row.get("centroid_fingerprint")
            if isinstance(centroid_fp, bytes):
                centroid_fp = np.array(
                    struct.unpack(f"<{FINGERPRINT_DIM}f", centroid_fp), dtype=np.float32
                )
            elif centroid_fp is None:
                centroid_fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)

            self._clusters[cluster_id] = ClusterInfo(
                cluster_id=cluster_id,
                label=row.get("label", f"Cluster {cluster_id}"),
                zone=row.get("dominant_zone", "unknown"),
                centroid_fingerprint=centroid_fp,
                centroid_xy=(
                    float(row.get("centroid_x", 0)),
                    float(row.get("centroid_y", 0)),
                ),
                size=int(row.get("size", 0)),
            )

            if cluster_id >= 0:  # Exclude noise cluster
                centroids.append(centroid_fp)
                centroid_ids.append(cluster_id)

        if centroids:
            self._centroids = np.vstack(centroids)
            self._centroid_ids = centroid_ids

        logger.info(f"  Loaded {len(self._clusters)} clusters")

    def reload(self):
        """Force reload of models (e.g., after new discovery run)."""
        with self._lock:
            self._loaded = False
            self._clusterer = None
            self._embedding_models = None
            self._clusters = None
            self._centroids = None
            self._centroid_ids = None
            self._run_id = None

        self._load_models()

    def classify(self, fingerprint: np.ndarray) -> ClassificationResult:
        """
        Classify a single fingerprint.

        Args:
            fingerprint: 20-dimensional fingerprint vector

        Returns:
            ClassificationResult with cluster_id, zone, confidence, etc.
        """
        self._load_models()

        fingerprint = np.asarray(fingerprint, dtype=np.float32)
        if fingerprint.shape != (FINGERPRINT_DIM,):
            raise ValueError(
                f"Expected fingerprint of shape ({FINGERPRINT_DIM},), got {fingerprint.shape}"
            )

        # Assign zone based on fingerprint features
        zone, zone_confidence = self._assign_zone(fingerprint)

        # Assign cluster
        if self.use_approximate_predict and self._clusterer is not None:
            cluster_id, cluster_prob = self._classify_hdbscan(fingerprint)
        else:
            cluster_id, cluster_prob = self._classify_centroid(fingerprint)

        # Get cluster label
        cluster_info = self._clusters.get(cluster_id)
        cluster_label = cluster_info.label if cluster_info else f"Cluster {cluster_id}"

        # Optionally compute embedding
        embedding = None
        if self.precompute_embeddings and self._embedding_models:
            embedding = self._compute_embedding(fingerprint)

        return ClassificationResult(
            cluster_id=cluster_id,
            cluster_label=cluster_label,
            cluster_probability=cluster_prob,
            zone=zone,
            zone_confidence=zone_confidence,
            embedding=embedding,
        )

    def classify_batch(self, fingerprints: np.ndarray) -> List[ClassificationResult]:
        """
        Classify a batch of fingerprints.

        Args:
            fingerprints: Array of shape (N, 20)

        Returns:
            List of ClassificationResult objects
        """
        self._load_models()

        fingerprints = np.asarray(fingerprints, dtype=np.float32)
        if fingerprints.ndim == 1:
            fingerprints = fingerprints.reshape(1, -1)

        results = []
        for fp in fingerprints:
            results.append(self.classify(fp))

        return results

    def _assign_zone(self, fingerprint: np.ndarray) -> Tuple[str, float]:
        """
        Assign zone label based on fingerprint features.

        Uses rotational_variance when available (v2 fingerprints) to improve
        zone classification accuracy:

        - syntax_floor: High local mass + low entropy + LOW rotational variance
          (attention to nearby tokens for grammar/syntax)

        - structure_ripple: High long-range mass + HIGH rotational variance
          (attention to distant tokens for structural patterns)

        - semantic_bridge: Balanced attention, medium rotational variance
          (mixed-distance attention for coreference/retrieval)

        NOTE: Rotational variance measures attention DISTANCE, not semantic vs positional.
        See fingerprint_schema.RV_SEMANTICS_DOC for full explanation.
        """
        local_mass = fingerprint[FP_LOCAL_MASS]
        mid_mass = fingerprint[FP_MID_MASS]
        long_mass = fingerprint[FP_LONG_MASS]
        entropy = fingerprint[FP_ENTROPY]

        # Check for rotational_variance (v2 fingerprint)
        has_rotational_variance = len(fingerprint) > FP_ROTATIONAL_VARIANCE
        rotational_variance = (
            fingerprint[FP_ROTATIONAL_VARIANCE] if has_rotational_variance else None
        )

        total_mass = max(local_mass + mid_mass + long_mass, 1e-6)

        # Check syntax_floor (high local, low entropy, optionally low rotational variance)
        sf_thresh = ZONE_THRESHOLDS["syntax_floor"]
        if local_mass > sf_thresh["local_mass_min"]:
            if entropy < sf_thresh["entropy_max"]:
                # If we have rotational variance, verify it's low (local attention to nearby tokens)
                if rotational_variance is not None:
                    if rotational_variance <= sf_thresh.get(
                        "rotational_variance_max", 1.0
                    ):
                        # Strong confidence when RV confirms local/short-range attention
                        confidence = min(
                            1.0,
                            local_mass
                            * (1.0 - entropy / 4.0)
                            * (1.0 - rotational_variance),
                        )
                        return "syntax_floor", confidence
                    # High RV with local mass suggests mixed pattern - fall through
                    # Fall through to check other zones
                else:
                    # No rotational variance, use original heuristic
                    confidence = min(1.0, local_mass * (1.0 - entropy / 4.0))
                    return "syntax_floor", confidence

        # Check structure_ripple (high long-range, optionally high rotational variance)
        sr_thresh = ZONE_THRESHOLDS["structure_ripple"]
        if long_mass > sr_thresh["long_mass_min"]:
            # If we have rotational variance, verify it's high (long-range attention)
            if rotational_variance is not None:
                rv_min = sr_thresh.get("rotational_variance_min", 0.0)
                if rotational_variance >= rv_min:
                    # High confidence when RV confirms long-range attention pattern
                    confidence = min(1.0, (long_mass + rotational_variance) / 2)
                    return "structure_ripple", confidence
                # Low RV with long mass = unusual pattern, still classify but lower confidence
                # Still return structure_ripple but with lower confidence
                confidence = min(1.0, long_mass * 0.8)
                return "structure_ripple", confidence
            else:
                confidence = min(1.0, long_mass * 1.5)
                return "structure_ripple", confidence

        # Default: semantic_bridge
        # When rotational variance is available, use it to boost confidence
        if rotational_variance is not None:
            sb_range = ZONE_THRESHOLDS["semantic_bridge"].get(
                "rotational_variance_range", (0.0, 1.0)
            )
            if sb_range[0] <= rotational_variance <= sb_range[1]:
                # Rotational variance in expected range for semantic bridging
                confidence = min(1.0, mid_mass / total_mass + 0.1)
            else:
                confidence = mid_mass / total_mass * 0.9
        else:
            confidence = mid_mass / total_mass

        return "semantic_bridge", confidence

    def _classify_hdbscan(self, fingerprint: np.ndarray) -> Tuple[int, float]:
        """Classify using HDBSCAN approximate_predict."""
        try:
            # Need to transform fingerprint through embedding pipeline first
            if self._embedding_models:
                scaler = self._embedding_models.get("scaler")
                pca = self._embedding_models.get("pca")
                umap_model = self._embedding_models.get("umap")

                fp_scaled = scaler.transform(fingerprint.reshape(1, -1))
                fp_pca = pca.transform(fp_scaled)
                fp_umap = umap_model.transform(fp_pca)

                labels, strengths = hdbscan.approximate_predict(
                    self._clusterer, fp_umap
                )
                return int(labels[0]), float(strengths[0])
            else:
                # Fall back to centroid
                return self._classify_centroid(fingerprint)

        except Exception as e:
            logger.warning(
                f"HDBSCAN approximate_predict failed: {e}, using centroid fallback"
            )
            return self._classify_centroid(fingerprint)

    def _classify_centroid(self, fingerprint: np.ndarray) -> Tuple[int, float]:
        """Classify by nearest centroid (cosine similarity)."""
        if self._centroids is None or len(self._centroids) == 0:
            return -1, 0.0

        # Normalize for cosine similarity
        fp_norm = fingerprint / (np.linalg.norm(fingerprint) + 1e-8)
        centroids_norm = self._centroids / (
            np.linalg.norm(self._centroids, axis=1, keepdims=True) + 1e-8
        )

        # Cosine similarity
        similarities = centroids_norm @ fp_norm

        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]

        cluster_id = self._centroid_ids[best_idx]

        # Convert similarity to probability-like confidence
        # (shift from [-1, 1] to [0, 1] range)
        confidence = (best_similarity + 1.0) / 2.0

        return cluster_id, float(confidence)

    def _compute_embedding(
        self, fingerprint: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Compute 2D embedding for a fingerprint."""
        if not self._embedding_models:
            return None

        try:
            scaler = self._embedding_models.get("scaler")
            pca = self._embedding_models.get("pca")
            umap_model = self._embedding_models.get("umap")

            fp_scaled = scaler.transform(fingerprint.reshape(1, -1))
            fp_pca = pca.transform(fp_scaled)
            fp_umap = umap_model.transform(fp_pca)

            return (float(fp_umap[0, 0]), float(fp_umap[0, 1]))

        except Exception as e:
            logger.warning(f"Embedding computation failed: {e}")
            return None

    def get_cluster_info(self, cluster_id: int) -> Optional[ClusterInfo]:
        """Get info about a specific cluster."""
        self._load_models()
        return self._clusters.get(cluster_id)

    def get_all_clusters(self) -> Dict[int, ClusterInfo]:
        """Get info about all clusters."""
        self._load_models()
        return self._clusters.copy()

    @property
    def run_id(self) -> Optional[str]:
        """Get the current discovery run ID."""
        self._load_models()
        return self._run_id

    @property
    def cluster_count(self) -> int:
        """Get number of clusters (excluding noise)."""
        self._load_models()
        return len([c for c in self._clusters.values() if c.cluster_id >= 0])


# =============================================================================
# SIDECAR INTEGRATION
# =============================================================================


class SidecarClassifier:
    """
    Classifier wrapper for sidecar service integration.

    Provides:
    - Schema v1 compatible output format
    - Automatic model reloading on file changes
    - Metrics collection
    """

    def __init__(
        self,
        discovery_dir: str,
        reload_interval_seconds: int = 300,
    ):
        self.classifier = OnlineClassifier(
            discovery_dir,
            use_approximate_predict=True,
            precompute_embeddings=False,  # Don't compute embeddings in real-time
        )
        self.reload_interval = reload_interval_seconds
        self._last_reload = 0
        self._classification_count = 0

        # Try initial load
        try:
            self.classifier._load_models()
        except Exception as e:
            logger.warning(f"Initial model load failed: {e}")

    def classify(self, fingerprint: np.ndarray) -> Dict[str, Any]:
        """
        Classify fingerprint and return schema v1 compatible dict.

        Returns dict compatible with SemanticMemory.update_from_sidecar():
        {
            'manifold': {
                'zone': 'semantic_bridge',
                'confidence': 0.85,
                'cluster_id': 3,
                'cluster_label': 'Code Explanation',
            },
            'control': {
                # Optional control signals
            },
            'schema_version': 1,
        }
        """
        import time

        # Check if reload needed
        now = time.time()
        if now - self._last_reload > self.reload_interval:
            self._check_for_new_run()
            self._last_reload = now

        try:
            result = self.classifier.classify(fingerprint)
            self._classification_count += 1

            # Convert numpy types to Python native types for JSON serialization
            return {
                "manifold": {
                    "zone": result.zone,
                    "confidence": float(result.zone_confidence),
                    "cluster_id": int(result.cluster_id),
                    "cluster_label": result.cluster_label,
                    "cluster_probability": (
                        float(result.cluster_probability)
                        if result.cluster_probability is not None
                        else None
                    ),
                },
                "schema_version": 1,
                "run_id": self.classifier.run_id,
            }

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return {
                "manifold": {
                    "zone": "unknown",
                    "confidence": 0.0,
                    "cluster_id": -1,
                    "cluster_label": "Error",
                },
                "error": str(e),
                "schema_version": 1,
            }

    def _check_for_new_run(self):
        """Check if a new discovery run is available and reload if so."""
        try:
            old_run_id = self.classifier.run_id
            latest_dir = self.classifier._get_latest_dir()

            # Read manifest to check run_id
            manifest_path = latest_dir / "manifest.json"
            if manifest_path.exists():
                import json

                with open(manifest_path) as f:
                    manifest = json.load(f)
                    new_run_id = manifest.get("run_id")

                    if new_run_id != old_run_id:
                        logger.info(f"New discovery run detected: {new_run_id}")
                        self.classifier.reload()

        except Exception as e:
            logger.warning(f"Error checking for new run: {e}")

    @property
    def stats(self) -> Dict[str, Any]:
        """Get classifier statistics."""
        return {
            "classification_count": self._classification_count,
            "run_id": self.classifier.run_id,
            "cluster_count": self.classifier.cluster_count,
        }


# =============================================================================
# CLI FOR TESTING
# =============================================================================


def main():
    """Test classifier from command line."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Test online classifier")
    parser.add_argument(
        "--discovery-dir",
        "-d",
        required=True,
        help="Path to discovery outputs directory",
    )
    parser.add_argument(
        "--fingerprint", "-f", help="JSON array of 20 floats to classify"
    )
    parser.add_argument(
        "--random", "-r", action="store_true", help="Classify random fingerprints"
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=10,
        help="Number of random fingerprints to classify",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    classifier = OnlineClassifier(args.discovery_dir)

    if args.fingerprint:
        fp = np.array(json.loads(args.fingerprint), dtype=np.float32)
        result = classifier.classify(fp)
        print(f"Classification result:")
        print(f"  Cluster: {result.cluster_id} ({result.cluster_label})")
        print(f"  Zone: {result.zone} (confidence: {result.zone_confidence:.2f})")
        print(f"  Cluster probability: {result.cluster_probability:.2f}")

    elif args.random:
        print(f"Classifying {args.count} random fingerprints...")
        for i in range(args.count):
            fp = np.random.rand(FINGERPRINT_DIM).astype(np.float32)
            result = classifier.classify(fp)
            print(
                f"  [{i}] cluster={result.cluster_id}, zone={result.zone}, "
                f"conf={result.zone_confidence:.2f}"
            )

    else:
        print("Classifier loaded successfully!")
        print(f"  Run ID: {classifier.run_id}")
        print(f"  Clusters: {classifier.cluster_count}")
        print("\nCluster info:")
        for cluster_id, info in sorted(classifier.get_all_clusters().items()):
            if cluster_id >= 0:
                print(
                    f"  [{cluster_id}] {info.label} - {info.zone} (size: {info.size})"
                )


if __name__ == "__main__":
    main()
