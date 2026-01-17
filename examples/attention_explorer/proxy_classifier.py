#!/usr/bin/env python3
"""
Centroid-Based Proxy Classifier for Online Attention Routing

A lightweight, O(k) classifier that runs in the scheduler hot path:
1. Receives fingerprint from GPU fingerprinter
2. Computes distance to cached cluster centroids
3. Returns manifold classification + sampling hints

Design goals:
- Microsecond latency (no network calls in hot path)
- Zero-copy from GPU fingerprinter
- Thread-safe centroid updates from sidecar

Integration with scheduler:
    classifier = ProxyClassifier()
    classifier.start_sync(sidecar_url="http://localhost:9000")

    # In decode loop:
    if fingerprint_ready:
        manifold, hints = classifier.classify(fingerprint_vector)
        if manifold == "structured":
            sampling_params.temperature = 0.2

Usage standalone:
    python proxy_classifier.py --sidecar http://localhost:9000 --test
"""

import argparse
import json
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class CachedCentroid:
    """Cached cluster centroid for fast lookup."""

    cluster_id: int
    centroid: np.ndarray
    traits: List[str]
    sampling_hint: Dict[str, float]


class ProxyClassifier:
    """
    Fast centroid-based classifier for online routing.

    Maintains a local cache of centroids synced from the RAPIDS sidecar.
    Classification is O(k) where k = number of clusters.
    """

    def __init__(
        self,
        max_distance: float = 2.0,
        sync_interval: float = 30.0,
        feature_dim: int = 20,
    ):
        """
        Initialize proxy classifier.

        Args:
            max_distance: Maximum distance for valid classification
            sync_interval: Seconds between centroid sync from sidecar
            feature_dim: Expected fingerprint dimension
        """
        self.max_distance = max_distance
        self.sync_interval = sync_interval
        self.feature_dim = feature_dim

        # Cached centroids (thread-safe via lock)
        self._centroids: Dict[int, CachedCentroid] = {}
        self._centroids_lock = threading.Lock()

        # Sync state
        self._sidecar_url: Optional[str] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_sync = 0

        # Stats
        self._classify_count = 0
        self._hit_count = 0
        self._miss_count = 0

    def start_sync(self, sidecar_url: str):
        """Start background sync with sidecar."""
        if not HAS_REQUESTS:
            print("Warning: requests not available, sync disabled")
            return

        self._sidecar_url = sidecar_url.rstrip("/")
        self._stop_event.clear()
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()

    def stop_sync(self):
        """Stop background sync."""
        self._stop_event.set()
        if self._sync_thread:
            self._sync_thread.join(timeout=5)

    def _sync_loop(self):
        """Background sync loop."""
        while not self._stop_event.is_set():
            try:
                self._sync_centroids()
            except Exception as e:
                print(f"Sync error: {e}")

            # Wait for next sync
            self._stop_event.wait(self.sync_interval)

    def _sync_centroids(self):
        """Fetch centroids from sidecar."""
        try:
            resp = requests.get(
                f"{self._sidecar_url}/centroids",
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()

            new_centroids = {}
            for cluster_id_str, centroid_data in data.items():
                cluster_id = int(cluster_id_str)
                new_centroids[cluster_id] = CachedCentroid(
                    cluster_id=cluster_id,
                    centroid=np.array(centroid_data["centroid"], dtype=np.float32),
                    traits=centroid_data.get("traits", []),
                    sampling_hint=centroid_data.get("sampling_hint", {}),
                )

            with self._centroids_lock:
                self._centroids = new_centroids

            self._last_sync = time.time()

        except Exception as e:
            pass  # Silently fail, keep using cached centroids

    def set_centroids(self, centroids: Dict[int, Dict]):
        """Manually set centroids (for testing or offline mode)."""
        new_centroids = {}
        for cluster_id, data in centroids.items():
            new_centroids[int(cluster_id)] = CachedCentroid(
                cluster_id=int(cluster_id),
                centroid=np.array(data["centroid"], dtype=np.float32),
                traits=data.get("traits", []),
                sampling_hint=data.get("sampling_hint", {}),
            )

        with self._centroids_lock:
            self._centroids = new_centroids

    def classify(
        self,
        fingerprint: np.ndarray,
    ) -> Tuple[Optional[int], float, List[str], Dict[str, float]]:
        """
        Classify fingerprint by nearest centroid.

        Args:
            fingerprint: Feature vector from GPU fingerprinter

        Returns:
            (cluster_id, distance, traits, sampling_hint)
            cluster_id is None if no centroids or distance > max_distance
        """
        self._classify_count += 1

        with self._centroids_lock:
            if not self._centroids:
                self._miss_count += 1
                return None, float("inf"), [], {}

            centroids = list(self._centroids.values())

        # Ensure numpy array
        if HAS_TORCH and isinstance(fingerprint, torch.Tensor):
            fingerprint = fingerprint.cpu().numpy()
        fingerprint = np.asarray(fingerprint, dtype=np.float32)

        # Find nearest centroid (O(k))
        best_match: Optional[CachedCentroid] = None
        best_dist = float("inf")

        for centroid in centroids:
            dist = np.linalg.norm(fingerprint - centroid.centroid)
            if dist < best_dist:
                best_dist = dist
                best_match = centroid

        # Check distance threshold
        if best_dist > self.max_distance or best_match is None:
            self._miss_count += 1
            return None, best_dist, [], {}

        self._hit_count += 1
        return (
            best_match.cluster_id,
            best_dist,
            best_match.traits,
            best_match.sampling_hint,
        )

    def classify_torch(
        self,
        fingerprint: "torch.Tensor",
    ) -> Tuple[Optional[int], float, List[str], Dict[str, float]]:
        """
        Classify fingerprint (GPU tensor version).

        Optimized path that stays on GPU if possible.
        """
        if not HAS_TORCH:
            raise ImportError("torch not available")

        with self._centroids_lock:
            if not self._centroids:
                self._miss_count += 1
                return None, float("inf"), [], {}

            # Stack centroids into tensor
            centroid_list = list(self._centroids.values())
            centroids_np = np.stack([c.centroid for c in centroid_list])
            centroids_tensor = torch.from_numpy(centroids_np).to(fingerprint.device)

        # Compute distances on GPU
        dists = torch.norm(fingerprint.unsqueeze(0) - centroids_tensor, dim=1)
        min_idx = dists.argmin().item()
        min_dist = dists[min_idx].item()

        self._classify_count += 1

        if min_dist > self.max_distance:
            self._miss_count += 1
            return None, min_dist, [], {}

        self._hit_count += 1
        best = centroid_list[min_idx]
        return best.cluster_id, min_dist, best.traits, best.sampling_hint

    def get_stats(self) -> Dict:
        """Get classifier statistics."""
        hit_rate = (
            self._hit_count / self._classify_count if self._classify_count > 0 else 0
        )
        return {
            "n_centroids": len(self._centroids),
            "classify_count": self._classify_count,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "last_sync": self._last_sync,
            "sidecar_url": self._sidecar_url,
        }


class ManifoldRouter:
    """
    Higher-level router that applies manifold-based routing decisions.

    Wraps ProxyClassifier and applies sampling parameter modifications.
    """

    def __init__(self, classifier: ProxyClassifier):
        self.classifier = classifier

        # Manifold-specific parameter overrides
        self.manifold_params = {
            "syntax_floor": {
                "temperature": 0.2,
                "top_p": 0.95,
                "repetition_penalty": 1.0,
            },
            "semantic_bridge": {
                "temperature": 0.5,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            },
            "long_range": {
                "temperature": 0.6,
                "top_p": 0.9,
                "repetition_penalty": 1.05,
            },
            "diffuse": {
                "temperature": 0.8,
                "top_p": 0.85,
                "repetition_penalty": 1.0,
            },
        }

    def route(
        self,
        fingerprint: np.ndarray,
        base_params: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Route request based on fingerprint.

        Args:
            fingerprint: Attention fingerprint vector
            base_params: Base sampling parameters (overridden by manifold)

        Returns:
            Modified sampling parameters
        """
        params = dict(base_params) if base_params else {}

        cluster_id, distance, traits, hints = self.classifier.classify(fingerprint)

        if cluster_id is None:
            return params

        # Apply hints from sidecar
        params.update(hints)

        # Apply manifold-specific overrides
        for trait in traits:
            if trait in self.manifold_params:
                params.update(self.manifold_params[trait])
                break  # Use first matching manifold

        return params


def test_classifier():
    """Test classifier with synthetic data."""
    print("=" * 60)
    print("Proxy Classifier Test")
    print("=" * 60)

    classifier = ProxyClassifier(max_distance=2.0)

    # Set up test centroids
    classifier.set_centroids(
        {
            0: {
                "centroid": [0.8, 0.1, 0.1, 0.3] + [0.0] * 16,
                "traits": ["syntax_floor", "local_attention"],
                "sampling_hint": {"temperature": 0.2},
            },
            1: {
                "centroid": [0.2, 0.6, 0.2, 0.5] + [0.0] * 16,
                "traits": ["semantic_bridge", "retrieval_heavy"],
                "sampling_hint": {"temperature": 0.5},
            },
            2: {
                "centroid": [0.2, 0.2, 0.6, 0.6] + [0.0] * 16,
                "traits": ["long_range", "context_aware"],
                "sampling_hint": {"temperature": 0.6},
            },
        }
    )

    # Test classifications
    test_cases = [
        ("Syntax-like", [0.75, 0.15, 0.1, 0.35] + [0.0] * 16),
        ("Semantic-like", [0.25, 0.55, 0.2, 0.45] + [0.0] * 16),
        ("Long-range-like", [0.15, 0.25, 0.6, 0.55] + [0.0] * 16),
        ("Unknown (far)", [0.5, 0.5, 0.5, 0.9] + [0.5] * 16),
    ]

    print("\nClassification Results:")
    print("-" * 60)

    for name, vec in test_cases:
        vec_np = np.array(vec, dtype=np.float32)
        cluster_id, dist, traits, hints = classifier.classify(vec_np)

        if cluster_id is not None:
            print(f"{name}:")
            print(f"  Cluster: {cluster_id}, Distance: {dist:.3f}")
            print(f"  Traits: {traits}")
            print(f"  Hints: {hints}")
        else:
            print(f"{name}: No match (distance: {dist:.3f})")

    # Test router
    print("\nRouter Test:")
    print("-" * 60)
    router = ManifoldRouter(classifier)

    for name, vec in test_cases[:3]:
        vec_np = np.array(vec, dtype=np.float32)
        params = router.route(vec_np, {"temperature": 0.7})
        print(f"{name}: temperature={params.get('temperature', 'N/A')}")

    # Stats
    print("\nStats:")
    print(json.dumps(classifier.get_stats(), indent=2))

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Proxy classifier for attention routing"
    )
    parser.add_argument(
        "--sidecar", default="http://localhost:9000", help="RAPIDS sidecar URL"
    )
    parser.add_argument(
        "--sync-interval",
        type=float,
        default=30.0,
        help="Seconds between centroid sync",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run test with synthetic data"
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    if args.test:
        test_classifier()
        return

    # Create classifier with sync
    classifier = ProxyClassifier(sync_interval=args.sync_interval)
    classifier.start_sync(args.sidecar)

    if args.interactive:
        print(f"Connected to sidecar: {args.sidecar}")
        print("Enter fingerprint vector (comma-separated), or 'quit' to exit")

        while True:
            try:
                line = input("\n> ").strip()
            except EOFError:
                break

            if line.lower() in ("quit", "exit", "q"):
                break

            if line == "stats":
                print(json.dumps(classifier.get_stats(), indent=2))
                continue

            try:
                vec = np.array([float(x) for x in line.split(",")], dtype=np.float32)
                cluster_id, dist, traits, hints = classifier.classify(vec)

                if cluster_id is not None:
                    print(f"Cluster: {cluster_id}, Distance: {dist:.3f}")
                    print(f"Traits: {traits}")
                    print(f"Hints: {hints}")
                else:
                    print(f"No match (distance: {dist:.3f})")
            except Exception as e:
                print(f"Error: {e}")

        classifier.stop_sync()


if __name__ == "__main__":
    main()
