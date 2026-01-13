#!/usr/bin/env python3
"""
HDBSCAN Clustering for Attention Fingerprints

Clusters attention patterns to discover cognitive "modes" in model behavior.
Uses fingerprint vectors (hubness, consensus, spectral, entropy + histogram).

Usage:
    # Cluster from saved fingerprints
    python attention_clustering.py --input fingerprints.jsonl --output clusters.json

    # Live clustering from server
    python attention_clustering.py --server http://localhost:8000 --prompts prompts.txt

    # Interactive mode
    python attention_clustering.py --interactive --server http://localhost:8000
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional imports with graceful fallback
try:
    import hdbscan

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    import umap

    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Local import
sys.path.insert(0, str(Path(__file__).parent))
from attention_fingerprint import AttentionFingerprint


class AttentionClusterer:
    """HDBSCAN-based clustering for attention fingerprints."""

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int = 3,
        cluster_selection_epsilon: float = 0.0,
        metric: str = "euclidean",
        use_umap: bool = True,
        umap_n_components: int = 3,
        umap_n_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        scale_features: bool = True,
    ):
        """
        Initialize the clusterer.

        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN
            min_samples: Minimum samples for core points
            cluster_selection_epsilon: Distance threshold for cluster merging
            metric: Distance metric for HDBSCAN
            use_umap: Whether to apply UMAP before clustering
            umap_n_components: UMAP output dimensions
            umap_n_neighbors: UMAP neighborhood size
            umap_min_dist: UMAP minimum distance
            scale_features: Whether to standardize features
        """
        if not HAS_HDBSCAN:
            raise ImportError("hdbscan not installed. Run: pip install hdbscan")

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.use_umap = use_umap and HAS_UMAP
        self.umap_n_components = umap_n_components
        self.umap_n_neighbors = umap_n_neighbors
        self.umap_min_dist = umap_min_dist
        self.scale_features = scale_features and HAS_SKLEARN

        self.scaler = None
        self.umap_reducer = None
        self.clusterer = None
        self.fingerprinter = AttentionFingerprint()

        # Fitted data
        self.fingerprints: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.labels: Optional[np.ndarray] = None
        self.probabilities: Optional[np.ndarray] = None
        self.embedding: Optional[np.ndarray] = None

    def add_fingerprint(
        self,
        attention_tokens: Any,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Add a fingerprint from attention tokens.

        Args:
            attention_tokens: Attention data (list or multi-layer dict)
            metadata: Optional metadata (prompt, response, etc.)

        Returns:
            The computed fingerprint dict
        """
        fp = self.fingerprinter.fingerprint(attention_tokens)
        vec = np.array(fp["full_vector"], dtype=np.float32)

        self.fingerprints.append(vec)
        self.metadata.append(metadata or {})

        return fp

    def add_vector(self, vector: np.ndarray, metadata: Optional[Dict] = None):
        """Add a pre-computed fingerprint vector."""
        self.fingerprints.append(np.array(vector, dtype=np.float32))
        self.metadata.append(metadata or {})

    def fit(self, min_samples_for_fit: int = 10) -> Dict:
        """
        Fit the clustering model on collected fingerprints.

        Args:
            min_samples_for_fit: Minimum samples required to fit

        Returns:
            Clustering results dict
        """
        if len(self.fingerprints) < min_samples_for_fit:
            return {
                "status": "insufficient_data",
                "n_samples": len(self.fingerprints),
                "min_required": min_samples_for_fit,
            }

        X = np.stack(self.fingerprints)

        # Scale features
        if self.scale_features:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X

        # UMAP dimensionality reduction
        if self.use_umap and len(X) >= self.umap_n_neighbors:
            self.umap_reducer = umap.UMAP(
                n_components=self.umap_n_components,
                n_neighbors=min(self.umap_n_neighbors, len(X) - 1),
                min_dist=self.umap_min_dist,
                metric=self.metric,
                random_state=42,
            )
            self.embedding = self.umap_reducer.fit_transform(X_scaled)
            X_cluster = self.embedding
        else:
            self.embedding = X_scaled[:, :3]  # First 3 dims for viz
            X_cluster = X_scaled

        # HDBSCAN clustering
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, min(self.min_cluster_size, len(X) // 2)),
            min_samples=max(1, min(self.min_samples, len(X) // 3)),
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric="euclidean",  # Use euclidean after UMAP
            cluster_selection_method="leaf",  # More granular clusters
            prediction_data=True,
        )
        self.labels = self.clusterer.fit_predict(X_cluster)
        self.probabilities = self.clusterer.probabilities_

        # Compute cluster statistics
        unique_labels = set(self.labels)
        n_clusters = len(unique_labels - {-1})
        n_noise = (self.labels == -1).sum()

        # Silhouette score (if more than 1 cluster)
        silhouette = None
        if HAS_SKLEARN and n_clusters > 1 and n_noise < len(self.labels):
            mask = self.labels != -1
            if mask.sum() > n_clusters:
                try:
                    silhouette = float(
                        silhouette_score(X_cluster[mask], self.labels[mask])
                    )
                except:
                    pass

        # Cluster centroids in fingerprint space
        cluster_stats = {}
        for label in unique_labels:
            if label == -1:
                continue
            mask = self.labels == label
            cluster_vecs = X[mask]
            centroid = cluster_vecs.mean(axis=0)

            # Interpret centroid
            cluster_stats[int(label)] = {
                "size": int(mask.sum()),
                "hubness": float(centroid[0]),
                "consensus": float(centroid[1]),
                "spectral": float(centroid[2]),
                "entropy": float(centroid[3]),
                "mean_probability": float(self.probabilities[mask].mean()),
            }

        return {
            "status": "fitted",
            "n_samples": len(X),
            "n_clusters": n_clusters,
            "n_noise": int(n_noise),
            "silhouette": silhouette,
            "cluster_stats": cluster_stats,
            "labels": self.labels.tolist(),
        }

    def predict(self, attention_tokens: Any) -> Dict:
        """
        Predict cluster for new attention tokens.

        Args:
            attention_tokens: New attention data

        Returns:
            Prediction dict with cluster label and probability
        """
        if self.clusterer is None:
            raise ValueError("Must call fit() before predict()")

        fp = self.fingerprinter.fingerprint(attention_tokens)
        vec = np.array(fp["full_vector"], dtype=np.float32).reshape(1, -1)

        if self.scaler is not None:
            vec = self.scaler.transform(vec)

        if self.umap_reducer is not None:
            vec = self.umap_reducer.transform(vec)

        labels, strengths = hdbscan.approximate_predict(self.clusterer, vec)

        return {
            "cluster": int(labels[0]),
            "probability": float(strengths[0]),
            "fingerprint": fp,
        }

    def get_cluster_summary(self) -> Dict:
        """Get human-readable summary of discovered clusters."""
        if self.labels is None:
            return {"error": "Not fitted yet"}

        summaries = {}
        unique_labels = set(self.labels) - {-1}

        for label in unique_labels:
            mask = self.labels == label
            indices = np.where(mask)[0]

            # Get cluster fingerprints
            cluster_fps = [self.fingerprints[i] for i in indices]
            mean_fp = np.mean(cluster_fps, axis=0)

            # Interpret the cluster
            hubness, consensus, spectral, entropy = mean_fp[:4]

            # Generate description based on fingerprint
            traits = []
            if hubness > 0.7:
                traits.append("anchor-focused")
            elif hubness < 0.4:
                traits.append("distributed-attention")

            if consensus > 0.6:
                traits.append("cross-layer-agreement")
            elif consensus < 0.3:
                traits.append("layer-divergent")

            if spectral > 0.6:
                traits.append("long-range")
            elif spectral < 0.4:
                traits.append("local-syntax")

            if entropy < 1.2:
                traits.append("concentrated")
            elif entropy > 1.6:
                traits.append("diffuse")

            summaries[int(label)] = {
                "size": int(mask.sum()),
                "traits": traits,
                "description": " + ".join(traits) if traits else "neutral",
                "centroid": {
                    "hubness": float(hubness),
                    "consensus": float(consensus),
                    "spectral": float(spectral),
                    "entropy": float(entropy),
                },
                "sample_indices": indices[:5].tolist(),  # First 5 samples
            }

        return summaries

    def save(self, path: str):
        """Save clusterer state to file."""
        import pickle

        state = {
            "fingerprints": [fp.tolist() for fp in self.fingerprints],
            "metadata": self.metadata,
            "labels": self.labels.tolist() if self.labels is not None else None,
            "probabilities": (
                self.probabilities.tolist() if self.probabilities is not None else None
            ),
            "embedding": (
                self.embedding.tolist() if self.embedding is not None else None
            ),
            "config": {
                "min_cluster_size": self.min_cluster_size,
                "min_samples": self.min_samples,
                "cluster_selection_epsilon": self.cluster_selection_epsilon,
                "metric": self.metric,
                "use_umap": self.use_umap,
                "umap_n_components": self.umap_n_components,
            },
        }

        with open(path, "w") as f:
            json.dump(state, f, indent=2)

        # Save sklearn/hdbscan models separately
        model_path = path.replace(".json", "_models.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "scaler": self.scaler,
                    "umap_reducer": self.umap_reducer,
                    "clusterer": self.clusterer,
                },
                f,
            )

    def load(self, path: str):
        """Load clusterer state from file."""
        import pickle

        with open(path) as f:
            state = json.load(f)

        self.fingerprints = [
            np.array(fp, dtype=np.float32) for fp in state["fingerprints"]
        ]
        self.metadata = state["metadata"]
        self.labels = np.array(state["labels"]) if state["labels"] else None
        self.probabilities = (
            np.array(state["probabilities"]) if state["probabilities"] else None
        )
        self.embedding = np.array(state["embedding"]) if state["embedding"] else None

        # Load models
        model_path = path.replace(".json", "_models.pkl")
        if Path(model_path).exists():
            with open(model_path, "rb") as f:
                models = pickle.load(f)
                self.scaler = models.get("scaler")
                self.umap_reducer = models.get("umap_reducer")
                self.clusterer = models.get("clusterer")

    def to_visualization_data(self) -> Dict:
        """Export data for visualization (e.g., Plotly, D3)."""
        if self.embedding is None:
            return {"error": "Not fitted yet"}

        points = []
        for i, (emb, label, prob) in enumerate(
            zip(self.embedding, self.labels, self.probabilities)
        ):
            fp = self.fingerprints[i]
            points.append(
                {
                    "x": float(emb[0]),
                    "y": float(emb[1]),
                    "z": float(emb[2]) if len(emb) > 2 else 0.0,
                    "cluster": int(label),
                    "probability": float(prob),
                    "hubness": float(fp[0]),
                    "consensus": float(fp[1]),
                    "spectral": float(fp[2]),
                    "entropy": float(fp[3]),
                    "metadata": self.metadata[i],
                }
            )

        return {
            "points": points,
            "n_clusters": len(set(self.labels) - {-1}),
            "cluster_summaries": self.get_cluster_summary(),
        }


def collect_fingerprints_from_server(
    server: str,
    prompts: List[str],
    max_tokens: int = 50,
    top_k_attention: int = 5,
) -> List[Tuple[Dict, Dict]]:
    """
    Collect fingerprints from server for multiple prompts.

    Returns list of (fingerprint, metadata) tuples.
    """
    import requests

    fp = AttentionFingerprint()
    results = []

    for i, prompt in enumerate(prompts):
        try:
            resp = requests.post(
                f"{server}/v1/completions",
                json={
                    "model": "default",
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "return_attention_tokens": True,
                    "top_k_attention": top_k_attention,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            attn = choice.get("attention_tokens", [])
            text = choice.get("text", "")

            fingerprint = fp.fingerprint(attn)
            metadata = {
                "prompt": prompt,
                "response": text,
                "index": i,
            }

            results.append((fingerprint, metadata))
            print(f"[{i+1}/{len(prompts)}] Collected fingerprint for: {prompt[:40]}...")

        except Exception as e:
            print(f"[{i+1}/{len(prompts)}] Error: {e}")

    return results


def demo_clustering():
    """Demo with synthetic data."""
    print("=" * 60)
    print("HDBSCAN Clustering Demo")
    print("=" * 60)

    if not HAS_HDBSCAN:
        print("ERROR: hdbscan not installed. Run: pip install hdbscan")
        return

    np.random.seed(42)

    # Create synthetic fingerprint clusters
    # Cluster 0: High hubness, high consensus (structured code)
    cluster0 = np.random.randn(20, 20) * 0.1
    cluster0[:, 0] += 0.8  # hubness
    cluster0[:, 1] += 0.7  # consensus
    cluster0[:, 2] += 0.5  # spectral
    cluster0[:, 3] += 1.0  # entropy

    # Cluster 1: Low hubness, low consensus (creative chat)
    cluster1 = np.random.randn(20, 20) * 0.1
    cluster1[:, 0] += 0.3  # hubness
    cluster1[:, 1] += 0.3  # consensus
    cluster1[:, 2] += 0.6  # spectral
    cluster1[:, 3] += 1.8  # entropy

    # Cluster 2: Medium everything (factual Q&A)
    cluster2 = np.random.randn(15, 20) * 0.1
    cluster2[:, 0] += 0.5  # hubness
    cluster2[:, 1] += 0.5  # consensus
    cluster2[:, 2] += 0.5  # spectral
    cluster2[:, 3] += 1.4  # entropy

    # Combine
    all_vectors = np.vstack([cluster0, cluster1, cluster2])
    true_labels = [0] * 20 + [1] * 20 + [2] * 15

    # Create clusterer
    clusterer = AttentionClusterer(
        min_cluster_size=5,
        min_samples=3,
        use_umap=HAS_UMAP,
    )

    # Add vectors with metadata
    for i, (vec, true_label) in enumerate(zip(all_vectors, true_labels)):
        clusterer.add_vector(vec, {"true_label": true_label, "index": i})

    # Fit
    print("\nFitting HDBSCAN...")
    results = clusterer.fit()

    print(f"\nResults:")
    print(f"  Samples: {results['n_samples']}")
    print(f"  Clusters found: {results['n_clusters']}")
    print(f"  Noise points: {results['n_noise']}")
    if results.get("silhouette"):
        print(f"  Silhouette score: {results['silhouette']:.3f}")

    print("\nCluster Statistics:")
    for label, stats in results.get("cluster_stats", {}).items():
        print(
            f"  Cluster {label}: size={stats['size']}, "
            f"hubness={stats['hubness']:.2f}, "
            f"consensus={stats['consensus']:.2f}"
        )

    print("\nCluster Summaries:")
    summaries = clusterer.get_cluster_summary()
    for label, summary in summaries.items():
        print(f"  Cluster {label}: {summary['description']} (n={summary['size']})")

    # Test prediction
    print("\nTesting prediction on new point (high hubness)...")
    new_vec = np.zeros(20)
    new_vec[0] = 0.85  # High hubness
    new_vec[1] = 0.75  # High consensus
    new_vec[3] = 1.0  # Low entropy

    # Create fake attention data for prediction
    fake_attn = [
        {
            "token_positions": [0, 1, 2],
            "attention_scores": [0.6, 0.3, 0.1],
            "layer_id": 23,
        }
    ]

    pred = clusterer.predict(fake_attn)
    print(
        f"  Predicted cluster: {pred['cluster']} (probability: {pred['probability']:.3f})"
    )

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="HDBSCAN clustering for attention fingerprints"
    )
    parser.add_argument("--input", "-i", help="JSONL file with fingerprints")
    parser.add_argument("--output", "-o", help="Output file for clustering results")
    parser.add_argument(
        "--server", "-s", default="http://localhost:8000", help="SGLang server URL"
    )
    parser.add_argument("--prompts", "-p", help="File with prompts (one per line)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument(
        "--demo", action="store_true", help="Run demo with synthetic data"
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=5, help="Minimum cluster size"
    )
    parser.add_argument(
        "--no-umap", action="store_true", help="Disable UMAP dimensionality reduction"
    )

    args = parser.parse_args()

    if args.demo:
        demo_clustering()
        return

    if not HAS_HDBSCAN:
        print("ERROR: hdbscan not installed. Run: pip install hdbscan")
        print("Optional: pip install umap-learn scikit-learn")
        sys.exit(1)

    clusterer = AttentionClusterer(
        min_cluster_size=args.min_cluster_size,
        use_umap=not args.no_umap,
    )

    # Load from file
    if args.input:
        print(f"Loading fingerprints from {args.input}...")
        with open(args.input) as f:
            for line in f:
                data = json.loads(line)
                vec = data.get("full_vector") or data.get("vector")
                if vec:
                    clusterer.add_vector(vec, data.get("metadata", {}))
        print(f"Loaded {len(clusterer.fingerprints)} fingerprints")

    # Collect from server
    elif args.prompts:
        print(f"Loading prompts from {args.prompts}...")
        with open(args.prompts) as f:
            prompts = [line.strip() for line in f if line.strip()]

        print(f"Collecting fingerprints from {args.server}...")
        results = collect_fingerprints_from_server(args.server, prompts)

        for fp, meta in results:
            clusterer.add_vector(fp["full_vector"], meta)

    # Interactive mode
    elif args.interactive:
        import requests

        print(
            "Interactive clustering mode. Type prompts to add, 'fit' to cluster, 'quit' to exit."
        )

        while True:
            try:
                cmd = input("\n> ").strip()
            except EOFError:
                break

            if cmd.lower() in ("quit", "exit", "q"):
                break
            elif cmd.lower() == "fit":
                if len(clusterer.fingerprints) < 10:
                    print(
                        f"Need at least 10 samples, have {len(clusterer.fingerprints)}"
                    )
                    continue
                results = clusterer.fit()
                print(json.dumps(results, indent=2))
                summaries = clusterer.get_cluster_summary()
                for label, s in summaries.items():
                    print(f"Cluster {label}: {s['description']}")
            elif cmd.lower() == "summary":
                print(json.dumps(clusterer.get_cluster_summary(), indent=2))
            elif cmd.lower().startswith("save "):
                path = cmd[5:].strip()
                clusterer.save(path)
                print(f"Saved to {path}")
            elif cmd:
                # Treat as prompt
                try:
                    resp = requests.post(
                        f"{args.server}/v1/completions",
                        json={
                            "model": "default",
                            "prompt": cmd,
                            "max_tokens": 50,
                            "return_attention_tokens": True,
                            "top_k_attention": 5,
                        },
                        timeout=30,
                    )
                    data = resp.json()
                    attn = data["choices"][0].get("attention_tokens", [])
                    text = data["choices"][0].get("text", "")

                    fp = clusterer.add_fingerprint(
                        attn, {"prompt": cmd, "response": text}
                    )
                    print(f"Response: {text[:100]}...")
                    print(
                        f"Fingerprint: H={fp['hubness']:.2f} C={fp['consensus']:.2f} "
                        f"S={fp['spectral']:.2f} E={fp['entropy']:.2f}"
                    )
                    print(f"Total samples: {len(clusterer.fingerprints)}")

                    # Predict if fitted
                    if clusterer.clusterer is not None:
                        pred = clusterer.predict(attn)
                        print(
                            f"Predicted cluster: {pred['cluster']} (p={pred['probability']:.2f})"
                        )
                except Exception as e:
                    print(f"Error: {e}")

        return

    else:
        print("Provide --input, --prompts, --interactive, or --demo")
        sys.exit(1)

    # Fit and output
    if len(clusterer.fingerprints) >= 10:
        print("\nFitting HDBSCAN...")
        results = clusterer.fit()
        print(
            f"Found {results['n_clusters']} clusters, {results['n_noise']} noise points"
        )

        summaries = clusterer.get_cluster_summary()
        print("\nCluster Summaries:")
        for label, s in summaries.items():
            print(f"  Cluster {label}: {s['description']} (n={s['size']})")

        if args.output:
            viz_data = clusterer.to_visualization_data()
            with open(args.output, "w") as f:
                json.dump(viz_data, f, indent=2)
            print(f"\nSaved visualization data to {args.output}")

            # Also save model
            model_path = args.output.replace(".json", "_model.json")
            clusterer.save(model_path)
            print(f"Saved model to {model_path}")
    else:
        print(
            f"Need at least 10 samples for clustering, have {len(clusterer.fingerprints)}"
        )


if __name__ == "__main__":
    main()
