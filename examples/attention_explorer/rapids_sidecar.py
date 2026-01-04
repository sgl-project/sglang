#!/usr/bin/env python3
"""
RAPIDS cuML Sidecar for GPU-Accelerated Attention Clustering

A separate background process that:
1. Receives fingerprint vectors from SGLang scheduler (via ZMQ or HTTP)
2. Runs GPU-accelerated HDBSCAN clustering via cuML
3. Publishes cluster centroids back to scheduler for online routing

Architecture:
    SGLang Scheduler  --fingerprints-->  RAPIDS Sidecar  --centroids-->  Proxy Router
         (ZMQ PUSH)        (ZMQ PULL)           |
         |<----------- manifold hints ----------|

Requirements:
    pip install pyzmq  # For ZMQ streaming
    pip install cuml-cu12  # or cuml-cu11 for CUDA 11
    # OR use CPU fallback: pip install hdbscan scikit-learn

Usage:
    # Start sidecar with ZMQ listener (matches SGLang's --attention-sidecar-url)
    python rapids_sidecar.py --zmq-bind tcp://*:9001 --port 9000

    # SGLang server connects to sidecar via ZMQ
    python -m sglang.launch_server \\
        --model-path your-model \\
        --return-attention-tokens \\
        --attention-fingerprint-mode \\
        --attention-sidecar-url tcp://localhost:9001 \\
        --disable-cuda-graph

    # Query centroids via HTTP
    curl http://localhost:9000/centroids

Alternative HTTP-only mode (for debugging):
    python rapids_sidecar.py --port 9000  # No ZMQ

    # Send fingerprints via HTTP
    curl -X POST http://localhost:9000/fingerprint -d '{
        "request_id": "req-123",
        "vector": [0.7, 0.5, 0.6, 1.2, ...],
        "metadata": {"prompt_type": "code"}
    }'
"""

import argparse
import json
import time
import threading
from collections import deque
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum

# Try ZMQ for high-performance streaming
try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False

# Try RAPIDS cuML first, fallback to CPU
try:
    import cudf
    import cuml
    from cuml.cluster import HDBSCAN as cuHDBSCAN
    HAS_RAPIDS = True
except ImportError:
    HAS_RAPIDS = False

try:
    import hdbscan
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Simple HTTP server
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse


class ClusteringBackend(Enum):
    RAPIDS = "rapids"
    CPU = "cpu"
    ONLINE = "online"  # Lightweight online clustering (no scipy/sklearn needed)
    NONE = "none"


class OnlineMicroCluster:
    """
    Micro-cluster for online clustering (DenStream-style).

    Maintains running statistics that can be updated incrementally:
    - Centroid (weighted average of points)
    - Weight (decayed sum of points)
    - Radius (approximate cluster spread)
    """

    def __init__(self, point: np.ndarray, cluster_id: int, decay: float = 0.99):
        self.cluster_id = cluster_id
        self.decay = decay
        self.centroid = point.copy()
        self.weight = 1.0
        self.sq_sum = np.sum(point ** 2)
        self.n_points = 1
        self.last_update = time.time()

    def distance(self, point: np.ndarray) -> float:
        """Euclidean distance from point to centroid."""
        return float(np.linalg.norm(point - self.centroid))

    def add_point(self, point: np.ndarray):
        """Add point to cluster with decay on old points."""
        # Decay existing weight
        self.weight *= self.decay
        self.sq_sum *= self.decay

        # Add new point
        self.centroid = (self.centroid * self.weight + point) / (self.weight + 1)
        self.weight += 1
        self.sq_sum += np.sum(point ** 2)
        self.n_points += 1
        self.last_update = time.time()

    @property
    def radius(self) -> float:
        """Approximate cluster radius (std dev from centroid)."""
        if self.weight < 2:
            return 0.0
        variance = max(0, self.sq_sum / self.weight - np.sum(self.centroid ** 2))
        return float(np.sqrt(variance))

    def is_stale(self, max_age: float = 300) -> bool:
        """Check if cluster hasn't been updated recently."""
        return time.time() - self.last_update > max_age


@dataclass
class ClusterCentroid:
    """Cluster centroid with metadata."""
    cluster_id: int
    centroid: List[float]
    size: int
    traits: List[str]
    sampling_hint: Dict[str, float]  # Suggested sampling params


@dataclass
class FingerprintEntry:
    """Stored fingerprint with metadata."""
    request_id: str
    vector: np.ndarray
    timestamp: float
    metadata: Dict


class RAPIDSSidecar:
    """
    GPU-accelerated clustering sidecar.

    Maintains a buffer of fingerprints and periodically re-clusters
    to discover attention manifolds.

    Supports two modes for receiving fingerprints:
    - ZMQ PULL: High-performance streaming from SGLang scheduler
    - HTTP POST: For debugging and manual testing
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        min_cluster_size: int = 10,
        recluster_interval: float = 60.0,
        feature_dim: int = 20,
        zmq_bind: Optional[str] = None,
        online_mode: bool = False,
        online_threshold: float = 2.0,  # Distance threshold for new cluster
        online_max_clusters: int = 50,
    ):
        self.buffer_size = buffer_size
        self.min_cluster_size = min_cluster_size
        self.recluster_interval = recluster_interval
        self.feature_dim = feature_dim
        self.zmq_bind = zmq_bind
        self.online_mode = online_mode
        self.online_threshold = online_threshold
        self.online_max_clusters = online_max_clusters

        # Fingerprint buffer (ring buffer)
        self.fingerprints: deque = deque(maxlen=buffer_size)
        self.lock = threading.Lock()

        # Clustering state
        self.centroids: Dict[int, ClusterCentroid] = {}
        self.last_cluster_time = 0
        self.cluster_labels: Optional[np.ndarray] = None

        # Online clustering state
        self._micro_clusters: Dict[int, OnlineMicroCluster] = {}
        self._next_cluster_id = 0

        # ZMQ receiver state
        self._zmq_context = None
        self._zmq_socket = None
        self._zmq_thread = None
        self._zmq_received = 0

        # Backend selection
        if online_mode:
            self.backend = ClusteringBackend.ONLINE
            print("Using ONLINE clustering mode (real-time updates)")
        elif HAS_RAPIDS:
            self.backend = ClusteringBackend.RAPIDS
            print("Using RAPIDS cuML backend (GPU)")
        elif HAS_SKLEARN:
            self.backend = ClusteringBackend.CPU
            print("Using CPU backend (hdbscan + sklearn)")
        else:
            self.backend = ClusteringBackend.NONE
            print("WARNING: No clustering backend available!")

        # Background clustering thread
        self._stop_event = threading.Event()
        self._cluster_thread = threading.Thread(target=self._cluster_loop, daemon=True)

    def start(self):
        """Start background threads (clustering + optional ZMQ receiver)."""
        self._cluster_thread.start()

        # Start ZMQ receiver if configured
        if self.zmq_bind and HAS_ZMQ:
            self._start_zmq_receiver()

    def stop(self):
        """Stop all background threads."""
        self._stop_event.set()
        self._cluster_thread.join(timeout=5)

        # Stop ZMQ receiver
        if self._zmq_socket:
            self._zmq_socket.close()
        if self._zmq_context:
            self._zmq_context.term()
        if self._zmq_thread:
            self._zmq_thread.join(timeout=2)

    def _start_zmq_receiver(self):
        """Start ZMQ PULL socket to receive fingerprints from scheduler."""
        try:
            self._zmq_context = zmq.Context()
            self._zmq_socket = self._zmq_context.socket(zmq.PULL)
            self._zmq_socket.bind(self.zmq_bind)
            self._zmq_socket.setsockopt(zmq.RCVHWM, 10000)  # High water mark
            print(f"ZMQ receiver bound to {self.zmq_bind}")

            self._zmq_thread = threading.Thread(target=self._zmq_loop, daemon=True)
            self._zmq_thread.start()
        except Exception as e:
            print(f"Failed to start ZMQ receiver: {e}")
            self._zmq_socket = None

    def _zmq_loop(self):
        """Background loop to receive fingerprints via ZMQ."""
        poller = zmq.Poller()
        poller.register(self._zmq_socket, zmq.POLLIN)

        while not self._stop_event.is_set():
            try:
                # Poll with 100ms timeout so we can check stop_event
                socks = dict(poller.poll(100))

                if self._zmq_socket in socks:
                    message = self._zmq_socket.recv(flags=zmq.NOBLOCK)
                    data = json.loads(message.decode())

                    self.add_fingerprint(
                        request_id=data.get("request_id", "unknown"),
                        vector=data["vector"],
                        metadata=data.get("metadata"),
                    )
                    self._zmq_received += 1

            except zmq.Again:
                pass  # No message available
            except Exception as e:
                if not self._stop_event.is_set():
                    print(f"ZMQ receive error: {e}")

    def add_fingerprint(
        self,
        request_id: str,
        vector: List[float],
        metadata: Optional[Dict] = None,
    ):
        """Add a fingerprint to the buffer (and update online clusters if enabled)."""
        vec = np.array(vector, dtype=np.float32)
        entry = FingerprintEntry(
            request_id=request_id,
            vector=vec,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        with self.lock:
            self.fingerprints.append(entry)

            # Update online clusters in real-time
            if self.online_mode:
                self._update_online_clusters(vec)

    def _update_online_clusters(self, point: np.ndarray):
        """Update online micro-clusters with a new point (called with lock held)."""
        # Find nearest micro-cluster
        best_cluster = None
        best_dist = float('inf')

        for mc in self._micro_clusters.values():
            dist = mc.distance(point)
            if dist < best_dist:
                best_dist = dist
                best_cluster = mc

        # Decide: add to existing cluster or create new one
        if best_cluster is not None and best_dist < self.online_threshold:
            # Add to nearest cluster
            best_cluster.add_point(point)
        else:
            # Create new cluster
            if len(self._micro_clusters) < self.online_max_clusters:
                new_mc = OnlineMicroCluster(point, self._next_cluster_id)
                self._micro_clusters[self._next_cluster_id] = new_mc
                self._next_cluster_id += 1
            else:
                # Too many clusters - merge with nearest
                if best_cluster is not None:
                    best_cluster.add_point(point)

        # Prune stale clusters periodically
        if len(self._micro_clusters) > 0:
            stale_ids = [cid for cid, mc in self._micro_clusters.items() if mc.is_stale()]
            for cid in stale_ids:
                del self._micro_clusters[cid]

        # Update centroids from micro-clusters
        self._sync_centroids_from_microclusters()

    def _sync_centroids_from_microclusters(self):
        """Convert micro-clusters to ClusterCentroid objects."""
        new_centroids = {}
        for cid, mc in self._micro_clusters.items():
            traits = self._interpret_centroid(mc.centroid)
            sampling_hint = self._get_sampling_hint(traits)
            new_centroids[cid] = ClusterCentroid(
                cluster_id=cid,
                centroid=mc.centroid.tolist(),
                size=mc.n_points,
                traits=traits,
                sampling_hint=sampling_hint,
            )
        self.centroids = new_centroids

    def get_centroids(self) -> Dict[int, Dict]:
        """Get current cluster centroids."""
        return {k: asdict(v) for k, v in self.centroids.items()}

    def predict_cluster(self, vector: List[float]) -> Tuple[int, float]:
        """
        Predict cluster for new fingerprint using nearest centroid.

        Returns (cluster_id, distance). Returns (-1, inf) if no centroids.
        """
        if not self.centroids:
            return -1, float('inf')

        vec = np.array(vector, dtype=np.float32)
        best_cluster = -1
        best_dist = float('inf')

        for cluster_id, centroid in self.centroids.items():
            cent = np.array(centroid.centroid, dtype=np.float32)
            dist = np.linalg.norm(vec - cent)
            if dist < best_dist:
                best_dist = dist
                best_cluster = cluster_id

        return best_cluster, best_dist

    def _cluster_loop(self):
        """Background clustering loop."""
        while not self._stop_event.is_set():
            try:
                time.sleep(1)  # Check every second

                # Check if it's time to recluster
                if time.time() - self.last_cluster_time < self.recluster_interval:
                    continue

                # Need minimum samples
                with self.lock:
                    n_samples = len(self.fingerprints)
                    if n_samples < self.min_cluster_size * 2:
                        continue

                    # Copy data for clustering
                    vectors = np.stack([fp.vector for fp in self.fingerprints])

                self._run_clustering(vectors)
                self.last_cluster_time = time.time()

            except Exception as e:
                print(f"Clustering error: {e}")

    def _run_clustering(self, vectors: np.ndarray):
        """Run HDBSCAN clustering on fingerprint vectors."""
        print(f"Clustering {len(vectors)} fingerprints...")

        if self.backend == ClusteringBackend.RAPIDS:
            labels, centroids = self._cluster_rapids(vectors)
        elif self.backend == ClusteringBackend.CPU:
            labels, centroids = self._cluster_cpu(vectors)
        else:
            return

        self.cluster_labels = labels

        # Build centroid objects
        new_centroids = {}
        for cluster_id, centroid in centroids.items():
            mask = labels == cluster_id
            size = int(mask.sum())

            # Interpret centroid to get traits
            traits = self._interpret_centroid(centroid)
            sampling_hint = self._get_sampling_hint(traits)

            new_centroids[cluster_id] = ClusterCentroid(
                cluster_id=cluster_id,
                centroid=centroid.tolist(),
                size=size,
                traits=traits,
                sampling_hint=sampling_hint,
            )

        self.centroids = new_centroids
        print(f"Found {len(new_centroids)} clusters")

    def _cluster_rapids(self, vectors: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """RAPIDS cuML clustering."""
        import cupy as cp

        # Scale features
        vectors_gpu = cp.asarray(vectors)
        mean = vectors_gpu.mean(axis=0)
        std = vectors_gpu.std(axis=0) + 1e-9
        vectors_scaled = (vectors_gpu - mean) / std

        # Convert to cuDF
        df = cudf.DataFrame(vectors_scaled)

        # HDBSCAN
        clusterer = cuHDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=max(1, self.min_cluster_size // 2),
            cluster_selection_method='leaf',
            prediction_data=True,
        )
        labels = clusterer.fit_predict(df)
        labels = cp.asnumpy(labels.values)

        # Compute centroids
        centroids = {}
        for cluster_id in set(labels) - {-1}:
            mask = labels == cluster_id
            centroids[int(cluster_id)] = vectors[mask].mean(axis=0)

        return labels, centroids

    def _cluster_cpu(self, vectors: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """CPU fallback clustering."""
        # Scale features
        scaler = StandardScaler()
        vectors_scaled = scaler.fit_transform(vectors)

        # HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=max(1, self.min_cluster_size // 2),
            cluster_selection_method='leaf',
        )
        labels = clusterer.fit_predict(vectors_scaled)

        # Compute centroids
        centroids = {}
        for cluster_id in set(labels) - {-1}:
            mask = labels == cluster_id
            centroids[int(cluster_id)] = vectors[mask].mean(axis=0)

        return labels, centroids

    def _interpret_centroid(self, centroid: np.ndarray) -> List[str]:
        """Interpret centroid to extract trait labels."""
        traits = []

        # First 4 elements are: local_mass, mid_mass, long_mass, entropy
        local_mass = centroid[0] if len(centroid) > 0 else 0
        mid_mass = centroid[1] if len(centroid) > 1 else 0
        long_mass = centroid[2] if len(centroid) > 2 else 0
        entropy = centroid[3] if len(centroid) > 3 else 0

        # Hubness (from histogram peak)
        histogram = centroid[4:] if len(centroid) > 4 else []
        if len(histogram) > 0:
            hubness = 1 - entropy  # Inverse of normalized entropy

        # Classify based on mass distribution
        if local_mass > 0.6:
            traits.append("syntax_floor")
            traits.append("local_attention")
        elif mid_mass > 0.4:
            traits.append("semantic_bridge")
            traits.append("retrieval_heavy")
        elif long_mass > 0.3:
            traits.append("long_range")
            traits.append("context_aware")

        # Entropy-based traits
        if entropy < 0.5:
            traits.append("focused")
        elif entropy > 0.7:
            traits.append("diffuse")

        # Check for periodicity (comb pattern in histogram)
        if len(histogram) > 4:
            even_sum = sum(histogram[::2])
            odd_sum = sum(histogram[1::2])
            if abs(even_sum - odd_sum) > 0.2:
                traits.append("periodic")

        return traits if traits else ["neutral"]

    def _get_sampling_hint(self, traits: List[str]) -> Dict[str, float]:
        """Get sampling parameter hints based on traits."""
        hints = {
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.0,
        }

        if "syntax_floor" in traits or "local_attention" in traits:
            # Structured output (code, JSON)
            hints["temperature"] = 0.2
            hints["top_p"] = 0.95

        if "semantic_bridge" in traits:
            # Reasoning/retrieval
            hints["temperature"] = 0.5
            hints["repetition_penalty"] = 1.1

        if "diffuse" in traits:
            # Creative/chat
            hints["temperature"] = 0.8
            hints["top_p"] = 0.85

        if "focused" in traits:
            hints["temperature"] = min(hints["temperature"], 0.3)

        return hints

    def get_stats(self) -> Dict:
        """Get sidecar statistics."""
        stats = {
            "backend": self.backend.value,
            "buffer_size": len(self.fingerprints),
            "buffer_capacity": self.buffer_size,
            "n_clusters": len(self.centroids),
            "last_cluster_time": self.last_cluster_time,
            "zmq_enabled": self.zmq_bind is not None,
            "zmq_received": self._zmq_received,
        }
        return stats


class SidecarHandler(BaseHTTPRequestHandler):
    """HTTP handler for sidecar API."""

    sidecar: RAPIDSSidecar = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/centroids":
            data = self.sidecar.get_centroids()
            self._send_json(data)

        elif parsed.path == "/stats":
            data = self.sidecar.get_stats()
            self._send_json(data)

        elif parsed.path == "/health":
            self._send_json({"status": "ok"})

        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/fingerprint":
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            data = json.loads(body)

            self.sidecar.add_fingerprint(
                request_id=data.get("request_id", "unknown"),
                vector=data["vector"],
                metadata=data.get("metadata"),
            )
            self._send_json({"status": "accepted"})

        elif parsed.path == "/predict":
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            data = json.loads(body)

            cluster_id, distance = self.sidecar.predict_cluster(data["vector"])
            centroid = self.sidecar.centroids.get(cluster_id)

            response = {
                "cluster_id": cluster_id,
                "distance": distance,
                "traits": centroid.traits if centroid else [],
                "sampling_hint": centroid.sampling_hint if centroid else {},
            }
            self._send_json(response)

        elif parsed.path == "/recluster":
            # Force recluster
            self.sidecar.last_cluster_time = 0
            self._send_json({"status": "triggered"})

        else:
            self.send_error(404)

    def _send_json(self, data):
        response = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        pass  # Suppress logging


def main():
    parser = argparse.ArgumentParser(description="RAPIDS clustering sidecar")
    parser.add_argument("--port", type=int, default=9000,
                        help="HTTP port for querying centroids (default: 9000)")
    parser.add_argument("--zmq-bind", type=str, default=None,
                        help="ZMQ bind address for fingerprint streaming. "
                             "Example: 'tcp://*:9001'. Must match SGLang's --attention-sidecar-url "
                             "(but use '*' instead of 'localhost' for binding)")
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="Fingerprint buffer size (default: 10000)")
    parser.add_argument("--min-cluster-size", type=int, default=10,
                        help="Minimum cluster size for HDBSCAN (default: 10)")
    parser.add_argument("--recluster-interval", type=float, default=60.0,
                        help="Seconds between reclustering (default: 60)")
    parser.add_argument("--online", action="store_true",
                        help="Enable online clustering mode (real-time updates, no batching)")
    parser.add_argument("--online-threshold", type=float, default=2.0,
                        help="Distance threshold for creating new cluster in online mode (default: 2.0)")
    parser.add_argument("--online-max-clusters", type=int, default=50,
                        help="Maximum number of clusters in online mode (default: 50)")

    args = parser.parse_args()

    # Check ZMQ availability
    if args.zmq_bind and not HAS_ZMQ:
        print("ERROR: pyzmq not installed. Install with: pip install pyzmq")
        print("Falling back to HTTP-only mode.")
        args.zmq_bind = None

    # Create sidecar
    sidecar = RAPIDSSidecar(
        buffer_size=args.buffer_size,
        min_cluster_size=args.min_cluster_size,
        recluster_interval=args.recluster_interval,
        zmq_bind=args.zmq_bind,
        online_mode=args.online,
        online_threshold=args.online_threshold,
        online_max_clusters=args.online_max_clusters,
    )
    sidecar.start()

    # Set up handler
    SidecarHandler.sidecar = sidecar

    # Start server
    server = HTTPServer(('0.0.0.0', args.port), SidecarHandler)
    print(f"\n{'='*60}")
    print(f"RAPIDS Clustering Sidecar")
    print(f"{'='*60}")
    print(f"HTTP API:         http://0.0.0.0:{args.port}")
    if args.zmq_bind:
        print(f"ZMQ Receiver:     {args.zmq_bind}")
    else:
        print(f"ZMQ Receiver:     disabled (use --zmq-bind to enable)")
    print(f"Backend:          {sidecar.backend.value}")
    if args.online:
        print(f"Online mode:      threshold={args.online_threshold}, max_clusters={args.online_max_clusters}")
    else:
        print(f"Buffer size:      {args.buffer_size}")
        print(f"Recluster:        every {args.recluster_interval}s")
    print(f"{'='*60}")
    print(f"\nEndpoints:")
    print(f"  GET  /centroids  - Current cluster centroids")
    print(f"  GET  /stats      - Sidecar statistics")
    print(f"  GET  /health     - Health check")
    print(f"  POST /fingerprint - Submit fingerprint (HTTP mode)")
    print(f"  POST /predict    - Predict cluster for fingerprint")
    print(f"  POST /recluster  - Force recluster")
    print(f"\nPress Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sidecar.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
