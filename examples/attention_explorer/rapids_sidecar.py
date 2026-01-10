#!/usr/bin/env python3
"""
RAPIDS cuML Sidecar for GPU-Accelerated Attention Clustering

A separate background process that:
1. Receives fingerprint vectors from SGLang scheduler (via ZMQ or HTTP)
2. Runs GPU-accelerated HDBSCAN clustering via cuML
3. Publishes cluster centroids back to scheduler for online routing
4. Persists fingerprints to SQLite for discovery job batch processing
5. Loads discovery artifacts for zone-based classification

Architecture:
    SGLang Scheduler  --fingerprints-->  RAPIDS Sidecar  --centroids-->  Proxy Router
         (ZMQ PUSH)        (ZMQ PULL)           |
         |<----------- manifold hints ----------|
                                                |
                                           SQLite DB
                                                |
                                                v
                                          Discovery Job
                                         (PCA/UMAP/HDBSCAN)
                                                |
                                                v
                                        Parquet Artifacts

Requirements:
    pip install pyzmq  # For ZMQ streaming
    pip install cuml-cu12  # or cuml-cu11 for CUDA 11
    # OR use CPU fallback: pip install hdbscan scikit-learn

Usage:
    # Start sidecar with ZMQ listener and SQLite storage
    python rapids_sidecar.py --zmq-bind tcp://*:9001 --port 9000 \\
        --db ./fingerprints.db --discovery-dir ./discovery_outputs

    # SGLang server connects to sidecar via ZMQ
    python -m sglang.launch_server \\
        --model-path your-model \\
        --return-attention-tokens \\
        --attention-fingerprint-mode \\
        --attention-sidecar-url tcp://localhost:9001 \\
        --disable-cuda-graph

    # Query centroids via HTTP
    curl http://localhost:9000/centroids

    # Classify using discovery artifacts
    curl -X POST http://localhost:9000/classify -d '{"vector": [0.7, ...]}'

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
import os
import shutil
import sqlite3
import struct
import tempfile
import time
import threading
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, asdict, field
from enum import Enum

# Import manifest for git/hardware info in /version endpoint
try:
    from schemas.manifest import get_git_sha, get_git_branch
    HAS_MANIFEST = True
except ImportError:
    HAS_MANIFEST = False
    def get_git_sha():
        return None
    def get_git_branch():
        return None

# Sidecar version
SIDECAR_VERSION = "1.0.0"

# Try to import discovery classifier
try:
    from discovery import SidecarClassifier
    HAS_DISCOVERY = True
except ImportError:
    HAS_DISCOVERY = False

# Try to import RoPE de-rotation for educational analysis
try:
    from discovery import (
        RoPEDerotator,
        RoPEConfig,
        get_glossary,
        get_term_explanation,
        explain_attention_step,
        ATTENTION_GLOSSARY,
    )
    HAS_DEROTATION = True
except ImportError:
    HAS_DEROTATION = False

# Try to import Compass Router for angle-based routing
try:
    from discovery import (
        CompassRouter,
        CompassRouterConfig,
        CompassAnalyzer,
        COMPASS_GLOSSARY,
        create_compass_router,
        analyze_attention_compass,
    )
    HAS_COMPASS = True
except ImportError:
    HAS_COMPASS = False

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


class FingerprintStorage:
    """
    SQLite-based fingerprint storage for discovery job batch processing.

    Uses the schema from discovery/schema.sql to store fingerprints,
    enabling the discovery job to run batch clustering/embedding.
    """

    FINGERPRINT_DIM = 20  # Schema v1 fingerprint size

    def __init__(self, db_path: str, session_id: Optional[str] = None):
        self.db_path = db_path
        self.session_id = session_id or f"sidecar_{int(time.time())}"
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        self._initialized = False
        self._write_count = 0
        self._batch_buffer: List[Tuple] = []
        self._batch_size = 100

    def _ensure_initialized(self):
        """Initialize database connection and session."""
        if self._initialized:
            return

        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        # Verify schema exists
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fingerprints'"
        )
        if cursor.fetchone() is None:
            raise RuntimeError(
                f"Database {self.db_path} missing fingerprints table. "
                f"Initialize with: sqlite3 {self.db_path} < discovery/schema.sql"
            )

        # Insert session record
        self._conn.execute(
            """INSERT OR IGNORE INTO sessions (session_id, name, model_id)
               VALUES (?, ?, ?)""",
            (self.session_id, "rapids_sidecar", "unknown"),
        )
        self._conn.commit()
        self._initialized = True

    def store(
        self,
        request_id: str,
        step: int,
        fingerprint: np.ndarray,
        zone: Optional[str] = None,
        cluster_id: Optional[int] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict] = None,
    ):
        """Store a fingerprint to the database (batched for efficiency)."""
        with self._lock:
            self._ensure_initialized()

            # Pack fingerprint as binary blob
            fp_list = fingerprint.tolist()[:self.FINGERPRINT_DIM]
            # Pad if needed
            while len(fp_list) < self.FINGERPRINT_DIM:
                fp_list.append(0.0)
            fp_blob = struct.pack(f'<{self.FINGERPRINT_DIM}f', *fp_list)

            self._batch_buffer.append((
                request_id,
                self.session_id,
                step,
                fp_blob,
                zone,
                confidence,
                cluster_id if cluster_id is not None else -1,
            ))

            if len(self._batch_buffer) >= self._batch_size:
                self._flush_batch()

    def _flush_batch(self):
        """Flush buffered fingerprints to database."""
        if not self._batch_buffer:
            return

        try:
            self._conn.executemany(
                """INSERT OR REPLACE INTO fingerprints
                   (request_id, session_id, step, fingerprint,
                    manifold_zone, manifold_confidence, cluster_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                self._batch_buffer,
            )
            self._conn.commit()
            self._write_count += len(self._batch_buffer)
            self._batch_buffer.clear()
        except Exception as e:
            print(f"SQLite write error: {e}")

    def flush(self):
        """Force flush any buffered fingerprints."""
        with self._lock:
            self._flush_batch()

    def get_stats(self) -> Dict:
        """Get storage statistics."""
        with self._lock:
            if not self._initialized:
                return {"initialized": False}

            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM fingerprints WHERE session_id = ?",
                (self.session_id,),
            )
            count = cursor.fetchone()[0]

            return {
                "initialized": True,
                "db_path": self.db_path,
                "session_id": self.session_id,
                "session_fingerprints": count,
                "total_writes": self._write_count,
                "buffer_size": len(self._batch_buffer),
            }

    def close(self):
        """Close database connection."""
        with self._lock:
            self._flush_batch()
            if self._conn:
                # Update session end time
                self._conn.execute(
                    "UPDATE sessions SET end_time = datetime('now') WHERE session_id = ?",
                    (self.session_id,),
                )
                self._conn.commit()
                self._conn.close()
                self._conn = None


class DiscoveryProgressTracker:
    """
    Tracks discovery job progress for SSE live updates.

    Stores recent progress events that can be streamed to clients
    via Server-Sent Events (SSE).
    """

    def __init__(self, max_events: int = 1000):
        self.max_events = max_events
        self._events: deque = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._subscribers: List[threading.Event] = []
        self._current_run_id: Optional[str] = None
        self._current_stage: int = 0
        self._current_stage_name: str = ""
        self._percent_complete: float = 0.0

    def post_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Post a new progress event."""
        event = {
            "type": event_type,
            "timestamp": time.time(),
            "data": data,
        }

        with self._lock:
            self._events.append(event)

            # Update current state
            if event_type == "progress":
                self._current_run_id = data.get("run_id")
                self._current_stage = data.get("stage", 0)
                self._current_stage_name = data.get("stage_name", "")
                self._percent_complete = data.get("percent_complete", 0.0)
            elif event_type == "run_start":
                self._current_run_id = data.get("run_id")
                self._current_stage = 0
                self._percent_complete = 0.0
            elif event_type in ("run_complete", "run_error"):
                self._current_run_id = None

            # Notify subscribers
            for subscriber in self._subscribers:
                subscriber.set()

    def get_events_since(self, since_timestamp: float = 0) -> List[Dict]:
        """Get events since a timestamp."""
        with self._lock:
            return [e for e in self._events if e["timestamp"] > since_timestamp]

    def get_current_status(self) -> Dict:
        """Get current discovery status."""
        with self._lock:
            return {
                "run_id": self._current_run_id,
                "stage": self._current_stage,
                "stage_name": self._current_stage_name,
                "percent_complete": self._percent_complete,
                "is_running": self._current_run_id is not None,
            }

    def subscribe(self) -> threading.Event:
        """Subscribe to event notifications."""
        event = threading.Event()
        with self._lock:
            self._subscribers.append(event)
        return event

    def unsubscribe(self, event: threading.Event) -> None:
        """Unsubscribe from event notifications."""
        with self._lock:
            if event in self._subscribers:
                self._subscribers.remove(event)


class DiscoveryIntegration:
    """
    Integration with discovery job artifacts for zone classification.

    Loads discovery artifacts (PCA/UMAP models, cluster data) and provides
    classification using the pre-trained manifold.
    """

    def __init__(self, discovery_dir: str, auto_reload_interval: float = 300.0):
        self.discovery_dir = Path(discovery_dir)
        self.auto_reload_interval = auto_reload_interval
        self._classifier: Optional['SidecarClassifier'] = None
        self._last_reload = 0
        self._lock = threading.Lock()
        self._reload_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def start(self):
        """Start auto-reload background thread."""
        self._load_classifier()
        if self.auto_reload_interval > 0:
            self._reload_thread = threading.Thread(
                target=self._reload_loop, daemon=True
            )
            self._reload_thread.start()

    def stop(self):
        """Stop auto-reload thread."""
        self._stop_event.set()
        if self._reload_thread:
            self._reload_thread.join(timeout=2)

    def _reload_loop(self):
        """Background loop to reload discovery artifacts."""
        while not self._stop_event.is_set():
            time.sleep(self.auto_reload_interval)
            if self._stop_event.is_set():
                break
            self._load_classifier()

    def _load_classifier(self):
        """Load or reload discovery classifier."""
        if not HAS_DISCOVERY:
            return

        latest_path = self.discovery_dir / "latest"
        if not latest_path.exists():
            return

        try:
            with self._lock:
                # SidecarClassifier handles hot-reload internally
                if self._classifier is None:
                    self._classifier = SidecarClassifier(str(self.discovery_dir))
                else:
                    # Force reload check
                    self._classifier._check_reload()
                self._last_reload = time.time()
        except Exception as e:
            print(f"Discovery classifier load error: {e}")

    def classify(self, fingerprint: np.ndarray) -> Optional[Dict]:
        """
        Classify fingerprint using discovery artifacts.

        Returns schema v1 classification result or None if not available.
        """
        with self._lock:
            if self._classifier is None:
                return None
            try:
                return self._classifier.classify(fingerprint)
            except Exception as e:
                print(f"Classification error: {e}")
                return None

    def is_available(self) -> bool:
        """Check if discovery classifier is loaded."""
        with self._lock:
            return self._classifier is not None

    def get_status(self) -> Dict:
        """Get discovery integration status."""
        with self._lock:
            if self._classifier is None:
                return {
                    "available": False,
                    "discovery_dir": str(self.discovery_dir),
                    "has_discovery_module": HAS_DISCOVERY,
                }

            stats = self._classifier.stats
            return {
                "available": True,
                "discovery_dir": str(self.discovery_dir),
                "run_id": stats.get("run_id"),
                "cluster_count": stats.get("cluster_count"),
                "classification_count": stats.get("classification_count"),
                "last_reload": self._last_reload,
            }

    def reload(self) -> bool:
        """Force reload discovery artifacts."""
        try:
            self._load_classifier()
            return True
        except Exception:
            return False


class BlessedConfigStorage:
    """
    JSON file-based storage for blessed quantization configurations.

    Persists approved configs to a JSON file for sharing across browser sessions
    and deployments. Thread-safe for concurrent read/write.
    """

    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self._lock = threading.Lock()
        self._configs: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        """Load configs from JSON file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path) as f:
                    data = json.load(f)
                self._configs = {c["id"]: c for c in data.get("configs", [])}
            except Exception as e:
                print(f"Warning: Failed to load blessed configs: {e}")
                self._configs = {}
        else:
            self._configs = {}

    def _save(self):
        """Save configs to JSON file with atomic write (temp file + rename)."""
        temp_fd = None
        temp_path = None
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first
            temp_fd, temp_path = tempfile.mkstemp(
                dir=self.storage_path.parent,
                prefix=".blessed_configs_",
                suffix=".tmp"
            )
            with os.fdopen(temp_fd, 'w') as f:
                temp_fd = None  # Prevent double-close
                json.dump({
                    "configs": list(self._configs.values()),
                    "updated_at": time.time(),
                }, f, indent=2)

            # Atomic rename (on POSIX systems)
            shutil.move(temp_path, self.storage_path)
            temp_path = None  # Success, don't delete
        except Exception as e:
            print(f"Warning: Failed to save blessed configs: {e}")
        finally:
            # Clean up temp file on failure
            if temp_fd is not None:
                try:
                    os.close(temp_fd)
                except Exception:
                    pass
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    def get_all(self) -> List[Dict]:
        """Get all blessed configs."""
        with self._lock:
            return list(self._configs.values())

    def get(self, config_id: str) -> Optional[Dict]:
        """Get a specific config by ID."""
        with self._lock:
            return self._configs.get(config_id)

    def add(self, config: Dict) -> bool:
        """Add or update a blessed config."""
        if "id" not in config:
            return False
        with self._lock:
            self._configs[config["id"]] = config
            self._save()
            return True

    def remove(self, config_id: str) -> bool:
        """Remove a blessed config."""
        with self._lock:
            if config_id in self._configs:
                del self._configs[config_id]
                self._save()
                return True
            return False

    def get_stats(self) -> Dict:
        """Get storage statistics."""
        with self._lock:
            return {
                "count": len(self._configs),
                "storage_path": str(self.storage_path),
                "exists": self.storage_path.exists(),
            }


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
        db_path: Optional[str] = None,
        discovery_dir: Optional[str] = None,
        discovery_reload_interval: float = 300.0,
        blessed_configs_path: Optional[str] = None,
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

        # SQLite storage (optional)
        self._storage: Optional[FingerprintStorage] = None
        if db_path:
            self._storage = FingerprintStorage(db_path)
            print(f"SQLite storage enabled: {db_path}")

        # Discovery integration (optional)
        self._discovery: Optional[DiscoveryIntegration] = None
        if discovery_dir:
            self._discovery = DiscoveryIntegration(
                discovery_dir, auto_reload_interval=discovery_reload_interval
            )
            print(f"Discovery integration enabled: {discovery_dir}")

        # Blessed configs storage (optional)
        self._blessed_storage: Optional[BlessedConfigStorage] = None
        if blessed_configs_path:
            self._blessed_storage = BlessedConfigStorage(blessed_configs_path)
            print(f"Blessed configs storage enabled: {blessed_configs_path}")

        # Discovery progress tracker for SSE live updates
        self._progress_tracker = DiscoveryProgressTracker()

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

        # Start discovery integration if configured
        if self._discovery:
            self._discovery.start()

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

        # Stop discovery integration
        if self._discovery:
            self._discovery.stop()

        # Close storage
        if self._storage:
            self._storage.close()

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
        step: int = 0,
    ) -> Optional[Dict]:
        """
        Add a fingerprint to the buffer (and update online clusters if enabled).

        Returns discovery classification result if discovery is available.
        """
        vec = np.array(vector, dtype=np.float32)
        entry = FingerprintEntry(
            request_id=request_id,
            vector=vec,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        # Classify using discovery if available
        classification = None
        zone = None
        cluster_id = None
        confidence = None
        if self._discovery and self._discovery.is_available():
            classification = self._discovery.classify(vec)
            if classification:
                manifold = classification.get("manifold", {})
                zone = manifold.get("zone")
                cluster_id = manifold.get("cluster_id")
                confidence = manifold.get("confidence")

        with self.lock:
            self.fingerprints.append(entry)

            # Update online clusters in real-time
            if self.online_mode:
                self._update_online_clusters(vec)

        # Store in SQLite if enabled (outside lock for better concurrency)
        if self._storage:
            self._storage.store(
                request_id=request_id,
                step=step,
                fingerprint=vec,
                zone=zone,
                cluster_id=cluster_id,
                confidence=confidence,
                metadata=metadata,
            )

        return classification

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
            dist = float(np.linalg.norm(vec - cent))  # Convert to Python float
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

    def classify_fingerprint(self, vector: List[float]) -> Dict:
        """
        Classify a fingerprint using discovery artifacts.

        This is a read-only operation that doesn't add to buffer or storage.
        Falls back to online centroid classification if discovery unavailable.
        """
        vec = np.array(vector, dtype=np.float32)

        # Try discovery classification first
        if self._discovery and self._discovery.is_available():
            result = self._discovery.classify(vec)
            if result:
                return result

        # Fall back to centroid classification
        cluster_id, distance = self.predict_cluster(vector)
        centroid = self.centroids.get(cluster_id)

        return {
            "manifold": {
                "zone": centroid.traits[0] if centroid and centroid.traits else "unknown",
                "confidence": max(0, 1 - distance / 5.0),  # Heuristic confidence
                "cluster_id": cluster_id,
                "cluster_label": None,
            },
            "schema_version": 1,
            "source": "centroid_fallback",
        }

    def get_discovery_status(self) -> Dict:
        """Get discovery integration status."""
        if self._discovery:
            return self._discovery.get_status()
        return {"available": False, "reason": "not_configured"}

    def reload_discovery(self) -> bool:
        """Force reload discovery artifacts."""
        if self._discovery:
            return self._discovery.reload()
        return False

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

        # Add storage stats
        if self._storage:
            stats["storage"] = self._storage.get_stats()
        else:
            stats["storage"] = {"enabled": False}

        # Add discovery stats
        if self._discovery:
            stats["discovery"] = self._discovery.get_status()
        else:
            stats["discovery"] = {"available": False}

        # Add blessed configs stats
        if self._blessed_storage:
            stats["blessed_configs"] = self._blessed_storage.get_stats()
        else:
            stats["blessed_configs"] = {"enabled": False}

        return stats

    # -------------------------------------------------------------------------
    # Blessed Configs API
    # -------------------------------------------------------------------------

    def get_blessed_configs(self) -> List[Dict]:
        """Get all blessed configs."""
        if self._blessed_storage:
            return self._blessed_storage.get_all()
        return []

    def get_blessed_config(self, config_id: str) -> Optional[Dict]:
        """Get a specific blessed config."""
        if self._blessed_storage:
            return self._blessed_storage.get(config_id)
        return None

    def add_blessed_config(self, config: Dict) -> bool:
        """Add or update a blessed config."""
        if self._blessed_storage:
            return self._blessed_storage.add(config)
        return False

    def remove_blessed_config(self, config_id: str) -> bool:
        """Remove a blessed config."""
        if self._blessed_storage:
            return self._blessed_storage.remove(config_id)
        return False

    # -------------------------------------------------------------------------
    # Discovery Progress API (for SSE live updates)
    # -------------------------------------------------------------------------

    def post_discovery_progress(self, event_type: str, data: Dict[str, Any]) -> None:
        """Post a discovery progress event for SSE streaming."""
        self._progress_tracker.post_event(event_type, data)

    def get_discovery_progress(self) -> Dict:
        """Get current discovery progress status."""
        return self._progress_tracker.get_current_status()

    def get_discovery_events(self, since_timestamp: float = 0) -> List[Dict]:
        """Get discovery events since a timestamp."""
        return self._progress_tracker.get_events_since(since_timestamp)

    def subscribe_discovery_progress(self) -> threading.Event:
        """Subscribe to discovery progress notifications."""
        return self._progress_tracker.subscribe()

    def unsubscribe_discovery_progress(self, event: threading.Event) -> None:
        """Unsubscribe from discovery progress notifications."""
        self._progress_tracker.unsubscribe(event)


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

        elif parsed.path == "/healthz":
            # Kubernetes-style health check alias
            self._send_json({"status": "ok"})

        elif parsed.path == "/version":
            # Version info with git SHA for reproducibility
            self._send_json({
                "version": SIDECAR_VERSION,
                "git_sha": get_git_sha(),
                "git_branch": get_git_branch(),
                "backend": self.sidecar.backend.value,
                "has_discovery": self.sidecar._discovery is not None,
                "has_storage": self.sidecar._storage is not None,
                "has_blessed_configs": self.sidecar._blessed_storage is not None,
                "has_derotation": HAS_DEROTATION,
                "has_compass": HAS_COMPASS,
            })

        elif parsed.path == "/discovery/status":
            # Get discovery integration status
            data = self.sidecar.get_discovery_status()
            self._send_json(data)

        elif parsed.path == "/discovery/progress":
            # Get current discovery progress (non-streaming)
            data = self.sidecar.get_discovery_progress()
            self._send_json(data)

        elif parsed.path == "/discovery/live" or parsed.path.startswith("/discovery/live?"):
            # SSE endpoint for live discovery updates
            self._handle_sse_discovery(parsed)

        elif parsed.path == "/blessed-configs":
            # Get all blessed configs
            data = self.sidecar.get_blessed_configs()
            self._send_json({"configs": data})

        elif parsed.path.startswith("/blessed-configs/"):
            # Get specific blessed config
            config_id = parsed.path.split("/blessed-configs/")[1]
            config = self.sidecar.get_blessed_config(config_id)
            if config:
                self._send_json(config)
            else:
                self.send_error(404, "Config not found")

        elif parsed.path == "/education/glossary":
            # Get educational glossary for attention concepts
            if HAS_DEROTATION:
                self._send_json({
                    "glossary": ATTENTION_GLOSSARY,
                    "terms": list(ATTENTION_GLOSSARY.keys()),
                    "version": "1.0.0",
                })
            else:
                self._send_json_error(501, "Derotation module not available")

        elif parsed.path.startswith("/education/glossary/"):
            # Get specific term explanation
            if HAS_DEROTATION:
                term = parsed.path.split("/education/glossary/")[1]
                query = urllib.parse.parse_qs(parsed.query)
                detail_level = query.get("level", ["simple"])[0]
                explanation = get_term_explanation(term, detail_level)
                if explanation:
                    entry = ATTENTION_GLOSSARY.get(term, {})
                    self._send_json({
                        "term": term,
                        "level": detail_level,
                        "explanation": explanation,
                        "full_entry": entry,
                    })
                else:
                    self.send_error(404, f"Term not found: {term}")
            else:
                self._send_json_error(501, "Derotation module not available")

        elif parsed.path == "/compass/glossary":
            # Get compass router educational glossary
            if HAS_COMPASS:
                self._send_json({
                    "glossary": COMPASS_GLOSSARY,
                    "terms": list(COMPASS_GLOSSARY.keys()),
                    "version": "1.0.0",
                })
            else:
                self._send_json_error(501, "Compass router not available")

        else:
            self.send_error(404)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/fingerprint":
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            data = json.loads(body)

            classification = self.sidecar.add_fingerprint(
                request_id=data.get("request_id", "unknown"),
                vector=data["vector"],
                metadata=data.get("metadata"),
                step=data.get("step", 0),
            )
            response = {"status": "accepted"}
            if classification:
                response["classification"] = classification
            self._send_json(response)

        elif parsed.path == "/classify":
            # Classify fingerprint using discovery artifacts (read-only)
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)
            data = json.loads(body)

            result = self.sidecar.classify_fingerprint(data["vector"])
            self._send_json(result)

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

        elif parsed.path == "/discovery/reload":
            # Force reload discovery artifacts
            success = self.sidecar.reload_discovery()
            self._send_json({
                "status": "reloaded" if success else "failed",
                "discovery": self.sidecar.get_discovery_status(),
            })

        elif parsed.path == "/storage/flush":
            # Flush storage buffer to SQLite
            if self.sidecar._storage:
                self.sidecar._storage.flush()
                self._send_json({
                    "status": "flushed",
                    "storage": self.sidecar._storage.get_stats(),
                })
            else:
                self._send_json({"status": "storage_disabled"})

        elif parsed.path == "/discovery/progress":
            # Post discovery progress event for SSE streaming
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)

            try:
                data = json.loads(body)
                event_type = data.get("type", "progress")
                event_data = data.get("data", data)
                self.sidecar.post_discovery_progress(event_type, event_data)
                self._send_json({"status": "posted", "type": event_type})
            except json.JSONDecodeError as e:
                self._send_json_error(400, f"Invalid JSON: {e}")

        elif parsed.path == "/blessed-configs":
            # Add or update a blessed config with schema validation
            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)

            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                self.send_error(400, f"Invalid JSON: {e}")
                return

            # Schema validation for blessed configs
            errors = self._validate_blessed_config(data)
            if errors:
                self._send_json_error(400, f"Validation failed: {'; '.join(errors)}")
                return

            success = self.sidecar.add_blessed_config(data)
            if success:
                self._send_json({
                    "status": "saved",
                    "config_id": data.get("id"),
                })
            else:
                self._send_json_error(400, "Storage disabled or write failed")

        elif parsed.path == "/education/derotate":
            # Analyze attention pattern with RoPE de-rotation
            # Educational endpoint to understand semantic vs positional attention
            if not HAS_DEROTATION:
                self._send_json_error(501, "Derotation module not available")
                return

            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)

            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                self._send_json_error(400, f"Invalid JSON: {e}")
                return

            # Validate required fields
            required = ["query_pos", "key_positions", "attention_scores"]
            missing = [f for f in required if f not in data]
            if missing:
                self._send_json_error(400, f"Missing fields: {missing}")
                return

            # Create derotator with optional config
            config = None
            if "config" in data:
                config = RoPEConfig(
                    head_dim=data["config"].get("head_dim", 128),
                    base=data["config"].get("base", 10000.0),
                    max_position=data["config"].get("max_position", 131072),
                )
            derotator = RoPEDerotator(config=config)

            # Run analysis
            result = derotator.analyze(
                query_pos=data["query_pos"],
                key_positions=data["key_positions"],
                attention_scores=data["attention_scores"],
                sink_threshold=data.get("sink_threshold", 5),
            )

            # Optionally generate natural language explanation
            response = result.to_dict()
            if "tokens" in data:
                query_token = data["tokens"].get("query", "token")
                key_tokens = data["tokens"].get("keys", [])
                if key_tokens:
                    response["explanation"] = explain_attention_step(
                        query_token=query_token,
                        key_tokens=key_tokens,
                        attention_scores=data["attention_scores"],
                        analysis=result,
                    )

            self._send_json(response)

        elif parsed.path == "/education/explain":
            # Get explanation for specific attention pattern
            if not HAS_DEROTATION:
                self._send_json_error(501, "Derotation module not available")
                return

            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)

            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                self._send_json_error(400, f"Invalid JSON: {e}")
                return

            # Quick explanation without full analysis
            derotator = RoPEDerotator()
            result = derotator.analyze(
                query_pos=data.get("query_pos", 0),
                key_positions=data.get("key_positions", []),
                attention_scores=data.get("attention_scores", []),
            )

            self._send_json({
                "pattern": result.interpretation.get("pattern", "unknown"),
                "meaning": result.interpretation.get("meaning", ""),
                "insight": result.explanations.get("insight", ""),
                "zone": result.manifold_zone,
                "zone_description": result.explanations.get("zone", ""),
                "dominant_mode": result.dominant_mode,
                "rotational_variance": result.rotational_variance,
            })

        elif parsed.path == "/compass/route":
            # Route query using Compass Router (angle-based routing)
            if not HAS_COMPASS:
                self._send_json_error(501, "Compass router not available")
                return

            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)

            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                self._send_json_error(400, f"Invalid JSON: {e}")
                return

            # Validate required fields
            required = ["query_pos", "key_positions", "attention_scores"]
            missing = [f for f in required if f not in data]
            if missing:
                self._send_json_error(400, f"Missing fields: {missing}")
                return

            # Create router with optional config
            config = None
            if "config" in data:
                config = CompassRouterConfig(
                    sink_threshold=data["config"].get("sink_threshold", 5),
                    low_variance_threshold=data["config"].get("low_variance_threshold", 0.3),
                    high_variance_threshold=data["config"].get("high_variance_threshold", 0.7),
                    small_model=data["config"].get("small_model", "Qwen/Qwen3-4B"),
                    medium_model=data["config"].get("medium_model", "Qwen/Qwen3-14B"),
                    large_model=data["config"].get("large_model", "Qwen/Qwen3-72B"),
                )
            router = CompassRouter(config)

            # Route the query
            decision = router.route(
                query_pos=data["query_pos"],
                key_positions=data["key_positions"],
                attention_scores=data["attention_scores"],
                rotational_variance=data.get("rotational_variance"),
            )

            self._send_json(decision.to_dict())

        elif parsed.path == "/compass/analyze":
            # Analyze attention pattern with compass (no routing decision)
            if not HAS_COMPASS:
                self._send_json_error(501, "Compass router not available")
                return

            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)

            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                self._send_json_error(400, f"Invalid JSON: {e}")
                return

            # Quick compass analysis
            reading = analyze_attention_compass(
                query_pos=data.get("query_pos", 0),
                key_positions=data.get("key_positions", []),
                attention_scores=data.get("attention_scores", []),
            )

            self._send_json(reading.to_dict())

        elif parsed.path == "/compass/fingerprint":
            # Route based on fingerprint vector
            if not HAS_COMPASS:
                self._send_json_error(501, "Compass router not available")
                return

            content_len = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_len)

            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                self._send_json_error(400, f"Invalid JSON: {e}")
                return

            if "fingerprint" not in data:
                self._send_json_error(400, "Missing 'fingerprint' field")
                return

            router = create_compass_router()
            fingerprint = np.array(data["fingerprint"])
            decision = router.route_fingerprint(fingerprint, data.get("metadata"))

            self._send_json(decision.to_dict())

        else:
            self.send_error(404)

    def do_DELETE(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path.startswith("/blessed-configs/"):
            # Delete a blessed config
            config_id = parsed.path.split("/blessed-configs/")[1]
            success = self.sidecar.remove_blessed_config(config_id)
            if success:
                self._send_json({"status": "deleted", "config_id": config_id})
            else:
                self.send_error(404, "Config not found or storage disabled")
        else:
            self.send_error(404)

    def _handle_sse_discovery(self, parsed) -> None:
        """
        Handle SSE (Server-Sent Events) for live discovery updates.

        Streams discovery progress events to the client in real-time.
        Client should connect with: EventSource('/discovery/live')
        """
        # Parse query parameters
        query = urllib.parse.parse_qs(parsed.query)
        since = float(query.get('since', ['0'])[0])

        # Send SSE headers
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        # Subscribe to updates
        notify_event = self.sidecar.subscribe_discovery_progress()

        try:
            # Send initial status
            status = self.sidecar.get_discovery_progress()
            self._send_sse_event('status', status)

            # Send any missed events
            events = self.sidecar.get_discovery_events(since)
            for event in events:
                self._send_sse_event(event['type'], event['data'])

            # Stream updates
            last_check = time.time()
            while True:
                # Wait for notification or timeout (30s heartbeat)
                notify_event.wait(timeout=30)

                # Check for new events
                events = self.sidecar.get_discovery_events(last_check)
                for event in events:
                    self._send_sse_event(event['type'], event['data'])
                    last_check = event['timestamp']

                # Send heartbeat if no events
                if not events:
                    self._send_sse_event('heartbeat', {'timestamp': time.time()})

                # Reset notification event
                notify_event.clear()

        except (BrokenPipeError, ConnectionResetError):
            # Client disconnected
            pass
        except Exception as e:
            print(f"SSE error: {e}")
        finally:
            self.sidecar.unsubscribe_discovery_progress(notify_event)

    def _send_sse_event(self, event_type: str, data: Dict) -> None:
        """Send a single SSE event."""
        try:
            event_str = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
            self.wfile.write(event_str.encode())
            self.wfile.flush()
        except Exception:
            raise BrokenPipeError("Client disconnected")

    def _send_json(self, data):
        response = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.end_headers()
        self.wfile.write(response)

    def _send_json_error(self, status_code: int, message: str):
        """Send a JSON error response."""
        response = json.dumps({"error": message}).encode()
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(response))
        self.end_headers()
        self.wfile.write(response)

    def _validate_blessed_config(self, config: Dict) -> List[str]:
        """
        Validate a blessed config against expected schema.

        Required fields:
        - id: string (unique identifier)
        - model_id: string (model identifier)
        - quantization: dict with at least 'method' field

        Returns list of validation errors (empty if valid).
        """
        errors = []

        # Check required top-level fields
        if not isinstance(config, dict):
            return ["Config must be a JSON object"]

        if "id" not in config:
            errors.append("Missing required field 'id'")
        elif not isinstance(config["id"], str) or not config["id"].strip():
            errors.append("'id' must be a non-empty string")

        if "model_id" not in config:
            errors.append("Missing required field 'model_id'")
        elif not isinstance(config["model_id"], str):
            errors.append("'model_id' must be a string")

        # Validate quantization config
        if "quantization" not in config:
            errors.append("Missing required field 'quantization'")
        elif not isinstance(config["quantization"], dict):
            errors.append("'quantization' must be an object")
        else:
            quant = config["quantization"]
            if "method" not in quant:
                errors.append("quantization missing required field 'method'")
            elif quant["method"] not in ["none", "sinq", "asinq", "awq", "gptq", "squeezellm", "fp8", "marlin"]:
                errors.append(f"quantization.method '{quant['method']}' is not a recognized method")

            # Validate optional numeric fields
            if "nbits" in quant and not isinstance(quant["nbits"], int):
                errors.append("quantization.nbits must be an integer")
            if "group_size" in quant and not isinstance(quant["group_size"], int):
                errors.append("quantization.group_size must be an integer")

        return errors

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

    # Storage and discovery options
    parser.add_argument("--db", type=str, default=None,
                        help="SQLite database path for fingerprint storage. "
                             "Initialize with: sqlite3 <db> < discovery/schema.sql")
    parser.add_argument("--discovery-dir", type=str, default=None,
                        help="Directory containing discovery job artifacts. "
                             "Should contain a 'latest' symlink to current run.")
    parser.add_argument("--discovery-reload-interval", type=float, default=300.0,
                        help="Seconds between discovery artifact reload checks (default: 300)")
    parser.add_argument("--blessed-configs", type=str, default=None,
                        help="JSON file path for blessed quantization configs storage. "
                             "Enables /blessed-configs API endpoints.")

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
        db_path=args.db,
        discovery_dir=args.discovery_dir,
        discovery_reload_interval=args.discovery_reload_interval,
        blessed_configs_path=args.blessed_configs,
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
    if args.db:
        print(f"SQLite storage:   {args.db}")
    if args.discovery_dir:
        print(f"Discovery dir:    {args.discovery_dir}")
        print(f"Discovery reload: every {args.discovery_reload_interval}s")
    if args.blessed_configs:
        print(f"Blessed configs:  {args.blessed_configs}")
    print(f"{'='*60}")
    print(f"\nEndpoints:")
    print(f"  GET  /centroids          - Current cluster centroids")
    print(f"  GET  /stats              - Sidecar statistics")
    print(f"  GET  /health             - Health check")
    print(f"  GET  /discovery/status   - Discovery integration status")
    print(f"  GET  /discovery/progress - Current discovery progress")
    print(f"  GET  /discovery/live     - SSE stream for live discovery updates")
    print(f"  GET  /blessed-configs    - List blessed quantization configs")
    print(f"  POST /fingerprint        - Submit fingerprint (stores + classifies)")
    print(f"  POST /classify           - Classify fingerprint (read-only)")
    print(f"  POST /predict            - Predict cluster for fingerprint")
    print(f"  POST /recluster          - Force recluster")
    print(f"  POST /discovery/reload   - Reload discovery artifacts")
    print(f"  POST /discovery/progress - Post discovery progress event")
    print(f"  POST /storage/flush      - Flush storage buffer to SQLite")
    print(f"  POST /blessed-configs    - Add/update blessed config")
    print(f"  DELETE /blessed-configs/:id - Remove blessed config")
    print(f"\nPress Ctrl+C to stop\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sidecar.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
