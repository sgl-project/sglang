"""
Shared fixtures for integration tests.

Provides sample databases, fingerprints, and discovery artifacts.
"""

import json
import os
import sqlite3
import struct
import tempfile
import shutil
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime, timedelta

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# CONSTANTS
# =============================================================================

FINGERPRINT_DIM = 20
SCHEMA_VERSION = 1


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def pack_fingerprint(arr: np.ndarray) -> bytes:
    """Pack numpy array to fingerprint blob."""
    return struct.pack(f'<{FINGERPRINT_DIM}f', *arr.astype(np.float32))


def generate_syntax_floor_fingerprint() -> np.ndarray:
    """Generate fingerprint typical of syntax_floor zone (high local mass, low entropy)."""
    fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
    fp[0] = np.random.uniform(0.6, 0.9)  # local_mass
    fp[1] = np.random.uniform(0.05, 0.2)  # mid_mass
    fp[2] = np.random.uniform(0.01, 0.1)  # long_mass
    fp[3] = np.random.uniform(0.5, 2.0)  # entropy (low)
    # Histogram with local bias
    fp[4:12] = np.array([0.5, 0.3, 0.1, 0.05, 0.02, 0.01, 0.01, 0.01])
    return fp


def generate_semantic_bridge_fingerprint() -> np.ndarray:
    """Generate fingerprint typical of semantic_bridge zone (mid-range retrieval)."""
    fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
    fp[0] = np.random.uniform(0.2, 0.4)  # local_mass
    fp[1] = np.random.uniform(0.4, 0.6)  # mid_mass (high)
    fp[2] = np.random.uniform(0.1, 0.2)  # long_mass
    fp[3] = np.random.uniform(2.5, 3.5)  # entropy (medium)
    # Balanced histogram
    fp[4:12] = np.array([0.15, 0.2, 0.25, 0.15, 0.1, 0.07, 0.05, 0.03])
    return fp


def generate_structure_ripple_fingerprint() -> np.ndarray:
    """Generate fingerprint typical of structure_ripple zone (long-range patterns)."""
    fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
    fp[0] = np.random.uniform(0.1, 0.25)  # local_mass
    fp[1] = np.random.uniform(0.15, 0.3)  # mid_mass
    fp[2] = np.random.uniform(0.35, 0.6)  # long_mass (high)
    fp[3] = np.random.uniform(3.0, 4.0)  # entropy (high)
    # Periodic histogram pattern (high variance)
    fp[4:12] = np.array([0.3, 0.05, 0.25, 0.05, 0.2, 0.05, 0.08, 0.02])
    return fp


def generate_random_fingerprint() -> np.ndarray:
    """Generate random fingerprint with valid mass distribution."""
    fp = np.zeros(FINGERPRINT_DIM, dtype=np.float32)
    masses = np.random.dirichlet([1, 1, 1])
    fp[0:3] = masses
    fp[3] = np.random.uniform(1.0, 4.0)
    fp[4:12] = np.random.dirichlet(np.ones(8))
    return fp


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_fingerprints():
    """Generate a mix of fingerprints from all zones."""
    fingerprints = []

    # Generate 100 fingerprints from each zone
    for _ in range(100):
        fingerprints.append(generate_syntax_floor_fingerprint())
    for _ in range(100):
        fingerprints.append(generate_semantic_bridge_fingerprint())
    for _ in range(100):
        fingerprints.append(generate_structure_ripple_fingerprint())

    return np.array(fingerprints, dtype=np.float32)


@pytest.fixture
def sample_fingerprints_large():
    """Generate larger dataset for clustering tests."""
    fingerprints = []

    # Create 3 distinct clusters with some noise
    np.random.seed(42)

    # Cluster 1: syntax_floor variants (300 points)
    for _ in range(300):
        fp = generate_syntax_floor_fingerprint()
        fp += np.random.normal(0, 0.05, FINGERPRINT_DIM).astype(np.float32)
        fingerprints.append(fp)

    # Cluster 2: semantic_bridge variants (300 points)
    for _ in range(300):
        fp = generate_semantic_bridge_fingerprint()
        fp += np.random.normal(0, 0.05, FINGERPRINT_DIM).astype(np.float32)
        fingerprints.append(fp)

    # Cluster 3: structure_ripple variants (300 points)
    for _ in range(300):
        fp = generate_structure_ripple_fingerprint()
        fp += np.random.normal(0, 0.05, FINGERPRINT_DIM).astype(np.float32)
        fingerprints.append(fp)

    # Noise points (100 random)
    for _ in range(100):
        fingerprints.append(generate_random_fingerprint())

    return np.array(fingerprints, dtype=np.float32)


@pytest.fixture
def temp_db_with_schema():
    """Create a temporary database with full schema."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_fingerprints.db")

        conn = sqlite3.connect(db_path)
        conn.executescript("""
            PRAGMA journal_mode = WAL;

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                name TEXT,
                model_id TEXT,
                request_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME
            );

            CREATE TABLE IF NOT EXISTS fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                session_id TEXT,
                step INTEGER NOT NULL,
                token_id INTEGER,
                token_text TEXT,
                think_phase TEXT,
                segment_idx INTEGER DEFAULT 0,
                fingerprint BLOB NOT NULL,
                manifold_zone TEXT,
                manifold_confidence REAL,
                cluster_id INTEGER DEFAULT -1,
                cluster_probability REAL,
                top_expert_ids BLOB,
                router_entropy REAL,
                expert_load_balance REAL,
                top_k_positions BLOB,
                top_k_scores BLOB,
                sink_token_mass REAL,
                capture_layer_ids BLOB,
                schema_version INTEGER DEFAULT 1,
                model_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(request_id, step)
            );

            CREATE INDEX IF NOT EXISTS idx_fingerprints_session ON fingerprints(session_id);
            CREATE INDEX IF NOT EXISTS idx_fingerprints_request ON fingerprints(request_id);
            CREATE INDEX IF NOT EXISTS idx_fingerprints_cluster ON fingerprints(cluster_id);
            CREATE INDEX IF NOT EXISTS idx_fingerprints_zone ON fingerprints(manifold_zone);
            CREATE INDEX IF NOT EXISTS idx_fingerprints_created ON fingerprints(created_at);

            CREATE TABLE IF NOT EXISTS request_summary (
                request_id TEXT PRIMARY KEY,
                session_id TEXT,
                total_steps INTEGER NOT NULL,
                think_steps INTEGER DEFAULT 0,
                output_steps INTEGER DEFAULT 0,
                syntax_floor_pct REAL DEFAULT 0,
                semantic_bridge_pct REAL DEFAULT 0,
                structure_ripple_pct REAL DEFAULT 0,
                unknown_pct REAL DEFAULT 0,
                dominant_zone TEXT,
                mean_entropy REAL,
                mean_local_mass REAL,
                mean_long_mass REAL,
                expert_entropy_mean REAL,
                expert_switch_count INTEGER DEFAULT 0,
                unique_experts_used INTEGER DEFAULT 0,
                dominant_cluster_id INTEGER DEFAULT -1,
                cluster_transitions INTEGER DEFAULT 0,
                prompt_preview TEXT,
                response_preview TEXT,
                model_id TEXT,
                started_at TIMESTAMP,
                finished_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS discovery_runs (
                run_id TEXT PRIMARY KEY,
                time_window_start TIMESTAMP,
                time_window_end TIMESTAMP,
                fingerprint_count INTEGER,
                request_count INTEGER,
                cluster_count INTEGER,
                noise_count INTEGER,
                embedding_method TEXT,
                clustering_method TEXT,
                min_cluster_size INTEGER,
                output_dir TEXT,
                status TEXT CHECK(status IN ('running', 'completed', 'failed')),
                error_message TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS clusters (
                cluster_id INTEGER PRIMARY KEY,
                run_id TEXT NOT NULL,
                label TEXT,
                description TEXT,
                dominant_zone TEXT,
                zone_confidence REAL,
                size INTEGER,
                persistence REAL,
                centroid_x REAL,
                centroid_y REAL,
                centroid_fingerprint BLOB,
                medoid_request_id TEXT,
                medoid_step INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            INSERT OR IGNORE INTO clusters (cluster_id, run_id, label, dominant_zone, size)
            VALUES (-1, 'initial', 'Unassigned', 'unknown', 0);
        """)
        conn.commit()
        conn.close()

        yield db_path


@pytest.fixture
def populated_db(temp_db_with_schema, sample_fingerprints_large):
    """Create database populated with sample fingerprints."""
    db_path = temp_db_with_schema
    conn = sqlite3.connect(db_path)

    # Insert session
    conn.execute(
        "INSERT INTO sessions (session_id, name, model_id) VALUES (?, ?, ?)",
        ("test-session", "Integration Test", "test-model")
    )

    # Insert fingerprints
    now = datetime.utcnow()
    for i, fp in enumerate(sample_fingerprints_large):
        request_id = f"req-{i // 10:04d}"
        step = i % 10

        # Vary timestamps to simulate real data
        created_at = now - timedelta(minutes=len(sample_fingerprints_large) - i)

        conn.execute(
            """INSERT INTO fingerprints
               (request_id, session_id, step, fingerprint, model_id, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (request_id, "test-session", step, pack_fingerprint(fp),
             "test-model", created_at.isoformat())
        )

    # Insert request summaries
    for req_num in range(len(sample_fingerprints_large) // 10):
        request_id = f"req-{req_num:04d}"
        conn.execute(
            """INSERT INTO request_summary
               (request_id, session_id, total_steps, prompt_preview, response_preview, model_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (request_id, "test-session", 10,
             f"Test prompt {req_num}", f"Test response {req_num}", "test-model")
        )

    conn.commit()
    conn.close()

    return db_path


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory for discovery artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def discovery_artifacts(temp_output_dir, sample_fingerprints_large):
    """Create mock discovery artifacts for classifier tests."""
    import pandas as pd

    output_dir = Path(temp_output_dir)
    run_id = "test_run_20260109"
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True)

    n_samples = len(sample_fingerprints_large)

    # Create embeddings
    np.random.seed(42)
    embeddings = np.random.randn(n_samples, 2).astype(np.float32)

    # Create labels (3 clusters + noise)
    labels = np.array(
        [0] * 300 + [1] * 300 + [2] * 300 + [-1] * 100,
        dtype=np.int32
    )

    # Create zones
    zones = np.array(
        ['syntax_floor'] * 300 +
        ['semantic_bridge'] * 300 +
        ['structure_ripple'] * 300 +
        ['unknown'] * 100
    )

    # Create embeddings parquet
    embeddings_df = pd.DataFrame({
        'id': range(n_samples),
        'request_id': [f"req-{i // 10:04d}" for i in range(n_samples)],
        'step': [i % 10 for i in range(n_samples)],
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'cluster_id': labels,
        'zone': zones,
    })
    embeddings_df.to_parquet(run_dir / "embeddings.parquet")

    # Create cluster metadata
    clusters = []
    for cluster_id in range(3):
        mask = labels == cluster_id
        centroid = sample_fingerprints_large[mask].mean(axis=0)
        clusters.append({
            'cluster_id': cluster_id,
            'label': f"Cluster {cluster_id}",
            'size': int(mask.sum()),
            'dominant_zone': zones[mask][0],
            'zone_confidence': 0.9,
            'centroid_x': float(embeddings[mask, 0].mean()),
            'centroid_y': float(embeddings[mask, 1].mean()),
        })

    with open(run_dir / "clusters.json", 'w') as f:
        json.dump(clusters, f, indent=2)

    # Create fingerprints parquet
    fingerprints_df = pd.DataFrame({
        'id': range(n_samples),
        'fingerprint': [fp.tobytes() for fp in sample_fingerprints_large],
        'cluster_id': labels,
        'zone': zones,
    })
    fingerprints_df.to_parquet(run_dir / "fingerprints.parquet")

    # Create centroids
    centroids = {}
    for cluster_id in range(3):
        mask = labels == cluster_id
        centroids[cluster_id] = sample_fingerprints_large[mask].mean(axis=0).tolist()

    with open(run_dir / "centroids.json", 'w') as f:
        json.dump(centroids, f)

    # Create manifest
    manifest = {
        'run_id': run_id,
        'created_at': datetime.utcnow().isoformat(),
        'fingerprint_count': n_samples,
        'cluster_count': 3,
        'noise_count': 100,
        'embedding_method': 'pca_50_umap_2',
        'clustering_method': 'hdbscan',
    }

    with open(run_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    # Create latest symlink
    latest_link = output_dir / "latest"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_dir)

    return {
        'output_dir': str(output_dir),
        'run_dir': str(run_dir),
        'run_id': run_id,
        'n_samples': n_samples,
        'embeddings': embeddings,
        'labels': labels,
        'zones': zones,
        'fingerprints': sample_fingerprints_large,
        'centroids': centroids,
    }
