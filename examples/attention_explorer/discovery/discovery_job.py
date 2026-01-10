#!/usr/bin/env python3
"""
Attention Fingerprint Discovery Job

Batch pipeline that processes stored fingerprints to:
1. Compute 2D embeddings (PCA → UMAP)
2. Cluster with HDBSCAN
3. Assign zone labels (syntax_floor / semantic_bridge / structure_ripple)
4. Generate artifacts for UI visualization
5. Export clusterer for online assignment

Usage:
    python discovery_job.py --db fingerprints.db --output ./discovery_outputs

    # With custom parameters
    python discovery_job.py --db fingerprints.db --output ./outputs \
        --hours 24 --min-cluster-size 50 --umap-neighbors 15

Dependencies:
    pip install numpy pandas pyarrow hdbscan umap-learn scikit-learn joblib

Author: SGLang Attention Explorer
"""

import argparse
import json
import logging
import os
import sqlite3
import struct
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import manifest module for reproducibility layer
try:
    from schemas.manifest import ExperimentManifest, RunType
    HAS_MANIFEST = True
except ImportError:
    HAS_MANIFEST = False

# Import coordinator for long-running jobs
try:
    from .coordinator import DiscoveryJobCoordinator, CoordinatorConfig, run_discovery as run_discovery_coordinated
    HAS_COORDINATOR = True
except ImportError:
    HAS_COORDINATOR = False

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Optional imports with graceful fallback
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    logging.warning("hdbscan not installed. Install with: pip install hdbscan")

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    logging.warning("umap-learn not installed. Install with: pip install umap-learn")

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =============================================================================
# CONFIGURATION
# =============================================================================

FINGERPRINT_DIM = 20  # Schema v1 fingerprint dimension
SCHEMA_VERSION = 1

# Fingerprint layout (schema v1):
# [0]  local_mass     - attention mass in local window (0-32 tokens)
# [1]  mid_mass       - attention mass in mid range (32-256 tokens)
# [2]  long_mass      - attention mass in long range (256+ tokens)
# [3]  entropy        - attention entropy (higher = more distributed)
# [4-11] histogram    - 8-bin distance histogram
# [12-19] layer_stats - per-layer entropy (up to 8 layers)

FP_LOCAL_MASS = 0
FP_MID_MASS = 1
FP_LONG_MASS = 2
FP_ENTROPY = 3
FP_HISTOGRAM_START = 4
FP_HISTOGRAM_END = 12
FP_LAYER_STATS_START = 12

# Zone thresholds (tuned for typical LLM attention patterns)
ZONE_THRESHOLDS = {
    'syntax_floor': {
        'local_mass_min': 0.5,
        'entropy_max': 2.5,
    },
    'structure_ripple': {
        'long_mass_min': 0.25,
        'histogram_variance_min': 0.1,  # Periodic patterns have high variance
    },
    # semantic_bridge is the default when others don't match
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DiscoveryConfig:
    """Configuration for discovery job."""
    db_path: str
    output_dir: str
    time_window_hours: int = 24
    min_cluster_size: int = 50
    min_samples: int = 10
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1
    pca_components: int = 50
    prototype_count: int = 5
    batch_size: int = 100000  # For large datasets


@dataclass
class DiscoveryResult:
    """Result of discovery job."""
    run_id: str
    fingerprint_count: int
    request_count: int
    cluster_count: int
    noise_count: int
    output_dir: str
    duration_seconds: float


# =============================================================================
# DATABASE OPERATIONS
# =============================================================================

def unpack_fingerprint(blob: bytes) -> np.ndarray:
    """Unpack fingerprint from blob to numpy array."""
    if blob is None:
        return np.zeros(FINGERPRINT_DIM, dtype=np.float32)
    return np.array(struct.unpack(f'<{FINGERPRINT_DIM}f', blob), dtype=np.float32)


def pack_fingerprint(arr: np.ndarray) -> bytes:
    """Pack numpy array to fingerprint blob."""
    return struct.pack(f'<{FINGERPRINT_DIM}f', *arr.astype(np.float32))


def extract_fingerprints(
    db_path: str,
    time_window_hours: int = 24,
    batch_size: int = 100000,
) -> pd.DataFrame:
    """
    Extract fingerprints from SQLite database.

    Args:
        db_path: Path to SQLite database
        time_window_hours: Only extract fingerprints from last N hours
        batch_size: Batch size for large datasets

    Returns:
        DataFrame with columns: id, request_id, step, token_text, think_phase,
                               fingerprint (as numpy array)
    """
    logger.info(f"Extracting fingerprints from {db_path} (last {time_window_hours}h)")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)

    query = """
        SELECT
            id,
            request_id,
            session_id,
            step,
            token_id,
            token_text,
            think_phase,
            fingerprint,
            manifold_zone,
            cluster_id,
            top_expert_ids,
            router_entropy,
            model_id,
            created_at
        FROM fingerprints
        WHERE created_at >= ?
        ORDER BY created_at ASC
    """

    rows = []
    cursor = conn.execute(query, (cutoff_time.isoformat(),))

    while True:
        batch = cursor.fetchmany(batch_size)
        if not batch:
            break

        for row in batch:
            fp = unpack_fingerprint(row['fingerprint'])
            rows.append({
                'id': row['id'],
                'request_id': row['request_id'],
                'session_id': row['session_id'],
                'step': row['step'],
                'token_id': row['token_id'],
                'token_text': row['token_text'],
                'think_phase': row['think_phase'] or 'unknown',
                'fingerprint': fp,
                'existing_zone': row['manifold_zone'],
                'existing_cluster': row['cluster_id'],
                'router_entropy': row['router_entropy'],
                'model_id': row['model_id'],
                'created_at': row['created_at'],
            })

        logger.info(f"  Loaded {len(rows)} fingerprints...")

    conn.close()

    df = pd.DataFrame(rows)
    if len(df) > 0:
        logger.info(f"Extracted {len(df)} fingerprints from {df['request_id'].nunique()} requests")
    else:
        logger.info("No fingerprints found in the specified time window")
    return df


def extract_request_summaries(db_path: str, request_ids: List[str]) -> Dict[str, dict]:
    """Extract request summaries for prototype display."""
    if not request_ids:
        return {}

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    placeholders = ','.join(['?' for _ in request_ids])
    query = f"""
        SELECT request_id, prompt_preview, response_preview, model_id
        FROM request_summary
        WHERE request_id IN ({placeholders})
    """

    summaries = {}
    for row in conn.execute(query, request_ids):
        summaries[row['request_id']] = {
            'prompt_preview': row['prompt_preview'],
            'response_preview': row['response_preview'],
            'model_id': row['model_id'],
        }

    conn.close()
    return summaries


# =============================================================================
# ZONE ASSIGNMENT
# =============================================================================

def assign_zone_labels(fingerprints: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign zone labels based on fingerprint features.

    The three zones represent different "attention programs":
    - syntax_floor: Local attention, low entropy (JSON repair, brackets, formatting)
    - semantic_bridge: Mid-range retrieval, balanced (coreference, task routing)
    - structure_ripple: Long-range periodic patterns (counting, tables, indentation)

    Args:
        fingerprints: Array of shape (N, 20) with fingerprint vectors

    Returns:
        Tuple of (zone_labels, zone_confidences) arrays
    """
    n = len(fingerprints)
    zones = np.empty(n, dtype=object)
    confidences = np.zeros(n, dtype=np.float32)

    # Extract features
    local_mass = fingerprints[:, FP_LOCAL_MASS]
    mid_mass = fingerprints[:, FP_MID_MASS]
    long_mass = fingerprints[:, FP_LONG_MASS]
    entropy = fingerprints[:, FP_ENTROPY]

    # Histogram variance (indicates periodic patterns)
    histogram = fingerprints[:, FP_HISTOGRAM_START:FP_HISTOGRAM_END]
    hist_variance = np.var(histogram, axis=1)

    # Normalize masses for confidence calculation
    total_mass = local_mass + mid_mass + long_mass
    total_mass = np.maximum(total_mass, 1e-6)  # Avoid division by zero

    # Zone assignment logic
    for i in range(n):
        # Check syntax_floor first (high local, low entropy)
        if local_mass[i] > ZONE_THRESHOLDS['syntax_floor']['local_mass_min']:
            if entropy[i] < ZONE_THRESHOLDS['syntax_floor']['entropy_max']:
                zones[i] = 'syntax_floor'
                # Confidence based on how clearly it matches
                confidences[i] = min(1.0, local_mass[i] * (1.0 - entropy[i] / 4.0))
                continue

        # Check structure_ripple (high long-range, periodic patterns)
        if long_mass[i] > ZONE_THRESHOLDS['structure_ripple']['long_mass_min']:
            if hist_variance[i] > ZONE_THRESHOLDS['structure_ripple']['histogram_variance_min']:
                zones[i] = 'structure_ripple'
                confidences[i] = min(1.0, long_mass[i] + hist_variance[i])
                continue

        # Default: semantic_bridge
        zones[i] = 'semantic_bridge'
        # Confidence based on mid-mass dominance
        confidences[i] = mid_mass[i] / total_mass[i]

    logger.info(f"Zone distribution: "
                f"syntax_floor={np.sum(zones == 'syntax_floor')}, "
                f"semantic_bridge={np.sum(zones == 'semantic_bridge')}, "
                f"structure_ripple={np.sum(zones == 'structure_ripple')}")

    return zones, confidences


def compute_zone_for_cluster(
    fingerprints: np.ndarray,
    labels: np.ndarray,
    cluster_id: int
) -> Tuple[str, float]:
    """Compute dominant zone for a cluster."""
    mask = labels == cluster_id
    if not np.any(mask):
        return 'unknown', 0.0

    cluster_fps = fingerprints[mask]
    zones, confidences = assign_zone_labels(cluster_fps)

    # Vote for dominant zone
    zone_counts = {}
    for z in zones:
        zone_counts[z] = zone_counts.get(z, 0) + 1

    dominant = max(zone_counts, key=zone_counts.get)
    confidence = zone_counts[dominant] / len(zones)

    return dominant, confidence


# =============================================================================
# EMBEDDING & CLUSTERING
# =============================================================================

def compute_embeddings(
    fingerprints: np.ndarray,
    pca_components: int = 50,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
) -> Tuple[np.ndarray, Any, Any]:
    """
    Compute 2D embeddings using PCA → UMAP pipeline.

    Args:
        fingerprints: Array of shape (N, 20)
        pca_components: Number of PCA components (intermediate step)
        umap_neighbors: UMAP n_neighbors parameter
        umap_min_dist: UMAP min_dist parameter

    Returns:
        Tuple of (embeddings_2d, pca_model, umap_model)
    """
    if not HAS_UMAP:
        raise ImportError("umap-learn required for embeddings. Install with: pip install umap-learn")

    logger.info(f"Computing embeddings for {len(fingerprints)} fingerprints...")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(fingerprints)

    # PCA for dimensionality reduction (helps UMAP stability)
    n_components = min(pca_components, fingerprints.shape[1], len(fingerprints) - 1)
    logger.info(f"  Running PCA (n_components={n_components})...")
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    logger.info(f"  PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    # UMAP for 2D embedding
    logger.info(f"  Running UMAP (neighbors={umap_neighbors}, min_dist={umap_min_dist})...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        metric='euclidean',
        random_state=42,
        low_memory=len(fingerprints) > 500000,
    )
    X_umap = reducer.fit_transform(X_pca)

    logger.info(f"  Embeddings computed: shape={X_umap.shape}")

    # Bundle models for later use
    models = {
        'scaler': scaler,
        'pca': pca,
        'umap': reducer,
    }

    return X_umap, models


def cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 50,
    min_samples: int = 10,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    """
    Cluster embeddings using HDBSCAN.

    Args:
        embeddings: 2D array of shape (N, 2)
        min_cluster_size: Minimum cluster size
        min_samples: Minimum samples for core points

    Returns:
        Tuple of (labels, probabilities, clusterer)
    """
    if not HAS_HDBSCAN:
        raise ImportError("hdbscan required for clustering. Install with: pip install hdbscan")

    logger.info(f"Clustering {len(embeddings)} points with HDBSCAN...")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom',  # Excess of Mass
        prediction_data=True,  # Enable approximate_predict
    )

    labels = clusterer.fit_predict(embeddings)
    probabilities = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)

    logger.info(f"  Found {n_clusters} clusters, {n_noise} noise points")

    return labels, probabilities, clusterer


# =============================================================================
# PROTOTYPE SELECTION
# =============================================================================

def select_prototypes(
    embeddings: np.ndarray,
    labels: np.ndarray,
    df: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    """
    Select representative prototypes for each cluster.

    For each cluster, selects:
    - The medoid (most central point)
    - k-1 nearest neighbors to the medoid

    Args:
        embeddings: 2D embeddings
        labels: Cluster labels
        df: Original dataframe with fingerprint info
        k: Number of prototypes per cluster

    Returns:
        DataFrame with prototype info
    """
    logger.info(f"Selecting {k} prototypes per cluster...")

    prototypes = []
    cluster_ids = sorted(set(labels) - {-1})  # Exclude noise

    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        indices = np.where(mask)[0]

        if len(indices) == 0:
            continue

        cluster_points = embeddings[indices]

        # Find medoid (point closest to centroid)
        centroid = cluster_points.mean(axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        sorted_indices = np.argsort(distances)

        # Select top-k closest to centroid
        for rank, local_idx in enumerate(sorted_indices[:k]):
            global_idx = indices[local_idx]
            row = df.iloc[global_idx]

            prototypes.append({
                'cluster_id': cluster_id,
                'rank': rank,
                'fingerprint_id': row['id'],
                'request_id': row['request_id'],
                'step': row['step'],
                'token_text': row['token_text'],
                'x': embeddings[global_idx, 0],
                'y': embeddings[global_idx, 1],
            })

    proto_df = pd.DataFrame(prototypes)
    logger.info(f"  Selected {len(proto_df)} total prototypes")
    return proto_df


# =============================================================================
# CLUSTER METADATA
# =============================================================================

def build_cluster_metadata(
    embeddings: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    fingerprints: np.ndarray,
    clusterer: Any,
) -> pd.DataFrame:
    """
    Build metadata for each cluster.

    Args:
        embeddings: 2D embeddings
        labels: Cluster labels
        probabilities: Cluster assignment probabilities
        fingerprints: Original fingerprint vectors
        clusterer: HDBSCAN clusterer object

    Returns:
        DataFrame with cluster metadata
    """
    logger.info("Building cluster metadata...")

    clusters = []
    cluster_ids = sorted(set(labels) - {-1})

    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        cluster_embeddings = embeddings[mask]
        cluster_fingerprints = fingerprints[mask]
        cluster_probs = probabilities[mask]

        # Centroid
        centroid_xy = cluster_embeddings.mean(axis=0)
        centroid_fp = cluster_fingerprints.mean(axis=0)

        # Zone assignment for cluster
        dominant_zone, zone_confidence = compute_zone_for_cluster(
            fingerprints, labels, cluster_id
        )

        # Get persistence from HDBSCAN (if available)
        persistence = 0.0
        if hasattr(clusterer, 'cluster_persistence_'):
            if cluster_id < len(clusterer.cluster_persistence_):
                persistence = clusterer.cluster_persistence_[cluster_id]

        clusters.append({
            'cluster_id': cluster_id,
            'label': f'Cluster {cluster_id}',  # Default label, can be overridden
            'description': '',
            'dominant_zone': dominant_zone,
            'zone_confidence': zone_confidence,
            'size': int(np.sum(mask)),
            'persistence': persistence,
            'centroid_x': centroid_xy[0],
            'centroid_y': centroid_xy[1],
            'centroid_fingerprint': centroid_fp,
            'mean_probability': float(cluster_probs.mean()),
        })

    cluster_df = pd.DataFrame(clusters)
    logger.info(f"  Built metadata for {len(cluster_df)} clusters")
    return cluster_df


# =============================================================================
# REQUEST-LEVEL EMBEDDINGS
# =============================================================================

def compute_request_embeddings(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    labels: np.ndarray,
    zones: np.ndarray,
) -> pd.DataFrame:
    """
    Compute request-level (aggregated) embeddings.

    For each request, computes:
    - Mean embedding of all tokens
    - Dominant zone and cluster
    - Summary statistics

    Args:
        df: Original fingerprints dataframe
        embeddings: 2D embeddings
        labels: Cluster labels
        zones: Zone labels

    Returns:
        DataFrame with request-level embeddings
    """
    logger.info("Computing request-level embeddings...")

    # Add embeddings and labels to df for groupby
    df = df.copy()
    df['x'] = embeddings[:, 0]
    df['y'] = embeddings[:, 1]
    df['cluster_id'] = labels
    df['zone'] = zones

    request_data = []

    for request_id, group in df.groupby('request_id'):
        # Mean embedding
        mean_x = group['x'].mean()
        mean_y = group['y'].mean()

        # Dominant zone (mode)
        zone_counts = group['zone'].value_counts()
        dominant_zone = zone_counts.index[0] if len(zone_counts) > 0 else 'unknown'

        # Dominant cluster (mode, excluding noise)
        valid_clusters = group[group['cluster_id'] >= 0]['cluster_id']
        dominant_cluster = int(valid_clusters.mode().iloc[0]) if len(valid_clusters) > 0 else -1

        # Think vs output cluster
        think_group = group[group['think_phase'] == 'think']
        output_group = group[group['think_phase'] == 'output']

        think_cluster = -1
        if len(think_group) > 0:
            think_valid = think_group[think_group['cluster_id'] >= 0]['cluster_id']
            if len(think_valid) > 0:
                think_cluster = int(think_valid.mode().iloc[0])

        output_cluster = -1
        if len(output_group) > 0:
            output_valid = output_group[output_group['cluster_id'] >= 0]['cluster_id']
            if len(output_valid) > 0:
                output_cluster = int(output_valid.mode().iloc[0])

        request_data.append({
            'request_id': request_id,
            'session_id': group['session_id'].iloc[0],
            'x': mean_x,
            'y': mean_y,
            'dominant_zone': dominant_zone,
            'dominant_cluster_id': dominant_cluster,
            'think_cluster_id': think_cluster,
            'output_cluster_id': output_cluster,
            'total_steps': len(group),
            'think_steps': len(think_group),
            'output_steps': len(output_group),
        })

    request_df = pd.DataFrame(request_data)
    logger.info(f"  Computed embeddings for {len(request_df)} requests")
    return request_df


# =============================================================================
# OUTPUT WRITERS
# =============================================================================

def write_parquet(path: str, df: pd.DataFrame, special_columns: Dict[str, str] = None):
    """
    Write DataFrame to Parquet file.

    Args:
        path: Output path
        df: DataFrame to write
        special_columns: Dict mapping column names to special handling
                        ('fingerprint' -> pack as bytes)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Handle special columns
    if special_columns:
        df = df.copy()
        for col, handling in special_columns.items():
            if col in df.columns and handling == 'fingerprint':
                df[col] = df[col].apply(
                    lambda x: pack_fingerprint(x) if isinstance(x, np.ndarray) else x
                )

    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression='snappy')
    logger.info(f"  Wrote {path} ({len(df)} rows)")


def write_manifest(path: str, manifest: dict):
    """Write manifest JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"  Wrote {path}")


def update_latest_symlink(output_dir: str, run_id: str):
    """Update 'latest' symlink to point to newest run."""
    latest_path = os.path.join(output_dir, 'latest')
    run_path = os.path.join(output_dir, run_id)

    # Remove existing symlink
    if os.path.islink(latest_path):
        os.unlink(latest_path)
    elif os.path.exists(latest_path):
        os.remove(latest_path)

    # Create new symlink
    os.symlink(run_id, latest_path)
    logger.info(f"  Updated symlink: latest -> {run_id}")


def update_database(
    db_path: str,
    run_id: str,
    config: DiscoveryConfig,
    result: DiscoveryResult,
    cluster_df: pd.DataFrame,
    proto_df: pd.DataFrame,
):
    """Update database with discovery results."""
    logger.info("Updating database with discovery results...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Insert discovery run
    cursor.execute("""
        INSERT INTO discovery_runs (
            run_id, time_window_start, time_window_end,
            fingerprint_count, request_count, cluster_count, noise_count,
            embedding_method, clustering_method, min_cluster_size,
            output_dir, status, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'completed', CURRENT_TIMESTAMP)
    """, (
        run_id,
        (datetime.utcnow() - timedelta(hours=config.time_window_hours)).isoformat(),
        datetime.utcnow().isoformat(),
        result.fingerprint_count,
        result.request_count,
        result.cluster_count,
        result.noise_count,
        f'pca_{config.pca_components}_umap_2',
        f'hdbscan_min{config.min_cluster_size}',
        config.min_cluster_size,
        result.output_dir,
    ))

    # Update clusters table
    cursor.execute("DELETE FROM clusters WHERE run_id != 'initial'")

    for _, row in cluster_df.iterrows():
        cursor.execute("""
            INSERT INTO clusters (
                cluster_id, run_id, label, description,
                dominant_zone, zone_confidence, size, persistence,
                centroid_x, centroid_y, centroid_fingerprint
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(row['cluster_id']),
            run_id,
            row['label'],
            row['description'],
            row['dominant_zone'],
            float(row['zone_confidence']),
            int(row['size']),
            float(row['persistence']),
            float(row['centroid_x']),
            float(row['centroid_y']),
            pack_fingerprint(row['centroid_fingerprint']),
        ))

    # Update prototypes table
    cursor.execute("DELETE FROM prototypes")

    for _, row in proto_df.iterrows():
        cursor.execute("""
            INSERT INTO prototypes (
                cluster_id, run_id, rank, fingerprint_id,
                request_id, step, token_text, x, y
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(row['cluster_id']),
            run_id,
            int(row['rank']),
            int(row['fingerprint_id']) if pd.notna(row['fingerprint_id']) else None,
            row['request_id'],
            int(row['step']) if pd.notna(row['step']) else None,
            row['token_text'],
            float(row['x']),
            float(row['y']),
        ))

    conn.commit()
    conn.close()
    logger.info(f"  Updated database with {len(cluster_df)} clusters, {len(proto_df)} prototypes")


# =============================================================================
# MAIN DISCOVERY JOB
# =============================================================================

def run_discovery(config: DiscoveryConfig) -> DiscoveryResult:
    """
    Run the complete discovery pipeline.

    Args:
        config: Discovery configuration

    Returns:
        DiscoveryResult with job summary
    """
    start_time = datetime.utcnow()
    run_id = start_time.strftime('%Y-%m-%dT%H-%M-%S')

    logger.info(f"Starting discovery job: {run_id}")
    logger.info(f"  Database: {config.db_path}")
    logger.info(f"  Output: {config.output_dir}")
    logger.info(f"  Time window: {config.time_window_hours} hours")

    # Create output directory
    run_dir = os.path.join(config.output_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # 1. Extract fingerprints
    df = extract_fingerprints(
        config.db_path,
        config.time_window_hours,
        config.batch_size,
    )

    if len(df) == 0:
        logger.warning("No fingerprints found in time window")
        return DiscoveryResult(
            run_id=run_id,
            fingerprint_count=0,
            request_count=0,
            cluster_count=0,
            noise_count=0,
            output_dir=run_dir,
            duration_seconds=0,
        )

    # Stack fingerprints into array
    fingerprints = np.vstack(df['fingerprint'].values)
    logger.info(f"Fingerprint matrix: {fingerprints.shape}")

    # 2. Compute embeddings
    embeddings, models = compute_embeddings(
        fingerprints,
        pca_components=config.pca_components,
        umap_neighbors=config.umap_neighbors,
        umap_min_dist=config.umap_min_dist,
    )

    # 3. Cluster
    labels, probabilities, clusterer = cluster_embeddings(
        embeddings,
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
    )

    # 4. Assign zones
    zones, zone_confidences = assign_zone_labels(fingerprints)

    # 5. Build cluster metadata
    cluster_df = build_cluster_metadata(
        embeddings, labels, probabilities, fingerprints, clusterer
    )

    # 6. Select prototypes
    proto_df = select_prototypes(embeddings, labels, df, k=config.prototype_count)

    # 7. Compute request-level embeddings
    request_df = compute_request_embeddings(df, embeddings, labels, zones)

    # 8. Build embeddings dataframe
    embeddings_df = pd.DataFrame({
        'fingerprint_id': df['id'],
        'request_id': df['request_id'],
        'step': df['step'],
        'x': embeddings[:, 0],
        'y': embeddings[:, 1],
        'cluster_id': labels,
        'cluster_prob': probabilities,
        'zone_label': zones,
        'zone_confidence': zone_confidences,
    })

    # 9. Write outputs
    logger.info(f"Writing outputs to {run_dir}...")

    write_parquet(f"{run_dir}/embeddings.parquet", embeddings_df)
    write_parquet(
        f"{run_dir}/clusters.parquet",
        cluster_df,
        special_columns={'centroid_fingerprint': 'fingerprint'}
    )
    write_parquet(f"{run_dir}/prototypes.parquet", proto_df)
    write_parquet(f"{run_dir}/request_embeddings.parquet", request_df)

    # Save models for online inference
    joblib.dump(models, f"{run_dir}/embedding_models.joblib")
    joblib.dump(clusterer, f"{run_dir}/clusterer.joblib")
    logger.info(f"  Saved models to {run_dir}")

    # Write manifest with full reproducibility metadata
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    if HAS_MANIFEST:
        # Use ExperimentManifest for comprehensive reproducibility
        manifest_obj = ExperimentManifest.create(
            run_type=RunType.DISCOVERY,
            run_id=run_id,
            capture_git=True,
            capture_hardware=True,
        )

        # Set source database info
        manifest_obj.set_source(
            database_path=config.db_path,
            time_window_hours=config.time_window_hours,
            fingerprint_count=len(df),
            request_count=df['request_id'].nunique(),
        )

        # Set discovery parameters
        manifest_obj.set_parameters({
            'time_window_hours': config.time_window_hours,
            'min_cluster_size': config.min_cluster_size,
            'min_samples': config.min_samples,
            'umap_neighbors': config.umap_neighbors,
            'umap_min_dist': config.umap_min_dist,
            'pca_components': config.pca_components,
            'prototype_count': config.prototype_count,
            'embedding_method': f'pca_{config.pca_components}_umap_2',
            'clustering_method': 'hdbscan',
        })

        # Set statistics
        manifest_obj.set_statistics({
            'fingerprint_count': len(df),
            'request_count': df['request_id'].nunique(),
            'cluster_count': n_clusters,
            'noise_count': n_noise,
            'zone_distribution': {
                'syntax_floor': int(np.sum(zones == 'syntax_floor')),
                'semantic_bridge': int(np.sum(zones == 'semantic_bridge')),
                'structure_ripple': int(np.sum(zones == 'structure_ripple')),
            },
            'time_range': {
                'start': (datetime.utcnow() - timedelta(hours=config.time_window_hours)).isoformat(),
                'end': datetime.utcnow().isoformat(),
            },
        })

        # Add artifact references
        manifest_obj.add_artifact("embeddings", f"{run_dir}/embeddings.parquet")
        manifest_obj.add_artifact("clusters", f"{run_dir}/clusters.parquet")
        manifest_obj.add_artifact("prototypes", f"{run_dir}/prototypes.parquet")
        manifest_obj.add_artifact("request_embeddings", f"{run_dir}/request_embeddings.parquet")
        manifest_obj.add_artifact("embedding_models", f"{run_dir}/embedding_models.joblib")
        manifest_obj.add_artifact("clusterer", f"{run_dir}/clusterer.joblib")

        # Finalize and save
        manifest_obj.finalize()
        manifest_obj.save(f"{run_dir}/manifest.json")
    else:
        # Fallback to simple dict manifest
        manifest = {
            'schema_version': SCHEMA_VERSION,
            'run_id': run_id,
            'created_at': datetime.utcnow().isoformat(),
            'fingerprint_count': len(df),
            'request_count': df['request_id'].nunique(),
            'cluster_count': n_clusters,
            'noise_count': n_noise,
            'embedding_method': f'pca_{config.pca_components}_umap_2',
            'clustering_method': 'hdbscan',
            'parameters': {
                'time_window_hours': config.time_window_hours,
                'min_cluster_size': config.min_cluster_size,
                'min_samples': config.min_samples,
                'umap_neighbors': config.umap_neighbors,
                'umap_min_dist': config.umap_min_dist,
                'pca_components': config.pca_components,
            },
            'zone_distribution': {
                'syntax_floor': int(np.sum(zones == 'syntax_floor')),
                'semantic_bridge': int(np.sum(zones == 'semantic_bridge')),
                'structure_ripple': int(np.sum(zones == 'structure_ripple')),
            },
            'time_range': {
                'start': (datetime.utcnow() - timedelta(hours=config.time_window_hours)).isoformat(),
                'end': datetime.utcnow().isoformat(),
            },
        }
        write_manifest(f"{run_dir}/manifest.json", manifest)

    # Update latest symlink
    update_latest_symlink(config.output_dir, run_id)

    # Update database
    result = DiscoveryResult(
        run_id=run_id,
        fingerprint_count=len(df),
        request_count=df['request_id'].nunique(),
        cluster_count=n_clusters,
        noise_count=n_noise,
        output_dir=run_dir,
        duration_seconds=(datetime.utcnow() - start_time).total_seconds(),
    )

    update_database(config.db_path, run_id, config, result, cluster_df, proto_df)

    logger.info(f"Discovery job completed in {result.duration_seconds:.1f}s")
    logger.info(f"  Fingerprints: {result.fingerprint_count}")
    logger.info(f"  Requests: {result.request_count}")
    logger.info(f"  Clusters: {result.cluster_count}")
    logger.info(f"  Noise: {result.noise_count}")

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Run attention fingerprint discovery job',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--db', '-d',
        required=True,
        help='Path to SQLite database',
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output directory for discovery artifacts',
    )
    parser.add_argument(
        '--hours', '-H',
        type=int,
        default=24,
        help='Time window in hours (process fingerprints from last N hours)',
    )
    parser.add_argument(
        '--min-cluster-size',
        type=int,
        default=50,
        help='HDBSCAN min_cluster_size parameter',
    )
    parser.add_argument(
        '--min-samples',
        type=int,
        default=10,
        help='HDBSCAN min_samples parameter',
    )
    parser.add_argument(
        '--umap-neighbors',
        type=int,
        default=15,
        help='UMAP n_neighbors parameter',
    )
    parser.add_argument(
        '--umap-min-dist',
        type=float,
        default=0.1,
        help='UMAP min_dist parameter',
    )
    parser.add_argument(
        '--pca-components',
        type=int,
        default=50,
        help='Number of PCA components (intermediate step)',
    )
    parser.add_argument(
        '--prototypes',
        type=int,
        default=5,
        help='Number of prototypes per cluster',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100000,
        help='Batch size for loading fingerprints',
    )

    # Long-running mode arguments
    parser.add_argument(
        '--long-running',
        action='store_true',
        help='Enable long-running mode with checkpointing and live updates',
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Run ID to resume from (long-running mode only)',
    )
    parser.add_argument(
        '--websocket-port',
        type=int,
        default=9010,
        help='WebSocket port for live updates (0 to disable)',
    )
    parser.add_argument(
        '--max-memory-gb',
        type=float,
        default=8.0,
        help='Maximum memory usage in GB',
    )
    parser.add_argument(
        '--max-hours',
        type=float,
        help='Maximum runtime in hours (for time-limited runs)',
    )
    parser.add_argument(
        '--checkpoint-db',
        type=str,
        help='Separate database for checkpoints (defaults to --db)',
    )
    parser.add_argument(
        '--zone-thresholds',
        type=str,
        help='Path to zone thresholds JSON file',
    )
    parser.add_argument(
        '--umap-sample-size',
        type=int,
        default=50000,
        help='Sample size for UMAP fitting (long-running mode)',
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.db):
        logger.error(f"Database not found: {args.db}")
        sys.exit(1)

    # Check for long-running mode
    if args.long_running or args.resume:
        if not HAS_COORDINATOR:
            logger.error(
                "Coordinator module not available. "
                "Ensure coordinator.py is in the discovery package."
            )
            sys.exit(1)

        import asyncio

        # Run with coordinator
        logger.info("Running in long-running mode with coordinator")

        result = asyncio.run(run_discovery_coordinated(
            db_path=args.db,
            output_dir=args.output,
            resume_from=args.resume,
            websocket_port=args.websocket_port,
            max_memory_gb=args.max_memory_gb,
            chunk_size=args.batch_size,
            umap_sample_size=args.umap_sample_size,
            zone_thresholds_path=args.zone_thresholds,
            max_runtime_hours=args.max_hours,
        ))

        if result.success:
            print(f"\nDiscovery completed successfully!")
            print(f"  Run ID: {result.run_id}")
            print(f"  Duration: {result.total_duration_seconds:.1f}s")
            print(f"  Stages completed: {result.stages_completed}")
            print(f"  Fingerprints: {result.total_fingerprints}")
            print(f"  Clusters: {result.total_clusters}")
            print(f"  Zone distribution: {result.zone_distribution}")
        else:
            print(f"\nDiscovery failed!")
            print(f"  Error: {result.error}")
            sys.exit(1)
    else:
        # Standard mode (existing implementation)
        config = DiscoveryConfig(
            db_path=args.db,
            output_dir=args.output,
            time_window_hours=args.hours,
            min_cluster_size=args.min_cluster_size,
            min_samples=args.min_samples,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            pca_components=args.pca_components,
            prototype_count=args.prototypes,
            batch_size=args.batch_size,
        )

        try:
            result = run_discovery(config)
            print(f"\nDiscovery completed successfully!")
            print(f"  Run ID: {result.run_id}")
            print(f"  Output: {result.output_dir}")
            print(f"  Clusters: {result.cluster_count}")
            print(f"  Duration: {result.duration_seconds:.1f}s")
        except Exception as e:
            logger.exception(f"Discovery job failed: {e}")
            sys.exit(1)


if __name__ == '__main__':
    main()
