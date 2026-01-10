"""
Discovery Job Coordinator

Orchestrates the 9-stage manifold discovery pipeline with:
- Checkpointing between stages
- Memory-bounded processing
- Live WebSocket updates
- Graceful shutdown and resume
"""

import asyncio
import logging
import signal
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import json

import numpy as np
import pandas as pd

# Fingerprint dimension (must match schema)
FINGERPRINT_DIM = 20


def unpack_fingerprint(blob: bytes) -> np.ndarray:
    """Unpack fingerprint from blob to numpy array."""
    if blob is None:
        return np.zeros(FINGERPRINT_DIM, dtype=np.float32)
    return np.array(struct.unpack(f'<{FINGERPRINT_DIM}f', blob), dtype=np.float32)

from .checkpoint import (
    CheckpointManager,
    CheckpointState,
    create_checkpoint_state,
    STAGE_NAMES,
)
from .bounded_umap import MemoryBoundedUMAP, MemoryMonitor
from .websocket_server import ManifoldWebSocketServer, create_websocket_server

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CoordinatorConfig:
    """Configuration for discovery coordinator."""
    # Database
    db_path: str
    output_dir: str

    # Processing
    chunk_size: int = 10000
    max_memory_gb: float = 8.0
    umap_sample_size: int = 50000

    # Checkpointing
    checkpoint_interval_seconds: float = 300.0  # 5 minutes
    checkpoint_db_path: Optional[str] = None  # Defaults to db_path

    # WebSocket
    websocket_port: int = 9010  # 0 to disable
    websocket_enabled: bool = True

    # Zone thresholds
    zone_thresholds_path: Optional[str] = None

    # Resume
    resume_run_id: Optional[str] = None

    # Timing
    max_runtime_hours: Optional[float] = None  # None for unlimited


@dataclass
class StageResult:
    """Result from a pipeline stage."""
    stage: int
    stage_name: str
    success: bool
    duration_seconds: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class DiscoveryResult:
    """Final result of discovery run."""
    run_id: str
    success: bool
    total_duration_seconds: float
    stages_completed: int
    total_fingerprints: int
    total_clusters: int
    zone_distribution: Dict[str, int]
    output_paths: Dict[str, str]
    stage_results: List[StageResult]
    error: Optional[str] = None


# =============================================================================
# COORDINATOR
# =============================================================================

class DiscoveryJobCoordinator:
    """
    Orchestrates the complete manifold discovery pipeline.

    Pipeline stages:
    0. extract     - Load fingerprints from database
    1. standardize - Standardize features
    2. pca         - PCA dimensionality reduction
    3. umap        - UMAP embedding
    4. cluster     - HDBSCAN clustering
    5. zones       - Zone assignment
    6. metadata    - Compute cluster metadata
    7. prototypes  - Select cluster prototypes
    8. export      - Export final artifacts
    9. complete    - Finalization

    Usage:
        config = CoordinatorConfig(db_path='fp.db', output_dir='./outputs')
        coordinator = DiscoveryJobCoordinator(config)

        # Run (blocking)
        result = asyncio.run(coordinator.run())

        # Resume interrupted run
        result = asyncio.run(coordinator.run(resume_from='run-123'))
    """

    def __init__(self, config: CoordinatorConfig):
        """
        Initialize coordinator.

        Args:
            config: CoordinatorConfig instance
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint manager
        checkpoint_db = config.checkpoint_db_path or config.db_path
        self._checkpoint_mgr = CheckpointManager(
            db_path=checkpoint_db,
            output_dir=str(self.output_dir),
            checkpoint_interval_seconds=config.checkpoint_interval_seconds,
        )

        # WebSocket server (created on run)
        self._ws_server: Optional[ManifoldWebSocketServer] = None

        # Memory monitor
        self._memory_monitor = MemoryMonitor(
            max_memory_gb=config.max_memory_gb,
            warning_threshold=0.8,
            critical_threshold=0.95,
        )

        # State
        self._run_id: str = ""
        self._current_state: Optional[CheckpointState] = None
        self._running = False
        self._shutdown_requested = False
        self._start_time: float = 0.0

        # Stage data (passed between stages)
        self._fingerprints: Optional[pd.DataFrame] = None
        self._scaled_features: Optional[np.ndarray] = None
        self._pca_features: Optional[np.ndarray] = None
        self._embeddings: Optional[np.ndarray] = None
        self._cluster_labels: Optional[np.ndarray] = None
        self._zone_labels: Optional[List[str]] = None
        self._cluster_metadata: Optional[Dict[str, Any]] = None

        # Results
        self._stage_results: List[StageResult] = []

    async def run(
        self,
        resume_from: Optional[str] = None,
    ) -> DiscoveryResult:
        """
        Run the discovery pipeline.

        Args:
            resume_from: Optional run ID to resume from

        Returns:
            DiscoveryResult with final metrics
        """
        self._running = True
        self._shutdown_requested = False
        self._start_time = time.time()

        # Generate or resume run ID
        if resume_from:
            self._run_id = resume_from
            self._current_state = self._checkpoint_mgr.load_checkpoint(resume_from)
            if not self._current_state:
                logger.warning(f"No checkpoint found for {resume_from}, starting fresh")
                self._run_id = self._generate_run_id()
                start_stage = 0
            else:
                start_stage = self._current_state.stage
                logger.info(f"Resuming from stage {start_stage}: {STAGE_NAMES[start_stage]}")
        else:
            self._run_id = self._generate_run_id()
            start_stage = 0

        # Setup signal handlers
        self._setup_signal_handlers()

        # Start WebSocket server
        if self.config.websocket_enabled and self.config.websocket_port > 0:
            self._ws_server = create_websocket_server(port=self.config.websocket_port)
            if self._ws_server:
                await self._ws_server.start()
                await self._ws_server.broadcast_run_start(self._run_id, {
                    'db_path': self.config.db_path,
                    'output_dir': str(self.output_dir),
                    'start_stage': start_stage,
                })

        try:
            # Run pipeline stages
            for stage in range(start_stage, len(STAGE_NAMES)):
                if self._shutdown_requested:
                    logger.info("Shutdown requested, saving checkpoint")
                    await self._save_checkpoint(stage)
                    break

                if self._check_timeout():
                    logger.info("Timeout reached, saving checkpoint")
                    await self._save_checkpoint(stage)
                    break

                result = await self._run_stage(stage)
                self._stage_results.append(result)

                if not result.success:
                    logger.error(f"Stage {stage} failed: {result.error}")
                    await self._broadcast_error(f"Stage {stage} failed: {result.error}")
                    break

            # Build final result
            return self._build_result()

        finally:
            # Cleanup
            if self._ws_server:
                await self._ws_server.stop()
            self._running = False

    async def _run_stage(self, stage: int) -> StageResult:
        """
        Run a single pipeline stage.

        Args:
            stage: Stage number (0-9)

        Returns:
            StageResult with metrics
        """
        stage_name = STAGE_NAMES[stage]
        start_time = time.time()

        logger.info(f"Starting stage {stage}: {stage_name}")
        await self._broadcast_stage_start(stage, stage_name)

        try:
            # Dispatch to stage handler
            if stage == 0:
                await self._stage_extract()
            elif stage == 1:
                await self._stage_standardize()
            elif stage == 2:
                await self._stage_pca()
            elif stage == 3:
                await self._stage_umap()
            elif stage == 4:
                await self._stage_cluster()
            elif stage == 5:
                await self._stage_zones()
            elif stage == 6:
                await self._stage_metadata()
            elif stage == 7:
                await self._stage_prototypes()
            elif stage == 8:
                await self._stage_export()
            elif stage == 9:
                await self._stage_complete()

            duration = time.time() - start_time

            # Save checkpoint
            await self._save_checkpoint(stage + 1)

            await self._broadcast_stage_complete(stage, stage_name)

            return StageResult(
                stage=stage,
                stage_name=stage_name,
                success=True,
                duration_seconds=duration,
                metrics=self._get_stage_metrics(stage),
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(f"Stage {stage} error")

            return StageResult(
                stage=stage,
                stage_name=stage_name,
                success=False,
                duration_seconds=duration,
                error=str(e),
            )

    # =========================================================================
    # STAGE IMPLEMENTATIONS
    # =========================================================================

    async def _stage_extract(self) -> None:
        """Stage 0: Extract fingerprints from database."""
        import sqlite3

        logger.info("Extracting fingerprints from database")

        # Load from resumed state if available
        if self._current_state and self._current_state.embeddings_path:
            path = self.output_dir / self._current_state.embeddings_path
            if path.exists():
                self._fingerprints = pd.read_parquet(path)
                logger.info(f"Loaded {len(self._fingerprints)} fingerprints from checkpoint")
                return

        # Load from database in chunks
        conn = sqlite3.connect(self.config.db_path)
        chunks = []
        total_loaded = 0

        query = """
            SELECT id, fingerprint, request_id
            FROM fingerprints
            ORDER BY id
        """

        for chunk in pd.read_sql_query(query, conn, chunksize=self.config.chunk_size):
            chunks.append(chunk)
            total_loaded += len(chunk)

            await self._broadcast_progress(
                stage=0,
                stage_name='extract',
                items_processed=total_loaded,
                total_items=0,  # Unknown until complete
                percent_complete=0,
            )

            # Check memory
            if self._memory_monitor.get_usage_gb() > self.config.max_memory_gb * 0.9:
                logger.warning("Memory pressure during extraction, stopping early")
                break

        conn.close()

        self._fingerprints = pd.concat(chunks, ignore_index=True)
        logger.info(f"Extracted {len(self._fingerprints)} fingerprints")

    async def _stage_standardize(self) -> None:
        """Stage 1: Standardize features."""
        from sklearn.preprocessing import StandardScaler

        logger.info("Standardizing features")

        if self._fingerprints is None:
            raise ValueError("No fingerprints loaded")

        # Parse fingerprint BLOB to feature array
        features_list = []
        for idx, row in self._fingerprints.iterrows():
            fp = row['fingerprint']
            if isinstance(fp, bytes):
                # Unpack from BLOB format (packed float32)
                fp = unpack_fingerprint(fp)
            elif isinstance(fp, str):
                # Legacy JSON format
                fp = json.loads(fp)
            features_list.append(fp)

            if idx % 10000 == 0:
                await self._broadcast_progress(
                    stage=1,
                    stage_name='standardize',
                    items_processed=idx,
                    total_items=len(self._fingerprints),
                    percent_complete=(idx / len(self._fingerprints)) * 100,
                )

        features_array = np.array(features_list, dtype=np.float32)

        # Standardize
        scaler = StandardScaler()
        self._scaled_features = scaler.fit_transform(features_array)

        logger.info(f"Standardized features shape: {self._scaled_features.shape}")

    async def _stage_pca(self) -> None:
        """Stage 2: PCA dimensionality reduction."""
        from sklearn.decomposition import PCA

        logger.info("Running PCA dimensionality reduction")

        if self._scaled_features is None:
            raise ValueError("No scaled features")

        # Determine components
        n_samples, n_features = self._scaled_features.shape
        n_components = min(50, n_features, n_samples)

        pca = PCA(n_components=n_components)
        self._pca_features = pca.fit_transform(self._scaled_features)

        # Report variance explained
        variance_explained = np.sum(pca.explained_variance_ratio_)
        logger.info(
            f"PCA complete: {n_components} components, "
            f"{variance_explained:.1%} variance explained"
        )

        await self._broadcast_progress(
            stage=2,
            stage_name='pca',
            items_processed=n_samples,
            total_items=n_samples,
            percent_complete=100,
        )

    async def _stage_umap(self) -> None:
        """Stage 3: UMAP embedding with memory-bounded processing."""
        logger.info("Running UMAP embedding")

        if self._pca_features is None:
            raise ValueError("No PCA features")

        # Use memory-bounded UMAP
        umap_wrapper = MemoryBoundedUMAP(
            n_components=2,
            sample_size=self.config.umap_sample_size,
            transform_chunk_size=self.config.chunk_size,
            max_memory_gb=self.config.max_memory_gb,
            n_neighbors=15,
            min_dist=0.1,
        )

        async def progress_callback(processed: int, total: int) -> None:
            await self._broadcast_progress(
                stage=3,
                stage_name='umap',
                items_processed=processed,
                total_items=total,
                percent_complete=(processed / total) * 100 if total > 0 else 0,
            )

        # Fit and transform
        self._embeddings = umap_wrapper.fit_transform_bounded(
            self._pca_features,
            progress_callback=lambda p, t: asyncio.create_task(progress_callback(p, t)),
        )

        logger.info(f"UMAP complete: {self._embeddings.shape}")

    async def _stage_cluster(self) -> None:
        """Stage 4: HDBSCAN clustering."""
        try:
            from hdbscan import HDBSCAN
        except ImportError:
            from sklearn.cluster import HDBSCAN

        logger.info("Running HDBSCAN clustering")

        if self._embeddings is None:
            raise ValueError("No embeddings")

        clusterer = HDBSCAN(
            min_cluster_size=50,
            min_samples=10,
            cluster_selection_method='eom',
        )

        self._cluster_labels = clusterer.fit_predict(self._embeddings)

        n_clusters = len(set(self._cluster_labels)) - (1 if -1 in self._cluster_labels else 0)
        n_noise = np.sum(self._cluster_labels == -1)

        logger.info(f"Clustering complete: {n_clusters} clusters, {n_noise} noise points")

        await self._broadcast_progress(
            stage=4,
            stage_name='cluster',
            items_processed=len(self._embeddings),
            total_items=len(self._embeddings),
            percent_complete=100,
        )

    async def _stage_zones(self) -> None:
        """Stage 5: Zone assignment."""
        logger.info("Assigning zones")

        if self._fingerprints is None or self._cluster_labels is None:
            raise ValueError("Missing data for zone assignment")

        # Load zone thresholds if provided
        zone_thresholds = None
        if self.config.zone_thresholds_path:
            with open(self.config.zone_thresholds_path) as f:
                data = json.load(f)
                zone_thresholds = data.get('thresholds', data)

        # Classify each fingerprint
        self._zone_labels = []

        for idx in range(len(self._fingerprints)):
            row = self._fingerprints.iloc[idx]
            fp = row['fingerprint']
            if isinstance(fp, str):
                fp = json.loads(fp)

            zone = self._classify_zone(fp, zone_thresholds)
            self._zone_labels.append(zone)

            if idx % 10000 == 0:
                await self._broadcast_progress(
                    stage=5,
                    stage_name='zones',
                    items_processed=idx,
                    total_items=len(self._fingerprints),
                    percent_complete=(idx / len(self._fingerprints)) * 100,
                )

        # Broadcast zone stats
        zone_dist = {}
        for zone in self._zone_labels:
            zone_dist[zone] = zone_dist.get(zone, 0) + 1

        if self._ws_server:
            await self._ws_server.broadcast_zone_stats(self._run_id, zone_dist)

        logger.info(f"Zone assignment complete: {zone_dist}")

    def _classify_zone(
        self,
        fingerprint: List[float],
        thresholds: Optional[Dict] = None,
    ) -> str:
        """Classify a fingerprint into a zone."""
        # Extract features from fingerprint
        # Assuming fingerprint format: [local_mass, mid_mass, long_mass, entropy, ...]
        if len(fingerprint) < 4:
            return 'semantic_bridge'

        local_mass = fingerprint[0]
        mid_mass = fingerprint[1]
        long_mass = fingerprint[2]
        entropy = fingerprint[3]

        # Use thresholds or defaults
        if thresholds is None:
            thresholds = {
                'syntax_floor': {'local_mass_min': 0.7, 'entropy_max': 2.0},
                'long_range': {'long_mass_min': 0.3, 'entropy_min': 3.0},
                'diffuse': {'entropy_min': 4.5},
            }

        sf = thresholds.get('syntax_floor', {})
        if local_mass >= sf.get('local_mass_min', 0.7) and entropy <= sf.get('entropy_max', 2.0):
            return 'syntax_floor'

        lr = thresholds.get('long_range', {})
        if long_mass >= lr.get('long_mass_min', 0.3) and entropy >= lr.get('entropy_min', 3.0):
            return 'long_range'

        df = thresholds.get('diffuse', {})
        if entropy >= df.get('entropy_min', 4.5):
            return 'diffuse'

        return 'semantic_bridge'

    async def _stage_metadata(self) -> None:
        """Stage 6: Compute cluster metadata."""
        logger.info("Computing cluster metadata")

        if self._embeddings is None or self._cluster_labels is None:
            raise ValueError("Missing data")

        unique_clusters = set(self._cluster_labels) - {-1}
        self._cluster_metadata = {}

        for cluster_id in unique_clusters:
            mask = self._cluster_labels == cluster_id
            cluster_points = self._embeddings[mask]

            centroid = np.mean(cluster_points, axis=0)
            size = int(np.sum(mask))

            # Zone distribution within cluster
            if self._zone_labels:
                cluster_zones = [self._zone_labels[i] for i in range(len(mask)) if mask[i]]
                zone_counts = {}
                for z in cluster_zones:
                    zone_counts[z] = zone_counts.get(z, 0) + 1
                dominant_zone = max(zone_counts, key=zone_counts.get)
            else:
                dominant_zone = 'unknown'

            self._cluster_metadata[int(cluster_id)] = {
                'centroid_x': float(centroid[0]),
                'centroid_y': float(centroid[1]),
                'size': size,
                'dominant_zone': dominant_zone,
            }

        # Broadcast cluster updates
        if self._ws_server:
            clusters = [
                {
                    'id': cid,
                    'centroid_x': meta['centroid_x'],
                    'centroid_y': meta['centroid_y'],
                    'size': meta['size'],
                    'zone': meta['dominant_zone'],
                }
                for cid, meta in self._cluster_metadata.items()
            ]
            await self._ws_server.broadcast_cluster_update(self._run_id, clusters)

        logger.info(f"Computed metadata for {len(self._cluster_metadata)} clusters")

    async def _stage_prototypes(self) -> None:
        """Stage 7: Select cluster prototypes."""
        logger.info("Selecting cluster prototypes")

        if self._cluster_metadata is None or self._embeddings is None:
            raise ValueError("Missing data")

        # Select fingerprint closest to centroid as prototype
        prototypes = []

        for cluster_id, meta in self._cluster_metadata.items():
            mask = self._cluster_labels == cluster_id
            cluster_indices = np.where(mask)[0]

            if len(cluster_indices) == 0:
                continue

            centroid = np.array([meta['centroid_x'], meta['centroid_y']])
            cluster_embeddings = self._embeddings[mask]

            # Find closest to centroid
            distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
            prototype_idx = cluster_indices[np.argmin(distances)]

            prototypes.append({
                'cluster_id': int(cluster_id),
                'fingerprint_idx': int(prototype_idx),
                'fingerprint_id': int(self._fingerprints.iloc[prototype_idx]['id']),
            })

        # Store prototypes
        self._prototypes = prototypes
        logger.info(f"Selected {len(prototypes)} prototypes")

    async def _stage_export(self) -> None:
        """Stage 8: Export all artifacts."""
        logger.info("Exporting artifacts")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = self.output_dir / f"discovery_{self._run_id}"
        export_dir.mkdir(parents=True, exist_ok=True)

        # Export embeddings
        if self._embeddings is not None and self._fingerprints is not None:
            df_embeddings = pd.DataFrame({
                'fingerprint_id': self._fingerprints['id'].values,
                'x': self._embeddings[:, 0],
                'y': self._embeddings[:, 1],
                'cluster_id': self._cluster_labels,
                'zone': self._zone_labels,
            })
            embeddings_path = export_dir / 'embeddings.parquet'
            df_embeddings.to_parquet(embeddings_path, index=False)

        # Export cluster metadata
        if self._cluster_metadata:
            metadata_path = export_dir / 'clusters.json'
            with open(metadata_path, 'w') as f:
                json.dump(self._cluster_metadata, f, indent=2)

        # Export prototypes
        if hasattr(self, '_prototypes'):
            prototypes_path = export_dir / 'prototypes.json'
            with open(prototypes_path, 'w') as f:
                json.dump(self._prototypes, f, indent=2)

        # Export manifest
        manifest = {
            'run_id': self._run_id,
            'timestamp': timestamp,
            'config': {
                'db_path': self.config.db_path,
                'chunk_size': self.config.chunk_size,
                'umap_sample_size': self.config.umap_sample_size,
            },
            'metrics': {
                'total_fingerprints': len(self._fingerprints) if self._fingerprints is not None else 0,
                'total_clusters': len(self._cluster_metadata) if self._cluster_metadata else 0,
                'zone_distribution': self._get_zone_distribution(),
            },
            'files': {
                'embeddings': 'embeddings.parquet',
                'clusters': 'clusters.json',
                'prototypes': 'prototypes.json',
            },
        }
        manifest_path = export_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        self._export_dir = export_dir
        logger.info(f"Exported artifacts to {export_dir}")

    async def _stage_complete(self) -> None:
        """Stage 9: Finalization."""
        logger.info("Finalizing discovery run")

        # Clear checkpoint (successful completion)
        self._checkpoint_mgr.clear_checkpoint(self._run_id)

        # Broadcast completion
        if self._ws_server:
            await self._ws_server.broadcast_run_complete(self._run_id, {
                'total_fingerprints': len(self._fingerprints) if self._fingerprints is not None else 0,
                'total_clusters': len(self._cluster_metadata) if self._cluster_metadata else 0,
                'duration_seconds': time.time() - self._start_time,
            })

        logger.info(f"Discovery run {self._run_id} complete")

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _setup_signal_handlers(self) -> None:
        """Setup graceful shutdown signal handlers."""
        def handler(signum, frame):
            logger.info(f"Received signal {signum}, requesting shutdown")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _check_timeout(self) -> bool:
        """Check if max runtime exceeded."""
        if self.config.max_runtime_hours is None:
            return False

        elapsed_hours = (time.time() - self._start_time) / 3600
        return elapsed_hours >= self.config.max_runtime_hours

    async def _save_checkpoint(self, stage: int) -> None:
        """Save checkpoint at current stage."""
        state = create_checkpoint_state(
            run_id=self._run_id,
            stage=stage,
            total_fingerprints=len(self._fingerprints) if self._fingerprints is not None else 0,
            elapsed_seconds=time.time() - self._start_time,
        )

        # Save partial artifacts for resume
        if self._scaled_features is not None and stage > 1:
            path = self._checkpoint_mgr.save_partial_artifact(
                self._run_id, 'scaled_features',
                pd.DataFrame(self._scaled_features), 'parquet'
            )
            state.embeddings_path = path

        self._checkpoint_mgr.save_checkpoint(state, force=True)
        self._current_state = state

    def _get_stage_metrics(self, stage: int) -> Dict[str, Any]:
        """Get metrics for a completed stage."""
        metrics = {
            'memory_gb': self._memory_monitor.get_usage_gb(),
        }

        if stage == 0 and self._fingerprints is not None:
            metrics['fingerprints_loaded'] = len(self._fingerprints)
        elif stage == 3 and self._embeddings is not None:
            metrics['embeddings_computed'] = len(self._embeddings)
        elif stage == 4 and self._cluster_labels is not None:
            n_clusters = len(set(self._cluster_labels)) - (1 if -1 in self._cluster_labels else 0)
            metrics['clusters_found'] = n_clusters

        return metrics

    def _get_zone_distribution(self) -> Dict[str, int]:
        """Get zone distribution from labels."""
        if not self._zone_labels:
            return {}

        dist = {}
        for zone in self._zone_labels:
            dist[zone] = dist.get(zone, 0) + 1
        return dist

    def _build_result(self) -> DiscoveryResult:
        """Build final discovery result."""
        return DiscoveryResult(
            run_id=self._run_id,
            success=all(r.success for r in self._stage_results),
            total_duration_seconds=time.time() - self._start_time,
            stages_completed=len([r for r in self._stage_results if r.success]),
            total_fingerprints=len(self._fingerprints) if self._fingerprints is not None else 0,
            total_clusters=len(self._cluster_metadata) if self._cluster_metadata else 0,
            zone_distribution=self._get_zone_distribution(),
            output_paths={
                'export_dir': str(getattr(self, '_export_dir', '')),
            },
            stage_results=self._stage_results,
            error=next((r.error for r in self._stage_results if r.error), None),
        )

    # =========================================================================
    # BROADCAST METHODS
    # =========================================================================

    async def _broadcast_progress(
        self,
        stage: int,
        stage_name: str,
        items_processed: int,
        total_items: int,
        percent_complete: float,
    ) -> None:
        """Broadcast progress update."""
        if not self._ws_server:
            return

        # Calculate ETA
        elapsed = time.time() - self._start_time
        if percent_complete > 0:
            eta = elapsed * (100 - percent_complete) / percent_complete
        else:
            eta = None

        await self._ws_server.broadcast_progress(
            run_id=self._run_id,
            stage=stage,
            stage_name=stage_name,
            percent_complete=percent_complete,
            items_processed=items_processed,
            total_items=total_items,
            eta_seconds=eta,
            memory_used_gb=self._memory_monitor.get_usage_gb(),
        )

    async def _broadcast_stage_start(self, stage: int, stage_name: str) -> None:
        """Broadcast stage start."""
        if self._ws_server:
            await self._ws_server.broadcast_stage_start(self._run_id, stage, stage_name)

    async def _broadcast_stage_complete(self, stage: int, stage_name: str) -> None:
        """Broadcast stage completion."""
        if self._ws_server:
            await self._ws_server.broadcast_stage_complete(self._run_id, stage, stage_name)

    async def _broadcast_error(self, error: str) -> None:
        """Broadcast error."""
        if self._ws_server:
            await self._ws_server.broadcast_run_error(self._run_id, error)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

async def run_discovery(
    db_path: str,
    output_dir: str,
    resume_from: Optional[str] = None,
    websocket_port: int = 9010,
    max_memory_gb: float = 8.0,
    chunk_size: int = 10000,
    umap_sample_size: int = 50000,
    zone_thresholds_path: Optional[str] = None,
    max_runtime_hours: Optional[float] = None,
) -> DiscoveryResult:
    """
    Run discovery pipeline with given configuration.

    Args:
        db_path: Path to fingerprints database
        output_dir: Output directory for artifacts
        resume_from: Optional run ID to resume
        websocket_port: WebSocket port (0 to disable)
        max_memory_gb: Memory limit
        chunk_size: Processing chunk size
        umap_sample_size: UMAP fit sample size
        zone_thresholds_path: Path to zone thresholds JSON
        max_runtime_hours: Maximum runtime in hours

    Returns:
        DiscoveryResult
    """
    config = CoordinatorConfig(
        db_path=db_path,
        output_dir=output_dir,
        websocket_port=websocket_port,
        max_memory_gb=max_memory_gb,
        chunk_size=chunk_size,
        umap_sample_size=umap_sample_size,
        zone_thresholds_path=zone_thresholds_path,
        max_runtime_hours=max_runtime_hours,
    )

    coordinator = DiscoveryJobCoordinator(config)
    return await coordinator.run(resume_from=resume_from)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run manifold discovery pipeline")
    parser.add_argument("--db", required=True, help="Path to fingerprints database")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--resume", help="Run ID to resume from")
    parser.add_argument("--websocket-port", type=int, default=9010, help="WebSocket port (0 to disable)")
    parser.add_argument("--max-memory-gb", type=float, default=8.0, help="Memory limit in GB")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Processing chunk size")
    parser.add_argument("--umap-sample-size", type=int, default=50000, help="UMAP fit sample size")
    parser.add_argument("--zone-thresholds", help="Path to zone thresholds JSON")
    parser.add_argument("--max-hours", type=float, help="Maximum runtime in hours")

    args = parser.parse_args()

    result = asyncio.run(run_discovery(
        db_path=args.db,
        output_dir=args.output,
        resume_from=args.resume,
        websocket_port=args.websocket_port,
        max_memory_gb=args.max_memory_gb,
        chunk_size=args.chunk_size,
        umap_sample_size=args.umap_sample_size,
        zone_thresholds_path=args.zone_thresholds,
        max_runtime_hours=args.max_hours,
    ))

    print(f"\nDiscovery {'COMPLETE' if result.success else 'FAILED'}")
    print(f"  Run ID: {result.run_id}")
    print(f"  Duration: {result.total_duration_seconds:.1f}s")
    print(f"  Stages completed: {result.stages_completed}")
    print(f"  Fingerprints: {result.total_fingerprints}")
    print(f"  Clusters: {result.total_clusters}")
    print(f"  Zones: {result.zone_distribution}")

    if result.error:
        print(f"  Error: {result.error}")
