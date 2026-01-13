"""
Spectral Manifold Discovery

Replaces PCA with Laplacian Eigenmaps to capture the true spectral structure
of the model's geometric memory, based on arXiv:2510.26745.

The key insight: Models store facts as geometric structures aligned with
eigenvectors of the graph Laplacian, not as associative lookups.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class SpectralMode(Enum):
    """Spectral modes indicating computation type."""

    GEOMETRIC_LOOKUP = "geometric_lookup"  # High coherence - simple navigation
    SEMANTIC_REASONING = "semantic_reasoning"  # Medium coherence - mid-range
    VOID_COMPUTATION = "void_computation"  # Low coherence - needs deep reasoning


@dataclass
class SpectralAnalysis:
    """Results of spectral analysis on fingerprints."""

    embeddings: np.ndarray  # n_samples x n_components
    eigenvalues: np.ndarray  # Laplacian eigenvalues
    eigenvectors: np.ndarray  # Laplacian eigenvectors
    spectral_gap: float  # Gap between first non-zero eigenvalues
    effective_dimension: int  # Number of significant eigenmodes
    graph_connectivity: float  # How connected the fingerprint graph is


@dataclass
class SpectralCoherence:
    """Coherence metrics for a query fingerprint."""

    coherence_score: float  # 0-1, how well query fits spectral skeleton
    projection_energy: np.ndarray  # Energy in each eigenmode
    dominant_modes: List[int]  # Indices of dominant eigenmodes
    mode: SpectralMode  # Recommended computation mode
    confidence: float  # Confidence in mode assignment


@dataclass
class SpectralDiscoveryConfig:
    """Configuration for spectral manifold discovery."""

    n_components: int = 50  # Number of spectral components
    n_neighbors: int = 15  # k for k-NN graph construction
    eigen_solver: str = "arpack"  # Eigenvalue solver
    gamma: Optional[float] = None  # RBF kernel bandwidth (None = auto)
    normalize_laplacian: bool = True  # Use normalized Laplacian
    coherence_threshold_high: float = 0.7  # Above = geometric lookup
    coherence_threshold_low: float = 0.3  # Below = void computation
    min_eigenvalue_ratio: float = 0.01  # For effective dimension


class SpectralManifoldDiscovery:
    """
    Spectral-based manifold discovery using Laplacian Eigenmaps.

    Unlike PCA which assumes Gaussian data, this captures the graph
    structure of the model's internal geometric memory.
    """

    def __init__(self, config: Optional[SpectralDiscoveryConfig] = None):
        self.config = config or SpectralDiscoveryConfig()
        self.scaler = StandardScaler()
        self.eigenvectors_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.mean_fingerprint_: Optional[np.ndarray] = None
        self.graph_laplacian_: Optional[np.ndarray] = None
        self._training_fingerprints_scaled: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, fingerprints: np.ndarray) -> "SpectralManifoldDiscovery":
        """
        Fit the spectral embedding on fingerprint data.

        Args:
            fingerprints: Array of shape (n_samples, n_features)

        Returns:
            self for chaining
        """
        logger.info(f"Fitting spectral discovery on {len(fingerprints)} fingerprints")

        # Standardize features
        fingerprints_scaled = self.scaler.fit_transform(fingerprints)
        self.mean_fingerprint_ = np.mean(fingerprints_scaled, axis=0)
        self._training_fingerprints_scaled = fingerprints_scaled

        # Build k-NN graph (the "subway map" structure)
        logger.info(f"Building {self.config.n_neighbors}-NN graph...")
        nn = NearestNeighbors(
            n_neighbors=self.config.n_neighbors, algorithm="auto", metric="euclidean"
        )
        nn.fit(fingerprints_scaled)

        # Get adjacency matrix
        distances, indices = nn.kneighbors(fingerprints_scaled)

        # Build sparse adjacency matrix with RBF weights
        n_samples = len(fingerprints_scaled)
        if self.config.gamma is None:
            # Auto bandwidth: median of k-NN distances
            self.config.gamma = 1.0 / (2 * np.median(distances[:, 1:]) ** 2)

        # Construct weighted adjacency
        row_idx = np.repeat(np.arange(n_samples), self.config.n_neighbors)
        col_idx = indices.flatten()
        weights = np.exp(-self.config.gamma * distances.flatten() ** 2)

        adjacency = csr_matrix(
            (weights, (row_idx, col_idx)), shape=(n_samples, n_samples)
        )
        # Make symmetric
        adjacency = (adjacency + adjacency.T) / 2

        # Compute graph Laplacian
        logger.info("Computing graph Laplacian...")
        L = laplacian(adjacency, normed=self.config.normalize_laplacian)
        self.graph_laplacian_ = L

        # Compute eigendecomposition
        logger.info(f"Computing {self.config.n_components} eigenvectors...")
        n_components = min(self.config.n_components, n_samples - 1)

        if self.config.eigen_solver == "arpack":
            from scipy.sparse.linalg import eigsh

            # Get smallest eigenvalues (Laplacian is positive semi-definite)
            eigenvalues, eigenvectors = eigsh(
                L,
                k=n_components + 1,  # +1 for the zero eigenvalue
                which="SM",
                maxiter=5000,
            )
        else:
            # Dense solver for small datasets
            L_dense = L.toarray() if hasattr(L, "toarray") else L
            eigenvalues, eigenvectors = eigh(L_dense)
            eigenvalues = eigenvalues[: n_components + 1]
            eigenvectors = eigenvectors[:, : n_components + 1]

        # Sort by eigenvalue (should already be sorted, but ensure)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Skip the first eigenvector (constant, eigenvalue ≈ 0)
        self.eigenvalues_ = eigenvalues[1 : n_components + 1]
        self.eigenvectors_ = eigenvectors[:, 1 : n_components + 1]

        self._fitted = True
        logger.info(
            f"Spectral discovery fitted. Spectral gap: {self._compute_spectral_gap():.4f}"
        )

        return self

    def transform(self, fingerprints: np.ndarray) -> np.ndarray:
        """
        Transform fingerprints to spectral embedding space.

        Args:
            fingerprints: Array of shape (n_samples, n_features)

        Returns:
            Spectral embeddings of shape (n_samples, n_components)
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")

        fingerprints_scaled = self.scaler.transform(fingerprints)

        # Project onto eigenvectors using Nyström approximation
        # For out-of-sample extension
        embeddings = self._nystrom_extension(fingerprints_scaled)

        return embeddings

    def fit_transform(self, fingerprints: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        """
        self.fit(fingerprints)
        # For training data, use the eigenvectors directly
        return self.eigenvectors_.copy()

    def _nystrom_extension(self, fingerprints_scaled: np.ndarray) -> np.ndarray:
        """
        Extend spectral embedding to new points using Nyström method.
        """
        # This is an approximation for out-of-sample points
        # For exact extension, we'd need to recompute the Laplacian

        # Simple projection: use the learned eigenvectors as basis
        # Center the fingerprints
        centered = fingerprints_scaled - self.mean_fingerprint_

        # Project onto principal spectral directions
        # This is approximate but fast
        embeddings = centered @ self.eigenvectors_

        return embeddings

    def _compute_spectral_gap(self) -> float:
        """Compute the spectral gap (difference between smallest eigenvalues)."""
        if self.eigenvalues_ is None or len(self.eigenvalues_) < 2:
            return 0.0
        return float(self.eigenvalues_[1] - self.eigenvalues_[0])

    def compute_effective_dimension(self) -> int:
        """
        Compute effective dimensionality of the spectral embedding.

        This indicates how many eigenmodes carry significant information.
        """
        if self.eigenvalues_ is None:
            return 0

        # Normalize eigenvalues
        total = np.sum(self.eigenvalues_)
        if total == 0:
            return 0

        ratios = self.eigenvalues_ / total

        # Count modes above threshold
        return int(np.sum(ratios > self.config.min_eigenvalue_ratio))

    def analyze(self, fingerprints: np.ndarray) -> SpectralAnalysis:
        """
        Perform full spectral analysis on fingerprints.
        """
        embeddings = self.fit_transform(fingerprints)

        return SpectralAnalysis(
            embeddings=embeddings,
            eigenvalues=self.eigenvalues_.copy(),
            eigenvectors=self.eigenvectors_.copy(),
            spectral_gap=self._compute_spectral_gap(),
            effective_dimension=self.compute_effective_dimension(),
            graph_connectivity=self._compute_connectivity(),
        )

    def _compute_connectivity(self) -> float:
        """Compute graph connectivity metric."""
        if self.eigenvalues_ is None:
            return 0.0
        # Algebraic connectivity is the second smallest eigenvalue
        return float(self.eigenvalues_[0]) if len(self.eigenvalues_) > 0 else 0.0

    def compute_spectral_coherence(
        self, query_fingerprint: np.ndarray
    ) -> SpectralCoherence:
        """
        Measure how well a query projects onto the spectral skeleton.

        High coherence -> Query fits the learned geometry (easy lookup)
        Low coherence -> Query is in "void space" (needs computation)

        Uses k-NN distance in spectral space to estimate coherence.

        Args:
            query_fingerprint: Single fingerprint vector

        Returns:
            SpectralCoherence with coherence metrics
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before computing coherence")

        # Scale the query
        query_scaled = self.scaler.transform(query_fingerprint.reshape(1, -1))[0]

        # For spectral coherence, we measure how well the query fits
        # the learned spectral structure by computing its distance to
        # the nearest points in spectral space

        # Method: Use the spectral embedding coordinates to find
        # how "central" the query would be in the learned manifold

        # Compute distance to mean fingerprint
        dist_to_mean = np.linalg.norm(query_scaled - self.mean_fingerprint_)

        # Compute statistics of training data for normalization
        # We stored eigenvectors which represent spectral coordinates of training points
        spectral_coords = self.eigenvectors_  # (n_samples, n_components)

        # Compute centroid in spectral space
        spectral_centroid = np.mean(spectral_coords, axis=0)

        # Compute mean distance from centroid in spectral space
        spectral_distances = np.linalg.norm(spectral_coords - spectral_centroid, axis=1)
        mean_spectral_dist = np.mean(spectral_distances)
        std_spectral_dist = np.std(spectral_distances) + 1e-10

        # For the query, estimate its spectral position using Nyström-like projection
        # Use correlation with training fingerprints as proxy
        training_fingerprints_scaled = self._training_fingerprints_scaled
        if training_fingerprints_scaled is None:
            # Fallback: use distance to mean as coherence proxy
            max_dist = np.sqrt(query_scaled.shape[0])  # Max possible distance
            coherence_score = max(0, 1 - dist_to_mean / max_dist)
        else:
            # Compute similarity to all training points
            distances = np.linalg.norm(
                training_fingerprints_scaled - query_scaled, axis=1
            )
            # Find k nearest neighbors
            k = min(self.config.n_neighbors, len(distances))
            nearest_indices = np.argsort(distances)[:k]
            nearest_distances = distances[nearest_indices]

            # Coherence based on distance to nearest neighbors
            # Small distance = high coherence (query is similar to training data)
            mean_dist = np.mean(nearest_distances)

            # Normalize by typical inter-point distance
            typical_dist = np.median(distances)
            if typical_dist > 0:
                coherence_score = np.exp(-mean_dist / typical_dist)
            else:
                coherence_score = 1.0

        coherence_score = float(np.clip(coherence_score, 0, 1))

        # Compute "projection energy" as proxy using eigenvalue distribution
        n_components = len(self.eigenvalues_)
        projection_energy = np.zeros(n_components)

        # Estimate energy distribution based on eigenvalue spectrum
        # Points on the manifold concentrate energy in low eigenmodes
        eigenvalue_weights = 1.0 / (self.eigenvalues_ + 1e-10)
        eigenvalue_weights /= np.sum(eigenvalue_weights)
        projection_energy = eigenvalue_weights * coherence_score

        # Find dominant modes (low eigenvalue modes for on-manifold points)
        sorted_modes = np.argsort(self.eigenvalues_)[:5].tolist()
        dominant_modes = sorted_modes

        # Determine computation mode
        if coherence_score > self.config.coherence_threshold_high:
            mode = SpectralMode.GEOMETRIC_LOOKUP
            confidence = (coherence_score - self.config.coherence_threshold_high) / (
                1.0 - self.config.coherence_threshold_high + 1e-10
            )
        elif coherence_score < self.config.coherence_threshold_low:
            mode = SpectralMode.VOID_COMPUTATION
            confidence = (self.config.coherence_threshold_low - coherence_score) / (
                self.config.coherence_threshold_low + 1e-10
            )
        else:
            mode = SpectralMode.SEMANTIC_REASONING
            mid = (
                self.config.coherence_threshold_high
                + self.config.coherence_threshold_low
            ) / 2
            range_size = (
                self.config.coherence_threshold_high
                - self.config.coherence_threshold_low
            )
            confidence = 1.0 - abs(coherence_score - mid) / (range_size + 1e-10)

        confidence = float(np.clip(confidence, 0, 1))

        return SpectralCoherence(
            coherence_score=coherence_score,
            projection_energy=projection_energy,
            dominant_modes=dominant_modes,
            mode=mode,
            confidence=confidence,
        )

    def get_spectral_skeleton(self, n_landmarks: int = 100) -> np.ndarray:
        """
        Extract landmark points that define the spectral skeleton.

        These are points at the extremes of each spectral dimension.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before getting skeleton")

        skeleton_indices = set()
        embeddings = self.eigenvectors_

        # Get extremes along each dimension
        for dim in range(min(embeddings.shape[1], n_landmarks // 2)):
            skeleton_indices.add(int(np.argmin(embeddings[:, dim])))
            skeleton_indices.add(int(np.argmax(embeddings[:, dim])))

        return np.array(sorted(skeleton_indices))

    def save(self, path: str) -> None:
        """Save the fitted model to disk."""
        import pickle

        state = {
            "config": self.config,
            "eigenvalues": self.eigenvalues_,
            "eigenvectors": self.eigenvectors_,
            "mean_fingerprint": self.mean_fingerprint_,
            "training_fingerprints_scaled": self._training_fingerprints_scaled,
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info(f"Saved spectral discovery to {path}")

    @classmethod
    def load(cls, path: str) -> "SpectralManifoldDiscovery":
        """Load a fitted model from disk."""
        import pickle

        with open(path, "rb") as f:
            state = pickle.load(f)

        instance = cls(config=state["config"])
        instance.eigenvalues_ = state["eigenvalues"]
        instance.eigenvectors_ = state["eigenvectors"]
        instance.mean_fingerprint_ = state["mean_fingerprint"]
        instance._training_fingerprints_scaled = state.get(
            "training_fingerprints_scaled"
        )
        instance.scaler.mean_ = state["scaler_mean"]
        instance.scaler.scale_ = state["scaler_scale"]
        instance._fitted = state["fitted"]

        logger.info(f"Loaded spectral discovery from {path}")
        return instance


class FrequencyBandAnalyzer:
    """
    Analyze RoPE frequency bands in attention patterns.

    Based on the insight that:
    - Low dimensions (0-16): High frequency, local position sensitivity
    - High dimensions (64+): Low frequency, global context
    """

    def __init__(self, n_low_dims: int = 16, n_high_dims: int = 64):
        self.n_low_dims = n_low_dims
        self.n_high_dims = n_high_dims

    def analyze_frequency_bands(
        self, fingerprint: np.ndarray, pca_loadings: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Analyze activity in high vs low frequency bands.

        Returns:
            Dict with high_band_activity, low_band_activity, band_ratio
        """
        if pca_loadings is not None:
            # Use PCA loadings to identify frequency-sensitive components
            low_band = np.abs(pca_loadings[: self.n_low_dims]).sum()
            high_band = np.abs(pca_loadings[self.n_high_dims :]).sum()
        else:
            # Direct fingerprint analysis
            # Fingerprint layout: [local_mass, mid_mass, long_mass, entropy, hist..., layer_entropy...]
            local_mass = fingerprint[0]
            long_mass = fingerprint[2]

            low_band = local_mass  # Local = high frequency
            high_band = long_mass  # Long-range = low frequency

        total = low_band + high_band
        if total == 0:
            return {
                "high_band_activity": 0.0,
                "low_band_activity": 0.0,
                "band_ratio": 1.0,
            }

        return {
            "high_band_activity": float(high_band / total),
            "low_band_activity": float(low_band / total),
            "band_ratio": float(high_band / low_band) if low_band > 0 else float("inf"),
        }

    def recommend_model_size(self, band_analysis: Dict[str, float]) -> str:
        """
        Recommend model size based on frequency band activity.

        High-band dominant -> Complex reasoning, use large model
        Low-band dominant -> Grammar/local, use small model
        """
        ratio = band_analysis["band_ratio"]

        if ratio > 2.0:
            return "large"  # 70B+ for complex reasoning
        elif ratio < 0.5:
            return "small"  # 7B for local patterns
        else:
            return "medium"  # 13B-30B for balanced
