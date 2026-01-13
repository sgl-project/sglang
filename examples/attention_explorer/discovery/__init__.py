"""
Attention Fingerprint Discovery Pipeline

This module provides tools for batch processing and real-time classification
of attention fingerprints from the SGLang inference server.

Components:
- schema.sql: SQLite schema for fingerprint storage
- discovery_job.py: Batch pipeline for embedding and clustering
- classifier.py: Online classifier for real-time sidecar use
- spectral_discovery.py: Laplacian Eigenmaps for geometric memory analysis
- spectral_router.py: Query routing based on spectral coherence

Usage:
    # Initialize database
    sqlite3 fingerprints.db < schema.sql

    # Run discovery job (hourly/daily)
    python discovery_job.py --db fingerprints.db --output ./discovery_outputs

    # Use classifier in sidecar
    from discovery import OnlineClassifier, SidecarClassifier

    classifier = SidecarClassifier('./discovery_outputs')
    result = classifier.classify(fingerprint_vector)

    # Use spectral router for model selection
    from discovery import SpectralRouter, create_router_from_fingerprints

    router = create_router_from_fingerprints(fingerprints)
    decision = router.route(query_fingerprint)
"""

from .classifier import (
    ClassificationResult,
    ClusterInfo,
    OnlineClassifier,
    SidecarClassifier,
)
from .compass_router import (
    COMPASS_GLOSSARY,
    CompassAnalyzer,
    CompassHeading,
    CompassReading,
    CompassRouter,
    CompassRouterConfig,
    CompassRoutingDecision,
    RoutingTier,
    SinqAnchor,
    analyze_attention_compass,
    create_compass_router,
)
from .manifold_firewall import (
    FIREWALL_GLOSSARY,
    ZONE_TRANSITIONS,
    AlertSeverity,
    DriftEvent,
    DriftType,
    FirewallCheckResult,
    FirewallConfig,
    FirewallState,
    ManifoldBatchAnalyzer,
    ManifoldFirewall,
    ManifoldPoint,
    create_firewall,
)
from .rope_derotation import (
    GLOSSARY as ATTENTION_GLOSSARY,  # Fingerprint integration (schema v2)
)
from .rope_derotation import (
    INTERPRETATION_TEMPLATES,
    AttentionMode,
    DerotatedAttention,
    RoPEConfig,
    RoPEDerotator,
    compute_rotational_variance_batch,
    compute_rotational_variance_for_fingerprint,
    explain_attention_step,
    extend_fingerprint_with_rotational_variance,
    extend_fingerprints_batch,
    get_glossary,
    get_term_explanation,
)
from .spectral_discovery import (
    FrequencyBandAnalyzer,
    SpectralAnalysis,
    SpectralCoherence,
    SpectralDiscoveryConfig,
    SpectralManifoldDiscovery,
    SpectralMode,
)
from .spectral_router import (
    AdaptiveSpectralRouter,
    ModelSize,
    RouterConfig,
    RoutingDecision,
    SpectralRouter,
    create_router_from_fingerprints,
)

__all__ = [
    # Classifier
    "ClassificationResult",
    "ClusterInfo",
    "OnlineClassifier",
    "SidecarClassifier",
    # Spectral Discovery
    "SpectralManifoldDiscovery",
    "SpectralDiscoveryConfig",
    "SpectralAnalysis",
    "SpectralCoherence",
    "SpectralMode",
    "FrequencyBandAnalyzer",
    # Spectral Router
    "SpectralRouter",
    "AdaptiveSpectralRouter",
    "RouterConfig",
    "RoutingDecision",
    "ModelSize",
    "create_router_from_fingerprints",
]

__version__ = "1.1.0"
