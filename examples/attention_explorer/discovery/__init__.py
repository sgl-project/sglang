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

from .spectral_discovery import (
    SpectralManifoldDiscovery,
    SpectralDiscoveryConfig,
    SpectralAnalysis,
    SpectralCoherence,
    SpectralMode,
    FrequencyBandAnalyzer,
)

from .spectral_router import (
    SpectralRouter,
    AdaptiveSpectralRouter,
    RouterConfig,
    RoutingDecision,
    ModelSize,
    create_router_from_fingerprints,
)

from .rope_derotation import (
    RoPEDerotator,
    RoPEConfig,
    DerotatedAttention,
    AttentionMode,
    GLOSSARY as ATTENTION_GLOSSARY,
    INTERPRETATION_TEMPLATES,
    get_glossary,
    get_term_explanation,
    explain_attention_step,
)

from .compass_router import (
    CompassRouter,
    CompassRouterConfig,
    CompassAnalyzer,
    CompassReading,
    CompassRoutingDecision,
    CompassHeading,
    RoutingTier,
    SinqAnchor,
    COMPASS_GLOSSARY,
    create_compass_router,
    analyze_attention_compass,
)

from .manifold_firewall import (
    ManifoldFirewall,
    ManifoldBatchAnalyzer,
    FirewallConfig,
    FirewallState,
    FirewallCheckResult,
    ManifoldPoint,
    DriftEvent,
    DriftType,
    AlertSeverity,
    FIREWALL_GLOSSARY,
    ZONE_TRANSITIONS,
    create_firewall,
)

__all__ = [
    # Classifier
    'ClassificationResult',
    'ClusterInfo',
    'OnlineClassifier',
    'SidecarClassifier',
    # Spectral Discovery
    'SpectralManifoldDiscovery',
    'SpectralDiscoveryConfig',
    'SpectralAnalysis',
    'SpectralCoherence',
    'SpectralMode',
    'FrequencyBandAnalyzer',
    # Spectral Router
    'SpectralRouter',
    'AdaptiveSpectralRouter',
    'RouterConfig',
    'RoutingDecision',
    'ModelSize',
    'create_router_from_fingerprints',
]

__version__ = '1.1.0'
