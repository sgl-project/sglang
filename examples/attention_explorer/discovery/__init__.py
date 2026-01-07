"""
Attention Fingerprint Discovery Pipeline

This module provides tools for batch processing and real-time classification
of attention fingerprints from the SGLang inference server.

Components:
- schema.sql: SQLite schema for fingerprint storage
- discovery_job.py: Batch pipeline for embedding and clustering
- classifier.py: Online classifier for real-time sidecar use

Usage:
    # Initialize database
    sqlite3 fingerprints.db < schema.sql

    # Run discovery job (hourly/daily)
    python discovery_job.py --db fingerprints.db --output ./discovery_outputs

    # Use classifier in sidecar
    from discovery import OnlineClassifier, SidecarClassifier

    classifier = SidecarClassifier('./discovery_outputs')
    result = classifier.classify(fingerprint_vector)
"""

from .classifier import (
    ClassificationResult,
    ClusterInfo,
    OnlineClassifier,
    SidecarClassifier,
)

__all__ = [
    'ClassificationResult',
    'ClusterInfo',
    'OnlineClassifier',
    'SidecarClassifier',
]

__version__ = '1.0.0'
