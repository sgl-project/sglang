#!/usr/bin/env python3
"""
Fingerprint Schema - Single Source of Truth

This module defines the canonical fingerprint format and semantics.
All other modules should import from here to ensure consistency.

Schema Versions:
- V1 (20 dimensions): Base fingerprint with attention mass, entropy, histogram
- V2 (21 dimensions): Adds rotational_variance at position 20

Author: SGLang Attention Explorer
"""

from typing import Optional, Tuple

import numpy as np

# =============================================================================
# SCHEMA VERSIONS
# =============================================================================

SCHEMA_VERSION = 2  # Current version

# Fingerprint dimensions
V1_DIM = 20  # Schema v1: base fingerprint
V2_DIM = 21  # Schema v2: adds rotational_variance

# Backward-compatible alias
FINGERPRINT_DIM = V1_DIM  # Default to v1 for backward compatibility


# =============================================================================
# FEATURE INDICES
# =============================================================================

# Attention mass distribution (positions 0-2)
FP_LOCAL_MASS = 0      # Attention mass in local window (0-32 tokens)
FP_MID_MASS = 1        # Attention mass in mid range (32-256 tokens)
FP_LONG_MASS = 2       # Attention mass in long range (256+ tokens)

# Attention entropy (position 3)
FP_ENTROPY = 3         # Attention entropy (higher = more distributed)

# Distance histogram (positions 4-11)
FP_HISTOGRAM_START = 4
FP_HISTOGRAM_END = 12

# Layer statistics (positions 12-19)
FP_LAYER_STATS_START = 12
FP_LAYER_STATS_END = 20

# Schema V2 extension (position 20)
FP_ROTATIONAL_VARIANCE = 20


# =============================================================================
# ROTATIONAL VARIANCE SEMANTICS
# =============================================================================
#
# Rotational Variance (RV) measures how LONG-RANGE the attention pattern is.
# It is computed by comparing raw attention scores to RoPE-derotated scores.
#
# CANONICAL DEFINITION:
#   Low RV (→0)  = LOCAL/SHORT-RANGE attention
#                  (tokens attend to nearby positions)
#   High RV (→1) = LONG-RANGE/DISTANT attention
#                  (tokens attend to far-away positions)
#
# IMPORTANT: RV measures DISTANCE, not semantic vs positional!
# Both local and long-range attention use RoPE position encoding.
# The difference is whether attended tokens are near or far.
#
# ZONE MAPPING:
#   syntax_floor:     Low RV (≤0.25) - local attention to nearby syntax
#   semantic_bridge:  Medium RV (0.15-0.5) - mid-range attention
#   structure_ripple: High RV (≥0.35) - long-range structural patterns
#
# =============================================================================

# RV thresholds for zone classification
RV_THRESHOLD_LOCAL = 0.25      # syntax_floor: RV must be ≤ this
RV_THRESHOLD_LONG_RANGE = 0.35 # structure_ripple: RV must be ≥ this
RV_RANGE_BRIDGE = (0.15, 0.5)  # semantic_bridge: typical RV range


# =============================================================================
# ZONE THRESHOLDS
# =============================================================================

ZONE_THRESHOLDS = {
    'syntax_floor': {
        'local_mass_min': 0.5,
        'entropy_max': 2.5,
        # Low RV = local attention to nearby tokens
        'rotational_variance_max': RV_THRESHOLD_LOCAL,
    },
    'structure_ripple': {
        'long_mass_min': 0.25,
        'histogram_variance_min': 0.1,
        # High RV = long-range attention to distant tokens
        'rotational_variance_min': RV_THRESHOLD_LONG_RANGE,
    },
    'semantic_bridge': {
        # Medium RV = balanced local/long-range attention
        'rotational_variance_range': RV_RANGE_BRIDGE,
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_version(fingerprint: np.ndarray) -> int:
    """
    Detect schema version from fingerprint dimension.

    Args:
        fingerprint: Fingerprint array

    Returns:
        Schema version (1 or 2)
    """
    dim = len(fingerprint) if fingerprint.ndim == 1 else fingerprint.shape[1]
    return 2 if dim >= V2_DIM else 1


def is_v2(fingerprint: np.ndarray) -> bool:
    """Check if fingerprint is schema v2 (has rotational_variance)."""
    return get_version(fingerprint) >= 2


def get_rotational_variance(fingerprint: np.ndarray) -> Optional[float]:
    """
    Extract rotational variance from fingerprint if available.

    Args:
        fingerprint: Fingerprint array (v1 or v2)

    Returns:
        Rotational variance value, or None if v1 fingerprint
    """
    if not is_v2(fingerprint):
        return None

    if fingerprint.ndim == 1:
        return float(fingerprint[FP_ROTATIONAL_VARIANCE])
    else:
        return fingerprint[:, FP_ROTATIONAL_VARIANCE]


def extend_v1_to_v2(
    fingerprint: np.ndarray,
    rotational_variance: float = 0.5,
) -> np.ndarray:
    """
    Extend a v1 fingerprint to v2 by adding rotational_variance.

    Args:
        fingerprint: V1 fingerprint (20 dimensions)
        rotational_variance: RV value to add (default 0.5 = neutral)

    Returns:
        V2 fingerprint (21 dimensions)
    """
    if is_v2(fingerprint):
        return fingerprint

    if fingerprint.ndim == 1:
        return np.append(fingerprint, rotational_variance)
    else:
        rv_col = np.full((len(fingerprint), 1), rotational_variance)
        return np.hstack([fingerprint, rv_col])


def validate_fingerprint(fingerprint: np.ndarray) -> Tuple[bool, str]:
    """
    Validate fingerprint format.

    Args:
        fingerprint: Fingerprint to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if fingerprint is None:
        return False, "Fingerprint is None"

    if fingerprint.ndim == 1:
        dim = len(fingerprint)
    elif fingerprint.ndim == 2:
        dim = fingerprint.shape[1]
    else:
        return False, f"Invalid dimensions: {fingerprint.ndim}"

    if dim < V1_DIM:
        return False, f"Fingerprint too small: {dim} < {V1_DIM}"

    if dim > V2_DIM:
        return False, f"Fingerprint too large: {dim} > {V2_DIM}"

    return True, ""


# =============================================================================
# DOCUMENTATION STRINGS
# =============================================================================

FINGERPRINT_DOC = """
Fingerprint Schema Documentation

The attention fingerprint is a compact representation of a token's attention
pattern, designed for clustering, routing, and eviction decisions.

Schema V1 (20 dimensions):
  [0]     local_mass   - Attention mass in local window (0-32 tokens)
  [1]     mid_mass     - Attention mass in mid range (32-256 tokens)
  [2]     long_mass    - Attention mass in long range (256+ tokens)
  [3]     entropy      - Attention entropy (higher = more distributed)
  [4-11]  histogram    - 8-bin distance histogram
  [12-19] layer_stats  - Per-layer entropy (up to 8 layers)

Schema V2 (21 dimensions):
  [0-19]  Same as V1
  [20]    rotational_variance - RoPE de-rotation effect magnitude
          Low RV (→0)  = local/short-range attention
          High RV (→1) = long-range/distant attention

Zone Classification:
  - syntax_floor:     Local attention, low entropy, low RV
  - semantic_bridge:  Mid-range attention, medium RV
  - structure_ripple: Long-range patterns, high RV
"""


RV_SEMANTICS_DOC = """
Rotational Variance (RV) Semantics

RV measures how LONG-RANGE the attention pattern is, NOT how "semantic" it is.

Computation:
  1. Compute raw attention scores
  2. Apply RoPE de-rotation to remove positional bias
  3. Measure difference: raw_diff = |raw_scores - derotated_scores|
  4. Invert and calibrate to [0, 1]: RV = (max_diff - raw_diff) / range

Interpretation:
  - Low raw_diff (small change) → tokens at SIMILAR distances → HIGH RV (long-range)
  - High raw_diff (big change) → tokens at VARYING distances → LOW RV (local)

The inversion ensures:
  - Local attention (nearby tokens) → Low RV
  - Long-range attention (distant tokens) → High RV

This aligns with zone thresholds:
  - syntax_floor: RV ≤ 0.25 (local attention)
  - structure_ripple: RV ≥ 0.35 (long-range attention)
  - semantic_bridge: RV in [0.15, 0.5] (balanced)
"""
