#!/usr/bin/env python3
"""
Run V2 Exploration with Rotational Variance

This script extends existing v1 fingerprints with computed rotational variance
and generates updated visualizations showing the RV-enhanced analysis.

Usage:
    python scripts/run_v2_exploration.py --db exploration_fingerprints.db --output ./v2_outputs

    # With custom parameters
    python scripts/run_v2_exploration.py --db exploration_fingerprints.db \
        --output ./v2_outputs \
        --skeleton-ratio 0.3 \
        --limit 50000

Author: SGLang Attention Explorer
"""

import argparse
import os
import sqlite3
import struct
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from discovery.fingerprint_schema import (
    V1_DIM,
    V2_DIM,
    FP_LOCAL_MASS,
    FP_MID_MASS,
    FP_LONG_MASS,
    FP_ENTROPY,
    FP_ROTATIONAL_VARIANCE,
    RV_THRESHOLD_LOCAL,
    RV_THRESHOLD_LONG_RANGE,
    ZONE_THRESHOLDS,
    is_v2,
)


def compute_synthetic_rv(fingerprint: np.ndarray) -> float:
    """
    Compute synthetic rotational variance from fingerprint features.

    Since we don't have the original attention scores, we estimate RV
    based on the attention mass distribution:
    - High local_mass → low RV (local attention)
    - High long_mass → high RV (long-range attention)
    - High entropy → moderate RV (distributed attention)

    This is an approximation for visualization purposes.
    """
    local_mass = fingerprint[FP_LOCAL_MASS]
    mid_mass = fingerprint[FP_MID_MASS]
    long_mass = fingerprint[FP_LONG_MASS]
    entropy = fingerprint[FP_ENTROPY]

    # Normalize masses
    total_mass = local_mass + mid_mass + long_mass
    if total_mass < 1e-6:
        return 0.5  # Neutral

    local_ratio = local_mass / total_mass
    long_ratio = long_mass / total_mass

    # RV estimation:
    # - Dominated by local → low RV
    # - Dominated by long → high RV
    # - Mixed → moderate RV
    # - High entropy shifts toward moderate

    base_rv = long_ratio * 0.8 + mid_mass / total_mass * 0.4

    # Entropy adjustment: high entropy → move toward 0.5
    entropy_factor = min(1.0, entropy / 4.0)  # Normalize entropy
    rv = base_rv * (1 - entropy_factor * 0.3) + 0.5 * entropy_factor * 0.3

    # If strongly local, push lower
    if local_ratio > 0.6:
        rv = rv * 0.6

    return max(0.0, min(1.0, rv))


def extend_fingerprint_to_v2(fingerprint: np.ndarray) -> np.ndarray:
    """Extend v1 fingerprint to v2 by adding computed RV."""
    if len(fingerprint) >= V2_DIM:
        return fingerprint  # Already v2

    rv = compute_synthetic_rv(fingerprint)
    return np.append(fingerprint, rv)


def classify_zone_v2(fingerprint: np.ndarray) -> Tuple[str, float]:
    """Classify zone using v2 schema with RV."""
    local_mass = fingerprint[FP_LOCAL_MASS]
    mid_mass = fingerprint[FP_MID_MASS]
    long_mass = fingerprint[FP_LONG_MASS]
    entropy = fingerprint[FP_ENTROPY]
    rv = fingerprint[FP_ROTATIONAL_VARIANCE] if len(fingerprint) > FP_ROTATIONAL_VARIANCE else 0.5

    total_mass = max(local_mass + mid_mass + long_mass, 1e-6)

    # Syntax floor: high local, low entropy, low RV
    if local_mass > 0.5 and entropy < 2.5 and rv <= RV_THRESHOLD_LOCAL:
        confidence = min(1.0, local_mass * (1.0 - entropy / 4.0) * (1.0 - rv))
        return 'syntax_floor', confidence

    # Structure ripple: high long-range, high RV
    if long_mass > 0.25 and rv >= RV_THRESHOLD_LONG_RANGE:
        confidence = min(1.0, (long_mass + rv) / 2)
        return 'structure_ripple', confidence

    # Semantic bridge: default
    confidence = mid_mass / total_mass
    return 'semantic_bridge', confidence


def load_fingerprints(db_path: str, limit: int = 100000) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load fingerprints from database."""
    print(f"Loading fingerprints from {db_path}...")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT fingerprint, manifold_zone, request_id
        FROM fingerprints
        ORDER BY id
        LIMIT {limit}
    """)

    rows = cursor.fetchall()
    conn.close()

    fingerprints = []
    zones = []
    request_ids = []

    for fp_blob, zone, req_id in rows:
        n_floats = len(fp_blob) // 4
        fp = np.array(struct.unpack(f'<{n_floats}f', fp_blob))
        fingerprints.append(fp)
        zones.append(zone or 'unknown')
        request_ids.append(req_id)

    fingerprints = np.array(fingerprints)
    print(f"Loaded {len(fingerprints)} fingerprints (original dim={fingerprints.shape[1]})")

    return fingerprints, zones, request_ids


def run_v2_exploration(
    db_path: str,
    output_dir: str,
    limit: int = 100000,
    skeleton_ratio: float = 0.3,
):
    """Run the v2 exploration with RV enhancement."""
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)

    # Load fingerprints
    fingerprints_v1, zones_v1, request_ids = load_fingerprints(db_path, limit)

    # Extend to v2
    print("\nExtending fingerprints to v2 schema...")
    fingerprints_v2 = np.array([extend_fingerprint_to_v2(fp) for fp in fingerprints_v1])
    print(f"Extended to {fingerprints_v2.shape[1]} dimensions")

    # Reclassify zones with v2 schema
    print("\nReclassifying zones with rotational variance...")
    zones_v2 = []
    confidences = []
    for fp in fingerprints_v2:
        zone, conf = classify_zone_v2(fp)
        zones_v2.append(zone)
        confidences.append(conf)

    # Zone distribution comparison
    def count_zones(zones):
        counts = {}
        for z in zones:
            counts[z] = counts.get(z, 0) + 1
        return counts

    v1_counts = count_zones(zones_v1)
    v2_counts = count_zones(zones_v2)

    print("\nZone distribution comparison:")
    print(f"{'Zone':<20} {'V1 Count':<12} {'V1 %':<10} {'V2 Count':<12} {'V2 %':<10} {'Delta'}")
    print("-" * 75)
    all_zones = set(v1_counts.keys()) | set(v2_counts.keys())
    for zone in sorted(all_zones):
        v1_c = v1_counts.get(zone, 0)
        v2_c = v2_counts.get(zone, 0)
        v1_pct = v1_c / len(zones_v1) * 100
        v2_pct = v2_c / len(zones_v2) * 100
        delta = v2_pct - v1_pct
        print(f"{zone:<20} {v1_c:<12} {v1_pct:<10.1f} {v2_c:<12} {v2_pct:<10.1f} {delta:+.1f}%")

    # RV statistics
    rv_values = fingerprints_v2[:, FP_ROTATIONAL_VARIANCE]
    print(f"\nRotational Variance statistics:")
    print(f"  Min: {rv_values.min():.3f}")
    print(f"  Max: {rv_values.max():.3f}")
    print(f"  Mean: {rv_values.mean():.3f}")
    print(f"  Std: {rv_values.std():.3f}")

    # RV by zone
    print("\nRV by zone:")
    for zone in sorted(set(zones_v2)):
        mask = np.array([z == zone for z in zones_v2])
        zone_rv = rv_values[mask]
        print(f"  {zone}: mean={zone_rv.mean():.3f}, std={zone_rv.std():.3f}")

    # Compute skeleton
    print("\nComputing spectral skeleton...")
    skeleton_indices = None
    try:
        from discovery.spectral_eviction import SpectralSkeletonComputer
        computer = SpectralSkeletonComputer(retention_ratio=skeleton_ratio)
        result = computer.compute_skeleton(fingerprints_v2, len(fingerprints_v2))
        skeleton_indices = result.skeleton_indices
        print(f"Skeleton computed: {len(skeleton_indices)} tokens retained ({len(skeleton_indices)/len(fingerprints_v2)*100:.1f}%)")

        # Skeleton zone distribution
        skeleton_zones = [zones_v2[i] for i in skeleton_indices]
        skeleton_counts = count_zones(skeleton_zones)
        print("\nSkeleton zone distribution:")
        for zone in sorted(skeleton_counts.keys()):
            c = skeleton_counts[zone]
            pct = c / len(skeleton_indices) * 100
            print(f"  {zone}: {c} ({pct:.1f}%)")
    except ImportError as e:
        print(f"Skeleton computation skipped: {e}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    try:
        from discovery.visualizations import create_exploration_report

        saved_files = create_exploration_report(
            fingerprints_v2,
            zones=zones_v2,
            skeleton_indices=skeleton_indices,
            output_dir=output_dir,
            prefix="v2",
        )
    except ImportError as e:
        print(f"Visualization generation failed: {e}")
        saved_files = {}

    # Write summary report
    summary_path = os.path.join(output_dir, "v2_exploration_summary.md")
    with open(summary_path, 'w') as f:
        f.write("# V2 Exploration Summary\n\n")
        f.write(f"**Source Database:** {db_path}\n")
        f.write(f"**Fingerprints:** {len(fingerprints_v2)}\n")
        f.write(f"**Schema:** V2 (21 dimensions with Rotational Variance)\n\n")

        f.write("## Zone Distribution Comparison\n\n")
        f.write("| Zone | V1 Count | V1 % | V2 Count | V2 % | Delta |\n")
        f.write("|------|----------|------|----------|------|-------|\n")
        for zone in sorted(all_zones):
            v1_c = v1_counts.get(zone, 0)
            v2_c = v2_counts.get(zone, 0)
            v1_pct = v1_c / len(zones_v1) * 100
            v2_pct = v2_c / len(zones_v2) * 100
            delta = v2_pct - v1_pct
            f.write(f"| {zone} | {v1_c} | {v1_pct:.1f}% | {v2_c} | {v2_pct:.1f}% | {delta:+.1f}% |\n")

        f.write("\n## Rotational Variance Statistics\n\n")
        f.write(f"- **Min:** {rv_values.min():.3f}\n")
        f.write(f"- **Max:** {rv_values.max():.3f}\n")
        f.write(f"- **Mean:** {rv_values.mean():.3f}\n")
        f.write(f"- **Std:** {rv_values.std():.3f}\n\n")

        f.write("### RV by Zone\n\n")
        f.write("| Zone | Mean RV | Std RV |\n")
        f.write("|------|---------|--------|\n")
        for zone in sorted(set(zones_v2)):
            mask = np.array([z == zone for z in zones_v2])
            zone_rv = rv_values[mask]
            f.write(f"| {zone} | {zone_rv.mean():.3f} | {zone_rv.std():.3f} |\n")

        if skeleton_indices:
            f.write(f"\n## Spectral Skeleton\n\n")
            f.write(f"- **Retention ratio:** {skeleton_ratio:.0%}\n")
            f.write(f"- **Tokens retained:** {len(skeleton_indices)}\n")
            f.write(f"- **Tokens evicted:** {len(fingerprints_v2) - len(skeleton_indices)}\n\n")

            f.write("### Skeleton Zone Distribution\n\n")
            f.write("| Zone | Count | % |\n")
            f.write("|------|-------|---|\n")
            for zone in sorted(skeleton_counts.keys()):
                c = skeleton_counts[zone]
                pct = c / len(skeleton_indices) * 100
                f.write(f"| {zone} | {c} | {pct:.1f}% |\n")

        if saved_files:
            f.write("\n## Generated Visualizations\n\n")
            for name, path in saved_files.items():
                rel_path = os.path.basename(path)
                f.write(f"- **{name}:** [{rel_path}]({rel_path})\n")

    print(f"\nSummary written to: {summary_path}")

    elapsed = time.time() - start_time
    print(f"\nV2 exploration completed in {elapsed:.1f}s")

    return {
        'fingerprints_v2': fingerprints_v2,
        'zones_v2': zones_v2,
        'skeleton_indices': skeleton_indices,
        'rv_values': rv_values,
        'saved_files': saved_files,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run V2 exploration with rotational variance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", "-d", required=True, help="Path to fingerprints database")
    parser.add_argument("--output", "-o", default="./v2_outputs", help="Output directory")
    parser.add_argument("--limit", "-n", type=int, default=100000, help="Max fingerprints to load")
    parser.add_argument("--skeleton-ratio", type=float, default=0.3, help="Skeleton retention ratio")

    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: Database not found: {args.db}")
        sys.exit(1)

    run_v2_exploration(
        db_path=args.db,
        output_dir=args.output,
        limit=args.limit,
        skeleton_ratio=args.skeleton_ratio,
    )


if __name__ == "__main__":
    main()
