#!/usr/bin/env python3
"""Generate manifold visualization for Qwen3-4B exploration."""

import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DB_PATH = "./exploration_4b_fingerprints.db"
OUTPUT_DIR = "./exploration_4b_outputs"


def classify_zone(fp):
    """Classify fingerprint into zone based on features."""
    # Fingerprint layout: [local_mass, mid_mass, long_mass, entropy, hist_0..hist_7, layer_entropy_0..layer_entropy_7]
    local_mass = fp[0]
    mid_mass = fp[1]
    long_mass = fp[2]
    entropy = fp[3]

    # Zone classification thresholds (from sidecar classifier)
    if local_mass > 0.5 and entropy < 2.0:
        return 'syntax_floor'
    elif long_mass > 0.4 or entropy > 3.5:
        return 'structure_ripple'
    else:
        return 'semantic_bridge'


def load_fingerprints():
    """Load fingerprints from database."""
    import struct

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT fingerprint, manifold_zone, request_id
        FROM fingerprints
        ORDER BY id
    """)

    rows = cursor.fetchall()
    conn.close()

    fingerprints = []
    zones = []
    request_ids = []

    for row in rows:
        fp_blob, zone, req_id = row
        # Parse fingerprint BLOB (packed float32[20])
        fp = np.array(struct.unpack('<20f', fp_blob))
        fingerprints.append(fp)
        # Compute zone from features if not stored
        computed_zone = zone if zone else classify_zone(fp)
        zones.append(computed_zone)
        request_ids.append(req_id)

    return np.array(fingerprints), zones, request_ids


def create_manifold_visualization():
    """Create 2D manifold visualization using PCA."""
    print("Loading fingerprints...")
    fingerprints, zones, request_ids = load_fingerprints()
    print(f"Loaded {len(fingerprints)} fingerprints")

    # Standardize
    scaler = StandardScaler()
    fp_scaled = scaler.fit_transform(fingerprints)

    # PCA to 2D
    print("Running PCA...")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(fp_scaled)

    # Color mapping
    zone_colors = {
        'syntax_floor': '#45b7d1',
        'semantic_bridge': '#4ecdc4',
        'structure_ripple': '#ff6b6b',
        'unknown': '#999999'
    }

    colors = [zone_colors.get(z, '#999999') for z in zones]

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Scatter plot by zone
    ax1 = axes[0, 0]
    handles = []
    labels = []
    for zone in ['structure_ripple', 'semantic_bridge', 'syntax_floor']:
        mask = np.array([z == zone for z in zones])
        if mask.sum() > 0:
            scatter = ax1.scatter(coords[mask, 0], coords[mask, 1],
                       c=zone_colors[zone], label=f"{zone} ({mask.sum():,})",
                       alpha=0.3, s=5)
            handles.append(scatter)
            labels.append(f"{zone.replace('_', ' ').title()} ({mask.sum():,})")
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.set_title('Qwen3-4B Attention Manifold by Zone', fontsize=14, fontweight='bold')
    if handles:
        ax1.legend(handles, labels, loc='upper right')
    ax1.grid(alpha=0.3)

    # 2. Density heatmap
    ax2 = axes[0, 1]
    hb = ax2.hexbin(coords[:, 0], coords[:, 1], gridsize=50, cmap='YlOrRd', mincnt=1)
    ax2.set_xlabel('PCA Component 1')
    ax2.set_ylabel('PCA Component 2')
    ax2.set_title('Fingerprint Density', fontsize=14, fontweight='bold')
    plt.colorbar(hb, ax=ax2, label='Count')

    # 3. Zone distribution pie chart
    ax3 = axes[1, 0]
    zone_counts = {}
    for z in zones:
        zone_counts[z] = zone_counts.get(z, 0) + 1

    labels = list(zone_counts.keys())
    sizes = list(zone_counts.values())
    colors_pie = [zone_colors.get(z, '#999999') for z in labels]

    wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        colors=colors_pie, startangle=90)
    ax3.set_title('Zone Distribution (109K tokens)', fontsize=14, fontweight='bold')

    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')

    total = len(fingerprints)
    unique_requests = len(set(request_ids))

    summary = f"""
    QWEN3-4B MANIFOLD SUMMARY
    ═══════════════════════════════════════

    Total Fingerprints: {total:,}
    Unique Requests: {unique_requests}

    Zone Distribution:
    ─────────────────────────────────────
    """

    for zone, count in sorted(zone_counts.items(), key=lambda x: -x[1]):
        pct = (count / total) * 100
        summary += f"    {zone:<20} {count:>8,} ({pct:>5.1f}%)\n"

    summary += f"""
    ─────────────────────────────────────

    PCA Explained Variance:
        PC1: {pca.explained_variance_ratio_[0]*100:.1f}%
        PC2: {pca.explained_variance_ratio_[1]*100:.1f}%
        Total: {sum(pca.explained_variance_ratio_)*100:.1f}%

    Key Observation:
    ─────────────────────────────────────
    The 4B model shows 84% structure_ripple
    dominance, indicating heavy reliance on
    long-range structural attention patterns.
    """

    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4b_manifold_visualization.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved: {OUTPUT_DIR}/4b_manifold_visualization.png")


if __name__ == "__main__":
    create_manifold_visualization()
