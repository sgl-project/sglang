#!/usr/bin/env python3
"""
Visualization utilities for attention fingerprint analysis.

Provides plotting functions for:
1. Rotational Variance over token position (with zone overlays)
2. Fingerprint manifold scatter (colored by RV or zone)
3. Skeleton view (retained vs evicted tokens)
4. Zone distribution charts

Usage:
    from discovery.visualizations import (
        plot_rv_timeline,
        plot_manifold_scatter,
        plot_skeleton_view,
        plot_zone_distribution,
        create_exploration_report,
    )

    # Single plot
    fig = plot_rv_timeline(fingerprints, title="RV over sequence")
    fig.savefig("rv_timeline.png")

    # Full report
    create_exploration_report(fingerprints, output_dir="./plots")

Dependencies:
    pip install matplotlib numpy scikit-learn

Author: SGLang Attention Explorer
"""

import os
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Import schema constants
try:
    from .fingerprint_schema import (
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
        get_rotational_variance,
    )
except ImportError:
    # Fallback for standalone use
    V1_DIM, V2_DIM = 20, 21
    FP_LOCAL_MASS, FP_MID_MASS, FP_LONG_MASS, FP_ENTROPY = 0, 1, 2, 3
    FP_ROTATIONAL_VARIANCE = 20
    RV_THRESHOLD_LOCAL, RV_THRESHOLD_LONG_RANGE = 0.25, 0.35

# Lazy import matplotlib to avoid import errors when not needed
_plt = None
_colors = None


def _get_plt():
    """Lazy import matplotlib."""
    global _plt, _colors
    if _plt is None:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        _plt = plt
        _colors = mcolors
    return _plt, _colors


# =============================================================================
# COLOR SCHEMES
# =============================================================================

ZONE_COLORS = {
    'syntax_floor': '#3498db',      # Blue - local/syntax
    'semantic_bridge': '#9b59b6',   # Purple - mid-range
    'structure_ripple': '#e74c3c',  # Red - long-range
    'long_range': '#e67e22',        # Orange
    'diffuse': '#95a5a6',           # Gray
    'unknown': '#bdc3c7',           # Light gray
}

RV_CMAP = 'RdYlBu_r'  # Red (high RV) to Blue (low RV)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def classify_zone_from_fingerprint(fp: np.ndarray) -> str:
    """Classify zone from fingerprint features."""
    local_mass = fp[FP_LOCAL_MASS]
    long_mass = fp[FP_LONG_MASS]
    entropy = fp[FP_ENTROPY]

    # Check for RV if available
    rv = fp[FP_ROTATIONAL_VARIANCE] if len(fp) > FP_ROTATIONAL_VARIANCE else None

    if local_mass > 0.5 and entropy < 2.5:
        if rv is None or rv <= RV_THRESHOLD_LOCAL:
            return 'syntax_floor'
    if long_mass > 0.25:
        if rv is None or rv >= RV_THRESHOLD_LONG_RANGE:
            return 'structure_ripple'
    return 'semantic_bridge'


def extract_rv_values(fingerprints: np.ndarray) -> Optional[np.ndarray]:
    """Extract rotational variance values from fingerprints."""
    if fingerprints.ndim == 1:
        fingerprints = fingerprints.reshape(1, -1)

    if fingerprints.shape[1] <= FP_ROTATIONAL_VARIANCE:
        return None

    return fingerprints[:, FP_ROTATIONAL_VARIANCE]


# =============================================================================
# PLOT 1: RV TIMELINE
# =============================================================================

def plot_rv_timeline(
    fingerprints: np.ndarray,
    zones: Optional[List[str]] = None,
    title: str = "Rotational Variance Over Sequence",
    figsize: Tuple[int, int] = (14, 6),
    show_thresholds: bool = True,
    show_zone_colors: bool = True,
) -> Any:
    """
    Plot rotational variance over token position with zone overlays.

    Args:
        fingerprints: Array of shape (n_tokens, dim) with fingerprints
        zones: Optional list of zone labels per token
        title: Plot title
        figsize: Figure size (width, height)
        show_thresholds: Show RV threshold lines
        show_zone_colors: Color background by zone

    Returns:
        matplotlib Figure object
    """
    plt, colors = _get_plt()

    rv_values = extract_rv_values(fingerprints)
    if rv_values is None:
        raise ValueError("Fingerprints don't have rotational variance (need v2 schema)")

    n_tokens = len(rv_values)
    positions = np.arange(n_tokens)

    # Compute zones if not provided
    if zones is None:
        zones = [classify_zone_from_fingerprint(fp) for fp in fingerprints]

    fig, ax = plt.subplots(figsize=figsize)

    # Background zone coloring
    if show_zone_colors:
        current_zone = zones[0]
        start_pos = 0

        for i, zone in enumerate(zones + [None]):
            if zone != current_zone or i == len(zones):
                color = ZONE_COLORS.get(current_zone, '#ffffff')
                ax.axvspan(start_pos, i, alpha=0.15, color=color, label=None)
                current_zone = zone
                start_pos = i

    # Plot RV line
    ax.plot(positions, rv_values, 'k-', linewidth=1.5, alpha=0.8, label='Rotational Variance')

    # Scatter points colored by zone
    for zone in set(zones):
        mask = np.array([z == zone for z in zones])
        if mask.any():
            ax.scatter(
                positions[mask],
                rv_values[mask],
                c=ZONE_COLORS.get(zone, '#333333'),
                s=20,
                alpha=0.6,
                label=zone.replace('_', ' ').title(),
            )

    # Threshold lines
    if show_thresholds:
        ax.axhline(y=RV_THRESHOLD_LOCAL, color='blue', linestyle='--',
                   alpha=0.5, label=f'Local threshold ({RV_THRESHOLD_LOCAL})')
        ax.axhline(y=RV_THRESHOLD_LONG_RANGE, color='red', linestyle='--',
                   alpha=0.5, label=f'Long-range threshold ({RV_THRESHOLD_LONG_RANGE})')

    ax.set_xlabel('Token Position', fontsize=12)
    ax.set_ylabel('Rotational Variance', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, n_tokens)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# PLOT 2: MANIFOLD SCATTER
# =============================================================================

def plot_manifold_scatter(
    fingerprints: np.ndarray,
    color_by: str = 'rv',  # 'rv', 'zone', 'entropy'
    zones: Optional[List[str]] = None,
    title: str = "Fingerprint Manifold",
    figsize: Tuple[int, int] = (10, 8),
    method: str = 'pca',  # 'pca', 'umap'
    n_components: int = 2,
) -> Any:
    """
    Plot 2D manifold scatter of fingerprints.

    Args:
        fingerprints: Array of shape (n_tokens, dim)
        color_by: What to color points by ('rv', 'zone', 'entropy')
        zones: Optional zone labels (required if color_by='zone')
        title: Plot title
        figsize: Figure size
        method: Dimensionality reduction method ('pca' or 'umap')
        n_components: Should be 2 for visualization

    Returns:
        matplotlib Figure object
    """
    plt, mcolors = _get_plt()
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Standardize fingerprints
    scaler = StandardScaler()
    fp_scaled = scaler.fit_transform(fingerprints)

    # Dimensionality reduction
    if method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
        coords = reducer.fit_transform(fp_scaled)
        explained_var = reducer.explained_variance_ratio_.sum()
        method_label = f"PCA ({explained_var:.1%} variance)"
    elif method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            coords = reducer.fit_transform(fp_scaled)
            method_label = "UMAP"
        except ImportError:
            print("UMAP not available, falling back to PCA")
            reducer = PCA(n_components=n_components, random_state=42)
            coords = reducer.fit_transform(fp_scaled)
            method_label = "PCA (UMAP unavailable)"
    else:
        raise ValueError(f"Unknown method: {method}")

    fig, ax = plt.subplots(figsize=figsize)

    # Color mapping
    if color_by == 'rv':
        rv_values = extract_rv_values(fingerprints)
        if rv_values is None:
            print("No RV data, falling back to entropy coloring")
            color_by = 'entropy'
        else:
            scatter = ax.scatter(
                coords[:, 0], coords[:, 1],
                c=rv_values,
                cmap=RV_CMAP,
                s=15,
                alpha=0.6,
                vmin=0, vmax=1,
            )
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Rotational Variance', fontsize=11)

    if color_by == 'zone':
        if zones is None:
            zones = [classify_zone_from_fingerprint(fp) for fp in fingerprints]

        for zone in set(zones):
            mask = np.array([z == zone for z in zones])
            if mask.any():
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=ZONE_COLORS.get(zone, '#333333'),
                    s=15,
                    alpha=0.6,
                    label=zone.replace('_', ' ').title(),
                )
        ax.legend(loc='upper right', fontsize=10)

    if color_by == 'entropy':
        entropy = fingerprints[:, FP_ENTROPY]
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=entropy,
            cmap='viridis',
            s=15,
            alpha=0.6,
        )
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Entropy', fontsize=11)

    ax.set_xlabel(f'{method_label} Dimension 1', fontsize=12)
    ax.set_ylabel(f'{method_label} Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# PLOT 3: SKELETON VIEW
# =============================================================================

def plot_skeleton_view(
    fingerprints: np.ndarray,
    skeleton_indices: List[int],
    zones: Optional[List[str]] = None,
    title: str = "Spectral Skeleton (Retained vs Evicted)",
    figsize: Tuple[int, int] = (14, 8),
    show_rv: bool = True,
) -> Any:
    """
    Visualize skeleton tokens (retained) vs evicted tokens.

    Args:
        fingerprints: Array of shape (n_tokens, dim)
        skeleton_indices: Indices of tokens to KEEP (skeleton)
        zones: Optional zone labels
        title: Plot title
        figsize: Figure size
        show_rv: Show RV values if available

    Returns:
        matplotlib Figure object
    """
    plt, _ = _get_plt()
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    n_tokens = len(fingerprints)
    skeleton_set = set(skeleton_indices)
    evicted_indices = [i for i in range(n_tokens) if i not in skeleton_set]

    # Compute zones if not provided
    if zones is None:
        zones = [classify_zone_from_fingerprint(fp) for fp in fingerprints]

    # Get RV values
    rv_values = extract_rv_values(fingerprints)

    # Create figure with subplots
    if show_rv and rv_values is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
    else:
        fig, ax1 = plt.subplots(figsize=(figsize[0], figsize[1] * 0.6))
        ax2 = None

    # --- Top plot: 2D manifold with skeleton highlighted ---
    scaler = StandardScaler()
    fp_scaled = scaler.fit_transform(fingerprints)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(fp_scaled)

    # Plot evicted tokens (gray, small)
    ax1.scatter(
        coords[evicted_indices, 0],
        coords[evicted_indices, 1],
        c='#cccccc',
        s=10,
        alpha=0.3,
        label=f'Evicted ({len(evicted_indices)})',
    )

    # Plot skeleton tokens (colored by zone, larger)
    for zone in set(zones):
        mask = np.array([i in skeleton_set and zones[i] == zone for i in range(n_tokens)])
        if mask.any():
            ax1.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=ZONE_COLORS.get(zone, '#333333'),
                s=50,
                alpha=0.8,
                edgecolors='black',
                linewidths=0.5,
                label=f'{zone.replace("_", " ").title()} (kept)',
            )

    retention_pct = len(skeleton_indices) / n_tokens * 100
    ax1.set_title(f'{title}\nRetention: {len(skeleton_indices)}/{n_tokens} ({retention_pct:.1f}%)', fontsize=12)
    ax1.set_xlabel('PCA Dimension 1')
    ax1.set_ylabel('PCA Dimension 2')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Bottom plot: Token position view with skeleton markers ---
    if ax2 is not None and rv_values is not None:
        positions = np.arange(n_tokens)

        # Plot RV line
        ax2.plot(positions, rv_values, 'k-', alpha=0.4, linewidth=1)

        # Evicted tokens (small gray)
        ax2.scatter(
            evicted_indices,
            rv_values[evicted_indices],
            c='#cccccc',
            s=8,
            alpha=0.3,
        )

        # Skeleton tokens (larger, colored)
        for zone in set(zones):
            mask = np.array([i in skeleton_set and zones[i] == zone for i in range(n_tokens)])
            if mask.any():
                ax2.scatter(
                    np.where(mask)[0],
                    rv_values[mask],
                    c=ZONE_COLORS.get(zone, '#333333'),
                    s=30,
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=0.3,
                )

        # Threshold lines
        ax2.axhline(y=RV_THRESHOLD_LOCAL, color='blue', linestyle='--', alpha=0.4)
        ax2.axhline(y=RV_THRESHOLD_LONG_RANGE, color='red', linestyle='--', alpha=0.4)

        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Rotational Variance')
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlim(0, n_tokens)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# =============================================================================
# PLOT 4: ZONE DISTRIBUTION
# =============================================================================

def plot_zone_distribution(
    zones: List[str],
    rv_values: Optional[np.ndarray] = None,
    title: str = "Zone Distribution",
    figsize: Tuple[int, int] = (12, 5),
) -> Any:
    """
    Plot zone distribution as bar chart with optional RV box plots.

    Args:
        zones: List of zone labels
        rv_values: Optional RV values for box plots
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    plt, _ = _get_plt()

    zone_counts = {}
    for zone in zones:
        zone_counts[zone] = zone_counts.get(zone, 0) + 1

    zone_names = sorted(zone_counts.keys())
    counts = [zone_counts[z] for z in zone_names]
    colors = [ZONE_COLORS.get(z, '#333333') for z in zone_names]

    if rv_values is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(figsize=(figsize[0] // 2, figsize[1]))
        ax2 = None

    # Bar chart
    bars = ax1.bar(zone_names, counts, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Zone')
    ax1.set_ylabel('Count')
    ax1.set_title(f'{title}\n(n={len(zones)})')

    # Add percentage labels
    total = sum(counts)
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f'{pct:.1f}%',
            ha='center',
            fontsize=9,
        )

    ax1.set_xticklabels([z.replace('_', '\n') for z in zone_names], fontsize=10)

    # Box plots of RV by zone
    if ax2 is not None and rv_values is not None:
        zone_rv_data = []
        zone_labels = []
        for zone in zone_names:
            mask = np.array([z == zone for z in zones])
            if mask.any():
                zone_rv_data.append(rv_values[mask])
                zone_labels.append(zone)

        bp = ax2.boxplot(zone_rv_data, labels=[z.replace('_', '\n') for z in zone_labels], patch_artist=True)
        for patch, zone in zip(bp['boxes'], zone_labels):
            patch.set_facecolor(ZONE_COLORS.get(zone, '#333333'))
            patch.set_alpha(0.6)

        ax2.axhline(y=RV_THRESHOLD_LOCAL, color='blue', linestyle='--', alpha=0.5, label='Local threshold')
        ax2.axhline(y=RV_THRESHOLD_LONG_RANGE, color='red', linestyle='--', alpha=0.5, label='Long-range threshold')
        ax2.set_ylabel('Rotational Variance')
        ax2.set_title('RV Distribution by Zone')
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(fontsize=8)

    plt.tight_layout()
    return fig


# =============================================================================
# FULL REPORT GENERATOR
# =============================================================================

def create_exploration_report(
    fingerprints: np.ndarray,
    zones: Optional[List[str]] = None,
    skeleton_indices: Optional[List[int]] = None,
    output_dir: str = "./plots",
    prefix: str = "",
) -> Dict[str, str]:
    """
    Create a full visualization report.

    Args:
        fingerprints: Array of fingerprints
        zones: Optional zone labels
        skeleton_indices: Optional skeleton token indices
        output_dir: Directory to save plots
        prefix: Filename prefix

    Returns:
        Dict mapping plot name to file path
    """
    plt, _ = _get_plt()

    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}

    # Compute zones if not provided
    if zones is None:
        zones = [classify_zone_from_fingerprint(fp) for fp in fingerprints]

    rv_values = extract_rv_values(fingerprints)
    has_rv = rv_values is not None

    prefix = f"{prefix}_" if prefix else ""

    # 1. Zone distribution
    fig = plot_zone_distribution(zones, rv_values, title="Zone Distribution")
    path = os.path.join(output_dir, f"{prefix}zone_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files['zone_distribution'] = path
    print(f"Saved: {path}")

    # 2. Manifold scatter (by zone)
    fig = plot_manifold_scatter(fingerprints, color_by='zone', zones=zones, title="Manifold by Zone")
    path = os.path.join(output_dir, f"{prefix}manifold_by_zone.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    saved_files['manifold_by_zone'] = path
    print(f"Saved: {path}")

    # 3. RV-specific plots (only if RV available)
    if has_rv:
        # RV timeline
        fig = plot_rv_timeline(fingerprints, zones=zones, title="Rotational Variance Timeline")
        path = os.path.join(output_dir, f"{prefix}rv_timeline.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files['rv_timeline'] = path
        print(f"Saved: {path}")

        # Manifold by RV
        fig = plot_manifold_scatter(fingerprints, color_by='rv', title="Manifold by Rotational Variance")
        path = os.path.join(output_dir, f"{prefix}manifold_by_rv.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files['manifold_by_rv'] = path
        print(f"Saved: {path}")

    # 4. Skeleton view (if skeleton provided)
    if skeleton_indices is not None:
        fig = plot_skeleton_view(fingerprints, skeleton_indices, zones=zones)
        path = os.path.join(output_dir, f"{prefix}skeleton_view.png")
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        saved_files['skeleton_view'] = path
        print(f"Saved: {path}")

    print(f"\nGenerated {len(saved_files)} plots in {output_dir}")
    return saved_files


# =============================================================================
# CLI
# =============================================================================

def main():
    """Generate visualizations from command line."""
    import argparse
    import struct
    import sqlite3

    parser = argparse.ArgumentParser(description="Generate fingerprint visualizations")
    parser.add_argument("--db", "-d", required=True, help="Path to fingerprints database")
    parser.add_argument("--output", "-o", default="./plots", help="Output directory")
    parser.add_argument("--limit", "-n", type=int, default=10000, help="Max fingerprints to load")
    parser.add_argument("--prefix", "-p", default="", help="Filename prefix")
    parser.add_argument("--skeleton-ratio", type=float, default=0.3, help="Skeleton retention ratio")

    args = parser.parse_args()

    print(f"Loading fingerprints from {args.db}...")

    # Load fingerprints from database
    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT fingerprint, manifold_zone
        FROM fingerprints
        ORDER BY id
        LIMIT {args.limit}
    """)

    rows = cursor.fetchall()
    conn.close()

    fingerprints = []
    zones = []

    for fp_blob, zone in rows:
        # Detect schema version from blob size
        n_floats = len(fp_blob) // 4
        fp = np.array(struct.unpack(f'<{n_floats}f', fp_blob))
        fingerprints.append(fp)
        zones.append(zone if zone else classify_zone_from_fingerprint(fp))

    fingerprints = np.array(fingerprints)
    print(f"Loaded {len(fingerprints)} fingerprints (dim={fingerprints.shape[1]})")

    # Compute skeleton if we have enough data
    skeleton_indices = None
    if len(fingerprints) >= 50:
        try:
            from .spectral_eviction import SpectralSkeletonComputer
            computer = SpectralSkeletonComputer(retention_ratio=args.skeleton_ratio)
            result = computer.compute_skeleton(fingerprints, len(fingerprints))
            skeleton_indices = result.skeleton_indices
            print(f"Computed skeleton: {len(skeleton_indices)} tokens retained")
        except ImportError:
            print("SpectralSkeletonComputer not available, skipping skeleton view")

    # Generate report
    create_exploration_report(
        fingerprints,
        zones=zones,
        skeleton_indices=skeleton_indices,
        output_dir=args.output,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
