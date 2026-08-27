"""Panel draw functions for tensor comparison visualization."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from sglang.srt.debug_utils.comparator.visualizer.figure import _PanelContext
from sglang.srt.debug_utils.comparator.visualizer.preprocessing import (
    _SCATTER_SAMPLE_SIZE,
    _format_log_ticks,
    _format_stats,
    _maybe_downsample_numpy,
    _safe_hist,
    _to_log10,
)


def _draw_baseline_heatmap(
    axes: np.ndarray, row_idx: int, ctx: _PanelContext
) -> Optional[str]:
    _draw_heatmap_pair(
        axes, row_idx=row_idx, t=ctx.baseline_2d, title=f"{ctx.name} Baseline"
    )
    return _format_stats("Baseline", ctx.baseline_2d)


def _draw_target_heatmap(
    axes: np.ndarray, row_idx: int, ctx: _PanelContext
) -> Optional[str]:
    _draw_heatmap_pair(
        axes, row_idx=row_idx, t=ctx.target_2d, title=f"{ctx.name} Target"
    )
    return _format_stats("Target", ctx.target_2d)


def _draw_diff_heatmap(
    axes: np.ndarray, row_idx: int, ctx: _PanelContext
) -> Optional[str]:
    assert ctx.diff is not None
    _draw_heatmap_pair(axes, row_idx=row_idx, t=ctx.diff, title=f"{ctx.name} Abs Diff")
    return _format_stats("Abs Diff", ctx.diff)


def _draw_diff_histogram(
    axes: np.ndarray, row_idx: int, ctx: _PanelContext
) -> Optional[str]:
    assert ctx.diff is not None
    _draw_histogram_pair(
        axes, row_idx=row_idx, diff=ctx.diff, label=f"{ctx.name} Abs Diff"
    )
    return None


def _draw_hist2d(axes: np.ndarray, row_idx: int, ctx: _PanelContext) -> Optional[str]:
    _draw_scatter_hist2d(
        axes,
        row_idx=row_idx,
        baseline=ctx.baseline_2d,
        target=ctx.target_2d,
        label=ctx.name,
    )
    return None


def _draw_sampled(axes: np.ndarray, row_idx: int, ctx: _PanelContext) -> Optional[str]:
    _draw_scatter_sampled(
        axes,
        row_idx=row_idx,
        baseline=ctx.baseline_2d,
        target=ctx.target_2d,
        label=ctx.name,
    )
    return None


# ────────────────────── internal drawing helpers ──────────────────────


def _draw_heatmap_pair(
    axes: np.ndarray,
    *,
    row_idx: int,
    t: torch.Tensor,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    ax_normal = axes[row_idx, 0]
    ax_log = axes[row_idx, 1]

    im = ax_normal.imshow(t.numpy(), aspect="auto", cmap="viridis")
    ax_normal.set_title(title)
    plt.colorbar(im, ax=ax_normal)

    im_log = ax_log.imshow(_to_log10(t).numpy(), aspect="auto", cmap="viridis")
    ax_log.set_title(f"{title} (Log10)")
    cbar = plt.colorbar(im_log, ax=ax_log)
    _format_log_ticks(cbar.ax, axis="y")


def _draw_histogram_pair(
    axes: np.ndarray,
    *,
    row_idx: int,
    diff: torch.Tensor,
    label: str,
) -> None:

    ax_normal = axes[row_idx, 0]
    ax_log = axes[row_idx, 1]

    diff_flat: np.ndarray = _maybe_downsample_numpy(diff.flatten())

    _safe_hist(ax_normal, diff_flat, bins=100, edgecolor="none")
    ax_normal.set_title(f"{label} Histogram")
    ax_normal.set_xlabel("Abs Diff")
    ax_normal.set_ylabel("Count")

    log_flat: np.ndarray = np.log10(np.abs(diff_flat) + 1e-10)
    _safe_hist(ax_log, log_flat, bins=100, edgecolor="none")
    ax_log.set_title(f"{label} Histogram (Log10)")
    ax_log.set_xlabel("Abs Diff")
    ax_log.set_ylabel("Count")
    _format_log_ticks(ax_log, axis="x")


def _draw_scatter_hist2d(
    axes: np.ndarray,
    *,
    row_idx: int,
    baseline: torch.Tensor,
    target: torch.Tensor,
    label: str,
) -> None:
    import matplotlib.pyplot as plt

    ax_normal = axes[row_idx, 0]
    ax_log = axes[row_idx, 1]

    b_flat: np.ndarray = _maybe_downsample_numpy(baseline.flatten())
    t_flat: np.ndarray = _maybe_downsample_numpy(target.flatten())
    min_len: int = min(len(b_flat), len(t_flat))
    b_flat = b_flat[:min_len]
    t_flat = t_flat[:min_len]

    # Normal scale
    lim: float = float(max(np.abs(b_flat).max(), np.abs(t_flat).max())) * 1.05
    if lim == 0:
        lim = 1.0
    _h, _xe, _ye, im = ax_normal.hist2d(
        b_flat,
        t_flat,
        bins=200,
        range=[[-lim, lim], [-lim, lim]],
        cmap="viridis",
        norm="log",
    )
    ax_normal.plot([-lim, lim], [-lim, lim], "r--", linewidth=0.5)
    ax_normal.set_title(f"{label} Hist2D")
    ax_normal.set_xlabel("Baseline")
    ax_normal.set_ylabel("Target")
    ax_normal.set_aspect("equal")
    plt.colorbar(im, ax=ax_normal)

    # Log scale
    b_log: np.ndarray = np.log10(np.abs(b_flat) + 1e-10)
    t_log: np.ndarray = np.log10(np.abs(t_flat) + 1e-10)
    vmin: float = float(min(b_log.min(), t_log.min())) - 0.5
    vmax: float = float(max(b_log.max(), t_log.max())) + 0.5
    _h2, _xe2, _ye2, im2 = ax_log.hist2d(
        b_log,
        t_log,
        bins=200,
        range=[[vmin, vmax], [vmin, vmax]],
        cmap="viridis",
        norm="log",
    )
    ax_log.plot([vmin, vmax], [vmin, vmax], "r--", linewidth=0.5)
    ax_log.set_title(f"{label} Hist2D (Log10 Abs)")
    ax_log.set_xlabel("Baseline")
    ax_log.set_ylabel("Target")
    ax_log.set_aspect("equal")
    plt.colorbar(im2, ax=ax_log)
    _format_log_ticks(ax_log, axis="both")


def _draw_scatter_sampled(
    axes: np.ndarray,
    *,
    row_idx: int,
    baseline: torch.Tensor,
    target: torch.Tensor,
    label: str,
) -> None:
    import matplotlib.pyplot as plt

    ax_baseline = axes[row_idx, 0]
    ax_target = axes[row_idx, 1]

    b_flat: np.ndarray = baseline.flatten().numpy()
    t_flat: np.ndarray = target.flatten().numpy()

    n_samples: int = min(_SCATTER_SAMPLE_SIZE, len(b_flat))
    rng: np.random.Generator = np.random.default_rng(seed=42)
    indices: np.ndarray = np.sort(rng.choice(len(b_flat), n_samples, replace=False))
    b_sampled: np.ndarray = b_flat[indices]
    t_sampled: np.ndarray = t_flat[indices]

    side: int = int(np.sqrt(n_samples))
    n_use: int = side * side
    b_2d: np.ndarray = b_sampled[:n_use].reshape(side, side)
    t_2d: np.ndarray = t_sampled[:n_use].reshape(side, side)

    vmin: float = float(min(b_2d.min(), t_2d.min()))
    vmax: float = float(max(b_2d.max(), t_2d.max()))

    im_b = ax_baseline.imshow(b_2d, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_baseline.set_title(f"{label} Baseline (10k sampled)")
    plt.colorbar(im_b, ax=ax_baseline)

    im_t = ax_target.imshow(t_2d, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_target.set_title(f"{label} Target (10k sampled)")
    plt.colorbar(im_t, ax=ax_target)
