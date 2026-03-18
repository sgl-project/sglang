"""Main orchestration logic for comparison figure generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from sglang.srt.debug_utils.comparator.visualizer.preprocessing import (
    _preprocess_tensor,
)


@dataclass(frozen=True)
class _PanelContext:
    baseline_2d: torch.Tensor
    target_2d: torch.Tensor
    diff: Optional[torch.Tensor]  # None when shapes differ
    name: str


@dataclass(frozen=True)
class _Panel:
    label: str
    requires_diff: bool
    draw: Callable[[np.ndarray, int, _PanelContext], Optional[str]]


def _build_panels() -> list[_Panel]:
    from sglang.srt.debug_utils.comparator.visualizer.panels import (
        _draw_baseline_heatmap,
        _draw_diff_heatmap,
        _draw_diff_histogram,
        _draw_hist2d,
        _draw_sampled,
        _draw_target_heatmap,
    )

    return [
        _Panel(
            label="Baseline Heatmap", requires_diff=False, draw=_draw_baseline_heatmap
        ),
        _Panel(label="Target Heatmap", requires_diff=False, draw=_draw_target_heatmap),
        _Panel(label="Abs Diff Heatmap", requires_diff=True, draw=_draw_diff_heatmap),
        _Panel(label="Abs Diff Hist", requires_diff=True, draw=_draw_diff_histogram),
        _Panel(label="Hist2D", requires_diff=True, draw=_draw_hist2d),
        _Panel(label="Sampled", requires_diff=True, draw=_draw_sampled),
    ]


def generate_comparison_figure(
    *,
    baseline: torch.Tensor,
    target: torch.Tensor,
    name: str,
    output_path: Path,
) -> None:
    """Generate a multi-panel comparison PNG for a baseline/target tensor pair.

    Panels (6 rows x 2 cols, left=normal, right=log10):
      Row 0: Baseline heatmap
      Row 1: Target heatmap
      Row 2: Abs Diff heatmap
      Row 3: Abs Diff histogram
      Row 4: Hist2D scatter (baseline vs target density)
      Row 5: Sampled scatter (10k sampled mini-heatmap)
    """
    import matplotlib.pyplot as plt

    baseline_f: torch.Tensor = baseline.detach().cpu().float()
    target_f: torch.Tensor = target.detach().cpu().float()

    can_diff: bool = baseline_f.shape == target_f.shape

    baseline_2d: torch.Tensor = _preprocess_tensor(baseline_f)
    target_2d: torch.Tensor = _preprocess_tensor(target_f)

    diff: Optional[torch.Tensor] = (baseline_2d - target_2d).abs() if can_diff else None

    ctx = _PanelContext(
        baseline_2d=baseline_2d,
        target_2d=target_2d,
        diff=diff,
        name=name,
    )

    panels: list[_Panel] = _build_panels()
    active: list[_Panel] = [p for p in panels if not p.requires_diff or can_diff]

    nrows: int = len(active)
    ncols: int = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    stats_lines: list[str] = []
    for i, panel in enumerate(active):
        stats_line: Optional[str] = panel.draw(axes, i, ctx)
        if stats_line is not None:
            stats_lines.append(stats_line)

    num_stats: int = len(stats_lines)
    title_height: float = 0.015 * num_stats + 0.015
    fig.suptitle(
        "\n".join(stats_lines),
        fontsize=9,
        family="monospace",
        y=1 - title_height / 2,
    )
    plt.tight_layout(rect=[0, 0, 1, 1 - title_height])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
