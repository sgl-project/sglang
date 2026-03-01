"""Per-token relative difference heatmap generator.

Produces a single PNG with rows = tensor names, columns = token positions,
color = log10(rel_diff).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sglang.srt.debug_utils.comparator.output_types import ComparisonRecord


def generate_per_token_heatmap(
    *,
    records: list[ComparisonRecord],
    output_path: Path,
) -> Optional[Path]:
    """Generate a per-token relative difference heatmap PNG.

    Returns the output path if a file was written, or None if no data was available.
    """
    rows_data: list[tuple[str, list[float]]] = _collect_per_token_data(records=records)
    if not rows_data:
        return None

    _render_heatmap(rows_data=rows_data, output_path=output_path)
    return output_path


def _collect_per_token_data(
    *,
    records: list[ComparisonRecord],
) -> list[tuple[str, list[float]]]:
    rows: list[tuple[str, list[float]]] = []
    for record in records:
        if record.diff is None or record.diff.per_token_rel_diff is None:
            continue
        rows.append((record.name, record.diff.per_token_rel_diff))
    return rows


def _render_heatmap(
    *,
    rows_data: list[tuple[str, list[float]]],
    output_path: Path,
) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    max_len: int = max(len(vals) for _, vals in rows_data)
    labels: list[str] = [label for label, _ in rows_data]

    matrix: np.ndarray = np.full((len(rows_data), max_len), np.nan, dtype=np.float64)
    for i, (_, vals) in enumerate(rows_data):
        matrix[i, : len(vals)] = vals

    fig_width: float = max(12.0, max_len * 0.15)
    fig_height: float = max(6.0, len(rows_data) * 0.3)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        np.log10(matrix + 1e-10), aspect="auto", cmap="hot", interpolation="nearest"
    )

    ax.set_xlabel("Token Position")
    ax.set_ylabel("Tensor")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    colorbar = fig.colorbar(im, ax=ax)
    colorbar.set_label("log10(rel_diff)")

    ax.set_title("Per-Token Relative Difference Heatmap")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
