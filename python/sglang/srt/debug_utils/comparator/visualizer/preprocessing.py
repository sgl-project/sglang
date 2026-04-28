"""Tensor preprocessing and utility functions for visualization."""

from __future__ import annotations

import math
import re

import numpy as np
import torch

_DOWNSAMPLE_THRESHOLD: int = 10_000_000
_SCATTER_SAMPLE_SIZE: int = 10_000


def _preprocess_tensor(tensor: torch.Tensor) -> torch.Tensor:
    t: torch.Tensor = tensor.squeeze()

    while t.ndim < 2:
        t = t.unsqueeze(0)
    if t.ndim > 2:
        t = t.reshape(-1, t.shape[-1])

    t = _reshape_to_balanced_aspect(t)
    return t


def _reshape_to_balanced_aspect(
    t: torch.Tensor, max_ratio: float = 5.0
) -> torch.Tensor:
    assert t.ndim == 2

    h, w = t.shape
    ratio: float = h / w if w > 0 else float("inf")

    if 1 / max_ratio <= ratio <= max_ratio:
        return t

    total: int = h * w
    target_side: int = int(math.sqrt(total))

    for new_h in range(target_side, 0, -1):
        if total % new_h == 0:
            new_w: int = total // new_h
            new_ratio: float = new_h / new_w
            if 1 / max_ratio <= new_ratio <= max_ratio:
                return t.reshape(new_h, new_w)

    return t.reshape(1, -1)


# ────────────────────── utility ──────────────────────


def _to_log10(t: torch.Tensor) -> torch.Tensor:
    return t.abs().clamp(min=1e-10).log10()


def _format_log_ticks(ax: object, axis: str = "both") -> None:
    from matplotlib.ticker import FuncFormatter

    formatter = FuncFormatter(
        lambda x, _: f"1e{int(x)}" if x == int(x) else f"1e{x:.1f}"
    )
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(formatter)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(formatter)


def _format_stats(name: str, t: torch.Tensor) -> str:
    return (
        f"{name}: shape={tuple(t.shape)}, "
        f"min={t.min().item():.4g}, max={t.max().item():.4g}, "
        f"mean={t.mean().item():.4g}, std={t.std().item():.4g}"
    )


def _safe_hist(
    ax: object, data: np.ndarray, *, bins: int = 100, **kwargs: object
) -> None:
    data_f64: np.ndarray = data.astype(np.float64)
    try:
        ax.hist(data_f64, bins=bins, **kwargs)
    except ValueError:
        ax.hist(data_f64, bins=max(1, len(np.unique(data_f64[:1000]))), **kwargs)


def _maybe_downsample_numpy(
    t: torch.Tensor,
    max_elements: int = _DOWNSAMPLE_THRESHOLD,
) -> np.ndarray:
    if t.numel() <= max_elements:
        return t.numpy()

    rng: np.random.Generator = np.random.default_rng(seed=0)
    indices: np.ndarray = rng.choice(t.numel(), max_elements, replace=False)
    return t.numpy()[indices]


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[/\.\s]+", "_", name).strip("_")
