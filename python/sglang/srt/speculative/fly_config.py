"""FLy verification settings (see sglang.kernels.ops.speculative.fly)."""

from __future__ import annotations

from typing import Optional

import msgspec


class FlyConfig(msgspec.Struct, frozen=True):
    window_size: int
    entropy_threshold: float = 0.3
    entropy_top_k: int = 3


def resolve_fly_config(
    *,
    gamma: int,
    window_size: Optional[int],
    entropy_threshold: float,
    entropy_top_k: int,
) -> FlyConfig:
    """Validate the FLy settings for ``gamma`` draft tokens per verify; an unset
    window defaults to ``min(6, gamma - 1)`` as in vLLM."""
    if gamma < 2:
        raise ValueError(
            f"FLy requires at least 2 draft tokens per verify, got gamma={gamma}."
        )
    window = min(6, gamma - 1) if window_size is None else int(window_size)
    if not 1 <= window < gamma:
        raise ValueError(
            f"FLy window size must be in [1, gamma - 1] = [1, {gamma - 1}], "
            f"got {window}."
        )
    if entropy_top_k < 1:
        raise ValueError(f"FLy entropy top-k must be positive, got {entropy_top_k}.")
    if entropy_threshold < 0:
        raise ValueError(
            f"FLy entropy threshold must be non-negative, got {entropy_threshold}."
        )
    return FlyConfig(
        window_size=window,
        entropy_threshold=float(entropy_threshold),
        entropy_top_k=int(entropy_top_k),
    )
