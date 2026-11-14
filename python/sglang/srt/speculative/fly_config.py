"""FLy verification settings (see sglang.kernels.ops.speculative.fly)."""

from __future__ import annotations

from typing import Optional

import msgspec

# The sampling pre-passes keep one accept bit per draft in an int32, and the
# entropy kernels unroll the top-k extraction once per walked draft position.
MAX_FLY_GAMMA = 31
MAX_FLY_ENTROPY_TOP_K = 16


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
    if not 2 <= gamma <= MAX_FLY_GAMMA:
        raise ValueError(
            f"FLy requires 2 to {MAX_FLY_GAMMA} draft tokens per verify, "
            f"got gamma={gamma}."
        )
    window = min(6, gamma - 1) if window_size is None else int(window_size)
    if not 1 <= window < gamma:
        raise ValueError(
            f"FLy window size must be in [1, gamma - 1] = [1, {gamma - 1}], "
            f"got {window}."
        )
    if not 1 <= entropy_top_k <= MAX_FLY_ENTROPY_TOP_K:
        raise ValueError(
            f"FLy entropy top-k must be in [1, {MAX_FLY_ENTROPY_TOP_K}], "
            f"got {entropy_top_k}."
        )
    if not entropy_threshold >= 0:
        raise ValueError(
            f"FLy entropy threshold must be non-negative, got {entropy_threshold}."
        )
    return FlyConfig(
        window_size=window,
        entropy_threshold=float(entropy_threshold),
        entropy_top_k=int(entropy_top_k),
    )
