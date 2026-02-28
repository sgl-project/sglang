from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange

from sglang.srt.debug_utils.comparator.dims import (
    _SingletonDimUtil,
    parse_dims,
)
from sglang.srt.debug_utils.comparator.utils import Pair, _FrozenBase
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink

# --- types ---


class AxisAlignerPlan(_FrozenBase):
    pattern: Pair[Optional[str]]  # einops pattern per side, None = no-op


# --- planner ---


def compute_axis_aligner_plan(
    dims_str_pair: Pair[Optional[str]],
) -> Optional[AxisAlignerPlan]:
    if dims_str_pair.x is None or dims_str_pair.y is None:
        return None

    # Pair[str] after None check
    dims_pair: Pair[str] = Pair(x=dims_str_pair.x, y=dims_str_pair.y)

    raw_names: Pair[list[str]] = dims_pair.map(
        lambda s: [spec.name for spec in parse_dims(s)]
    )
    filtered_names: Pair[list[str]] = dims_pair.map(
        lambda s: [spec.name for spec in _SingletonDimUtil.filter_out(parse_dims(s))]
    )

    target_order: Optional[list[str]] = _resolve_target_order(
        x_names=filtered_names.x, y_names=filtered_names.y
    )
    if target_order is None:
        return None

    pattern: Pair[Optional[str]] = raw_names.map(
        lambda names: _build_pattern(source=names, target=target_order)
    )

    if pattern.x is None and pattern.y is None:
        return None

    return AxisAlignerPlan(pattern=pattern)


def _resolve_target_order(
    x_names: list[str], y_names: list[str]
) -> Optional[list[str]]:
    """Determine the canonical dim order both sides should align to.

    Returns y_names (the target ordering) if name sets match, or None on mismatch.
    If both sides are identical, returns the shared order.
    """
    if x_names == y_names:
        return y_names

    if set(x_names) != set(y_names):
        # Local import to avoid circular dependency:
        # output_types -> aligner/entrypoint/types -> axis_aligner -> output_types
        from sglang.srt.debug_utils.comparator.output_types import GeneralWarning

        warning_sink.add(
            GeneralWarning(
                category="axis_aligner_dim_mismatch",
                message=(
                    f"AxisAligner: dim name sets differ (x={x_names}, y={y_names}), "
                    f"skipping axis swap"
                ),
            )
        )
        return None

    return y_names


def _build_pattern(*, source: list[str], target: list[str]) -> Optional[str]:
    """Build an einops rearrange pattern from source dim names to target dim names.

    Returns None if source already matches target (no rearrange needed).
    """
    if source == target:
        return None

    return f"{' '.join(source)} -> {' '.join(target)}"


# --- executor ---


def execute_axis_aligner_plan(
    tensor: torch.Tensor, plan: AxisAlignerPlan, *, side: str
) -> torch.Tensor:
    pattern: Optional[str] = plan.pattern.x if side == "x" else plan.pattern.y

    if pattern is not None:
        tensor = rearrange(tensor.rename(None), pattern)

    return tensor
