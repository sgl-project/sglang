from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange

from sglang.srt.debug_utils.comparator.dims import parse_dims
from sglang.srt.debug_utils.comparator.utils import Pair, _FrozenBase
from sglang.srt.debug_utils.comparator.warning_sink import warning_sink

# --- types ---


class AxisSwapperPlan(_FrozenBase):
    pattern: str  # einops pattern, e.g. "t h d -> t d h"


# --- planner ---


def compute_axis_swapper_plan(
    dims_str_pair: Pair[Optional[str]],
) -> Optional[AxisSwapperPlan]:
    if dims_str_pair.x is None or dims_str_pair.y is None:
        return None

    x_names: list[str] = [spec.name for spec in parse_dims(dims_str_pair.x)]
    y_names: list[str] = [spec.name for spec in parse_dims(dims_str_pair.y)]

    if x_names == y_names:
        return None

    if set(x_names) != set(y_names):
        # Local import to avoid circular dependency:
        # output_types -> aligner/entrypoint/types -> axis_swapper -> output_types
        from sglang.srt.debug_utils.comparator.output_types import GeneralWarning

        warning_sink.add(
            GeneralWarning(
                category="axis_swapper_dim_mismatch",
                message=(
                    f"AxisSwapper: dim name sets differ (x={x_names}, y={y_names}), "
                    f"skipping axis swap"
                ),
            )
        )
        return None

    pattern: str = f"{' '.join(x_names)} -> {' '.join(y_names)}"
    return AxisSwapperPlan(pattern=pattern)


# --- executor ---


def execute_axis_swapper_plan(
    tensor: torch.Tensor, plan: AxisSwapperPlan
) -> torch.Tensor:
    return rearrange(tensor.rename(None), plan.pattern)
