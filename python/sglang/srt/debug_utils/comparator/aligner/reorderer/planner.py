from sglang.srt.debug_utils.comparator.aligner.reorderer.types import (
    ReordererPlan,
    ZigzagToNaturalParams,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import AxisInfo
from sglang.srt.debug_utils.comparator.dims import DimSpec, Ordering, ParallelAxis

_ALLOWED_ZIGZAG_DIM_NAMES: set[str] = {"s"}


def compute_reorderer_plans(
    dim_specs: list[DimSpec],
    parallel_infos: list[dict[ParallelAxis, AxisInfo]],
) -> list[ReordererPlan]:
    plans: list[ReordererPlan] = []

    for dim_index, spec in enumerate(dim_specs):
        if (
            spec.ordering is not None
            and spec.ordering != Ordering.NATURAL
            and spec.parallel is not None
        ):
            if spec.name not in _ALLOWED_ZIGZAG_DIM_NAMES:
                raise ValueError(
                    f"Zigzag ordering is only supported on sequence dims "
                    f"(bshd/sbhd format, dim name must be one of "
                    f"{sorted(_ALLOWED_ZIGZAG_DIM_NAMES)}), "
                    f"but got dim name {spec.name!r} in {spec}"
                )

            assert spec.ordering == Ordering.ZIGZAG
            axis_size: int = parallel_infos[0][spec.parallel].axis_size
            plans.append(
                ReordererPlan(
                    params=ZigzagToNaturalParams(dim=dim_index, cp_size=axis_size),
                )
            )

    return plans
