from typing import Optional

from sglang.srt.debug_utils.comparator.aligner.reorderer.types import (
    ReordererPlan,
    ZigzagToNaturalParams,
    ZigzagToNaturalThdParams,
)
from sglang.srt.debug_utils.comparator.aligner.unsharder.types import AxisInfo
from sglang.srt.debug_utils.comparator.dims import (
    SEQ_DIM_NAME,
    TOKEN_DIM_NAME,
    DimSpec,
    Ordering,
    ParallelAxis,
)

_ALLOWED_ZIGZAG_DIM_NAMES: set[str] = {SEQ_DIM_NAME, TOKEN_DIM_NAME}


def compute_reorderer_plans(
    dim_specs: list[DimSpec],
    parallel_infos: list[dict[ParallelAxis, AxisInfo]],
    *,
    thd_global_seq_lens: Optional[list[int]] = None,
) -> list[ReordererPlan]:
    plans: list[ReordererPlan] = []

    for spec in dim_specs:
        if (
            spec.ordering is not None
            and spec.ordering != Ordering.NATURAL
            and spec.parallel is not None
        ):
            if spec.name not in _ALLOWED_ZIGZAG_DIM_NAMES:
                raise ValueError(
                    f"Zigzag ordering is only supported on sequence dims "
                    f"(dim name must be one of "
                    f"{sorted(_ALLOWED_ZIGZAG_DIM_NAMES)}), "
                    f"but got dim name {spec.name!r} in {spec}"
                )

            if spec.ordering != Ordering.ZIGZAG:
                raise ValueError(
                    f"Unsupported ordering {spec.ordering!r} for dim {spec.name!r}"
                )
            axis_size: int = parallel_infos[0][spec.parallel].axis_size

            if spec.name == TOKEN_DIM_NAME:
                if thd_global_seq_lens is None:
                    raise ValueError(
                        "thd_global_seq_lens is required for zigzag reorder on 't' dimension"
                    )
                params = ZigzagToNaturalThdParams(
                    dim_name=spec.name,
                    cp_size=axis_size,
                    seq_lens=thd_global_seq_lens,
                )
            elif spec.name == SEQ_DIM_NAME:
                params = ZigzagToNaturalParams(dim_name=spec.name, cp_size=axis_size)
            else:
                raise ValueError(
                    f"Unsupported zigzag dim name {spec.name!r}, "
                    f"expected one of {sorted(_ALLOWED_ZIGZAG_DIM_NAMES)}"
                )

            plans.append(ReordererPlan(params=params))

    return plans
