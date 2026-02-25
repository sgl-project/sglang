from typing import Optional

from sglang.srt.debug_utils.comparator.aligner.unsharder.types import AxisInfo
from sglang.srt.debug_utils.comparator.dims import ParallelAxis

_PARALLEL_INFO_KEYS = ("sglang_parallel_info", "megatron_parallel_info")


def normalize_parallel_info(meta: dict) -> dict[ParallelAxis, AxisInfo]:
    """Extract unified parallel info from dump meta."""
    info: Optional[dict] = None
    for key in _PARALLEL_INFO_KEYS:
        value = meta.get(key)
        if isinstance(value, dict) and value:
            if info is not None:
                raise ValueError(
                    f"Meta contains multiple parallel_info keys among {_PARALLEL_INFO_KEYS}"
                )
            info = value

    if info is None:
        return {}

    result: dict[ParallelAxis, AxisInfo] = {}
    for axis in ParallelAxis:
        axis_rank = info.get(f"{axis.value}_rank")
        axis_size = info.get(f"{axis.value}_size")
        if axis_rank is not None and axis_size is not None and axis_size > 1:
            result[axis] = AxisInfo(
                axis_rank=axis_rank,
                axis_size=axis_size,
            )

    return result
