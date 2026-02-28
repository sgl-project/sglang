from typing import Optional

from sglang.srt.debug_utils.comparator.aligner.unsharder.types import AxisInfo
from sglang.srt.debug_utils.comparator.dims import ParallelAxis

_PARALLEL_INFO_KEYS = ("sglang_parallel_info", "megatron_parallel_info")


def _is_error_sentinel(value: dict) -> bool:
    """Check if a parallel_info dict is an error sentinel (e.g. {'megatron_error': True})."""
    return any(k.endswith("_error") for k in value)


def normalize_parallel_info(meta: dict) -> dict[ParallelAxis, AxisInfo]:
    """Extract unified parallel info from dump meta."""
    info: Optional[dict] = None
    for key in _PARALLEL_INFO_KEYS:
        value = meta.get(key)
        if isinstance(value, dict) and value and not _is_error_sentinel(value):
            if info is not None:
                raise ValueError(
                    f"Meta contains multiple parallel_info keys among {_PARALLEL_INFO_KEYS}"
                )
            info = value

    if info is None:
        info = {}

    result: dict[ParallelAxis, AxisInfo] = {}
    for axis in ParallelAxis:
        axis_rank = info.get(f"{axis.value}_rank")
        axis_size = info.get(f"{axis.value}_size")

        # Recompute pseudo-axis lives at top-level meta, not inside parallel_info
        if axis_rank is None:
            axis_rank = meta.get(f"{axis.value}_rank")
            axis_size = meta.get(f"{axis.value}_size")

        if axis_rank is not None and axis_size is not None and axis_size > 1:
            result[axis] = AxisInfo(
                axis_rank=axis_rank,
                axis_size=axis_size,
            )

    return result
