from typing import Optional

from sglang.srt.debug_utils.comparator.dims import ParallelAxis
from sglang.srt.debug_utils.comparator.unshard.types import AxisInfo

_PARALLEL_INFO_KEYS = ("sglang_parallel_info", "megatron_parallel_info")

_AXIS_PREFIXES = [e.value for e in ParallelAxis]


def normalize_parallel_info(meta: dict) -> dict[str, AxisInfo]:
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

    result: dict[str, AxisInfo] = {}
    for prefix in _AXIS_PREFIXES:
        rank_key = f"{prefix}_rank"
        size_key = f"{prefix}_size"
        if rank_key in info and size_key in info:
            axis_size = info[size_key]
            if axis_size > 1:
                result[prefix] = AxisInfo(
                    axis_rank=info[rank_key],
                    axis_size=axis_size,
                )

    return result
