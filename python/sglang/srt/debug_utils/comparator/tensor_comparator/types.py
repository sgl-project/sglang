from typing import Optional

from sglang.srt.debug_utils.comparator.utils import _StrictBase

DEFAULT_PERCENTILES: tuple[int, ...] = (1, 5, 50, 95, 99)


class TensorStats(_StrictBase):
    mean: float
    abs_mean: float
    std: float
    min: float
    max: float
    percentiles: dict[int, float] = {}


class TensorInfo(_StrictBase):
    shape: list[int]
    dtype: str
    stats: TensorStats
    sample: Optional[str] = None


class DiffInfo(_StrictBase):
    rel_diff: float
    max_abs_diff: float
    mean_abs_diff: float
    abs_diff_percentiles: dict[int, float] = {}
    max_diff_coord: list[int]
    baseline_at_max: float
    target_at_max: float
    diff_threshold: float
    passed: bool
    per_token_rel_diff: Optional[list[float]] = None


class TensorComparisonInfo(_StrictBase):
    name: str
    baseline: TensorInfo
    target: TensorInfo
    unified_shape: Optional[list[int]]
    shape_mismatch: bool
    diff: Optional[DiffInfo] = None
    diff_downcast: Optional[DiffInfo] = None
    downcast_dtype: Optional[str] = None
