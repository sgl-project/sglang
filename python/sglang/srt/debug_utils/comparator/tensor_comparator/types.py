from typing import Optional

from sglang.srt.debug_utils.comparator.utils import _StrictBase


class TensorStats(_StrictBase):
    mean: float
    std: float
    min: float
    max: float
    p1: Optional[float] = None
    p5: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None


class TensorInfo(_StrictBase):
    shape: list[int]
    dtype: str
    stats: TensorStats
    sample: Optional[str] = None


class DiffInfo(_StrictBase):
    rel_diff: float
    max_abs_diff: float
    mean_abs_diff: float
    max_diff_coord: list[int]
    baseline_at_max: float
    target_at_max: float
    diff_threshold: float
    passed: bool


class TensorComparisonInfo(_StrictBase):
    name: str
    baseline: TensorInfo
    target: TensorInfo
    unified_shape: Optional[list[int]]
    shape_mismatch: bool
    diff: Optional[DiffInfo] = None
    diff_downcast: Optional[DiffInfo] = None
    downcast_dtype: Optional[str] = None
