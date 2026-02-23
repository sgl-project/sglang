from typing import Optional

from pydantic import BaseModel, ConfigDict


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


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


class TensorComparisonInfo(_StrictBase):
    name: str
    baseline: TensorInfo
    target: TensorInfo
    unified_shape: Optional[list[int]]
    shape_mismatch: bool
    diff: Optional[DiffInfo] = None
    diff_downcast: Optional[DiffInfo] = None
    downcast_dtype: Optional[str] = None
