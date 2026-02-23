from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class TensorStats:
    mean: float
    std: float
    min: float
    max: float
    p1: Optional[float] = None
    p5: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None


@dataclass
class TensorInfo:
    shape: torch.Size
    dtype: torch.dtype
    stats: TensorStats
    sample: Optional[str] = None


@dataclass
class DiffInfo:
    rel_diff: float
    max_abs_diff: float
    mean_abs_diff: float
    max_diff_coord: Tuple[int, ...]
    baseline_at_max: float
    target_at_max: float


@dataclass
class TensorComparisonInfo:
    name: str
    baseline: TensorInfo
    target: TensorInfo
    unified_shape: Optional[torch.Size]
    shape_mismatch: bool
    diff: Optional[DiffInfo] = None
    diff_downcast: Optional[DiffInfo] = None
    downcast_dtype: Optional[torch.dtype] = None
