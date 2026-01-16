from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True, kw_only=True)
class MarconiConfig:
    enable: bool
    eviction_policy: str
    eff_weight: float
    bootstrap_window_size: Optional[int]
    bootstrap_multiplier: int
    tuning_interval: int
    tuning_weights: Optional[Sequence[float]] = None
