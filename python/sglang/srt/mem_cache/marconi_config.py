from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True, kw_only=True)
class MarconiModelStats:
    model_dim: int
    ssm_state_size: int
    num_mamba_layers: int
    num_attn_layers: int
    num_mlp_layers: int
    kv_cache_dtype_size: int
    mamba_state_size_bytes: int


@dataclass(frozen=True, kw_only=True)
class MarconiConfig:
    enable: bool
    eff_weight: float
    bootstrap_window_size: Optional[int]
    bootstrap_multiplier: int
    tuning_interval: int
    tuning_weights: Optional[Sequence[float]] = None
    model_stats: Optional[MarconiModelStats] = None
