from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union


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
    admission_min_hits: int = 2
    admission_min_success_ratio: float = 0.1
    admission_decay: float = 0.995
    admission_score_threshold: float = 1.0
    admission_max_nodes: Optional[int] = None
    admission_max_tokens: Optional[int] = None
    admission_prune_interval: int = 200
    eviction_hot_weight: float = 0.2
    eviction_pin_threshold: int = 8
    eviction_pin_ttl: int = 500
    eviction_regret_window: int = 1000
    eviction_regret_max_entries: int = 20000
    eviction_latency_weights: Optional[Sequence[float]] = None
    tuning_max_workers: Optional[int] = None
    attn_only_reuse: bool = False
    track_buffer_size: Optional[int] = None
    track_max_points: Optional[int] = None
    mamba_layer_mask: Optional[Union[str, Sequence[int]]] = None
