from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

DEFAULT_MARCONI_EFF_WEIGHT = 0.75
DEFAULT_MAMBA_TRACK_BUFFER_SIZE = 2
DEFAULT_MAMBA_TRACK_MAX_POINTS = 2


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
    model_stats: Optional[MarconiModelStats] = None
    eviction_hot_weight: float = 0.0
    mamba_layer_mask: Optional[Union[str, Sequence[int]]] = None
    two_pass_branch_prefill: bool = True
    admission_max_nodes: Optional[int] = None
    admission_max_tokens: Optional[int] = None
    admission_prune_interval: int = 200
