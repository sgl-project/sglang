from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

DEFAULT_MARCONI_ADMISSION_DECAY = 0.99
DEFAULT_MARCONI_ADMISSION_MIN_HITS = 2
DEFAULT_MARCONI_ADMISSION_MIN_SUCCESS_RATIO = 0.1
DEFAULT_MARCONI_ADMISSION_SCORE_THRESHOLD = 0.5
DEFAULT_MARCONI_EFF_WEIGHT_TAXONOMY = 0.75
DEFAULT_MARCONI_EFF_WEIGHT_THRESHOLD = 0.0
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
    admission_policy: str = "thresholded"
    admission_min_hits: int = DEFAULT_MARCONI_ADMISSION_MIN_HITS
    admission_min_success_ratio: float = DEFAULT_MARCONI_ADMISSION_MIN_SUCCESS_RATIO
    admission_score_threshold: float = DEFAULT_MARCONI_ADMISSION_SCORE_THRESHOLD
    admission_decay: float = DEFAULT_MARCONI_ADMISSION_DECAY
    eviction_hot_weight: float = 0.2
    mamba_layer_mask: Optional[Union[str, Sequence[int]]] = None
    two_pass_branch_prefill: bool = True
