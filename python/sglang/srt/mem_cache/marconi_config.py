from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

DEFAULT_MARCONI_EFF_WEIGHT = 0.75
DEFAULT_MAMBA_TRACK_BUFFER_SIZE = 2


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

    @classmethod
    def enabled(
        cls,
        *,
        model_stats: Optional[MarconiModelStats] = None,
        eff_weight: Optional[float] = None,
    ) -> "MarconiConfig":
        return cls(
            enable=True,
            eff_weight=(
                eff_weight if eff_weight is not None else DEFAULT_MARCONI_EFF_WEIGHT
            ),
            model_stats=model_stats,
        )


def get_marconi_branch_align_interval(
    page_size: Optional[int] = None, *, align_interval: int = 512
) -> int:
    if page_size is not None and page_size > 0:
        if align_interval % page_size != 0:
            raise ValueError("Marconi branch alignment must be divisible by page_size.")
    return align_interval
