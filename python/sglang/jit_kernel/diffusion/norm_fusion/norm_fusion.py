from enum import Enum
from typing import Optional

import torch


# Fused kernels (norm fusion) often involve multiple inputs with different shapes.
# Different shapes require different indexing logic, shape validation, and
# special constraints, which easily leads to fragmented and hard-to-maintain
# implementations.
# IndexEnum consolidates all shape patterns currently encountered and provides
# a unified abstraction for their corresponding indexing rules, shape checks,
# and related constraints:
# - Scalar: [1]
# - NoBroadcast: [B, S, D]
# - BroadcastB: [1, S, D]
# - BroadcastS: [B, D], [B, 1, D]
# - BroadcastBS: [D], [1, D], [1, 1, D]
# - BF1D: [B, F, 1, D]
class IndexEnum(Enum):
    NotATensor = -1
    Scalar = 0
    NoBroadcast = 1
    BroadcastB = 2
    BroadcastS = 3
    BroadcastBS = 4
    BF1D = 5


def get_index_enum(t: Optional[torch.Tensor]) -> IndexEnum:
    if not isinstance(t, torch.Tensor):
        return IndexEnum.NotATensor
    ndim = t.ndim
    shape = t.shape
    # 1D cases
    # [1]        -> Scalar
    # [D]        -> BroadcastBS
    if ndim == 1:
        if shape[0] == 1:
            return IndexEnum.Scalar
        else:
            return IndexEnum.BroadcastBS
    # 2D cases
    # [B, D]        -> BroadcastS
    # [1, D]        -> BroadcastBS
    if ndim == 2:
        if shape[0] == 1:
            return IndexEnum.BroadcastBS
        else:
            return IndexEnum.BroadcastS
    # 3D cases
    # [B, S, D]     -> NoBroadcast
    # [1, S, D]     -> BroadcastB
    # [B, 1, D]     -> BroadcastS
    # [1, 1, D]     -> BroadcastBS
    if ndim == 3:
        if shape[0] == 1 and shape[1] == 1:
            return IndexEnum.BroadcastBS
        if shape[0] == 1:
            return IndexEnum.BroadcastB
        if shape[1] == 1:
            return IndexEnum.BroadcastS
        return IndexEnum.NoBroadcast
    # 4D case
    # [B, F, 1, D]  -> BF1D
    if ndim == 4:
        if shape[2] == 1:
            return IndexEnum.BF1D
    return IndexEnum.NotATensor


class NormEnum(Enum):
    LayerNorm = 0
    RMSNorm = 1


def get_norm_enum(norm_str: str) -> NormEnum:
    if norm_str == "layer":
        return NormEnum.LayerNorm
    elif norm_str == "rms":
        return NormEnum.RMSNorm
    raise ValueError(
        f"Unsupported norm type: '{norm_str}'. Expected one of: 'layer', 'rms'."
    )
