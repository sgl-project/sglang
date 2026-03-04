from __future__ import annotations

from enum import Enum
from typing import Optional

from sglang.srt.debug_utils.comparator.utils import _FrozenBase

TOKEN_DIM_NAME: str = "t"
BATCH_DIM_NAME: str = "b"
SEQ_DIM_NAME: str = "s"
SQUEEZE_DIM_NAME: str = "1"


class TokenLayout(Enum):
    T = "t"  # single flat token dim
    BS = "bs"  # separate batch + seq dims, need collapse


class ParallelAxis(Enum):
    TP = "tp"
    CP = "cp"
    EP = "ep"
    SP = "sp"
    RECOMPUTE_PSEUDO = "recompute_pseudo"


class Ordering(Enum):
    ZIGZAG = "zigzag"
    NATURAL = "natural"


class Reduction(Enum):
    PARTIAL = "partial"


class ParallelModifier(_FrozenBase):
    axis: ParallelAxis
    ordering: Optional[Ordering] = None
    reduction: Optional[Reduction] = None


_AXIS_LOOKUP: dict[str, ParallelAxis] = {m.value: m for m in ParallelAxis}
_QUALIFIER_LOOKUP: dict[str, Ordering | Reduction] = {
    **{m.value: m for m in Ordering},
    **{m.value: m for m in Reduction},
}

_FUSED_NAME_SEP: str = "___"


class DimSpec(_FrozenBase):
    name: str
    parallel_modifiers: list[ParallelModifier] = []

    @property
    def sub_dims(self) -> list[str]:
        """Sub-dim names. Fused: ``["num_heads", "head_dim"]``; plain: ``["h"]``."""
        return self.name.split("*")

    @property
    def is_fused(self) -> bool:
        return len(self.sub_dims) > 1

    @property
    def sanitized_name(self) -> str:
        """Name safe for PyTorch named tensors (``*`` → ``___``)."""
        if self.is_fused:
            return _FUSED_NAME_SEP.join(self.sub_dims)
        return self.name


class DimsSpec(_FrozenBase):
    """Parsed result of a full dims string like ``"b s h[tp] # dp:=moe_dp"``."""

    dims: list[DimSpec]
    dp_group_alias: Optional[str] = None
    replicated_axes: frozenset[ParallelAxis] = frozenset()
