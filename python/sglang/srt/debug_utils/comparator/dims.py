import re
from enum import Enum
from typing import Optional

import torch

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


class DimSpec(_FrozenBase):
    name: str
    parallel_modifiers: list[ParallelModifier] = []


class DimsSpec(_FrozenBase):
    """Parsed result of a full dims string like ``"b s h(tp) # dp:=moe_dp"``."""

    dims: list[DimSpec]
    dp_group_alias: Optional[str] = None


class _SingletonDimUtil:
    """Utilities for squeeze dims (name="1") and their singleton tensor-name mapping."""

    PREFIX: str = "singleton"

    @staticmethod
    def is_squeeze(spec: DimSpec) -> bool:
        return spec.name == SQUEEZE_DIM_NAME

    @staticmethod
    def filter_out(dim_specs: list[DimSpec]) -> list[DimSpec]:
        return [s for s in dim_specs if not _SingletonDimUtil.is_squeeze(s)]

    @staticmethod
    def make_name(index: int) -> str:
        return f"{_SingletonDimUtil.PREFIX}{index}"

    @staticmethod
    def is_singleton_name(name: str) -> bool:
        return (
            name.startswith(_SingletonDimUtil.PREFIX)
            and name[len(_SingletonDimUtil.PREFIX) :].isdigit()
        )

    @staticmethod
    def sanitize_names(names: list[str]) -> list[str]:
        """Replace '1' with 'singleton0', 'singleton1', ... for named tensor compatibility."""
        result: list[str] = []
        sq_idx: int = 0

        for name in names:
            if name == SQUEEZE_DIM_NAME:
                result.append(_SingletonDimUtil.make_name(sq_idx))
                sq_idx += 1
            else:
                result.append(name)

        return result


_DIM_PATTERN = re.compile(r"^(?P<name>[a-zA-Z_]\w*)(?:\((?P<modifiers>[^)]+)\))?$")

_AXIS_LOOKUP: dict[str, ParallelAxis] = {m.value: m for m in ParallelAxis}
_QUALIFIER_LOOKUP: dict[str, Ordering | Reduction] = {
    **{m.value: m for m in Ordering},
    **{m.value: m for m in Reduction},
}


def _parse_modifier_token(modifier_token: str, dim_token: str) -> ParallelModifier:
    """Parse 'sp', 'cp:zigzag', 'tp:partial', or 'cp:zigzag+partial' → ParallelModifier.

    Format: ``axis`` or ``axis:qual`` or ``axis:qual+qual``.
    Colon separates axis from qualifiers; ``+`` separates multiple qualifiers.
    """
    axis_str: str
    qualifiers_str: str
    if ":" in modifier_token:
        axis_str, qualifiers_str = modifier_token.split(":", maxsplit=1)
    else:
        axis_str, qualifiers_str = modifier_token, ""

    axis_str = axis_str.strip()
    axis: Optional[ParallelAxis] = _AXIS_LOOKUP.get(axis_str)
    if axis is None:
        raise ValueError(
            f"Unknown axis {axis_str!r} in modifier {modifier_token!r} "
            f"of dim spec: {dim_token!r}"
        )

    ordering: Optional[Ordering] = None
    reduction: Optional[Reduction] = None

    for q_str in (q.strip() for q in qualifiers_str.split("+") if q.strip()):
        qualifier: Optional[Ordering | Reduction] = _QUALIFIER_LOOKUP.get(q_str)
        if qualifier is None:
            raise ValueError(
                f"Unknown qualifier {q_str!r} in modifier "
                f"{modifier_token!r} of dim spec: {dim_token!r}"
            )
        if isinstance(qualifier, Ordering):
            if ordering is not None:
                raise ValueError(
                    f"Multiple ordering values in modifier "
                    f"{modifier_token!r} of dim spec: {dim_token!r}"
                )
            ordering = qualifier
        else:
            if reduction is not None:
                raise ValueError(
                    f"Multiple reduction values in modifier "
                    f"{modifier_token!r} of dim spec: {dim_token!r}"
                )
            reduction = qualifier

    return ParallelModifier(axis=axis, ordering=ordering, reduction=reduction)


def parse_dim(token: str) -> DimSpec:
    if token == SQUEEZE_DIM_NAME:
        return DimSpec(name=SQUEEZE_DIM_NAME)

    match = _DIM_PATTERN.match(token)
    if match is None:
        raise ValueError(f"Invalid dim token: {token!r}")

    name: str = match.group("name")
    modifiers_str: Optional[str] = match.group("modifiers")

    if modifiers_str is None:
        return DimSpec(name=name)

    modifiers: list[ParallelModifier] = []
    seen_axes: set[ParallelAxis] = set()

    for modifier_token in (p.strip() for p in modifiers_str.split(",")):
        modifier: ParallelModifier = _parse_modifier_token(modifier_token, token)
        if modifier.axis in seen_axes:
            raise ValueError(
                f"Duplicate axis {modifier.axis.value!r} in dim spec: {token!r}"
            )
        seen_axes.add(modifier.axis)
        modifiers.append(modifier)

    return DimSpec(name=name, parallel_modifiers=modifiers)


def parse_dims(dims_str: str) -> DimsSpec:
    """Parse ``"b s(cp:zigzag) h(tp) d # dp:=moe_dp"`` → :class:`DimsSpec`.

    The shape part (before ``#``) produces :pyattr:`DimsSpec.dims`.
    The declaration part (after ``#``) is scanned for ``dp:=<group>``
    which populates :pyattr:`DimsSpec.dp_group_alias`.
    """
    parts: list[str] = dims_str.split("#", maxsplit=1)
    raw: str = parts[0]

    if not raw.strip():
        raise ValueError("dims string must not be empty")

    dims: list[DimSpec] = [parse_dim(token) for token in raw.strip().split()]

    non_squeeze_names: list[str] = [
        spec.name for spec in dims if not _SingletonDimUtil.is_squeeze(spec)
    ]
    if len(non_squeeze_names) != len(set(non_squeeze_names)):
        duplicates = sorted(
            {n for n in non_squeeze_names if non_squeeze_names.count(n) > 1}
        )
        raise ValueError(f"Duplicate dim names: {duplicates}")

    dp_group_alias: Optional[str] = (
        _extract_dp_group_alias(parts[1]) if len(parts) > 1 else None
    )

    return DimsSpec(dims=dims, dp_group_alias=dp_group_alias)


def resolve_dim_names(dims_str: str) -> list[str]:
    """Parse dims string and return tensor-compatible names ('1' → 'singleton0', ...)."""
    names: list[str] = [spec.name for spec in parse_dims(dims_str).dims]
    return _SingletonDimUtil.sanitize_names(names)


def find_dim_index(dim_specs: list[DimSpec], name: str) -> Optional[int]:
    names: list[str] = [spec.name for spec in dim_specs]
    return names.index(name) if name in names else None


def resolve_dim_by_name(tensor: torch.Tensor, name: str) -> int:
    if tensor.names[0] is None:
        raise ValueError(f"Tensor has no names, cannot resolve {name!r}")

    names: tuple[Optional[str], ...] = tensor.names
    try:
        return list(names).index(name)
    except ValueError:
        raise ValueError(f"Dim name {name!r} not in tensor names {names}")


def apply_dim_names(tensor: torch.Tensor, dim_names: list[str]) -> torch.Tensor:
    return tensor.refine_names(*dim_names)


def strip_dim_names(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.rename(None)


_DP_ALIAS_PATTERN = re.compile(r"^dp:=(\w+)$")


def _extract_dp_group_alias(declaration_part: str) -> Optional[str]:
    """Scan the ``#`` declaration section for a ``dp:=<group>`` token."""
    for token in declaration_part.strip().split():
        match = _DP_ALIAS_PATTERN.match(token)
        if match is not None:
            return match.group(1)

    return None
