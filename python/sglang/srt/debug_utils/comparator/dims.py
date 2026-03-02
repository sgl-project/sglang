from __future__ import annotations

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


_DIM_PATTERN = re.compile(r"^(?P<name>[a-zA-Z_]\w*)(?:\[(?P<modifiers>[^\]]+)\])?$")

_FUSED_DIM_PATTERN = re.compile(r"^\((?P<inner>[^)]+)\)(?:\[(?P<modifiers>[^\]]+)\])?$")

_SUB_DIM_NAME_PATTERN = re.compile(r"^[a-zA-Z_]\w*$")

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

    fused_match = _FUSED_DIM_PATTERN.match(token)
    if fused_match is not None:
        return _parse_fused_dim(token=token, fused_match=fused_match)

    return _parse_single_dim(token)


def _parse_single_dim(token: str) -> DimSpec:
    match = _DIM_PATTERN.match(token)
    if match is None:
        raise ValueError(f"Invalid dim token: {token!r}")

    name: str = match.group("name")
    modifiers: list[ParallelModifier] = _parse_modifiers(
        modifiers_str=match.group("modifiers"), dim_token=token
    )
    return DimSpec(name=name, parallel_modifiers=modifiers)


def _parse_fused_dim(*, token: str, fused_match: re.Match[str]) -> DimSpec:
    inner: str = fused_match.group("inner")
    modifiers_str: Optional[str] = fused_match.group("modifiers")

    sub_names: list[str] = [s.strip() for s in inner.split("*")]
    for sub_name in sub_names:
        if not _SUB_DIM_NAME_PATTERN.match(sub_name):
            raise ValueError(
                f"Invalid sub-dim {sub_name!r} in fused dim token: {token!r}"
            )

    if len(sub_names) != len(set(sub_names)):
        raise ValueError(f"Duplicate sub-dim names in fused dim token: {token!r}")

    if len(sub_names) < 2:
        raise ValueError(
            f"Fused dim must have at least 2 sub-dims, got {len(sub_names)} in: {token!r}"
        )

    fused_name: str = "*".join(sub_names)
    modifiers: list[ParallelModifier] = _parse_modifiers(
        modifiers_str=modifiers_str, dim_token=token
    )
    return DimSpec(name=fused_name, parallel_modifiers=modifiers)


def _parse_modifiers(
    *, modifiers_str: Optional[str], dim_token: str
) -> list[ParallelModifier]:
    if modifiers_str is None:
        return []

    modifiers: list[ParallelModifier] = []
    seen_axes: set[ParallelAxis] = set()

    for modifier_token in (p.strip() for p in modifiers_str.split(",")):
        modifier: ParallelModifier = _parse_modifier_token(modifier_token, dim_token)
        if modifier.axis in seen_axes:
            raise ValueError(
                f"Duplicate axis {modifier.axis.value!r} in dim spec: {dim_token!r}"
            )
        seen_axes.add(modifier.axis)
        modifiers.append(modifier)

    return modifiers


def parse_dims(dims_str: str) -> DimsSpec:
    """Parse ``"b s[cp:zigzag] h[tp] d # dp:=moe_dp"`` → :class:`DimsSpec`.

    The shape part (before ``#``) produces :pyattr:`DimsSpec.dims`.
    The declaration part (after ``#``) is scanned for ``dp:=<group>``
    which populates :pyattr:`DimsSpec.dp_group_alias`.
    """
    parts: list[str] = dims_str.split("#", maxsplit=1)
    raw: str = parts[0]

    if not raw.strip():
        raise ValueError("dims string must not be empty")

    dims: list[DimSpec] = [parse_dim(token) for token in raw.strip().split()]

    # Collect all semantic names (expanding fused sub-dims) for duplicate detection
    semantic_names: list[str] = []
    for spec in dims:
        if _SingletonDimUtil.is_squeeze(spec):
            continue
        semantic_names.extend(spec.sub_dims)

    if len(semantic_names) != len(set(semantic_names)):
        duplicates = sorted({n for n in semantic_names if semantic_names.count(n) > 1})
        raise ValueError(f"Duplicate dim names: {duplicates}")

    dp_group_alias: Optional[str] = (
        _extract_dp_group_alias(parts[1]) if len(parts) > 1 else None
    )

    return DimsSpec(dims=dims, dp_group_alias=dp_group_alias)


def resolve_dim_names(dims_str: str) -> list[str]:
    """Parse dims string and return tensor-compatible names ('1' → 'singleton0', ...)."""
    specs: list[DimSpec] = parse_dims(dims_str).dims
    names: list[str] = [spec.sanitized_name for spec in specs]
    return _SingletonDimUtil.sanitize_names(names)


def find_dim_index(dim_specs: list[DimSpec], name: str) -> Optional[int]:
    """Find index by name. Accepts both ``*``-form and ``___``-form for fused dims."""
    for i, spec in enumerate(dim_specs):
        if spec.name == name or spec.sanitized_name == name:
            return i
    return None


def resolve_dim_by_name(tensor: torch.Tensor, name: str) -> int:
    if tensor.names[0] is None:
        raise ValueError(f"Tensor has no names, cannot resolve {name!r}")

    names: tuple[Optional[str], ...] = tensor.names
    try:
        return list(names).index(name)
    except ValueError:
        raise ValueError(f"Dim name {name!r} not in tensor names {names}")


def apply_dim_names(tensor: torch.Tensor, dim_names: list[str]) -> torch.Tensor:
    if tensor.ndim != len(dim_names):
        raise ValueError(
            f"dims metadata mismatch: tensor has {tensor.ndim} dims (shape {list(tensor.shape)}) "
            f"but dims string specifies {len(dim_names)} names {dim_names}. "
            f"Please fix the dims string in the dumper.dump() call to match the actual tensor shape."
        )
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
