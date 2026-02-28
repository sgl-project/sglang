import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

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


@dataclass(frozen=True)
class DimSpec:
    name: str
    parallel: Optional[ParallelAxis] = None
    ordering: Optional[Ordering] = None
    reduction: Optional[Reduction] = None


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

_MODIFIER_FIELDS: list[tuple[type[Enum], str]] = [
    (ParallelAxis, "parallel"),
    (Ordering, "ordering"),
    (Reduction, "reduction"),
]

_MODIFIER_LOOKUP: dict[str, tuple[str, Enum]] = {}
for _enum_cls, _field in _MODIFIER_FIELDS:
    for _member in _enum_cls:
        _MODIFIER_LOOKUP[_member.value] = (_field, _member)


def parse_dim(token: str) -> DimSpec:
    if token == SQUEEZE_DIM_NAME:
        return DimSpec(name=SQUEEZE_DIM_NAME)

    match = _DIM_PATTERN.match(token)
    if match is None:
        raise ValueError(f"Invalid dim token: {token!r}")

    name = match.group("name")
    modifiers_str = match.group("modifiers")

    if modifiers_str is None:
        return DimSpec(name=name)

    fields: dict[str, Enum] = {}
    for part in (p.strip() for p in modifiers_str.split(",")):
        if part not in _MODIFIER_LOOKUP:
            raise ValueError(f"Unknown modifier {part!r} in dim spec: {token!r}")
        field_name, enum_value = _MODIFIER_LOOKUP[part]
        if field_name in fields:
            raise ValueError(f"Multiple {field_name} values in dim token: {token!r}")
        fields[field_name] = enum_value

    return DimSpec(name=name, **fields)


def parse_dims(dims_str: str) -> list[DimSpec]:
    """Parse 'b s(cp,zigzag) h(tp) d' -> list[DimSpec]."""
    if not dims_str.strip():
        raise ValueError("dims string must not be empty")

    result = [parse_dim(token) for token in dims_str.strip().split()]

    non_squeeze_names: list[str] = [
        spec.name for spec in result if not _SingletonDimUtil.is_squeeze(spec)
    ]
    if len(non_squeeze_names) != len(set(non_squeeze_names)):
        duplicates = sorted(
            {n for n in non_squeeze_names if non_squeeze_names.count(n) > 1}
        )
        raise ValueError(f"Duplicate dim names: {duplicates}")

    return result


def resolve_dim_names(dims_str: str) -> list[str]:
    """Parse dims string and return tensor-compatible names ('1' → 'singleton0', ...)."""
    names: list[str] = [spec.name for spec in parse_dims(dims_str)]
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
