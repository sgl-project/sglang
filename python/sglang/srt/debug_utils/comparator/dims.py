import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ParallelAxis(Enum):
    TP = "tp"
    CP = "cp"
    EP = "ep"
    SP = "sp"


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

    names = [spec.name for spec in result]
    if len(names) != len(set(names)):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"Duplicate dim names: {duplicates}")

    return result
