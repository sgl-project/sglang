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


_DIM_PATTERN = re.compile(r"^([a-zA-Z_]\w*)(?:\(([^)]+)\))?$")

_ORDERING_VALUES = {e.value for e in Ordering}
_REDUCTION_VALUES = {e.value for e in Reduction}


def parse_dims(dims_str: str) -> list[DimSpec]:
    """Parse 'b s(cp,zigzag) h(tp) d' -> list[DimSpec]."""
    if not dims_str.strip():
        raise ValueError("dims string must not be empty")

    tokens = dims_str.strip().split()
    seen_names: set[str] = set()
    result: list[DimSpec] = []

    for token in tokens:
        match = _DIM_PATTERN.match(token)
        if match is None:
            raise ValueError(f"Invalid dim token: {token!r}")

        name = match.group(1)
        if name in seen_names:
            raise ValueError(f"Duplicate dim name: {name!r}")
        seen_names.add(name)

        paren_content = match.group(2)
        if paren_content is None:
            result.append(DimSpec(name=name))
            continue

        parts = [p.strip() for p in paren_content.split(",")]
        if parts == [""]:
            raise ValueError(f"Empty parentheses in dim token: {token!r}")

        parallel = _parse_parallel_axis(parts[0], token)
        ordering: Optional[Ordering] = None
        reduction: Optional[Reduction] = None

        for part in parts[1:]:
            if part in _ORDERING_VALUES:
                if ordering is not None:
                    raise ValueError(
                        f"Multiple ordering values in dim token: {token!r}"
                    )
                ordering = Ordering(part)
            elif part in _REDUCTION_VALUES:
                if reduction is not None:
                    raise ValueError(
                        f"Multiple reduction values in dim token: {token!r}"
                    )
                reduction = Reduction(part)
            else:
                raise ValueError(f"Unknown token {part!r} in dim spec: {token!r}")

        result.append(
            DimSpec(
                name=name,
                parallel=parallel,
                ordering=ordering,
                reduction=reduction,
            )
        )

    return result


def _parse_parallel_axis(value: str, context: str) -> ParallelAxis:
    try:
        return ParallelAxis(value)
    except ValueError:
        valid = [e.value for e in ParallelAxis]
        raise ValueError(
            f"Unknown parallel axis {value!r} in {context!r}. Valid: {valid}"
        ) from None
