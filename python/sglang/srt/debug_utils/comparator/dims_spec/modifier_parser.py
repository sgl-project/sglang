from __future__ import annotations

from typing import Optional

from sglang.srt.debug_utils.comparator.dims_spec.types import (
    _AXIS_LOOKUP,
    _QUALIFIER_LOOKUP,
    Ordering,
    ParallelAxis,
    ParallelModifier,
    Reduction,
)


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
        if q_str == "sharded":
            continue
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
