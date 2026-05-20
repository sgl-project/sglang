from __future__ import annotations

import re
from typing import Optional

from sglang.srt.debug_utils.comparator.dims_spec.modifier_parser import (
    _parse_modifiers,
)
from sglang.srt.debug_utils.comparator.dims_spec.types import (
    SQUEEZE_DIM_NAME,
    DimSpec,
    ParallelModifier,
)

_DIM_PATTERN = re.compile(r"^(?P<name>[a-zA-Z_]\w*)(?:\[(?P<modifiers>[^\]]+)\])?$")

_FUSED_DIM_PATTERN = re.compile(r"^\((?P<inner>[^)]+)\)(?:\[(?P<modifiers>[^\]]+)\])?$")

_SUB_DIM_NAME_PATTERN = re.compile(r"^[a-zA-Z_]\w*$")


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
