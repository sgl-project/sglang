from __future__ import annotations

from typing import Optional

from sglang.srt.debug_utils.comparator.dims_spec.comment_parser import (
    _CommentSuffix,
    _parse_comment_suffix,
)
from sglang.srt.debug_utils.comparator.dims_spec.dim_parser import parse_dim
from sglang.srt.debug_utils.comparator.dims_spec.types import (
    SQUEEZE_DIM_NAME,
    DimSpec,
    DimsSpec,
    ParallelAxis,
)


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


def parse_dims(dims_str: str) -> DimsSpec:
    """Parse ``"b s[cp:zigzag] h[tp] d # dp:=moe_dp ep:replicated"`` → :class:`DimsSpec`.

    The shape part (before ``#``) produces :pyattr:`DimsSpec.dims`.
    The declaration part (after ``#``) is scanned for:
    - ``dp:=<group>`` → :pyattr:`DimsSpec.dp_group_alias`
    - ``axis:replicated`` → :pyattr:`DimsSpec.replicated_axes`
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

    comment_suffix: _CommentSuffix = (
        _parse_comment_suffix(parts[1]) if len(parts) > 1 else _CommentSuffix()
    )
    dp_group_alias: Optional[str] = comment_suffix.dp_group_alias
    replicated_axes: frozenset[ParallelAxis] = comment_suffix.replicated_axes

    sharded_axes: set[ParallelAxis] = {
        m.axis for spec in dims for m in spec.parallel_modifiers
    }
    conflict: frozenset[ParallelAxis] = replicated_axes & sharded_axes
    if conflict:
        conflict_names: str = ", ".join(sorted(a.value for a in conflict))
        raise ValueError(
            f"Axes declared as both sharded (in dim spec) and replicated "
            f"(in # declaration): {conflict_names}"
        )

    return DimsSpec(
        dims=dims,
        dp_group_alias=dp_group_alias,
        replicated_axes=replicated_axes,
    )


def resolve_dim_names(dims_str: str) -> list[str]:
    """Parse dims string and return tensor-compatible names ('1' → 'singleton0', ...)."""
    specs: list[DimSpec] = parse_dims(dims_str).dims
    names: list[str] = [spec.sanitized_name for spec in specs]
    return _SingletonDimUtil.sanitize_names(names)
