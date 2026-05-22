from __future__ import annotations

from typing import Optional

import torch
from einops import rearrange

from sglang.srt.debug_utils.comparator.dims_spec import (
    _FUSED_NAME_SEP,
    SEQ_DIM_NAME,
    TOKEN_DIM_NAME,
    DimSpec,
    _SingletonDimUtil,
    parse_dims,
)
from sglang.srt.debug_utils.comparator.log_sink import log_sink
from sglang.srt.debug_utils.comparator.utils import Pair, _FrozenBase

# --- types ---


class AxisAlignerPlan(_FrozenBase):
    pattern: Pair[Optional[str]]  # einops pattern per side, None = no-op


# --- planner ---


def compute_axis_aligner_plan(
    dims_str_pair: Pair[Optional[str]],
) -> Optional[AxisAlignerPlan]:
    if dims_str_pair.x is None or dims_str_pair.y is None:
        return None

    dims_pair: Pair[str] = Pair(x=dims_str_pair.x, y=dims_str_pair.y)
    specs_pair: Pair[list[DimSpec]] = dims_pair.map(lambda s: parse_dims(s).dims)

    if not _semantic_names_match(specs_pair):
        return None

    # Canonical dim order follows y; fused groups stay fused (flatten, not unflatten).
    canonical_order: Optional[list[str]] = _build_canonical_order(specs_pair)
    if canonical_order is None:
        return None

    pattern: Pair[Optional[str]] = specs_pair.map(
        lambda specs: _build_side_pattern(specs=specs, canonical_order=canonical_order)
    )

    if pattern.x is None and pattern.y is None:
        return None

    return AxisAlignerPlan(pattern=pattern)


_SEQ_DIM_EQUIVALENCES: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({SEQ_DIM_NAME, TOKEN_DIM_NAME}),  # s ≡ t
    }
)


def _normalize_dim_name(name: str) -> str:
    for equiv_set in _SEQ_DIM_EQUIVALENCES:
        if name in equiv_set:
            return min(equiv_set)
    return name


def _semantic_names_match(specs_pair: Pair[list[DimSpec]]) -> bool:
    """Check that both sides share the same semantic name set (ignoring squeeze dims)."""
    names_pair: Pair[list[str]] = specs_pair.map(_expand_and_skip_squeeze)

    if set(map(_normalize_dim_name, names_pair.x)) == set(
        map(_normalize_dim_name, names_pair.y)
    ):
        return True

    # Local import to avoid circular dependency:
    # output_types -> aligner/entrypoint/types -> axis_aligner -> output_types
    from sglang.srt.debug_utils.comparator.output_types import ErrorLog

    log_sink.add(
        ErrorLog(
            category="axis_aligner_dim_mismatch",
            message=(
                f"AxisAligner: dim name sets differ (x={names_pair.x}, y={names_pair.y}), "
                f"skipping axis swap"
            ),
        )
    )
    return False


def _expand_and_skip_squeeze(specs: list[DimSpec]) -> list[str]:
    """Expand DimSpecs to flat semantic names, skipping squeeze dims."""
    return [
        name
        for spec in specs
        if not _SingletonDimUtil.is_squeeze(spec)
        for name in spec.sub_dims
    ]


def _build_canonical_order(specs_pair: Pair[list[DimSpec]]) -> Optional[list[str]]:
    """Build canonical dim order following y, preferring fused representation.

    Each element is either a plain name (``"c"``) or a fused placeholder (``"a___b"``).
    Fused groups from *either* side are merged — the separate side must flatten.
    Squeeze dims are excluded.

    Returns ``None`` if the two sides have overlapping but incompatible fused groups
    (e.g. x fuses ``(a*b)`` while y fuses ``(b*c)``).
    """
    # Map each sub-dim name → (placeholder, siblings) from both sides
    fused_lookup: dict[str, tuple[str, frozenset[str]]] = {}
    for spec in (*specs_pair.x, *specs_pair.y):
        if spec.is_fused:
            placeholder: str = spec.sanitized_name
            siblings: frozenset[str] = frozenset(spec.sub_dims)
            for sub_name in spec.sub_dims:
                existing: Optional[tuple[str, frozenset[str]]] = fused_lookup.get(
                    sub_name
                )
                if existing is not None and existing[1] != siblings:
                    from sglang.srt.debug_utils.comparator.output_types import ErrorLog

                    log_sink.add(
                        ErrorLog(
                            category="axis_aligner_fused_conflict",
                            message=(
                                f"AxisAligner: overlapping fused groups for sub-dim {sub_name!r} "
                                f"({existing[0]} vs {placeholder}), skipping axis alignment"
                            ),
                        )
                    )
                    return None
                fused_lookup.setdefault(sub_name, (placeholder, siblings))

    result: list[str] = []
    consumed: set[str] = set()

    for spec in specs_pair.y:
        if _SingletonDimUtil.is_squeeze(spec):
            continue

        names: list[str] = spec.sub_dims
        if any(n in consumed for n in names):
            continue

        entry: Optional[tuple[str, frozenset[str]]] = fused_lookup.get(names[0])
        if entry is not None:
            fused_placeholder, sibs = entry
            result.append(fused_placeholder)
            consumed.update(sibs)
        else:
            result.append(_normalize_dim_name(spec.name))
            consumed.update(names)

    return result


def _build_side_pattern(
    *, specs: list[DimSpec], canonical_order: list[str]
) -> Optional[str]:
    """Build an einops pattern for one side to reach ``canonical_order``.

    Fused specs become their placeholder; separate specs that belong to a fused group
    stay as individual names on the LHS and become ``(a b)`` on the RHS (einops flatten).
    Squeeze dims (``1``) appear on the LHS but are dropped from the RHS.
    """
    source_tokens: list[str] = [spec.sanitized_name for spec in specs]

    # Map normalized dim names back to this side's original names so that
    # einops patterns use consistent identifiers on LHS and RHS.
    norm_to_original: dict[str, str] = {
        _normalize_dim_name(spec.name): spec.name for spec in specs
    }

    def _to_side_name(token: str) -> str:
        return norm_to_original.get(token, token)

    # Build per-side target: replace fused placeholders with ``(a b)`` only if this side
    # has the sub-dims as separate (non-fused) names in the source
    fused_placeholders: set[str] = {
        spec.sanitized_name for spec in specs if spec.is_fused
    }
    translated_order: list[str] = [_to_side_name(t) for t in canonical_order]
    target_tokens: list[str] = [
        (
            f"({t.replace(_FUSED_NAME_SEP, ' ')})"
            if _FUSED_NAME_SEP in t and t not in fused_placeholders
            else t
        )
        for t in translated_order
    ]

    if source_tokens == target_tokens:
        return None

    return f"{' '.join(source_tokens)} -> {' '.join(target_tokens)}"


# --- executor ---


def execute_axis_aligner_plan(
    tensor: torch.Tensor, plan: AxisAlignerPlan, *, side: str
) -> torch.Tensor:
    if side not in ("x", "y"):
        raise ValueError(f"side must be 'x' or 'y', got {side!r}")

    pattern: Optional[str] = plan.pattern.x if side == "x" else plan.pattern.y

    if pattern is not None:
        tensor = rearrange(tensor.rename(None), pattern)

    return tensor
