"""Shared availability and replica-consistency rules for external L3 routers."""

from typing import Optional, Sequence


def disabled_l3_layout(
    reason: str, *, backend: Optional[str] = None, page_size: Optional[int] = None
) -> dict:
    return {
        "enabled": False,
        "backend": backend,
        "page_size": page_size,
        "hicache_object_layout": {"pools": []},
        "reason": reason,
    }


def _without_pp_rank(layout: Optional[dict]) -> Optional[dict]:
    # PP stages legitimately differ only in pp_rank.
    inner = layout.get("hicache_object_layout") if isinstance(layout, dict) else None
    if not isinstance(inner, dict):
        return layout
    rest = {key: value for key, value in inner.items() if key != "pp_rank"}
    return {**layout, "hicache_object_layout": rest}


def merge_l3_layouts(layouts: Sequence[Optional[dict]]) -> Optional[dict]:
    if not layouts or all(layout is None for layout in layouts):
        # Absence is reserved for older engines, not an explicitly disabled L3.
        return None
    first = layouts[0]
    if any(_without_pp_rank(layout) != _without_pp_rank(first) for layout in layouts):
        return disabled_l3_layout("Inconsistent HiCache L3 layouts across schedulers")
    return first
