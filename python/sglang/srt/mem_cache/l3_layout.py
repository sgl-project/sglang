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


def merge_l3_layouts(layouts: Sequence[Optional[dict]]) -> Optional[dict]:
    if not layouts or all(layout is None for layout in layouts):
        # Absence is reserved for older engines, not an explicitly disabled L3.
        return None
    first = layouts[0]
    if any(layout != first for layout in layouts):
        return disabled_l3_layout("Inconsistent HiCache L3 layouts across schedulers")
    return first
