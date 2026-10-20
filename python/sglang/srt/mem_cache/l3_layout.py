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


def _pp_rank(layout: dict) -> Optional[int]:
    inner = layout.get("hicache_object_layout")
    return inner.get("pp_rank") if isinstance(inner, dict) else None


def _stage_independent(layout: dict) -> dict:
    """The part of a reply that must agree across pipeline stages."""
    inner = layout.get("hicache_object_layout")
    if not isinstance(inner, dict):
        return layout
    rest = {k: v for k, v in inner.items() if k not in ("pp_rank", "pools")}
    return {**layout, "hicache_object_layout": rest}


def _unavailable(layout: dict, reason: str) -> dict:
    inner = {**layout["hicache_object_layout"], "pools": []}
    return {
        **layout,
        "enabled": False,
        "hicache_object_layout": inner,
        "reason": reason,
    }


def merge_l3_layouts(layouts: Sequence[Optional[dict]]) -> Optional[dict]:
    """Merge scheduler replies: replicas must match, PP stages are concatenated."""
    if not layouts or all(layout is None for layout in layouts):
        # Absence is reserved for older engines, not an explicitly disabled L3.
        return None
    inconsistent = "Inconsistent HiCache L3 layouts across schedulers"
    if any(layout is None for layout in layouts):
        return disabled_l3_layout(inconsistent)
    stages = {}
    for layout in layouts:
        if stages.setdefault(_pp_rank(layout), layout) != layout:
            return disabled_l3_layout(inconsistent)
    if None in stages:
        return layouts[0] if len(stages) == 1 else disabled_l3_layout(inconsistent)
    ordered = [stages[rank] for rank in sorted(stages)]
    first = ordered[0]
    if any(_stage_independent(s) != _stage_independent(first) for s in ordered[1:]):
        return disabled_l3_layout(inconsistent)
    if not first.get("enabled"):
        return first
    inner = first["hicache_object_layout"]
    pp_size = inner.get("pp_size", 1)
    if type(pp_size) is not int:
        return disabled_l3_layout(inconsistent)
    if sorted(stages) != list(range(pp_size)):
        return _unavailable(first, "Incomplete pipeline-parallel stage layouts")
    if pp_size == 1:
        return first
    # A layout spanning several stages belongs to none of them: pp_rank is null.
    pools = [p for s in ordered for p in s["hicache_object_layout"]["pools"]]
    return {
        **first,
        "hicache_object_layout": {**inner, "pools": pools, "pp_rank": None},
    }


def merge_scheduler_info_layouts(infos: Sequence[dict]) -> None:
    """Give every startup info the merged layout, so any entry is representative."""
    if any("hicache_object_layout" in info for info in infos):
        layout = merge_l3_layouts([info.get("hicache_object_layout") for info in infos])
        for info in infos:
            info["hicache_object_layout"] = layout
