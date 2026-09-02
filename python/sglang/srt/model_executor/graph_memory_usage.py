from __future__ import annotations

from collections.abc import Mapping

GRAPH_MEMORY_USAGE_KEYS = (
    "prefill",
    "decode",
    "target_verify",
    "draft_prefill",
    "draft_decode",
    "draft_extend",
)


def empty_graph_memory_usage() -> dict[str, float]:
    return dict.fromkeys(GRAPH_MEMORY_USAGE_KEYS, 0.0)


def merge_graph_memory_usage(
    *usages: Mapping[str, float] | None,
) -> dict[str, float]:
    """Sum graph-capture memory by phase and keep a stable base schema."""
    merged = empty_graph_memory_usage()
    for usage in usages:
        if usage is None:
            continue
        for phase, value in usage.items():
            merged[phase] = merged.get(phase, 0.0) + value
    return merged


def replace_graph_memory_usage(
    current: Mapping[str, float] | None,
    replacement: Mapping[str, float],
    *,
    phases: tuple[str, ...],
) -> dict[str, float]:
    """Replace one capture family while preserving measurements for the rest."""
    updated = merge_graph_memory_usage(current)
    for phase in phases:
        updated[phase] = 0.0
    updated.update(replacement)
    return updated


def merge_graph_time_usage(
    *usages: Mapping[str, float] | None,
) -> dict[str, float]:
    return merge_graph_memory_usage(*usages)


def empty_graph_time_usage() -> dict[str, float]:
    return empty_graph_memory_usage()


def replace_graph_time_usage(
    current: Mapping[str, float] | None,
    replacement: Mapping[str, float],
    *,
    phases: tuple[str, ...],
) -> dict[str, float]:
    return replace_graph_memory_usage(
        current,
        replacement,
        phases=phases,
    )
