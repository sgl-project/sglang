from __future__ import annotations

from collections.abc import Mapping

from sglang.srt.model_executor.graph_memory_usage import merge_graph_memory_usage


def combine_graph_memory_usage(
    target: Mapping[str, float] | None,
    draft: Mapping[str, float] | None,
) -> dict[str, float]:
    return merge_graph_memory_usage(target, draft)


def build_memory_usage(
    *,
    weight_gb: float,
    kv_cache_gb: float,
    startup_available_gb: float,
    token_capacity: int,
    token_capacity_window: Mapping[str, int],
    target_graph_memory_usage: Mapping[str, float] | None,
    draft_graph_memory_usage: Mapping[str, float] | None,
) -> dict:
    graph_memory_usage = combine_graph_memory_usage(
        target_graph_memory_usage,
        draft_graph_memory_usage,
    )
    return {
        "weight": round(weight_gb, 3),
        "kvcache": round(kv_cache_gb, 3),
        "startup_available": round(startup_available_gb, 3),
        "token_capacity": int(token_capacity),
        # Each sliding-window pool's capacity, by the window's name.
        "token_capacity_window": {
            name: int(capacity) for name, capacity in token_capacity_window.items()
        },
        "graph": {
            phase: round(memory_gb, 3)
            for phase, memory_gb in graph_memory_usage.items()
        },
    }
