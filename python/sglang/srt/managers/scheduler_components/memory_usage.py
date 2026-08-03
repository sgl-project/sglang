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
    token_capacity_swa: int | None,
    target_graph_memory_usage: Mapping[str, float] | None,
    draft_graph_memory_usage: Mapping[str, float] | None,
) -> dict:
    graph_memory_usage = combine_graph_memory_usage(
        target_graph_memory_usage,
        draft_graph_memory_usage,
    )
    return {
        "weight": round(weight_gb, 2),
        "kvcache": round(kv_cache_gb, 2),
        "startup_available": round(startup_available_gb, 2),
        "token_capacity": int(token_capacity),
        "token_capacity_swa": (
            None if token_capacity_swa is None else int(token_capacity_swa)
        ),
        "graph": {
            phase: round(memory_gb, 2)
            for phase, memory_gb in graph_memory_usage.items()
        },
    }
