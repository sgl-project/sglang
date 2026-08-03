from __future__ import annotations

from collections.abc import Iterable, Mapping

from sglang.srt.model_executor.graph_memory_usage import (
    empty_graph_time_usage,
    merge_graph_time_usage,
)


def build_scheduler_startup_time(
    *,
    target_load_weight: float,
    draft_load_weight: float,
    kv_cache_allocation: float,
    scheduler_e2e: float,
    target_cuda_graph: Mapping[str, float] | None,
    draft_cuda_graph: Mapping[str, float] | None,
) -> dict:
    return {
        "load_weight": target_load_weight + draft_load_weight,
        "kv_cache_allocation": kv_cache_allocation,
        "scheduler_e2e": scheduler_e2e,
        "cuda_graph": merge_graph_time_usage(
            target_cuda_graph,
            draft_cuda_graph,
        ),
    }


def aggregate_scheduler_startup_times(
    startup_times: Iterable[Mapping | None],
) -> dict:
    """Return critical-path startup durations across scheduler ranks."""
    result = {
        "load_weight": 0.0,
        "kv_cache_allocation": 0.0,
        "scheduler_e2e": 0.0,
        "cuda_graph": empty_graph_time_usage(),
    }
    for startup_time in startup_times:
        if not startup_time:
            continue
        result["load_weight"] = max(
            result["load_weight"], float(startup_time.get("load_weight", 0.0))
        )
        result["kv_cache_allocation"] = max(
            result["kv_cache_allocation"],
            float(startup_time.get("kv_cache_allocation", 0.0)),
        )
        result["scheduler_e2e"] = max(
            result["scheduler_e2e"],
            float(startup_time.get("scheduler_e2e", 0.0)),
        )
        for phase, duration in startup_time.get("cuda_graph", {}).items():
            result["cuda_graph"][phase] = max(
                result["cuda_graph"].get(phase, 0.0), float(duration)
            )
    return result


def build_engine_startup_time(
    scheduler_startup_times: Iterable[Mapping | None],
    *,
    e2e: float,
) -> dict:
    result = aggregate_scheduler_startup_times(scheduler_startup_times)
    result["e2e"] = e2e
    return result
