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
        for key in ("load_weight", "kv_cache_allocation", "scheduler_e2e"):
            val = startup_time.get(key)
            if val is not None:
                result[key] = max(result[key], float(val))
        cuda_graph = startup_time.get("cuda_graph")
        if isinstance(cuda_graph, Mapping):
            for phase, duration in cuda_graph.items():
                if duration is not None:
                    result["cuda_graph"][phase] = max(
                        result["cuda_graph"].get(phase, 0.0), float(duration)
                    )
    return result


def build_engine_startup_time(
    scheduler_startup_times: Iterable[Mapping | None],
    *,
    tokenizer_e2e: float,
) -> dict:
    result = aggregate_scheduler_startup_times(scheduler_startup_times)
    result["tokenizer_e2e"] = tokenizer_e2e
    return result
