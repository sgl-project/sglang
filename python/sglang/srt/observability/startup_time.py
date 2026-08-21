from __future__ import annotations

from collections.abc import Iterable, Mapping

from sglang.srt.model_executor.graph_memory_usage import (
    empty_graph_time_usage,
    merge_graph_time_usage,
)
from sglang.srt.observability.startup_phase_registry import freeze_startup_phases


def build_scheduler_startup_time(
    *,
    target_load_weight: float,
    draft_load_weight: float,
    scheduler_e2e: float,
    target_cuda_graph: Mapping[str, float] | None,
    draft_cuda_graph: Mapping[str, float] | None,
) -> dict:
    """Build one scheduler rank's startup-time dict.

    Building it closes the registry's cold-start snapshot.
    """
    return {
        **freeze_startup_phases(),
        "load_weight": target_load_weight + draft_load_weight,
        "scheduler_e2e": scheduler_e2e,
        "cuda_graph": merge_graph_time_usage(
            target_cuda_graph,
            draft_cuda_graph,
        ),
    }


def aggregate_scheduler_startup_times(
    startup_times: Iterable[Mapping | None],
) -> dict:
    """Return critical-path (max across ranks) startup durations."""
    result = {
        "load_weight": 0.0,
        "kv_cache_allocation": 0.0,
        "scheduler_e2e": 0.0,
        "cuda_graph": empty_graph_time_usage(),
    }
    for startup_time in startup_times:
        if not startup_time:
            continue
        for phase, duration in startup_time.items():
            if phase == "cuda_graph":
                for graph_phase, graph_duration in duration.items():
                    result["cuda_graph"][graph_phase] = max(
                        result["cuda_graph"].get(graph_phase, 0.0),
                        float(graph_duration),
                    )
            else:
                result[phase] = max(result.get(phase, 0.0), float(duration))
    return result


def build_engine_startup_time(
    scheduler_startup_times: Iterable[Mapping | None],
    *,
    tokenizer_e2e: float,
) -> dict:
    result = aggregate_scheduler_startup_times(scheduler_startup_times)
    result["tokenizer_e2e"] = tokenizer_e2e
    return result
