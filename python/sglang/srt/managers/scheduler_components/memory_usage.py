from __future__ import annotations

from collections.abc import Mapping

from sglang.srt.model_executor.graph_memory_usage import merge_graph_memory_usage
from sglang.srt.runtime_context import get_schedule


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
        "weight": round(weight_gb, 3),
        "kvcache": round(kv_cache_gb, 3),
        "startup_available": round(startup_available_gb, 3),
        "token_capacity": int(token_capacity),
        "token_capacity_swa": (
            None if token_capacity_swa is None else int(token_capacity_swa)
        ),
        "graph": {
            phase: round(memory_gb, 3)
            for phase, memory_gb in graph_memory_usage.items()
        },
    }


def emit_startup_metrics(
    metrics_collector,
    *,
    tp_worker,
    draft_worker,
    token_to_kv_pool_allocator,
    max_total_num_tokens: int,
    swa_tokens_per_layer: int | None,
    context_len: int,
    startup_available_gpu_memory_gb: float,
) -> None:
    """Report the capacity and memory the scheduler started with."""
    metrics_collector.emit_constants(
        max_total_num_tokens=max_total_num_tokens,
        max_total_num_tokens_swa=swa_tokens_per_layer,
        weight_memory_usage_gb=tp_worker.model_runner.weight_load_mem_usage,
        kv_cache_memory_usage_gb=(token_to_kv_pool_allocator.get_kvcache().mem_usage),
        graph_memory_usage_gb=combine_graph_memory_usage(
            tp_worker.graph_memory_usage,
            (None if draft_worker is None else draft_worker.graph_memory_usage),
        ),
        # TODO: max_running_requests_under_SLO has no setter — dead chain.
        max_running_requests_under_SLO=None,
        page_size=get_schedule().page_size,
        num_pages=max_total_num_tokens // token_to_kv_pool_allocator.page_size,
        context_len=context_len,
        startup_available_gpu_memory_gb=startup_available_gpu_memory_gb,
    )
