from __future__ import annotations

import contextlib
import logging
from collections import Counter
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_graph_counts: Counter[str] = Counter()


def is_dsv4_index_cache_profile_enabled() -> bool:
    return envs.SGLANG_DSV4_INDEXCACHE_PROFILE.get()


def profile_region(name: str, layer_id: Optional[int] = None):
    if not is_dsv4_index_cache_profile_enabled():
        return contextlib.nullcontext()
    label = f"dsv4_indexcache.{name}"
    if layer_id is not None:
        label = f"{label}.layer_{layer_id}"
    return torch.profiler.record_function(label)


def record_cuda_graph_path(mode: str, can_run_graph: bool) -> None:
    if not is_dsv4_index_cache_profile_enabled():
        return
    path = "replay" if can_run_graph else "fallback"
    with torch.profiler.record_function(f"dsv4_indexcache.cuda_graph.{mode}.{path}"):
        pass
    _graph_counts[f"{mode}.{path}"] += 1
    total = sum(_graph_counts.values())
    interval = envs.SGLANG_DSV4_INDEXCACHE_PROFILE_LOG_INTERVAL.get()
    if interval > 0 and total % interval == 0:
        logger.info(
            "[DSV4 IndexCache profile] CUDA graph path counts: %s",
            dict(_graph_counts),
        )
