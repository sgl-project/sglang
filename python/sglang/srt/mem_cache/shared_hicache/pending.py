from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

from sglang.srt.mem_cache.radix_cache import TreeNode
from sglang.srt.mem_cache.shared_hicache.plan import SharedHiCachePlan
from sglang.srt.mem_cache.shared_hicache.source import ResolvedHostPage


@dataclass
class SharedHiCachePendingFetch:
    plan: SharedHiCachePlan
    plan_offset: int
    target_start_block: int
    expected_hashes: tuple[int, ...]
    transfer: Any
    device_indices: Optional[torch.Tensor] = None
    locked_node: Optional[TreeNode] = None
    backend: str = "unknown"
    bytes_per_page: int = 0
    submitted_at: float = 0.0
    done_at: float = 0.0


def format_optional_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def pending_wait_ms(pending: SharedHiCachePendingFetch) -> Optional[float]:
    submitted_at = getattr(pending, "submitted_at", 0.0)
    if submitted_at <= 0:
        return None
    return (time.perf_counter() - submitted_at) * 1000


def pending_ready_wait_ms(pending: SharedHiCachePendingFetch) -> Optional[float]:
    done_at = pending.done_at or float(getattr(pending.transfer, "done_at", 0.0))
    if done_at <= 0:
        return None
    return max(0.0, (time.perf_counter() - done_at) * 1000)


def pending_timeout_secs(
    pending: SharedHiCachePendingFetch,
    *,
    page_size: int,
    timeout_secs: float,
    prefetch_timeout_config,
) -> float:
    if prefetch_timeout_config is None:
        return float(timeout_secs)
    num_tokens = len(pending.expected_hashes) * page_size
    return float(
        min(
            prefetch_timeout_config.max,
            prefetch_timeout_config.base
            + prefetch_timeout_config.per_ki_token * num_tokens / 1024,
        )
    )


def pending_should_stop_waiting(
    pending: SharedHiCachePendingFetch,
    *,
    policy: str,
    page_size: int,
    timeout_secs: float,
    prefetch_timeout_config,
) -> tuple[bool, str]:
    if policy == "best_effort":
        return True, "best_effort_incomplete"
    if policy == "wait_complete":
        return False, ""
    if policy == "timeout":
        timeout = pending_timeout_secs(
            pending,
            page_size=page_size,
            timeout_secs=timeout_secs,
            prefetch_timeout_config=prefetch_timeout_config,
        )
        elapsed = time.perf_counter() - pending.submitted_at
        if timeout >= 0 and elapsed > timeout:
            return True, "prefetch_timeout"
        return False, ""
    return True, "unknown_prefetch_policy"


def transfer_bytes_for_pages(
    pending: SharedHiCachePendingFetch, pages: list[ResolvedHostPage]
) -> int:
    bytes_per_page = int(getattr(pending, "bytes_per_page", 0) or 0)
    if bytes_per_page > 0:
        return len(pages) * bytes_per_page
    return sum(len(page.data) for page in pages)
