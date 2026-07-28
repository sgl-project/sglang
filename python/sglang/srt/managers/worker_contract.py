"""Typed values crossing the framework-to-worker boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig


class UnsupportedWorkerOperation(RuntimeError):
    """A worker was asked to perform an operation its backend does not support."""

    def __init__(self, operation: str, worker_type: str):
        super().__init__(f"{operation} is not supported by {worker_type}")
        self.operation = operation
        self.worker_type = worker_type


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerPoolState:
    """Framework-visible handles for initialized scheduler bookkeeping pools.

    Native backends may not use a framework MemoryPoolConfig, but they still
    provide the request pool and token-slot allocator used by the scheduler.
    """

    config: MemoryPoolConfig | None
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator


@dataclass(frozen=True, slots=True, kw_only=True)
class KVCacheLayout:
    """Scheduler-relevant KV-cache layout without exposing a model runner."""

    is_hybrid_swa: bool
    prefill_aware_swa: bool
    sliding_window_size: int | None
    full_tokens_per_layer: int | None
    swa_tokens_per_layer: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class AttentionRequirements:
    """Attention requirements consumed by generic scheduling code."""

    needs_cpu_seq_lens: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class WorkerMemoryUsage:
    """Backend-reported model memory; None means the value is unavailable."""

    weight_gb: float | None
    graph_gb: float | None
