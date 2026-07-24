from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_RADIX_CACHE,
    RadixCacheMetricsCollector,
    resolve_collector_class,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.hicache_storage import SidecarPoolSpec
    from sglang.srt.mem_cache.memory_pool_host import PoolEntry
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache_components.tree_component import (
        ComponentType,
    )


@runtime_checkable
class PrefixCacheTrait(Protocol):
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator
    page_size: int
    disable: bool


@dataclasses.dataclass
class MatchPrefixParams:
    """Unified parameters for match_prefix across different cache types"""

    key: RadixKey

    # Mamba specific
    cow_mamba: bool = False
    req: Optional[Req] = None


@dataclasses.dataclass
class InsertParams:
    """Unified parameters for insert across different cache types"""

    key: Optional[RadixKey] = None
    value: Optional[torch.Tensor] = None

    # Mamba specific
    mamba_value: Optional[torch.Tensor] = None

    # SWA specific
    prev_prefix_len: int = 0
    swa_evicted_seqlen: int = 0

    # General
    chunked: bool = False
    priority: int = 0


@dataclasses.dataclass
class InsertResult:
    """Result of an insert operation"""

    prefix_len: int
    total_len: int = 0
    last_device_node: Any = None
    mamba_exist: bool = False
    inserted_host_node: Any = None


@dataclasses.dataclass
class EvictParams:
    """Unified parameters for evict across different cache types"""

    num_tokens: int = 0
    swa_num_tokens: int = 0
    mamba_num: int = 0


@dataclasses.dataclass
class EvictResult:
    """Result of an evict operation"""

    num_tokens_evicted: int = 0
    swa_num_tokens_evicted: int = 0
    mamba_num_evicted: int = 0


@dataclasses.dataclass
class IncLockRefResult:
    """Result of an inc_lock_ref operation."""

    delta: Optional[int] = None
    swa_uuid_for_lock: Optional[int] = None
    swa_uuid_for_host_lock: Optional[int] = None
    # Component nodes that were tombstones at acquire time. Replaying this set
    # at release prevents a short-lived lock from consuming a later load-back or
    # request lock after that tombstone becomes a valid device value.
    skip_lock_node_ids: dict[ComponentType, set[int]] = dataclasses.field(
        default_factory=dict
    )

    def to_dec_params(self) -> DecLockRefParams:
        """Convert to the corresponding DecLockRefParams for dec_lock_ref."""
        return DecLockRefParams(
            swa_uuid_for_lock=self.swa_uuid_for_lock,
            swa_uuid_for_host_lock=self.swa_uuid_for_host_lock,
            skip_lock_node_ids={
                component_type: set(node_ids)
                for component_type, node_ids in self.skip_lock_node_ids.items()
            },
        )


@dataclasses.dataclass
class DecLockRefParams:
    """Parameters for dec_lock_ref operation."""

    swa_uuid_for_lock: Optional[int] = None
    swa_uuid_for_host_lock: Optional[int] = None
    skip_lock_node_ids: dict[ComponentType, set[int]] = dataclasses.field(
        default_factory=dict
    )


@dataclasses.dataclass
class DecLockRefResult:
    """Result of an dec_lock_ref operation."""

    delta: Optional[int] = None


@dataclasses.dataclass
class InitLoadBackParams:
    """Unified parameters for init_load_back across different cache types."""

    best_match_node: Any
    host_hit_length: int
    mem_quota: Optional[int] = None
    req: Optional[Req] = None


class MatchResult(NamedTuple):
    """Result of a prefix match operation.

    Attributes:
        device_indices  :   Indices of the KV cache on the device matched by common prefix.
        last_device_node:   The last TreeNode on the device that was matched.
        last_host_node  :   The last TreeNode on the host that was matched.
                            Note that if HiCache is not enabled,
                            this **must** be the same as `last_device_node`.
                            Reserved for L3 storage prefetch anchoring; L2 load_back
                            uses `best_match_node` instead.
        best_match_node :   Deepest node accepted by all component validators
                            during match_prefix. Anchor for every L2 host->device
                            load_back walk (FULL / SWA / ...). For legacy caches
                            that don't run multi-component validation, set this
                            equal to `last_host_node`.
        host_hit_length :   Number of Full-KV tokens that hit on host (CPU) and need to be
                            loaded back to device. Pure-KV cache semantics;
        swa_host_hit_length  :   Number of SWA tokens that hit on host (within the sliding
                            window) and will be load-back into the SWA device pool.
        mamba_host_hit_length:   Number of Mamba slots that hit on host and will be load-back
                            into the Mamba device pool. Typically 0 or 1.
        mamba_branching_seqlen: The mamba radix cache branching point, which is the longest
                                page-aligned position that could've been cache hit if there
                                exists a mamba state.
    """

    device_indices: torch.Tensor
    last_device_node: Any
    last_host_node: Any
    best_match_node: Any
    host_hit_length: int = 0
    swa_host_hit_length: int = 0
    mamba_host_hit_length: int = 0
    mamba_branching_seqlen: Optional[int] = None
    cache_protected_len: Optional[int] = None


def zero_match_result(tree_cache, match_result: MatchResult) -> MatchResult:
    if tree_cache.is_chunk_cache():
        # Chunk caches' match_prefix already returns a miss; no root_node to walk back to.
        return match_result
    root = tree_cache.root_node
    return match_result._replace(
        # [:0] keeps dtype and device of the original tensor (e.g. CUDA int64)
        # without allocating a fresh empty tensor.
        device_indices=match_result.device_indices[:0],
        last_device_node=root,
        last_host_node=root,
        best_match_node=root,
        host_hit_length=0,
        swa_host_hit_length=0,
        mamba_host_hit_length=0,
    )


class BasePrefixCache(ABC, PrefixCacheTrait):
    """Cache can be indexed by either rid or key."""

    metrics_collector: Optional[RadixCacheMetricsCollector] = (
        None  # metrics collector for the cache
    )

    def init_metrics_collector(self):
        from sglang.srt.runtime_context import get_server_args

        server_args = get_server_args()
        labels = {"cache_type": self.__class__.__name__}
        if server_args.extra_metric_labels:
            labels.update(server_args.extra_metric_labels)
        radix_cache_cls = resolve_collector_class(
            server_args,
            STAT_LOGGER_ROLE_RADIX_CACHE,
            RadixCacheMetricsCollector,
        )
        self.metrics_collector = radix_cache_cls(labels=labels)

    def update_eviction_metrics(self, num_evicted: int, start_time: float):
        if self.metrics_collector is not None and num_evicted > 0:
            self.metrics_collector.observe_eviction_duration(
                time.perf_counter() - start_time
            )
            self.metrics_collector.increment_eviction_num_tokens(num_evicted)

    def release_host_resources(self) -> None:
        """Release pinned host buffers in userspace on graceful shutdown.

        Kernel-side unpinning during process reclaim can stall teardown for
        tens of seconds (see HostKVCache.destroy). Idempotent.
        """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        pass

    def supports_fast_match_prefix(self) -> bool:
        return False

    def supports_dynamic_hicache_sidecars(self) -> bool:
        return False

    def register_hicache_draft_pools(
        self, specs: list[SidecarPoolSpec], entries: list[PoolEntry]
    ) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} does not support dynamic HiCache sidecars."
        )

    @abstractmethod
    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        pass

    @abstractmethod
    def cache_unfinished_req(self, req: Req, **kwargs):
        pass

    @abstractmethod
    def evict(self, params: EvictParams) -> EvictResult:
        pass

    @abstractmethod
    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        pass

    @abstractmethod
    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        pass

    def evictable_size(self):
        return 0

    def full_evictable_size(self):
        return 0

    def swa_evictable_size(self):
        return 0

    def protected_size(self):
        return 0

    def full_protected_size(self):
        return 0

    def swa_protected_size(self):
        return 0

    def total_size(self):
        raise NotImplementedError()

    def pretty_print(self):
        raise NotImplementedError()

    def init_load_back(
        self,
        params: InitLoadBackParams,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Preparing KV cache loading from host to device.
        """
        raise NotImplementedError()

    def ready_to_load_host_cache(self) -> Any:
        """
        Notify the cache controller to start the KV cache loading
        """
        raise NotImplementedError()

    def flush_write_through_acks(self) -> None:
        """Release lock_ref on radix-tree nodes whose write-through has completed.

        Lightweight operation that only processes finished write acks.
        No-op for caches without hierarchical write-through support.
        """
        pass

    def check_hicache_events(self) -> Any:
        """
        Check HiCache related activities to update radix tree and synchronize across TP workers if needed
        """
        raise NotImplementedError()

    def take_events(self):
        return []

    def supports_swa(self) -> bool:
        return False

    def swa_reprefill_tail_tokens(self) -> int:
        # Only the unified_kv compress-only HiCache layout needs to hold back a
        # trailing sliding window for re-prefill; every other cache keeps SWA
        # content-stable and overrides this where relevant.
        return 0

    def supports_mamba(self) -> bool:
        return False

    def supports_streaming_session(self) -> bool:
        return False

    def release_session(self, session_id: str) -> None:
        pass

    def release_radix_session(self, session_id: str) -> None:
        pass

    def session_held_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_full_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_swa_tokens(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_req_count(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def session_held_mamba_slots(self, active_pool_idxs: Optional[set] = None) -> int:
        return 0

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return not self.is_chunk_cache()

    def available_and_evictable_str(self) -> str:
        available_size = self.token_to_kv_pool_allocator.available_size()
        evictable_size = self.evictable_size()
        return f"Available tokens: {available_size + evictable_size} ({available_size=} + {evictable_size=})\n"
