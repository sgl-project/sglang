from __future__ import annotations

import dataclasses
import time
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import torch

from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.events import KVCacheEventRecorder
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_RADIX_CACHE,
    RadixCacheMetricsCollector,
    resolve_collector_class,
)
from sglang.srt.runtime_context import get_observability

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import HiCacheController
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
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

    # DSV4 NPU C128 sidecar pages, one page id per physical C128 page group.
    c128_value: Optional[torch.Tensor] = None

    # SWA specific
    prev_prefix_len: int = 0
    swa_evicted_seqlen: int = 0
    swa_branching_seqlen: Optional[int] = None

    # General
    chunked: bool = False
    priority: int = 0
    track_adopted_ranges: bool = False


@dataclasses.dataclass
class InsertResult:
    """Result of an insert operation"""

    prefix_len: int
    total_len: int = 0
    last_device_node: Any = None
    mamba_exist: bool = False
    swa_branch_inserted: bool = False
    inserted_host_node: Any = None
    host_insert_dropped: bool = False
    adopted_ranges: Optional[dict[ComponentType, list[tuple[int, int]]]] = None
    # Controller-applied actions from the non-stepped channels (e.g. insert_host); the stepped insert emits via InsertStepResult.actions.
    cache_actions: list[CacheAction | ComponentAction] = dataclasses.field(
        default_factory=list
    )

    def record_adopted_range(
        self, component_type: ComponentType, start: int, end: int
    ) -> None:
        if self.adopted_ranges is None or start >= end:
            return
        ranges = self.adopted_ranges.setdefault(component_type, [])
        if ranges and start <= ranges[-1][1]:
            prev_start, prev_end = ranges[-1]
            ranges[-1] = (min(prev_start, start), max(prev_end, end))
        else:
            ranges.append((start, end))


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
        swa_branching_seqlen: The SWA radix cache branching point, which is the longest
                              page-aligned position that could've been cache hit if there
                              exists an SWA window.
        mamba_host_hit_length:   Number of Mamba slots that hit on host and will be load-back
                            into the Mamba device pool. Typically 0 or 1.
        mamba_branching_seqlen: The mamba radix cache branching point, which is the longest
                                page-aligned position that could've been cache hit if there
                                exists a mamba state.
        full_kv_hit_length: Longest Full-KV prefix available on either device or
                            host, independent of other components.
    """

    device_indices: torch.Tensor
    last_device_node: Any
    last_host_node: Any
    best_match_node: Any
    host_hit_length: int = 0
    swa_host_hit_length: int = 0
    swa_branching_seqlen: Optional[int] = None
    mamba_host_hit_length: int = 0
    mamba_branching_seqlen: Optional[int] = None
    cache_protected_len: Optional[int] = None
    full_kv_hit_length: int = 0
    # Actions the Controller applies: CacheActions itself, ComponentActions routed to the owning component.
    cache_actions: Sequence[CacheAction | ComponentAction] = ()


def zero_match_result(
    tree_cache, match_result: MatchResult, extra_key: Optional[str] = None
) -> MatchResult:
    if tree_cache.is_chunk_cache():
        # Chunk caches' match_prefix already returns a miss; no root_node to walk back to.
        return match_result
    root = tree_cache.root_node_handle(extra_key=extra_key)
    return match_result._replace(
        # [:0] keeps dtype and device of the original tensor (e.g. CUDA int64)
        # without allocating a fresh empty tensor.
        device_indices=match_result.device_indices[:0],
        last_device_node=root,
        last_host_node=root,
        best_match_node=root,
        host_hit_length=0,
        swa_host_hit_length=0,
        swa_branching_seqlen=None,
        mamba_host_hit_length=0,
        full_kv_hit_length=0,
    )


def _dfs_weight_order(
    root_node: Any,
    node_handles: Sequence[Any],
    resolve_node_handle: Callable[[Any], Any],
) -> list[int]:
    last_node_to_indices: dict[Any, list[int]] = {}
    for index, node_handle in enumerate(node_handles):
        node = resolve_node_handle(node_handle)
        last_node_to_indices.setdefault(node, []).append(index)

    node_to_weight: dict[Any, int] = {
        node: len(indices) for node, indices in last_node_to_indices.items()
    }

    def calc_weight(node: Any) -> None:
        for child in node.children.values():
            calc_weight(child)
            node_to_weight[node] = node_to_weight.get(node, 0) + node_to_weight.get(
                child, 0
            )

    calc_weight(root_node)

    order: list[int] = []

    def append_dfs(node: Any) -> None:
        children = list(node.children.values())
        children.sort(key=lambda child: -node_to_weight.get(child, 0))
        for child in children:
            append_dfs(child)
        order.extend(last_node_to_indices.get(node, ()))

    append_dfs(root_node)
    return order


class BasePrefixCache(ABC, PrefixCacheTrait):
    """Cache can be indexed by either rid or key."""

    metrics_collector: Optional[RadixCacheMetricsCollector] = (
        None  # metrics collector for the cache
    )
    cache_controller: Optional[HiCacheController] = None
    # Set by caches that publish KV placement events; None means they don't.
    kv_events: Optional[KVCacheEventRecorder] = None

    def init_metrics_collector(self):
        labels = {"cache_type": self.__class__.__name__}
        if get_observability().extra_metric_labels:
            labels.update(get_observability().extra_metric_labels)
        radix_cache_cls = resolve_collector_class(
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

    def dfs_weight_order(self, node_handles: Sequence[Any]) -> list[int]:
        """Return request indices in depth-first, subtree-weight order."""
        return _dfs_weight_order(self.root_node, node_handles, self.resolve_node_handle)

    def resolve_node_handle(self, node_handle: Any) -> Any:
        """Map a node handle to its node -- e.g. UnifiedRadixCache looks up the
        node object from its NodeId. Temporary API for the Unified Radix Cache
        split migration.

        TODO(Jialin): Remove after the Unified Radix Cache split.
        """
        return node_handle

    def root_node_handle(self, extra_key: Optional[str] = None) -> Any:
        """The root handle as match results carry it -- the raw node by default,
        the root's NodeId for UnifiedRadixCache. extra_key scopes the root for
        implementations that shard trees per cache namespace."""
        return self.root_node

    def is_backuped(self, node: Any) -> bool:
        """Whether the node's Full KV is present on host."""
        return node.backuped

    def is_root(self, node: Any) -> bool:
        """Whether the node is a tree root."""
        return node is self.root_node

    def get_last_hash_value(self, node: Any) -> Optional[str]:
        """The node's last page hash, or None when it was never hashed."""
        return node.get_last_hash_value()

    def get_prefix_hash_values(self, node: Any) -> list[str]:
        """The hash chain of the node's ancestors, in root-to-parent order."""
        return node.get_prefix_hash_values(node.parent)

    @abstractmethod
    def cache_finished_req(self, req: Req, is_insert: bool = True, **kwargs):
        pass

    @abstractmethod
    def cache_unfinished_req(self, req: Req, **kwargs):
        pass

    def free_kv_row(self, kv: Any, ranges: list[tuple[int, int]]) -> None:
        """Give back ascending, disjoint, half-open row-position ranges
        of the ``kv`` record's row; one call keeps a shared page freed once.
        """
        from sglang.srt.mem_cache.common import free_kv_row_segments

        row = self.req_to_token_pool.req_to_token[kv.req_pool_idx]
        free_kv_row_segments(
            self.token_to_kv_pool_allocator,
            [(row[start:end], start) for start, end in ranges],
            swa_evicted_seqlen=kv.swa_evicted_seqlen,
        )

    @abstractmethod
    def evict(self, params: EvictParams) -> EvictResult:
        pass

    def evict_for_alloc(self, params: EvictParams) -> EvictResult:
        """Evict cache entries to cover allocator shortfalls.

        The default implementation preserves the component-count semantics of
        :meth:`evict`. Multi-component caches backed by shared memory can
        override this entry point to stop once collateral frees make the
        requested allocation feasible.
        """
        return self.evict(params)

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

    def finish_storage_prefetch_admission(
        self, req_id: str, fulfilled_tokens: int, reason: Optional[str]
    ) -> None:
        """Resolve storage-hit accounting once a request is admitted.

        Non-storage caches have no lifecycle state to resolve.
        """

    def discard_storage_prefetch_accounting(self, req_id: str) -> None:
        """Forget storage-hit lifecycle state without emitting a result."""

    def pop_prefetch_loaded_span(self, req_id: str) -> tuple[int, Optional[int]]:
        """Pop L3-loaded tokens and their absolute prefix start, if known."""
        return self.pop_prefetch_loaded_tokens(req_id), None

    def ready_to_load_host_cache(self) -> Any:
        """
        Notify the cache controller to start the KV cache loading
        """
        raise NotImplementedError()

    def check_hicache_events(self) -> Any:
        """
        Check HiCache related activities to update radix tree and synchronize across TP workers if needed
        """
        raise NotImplementedError()

    def take_events(self):
        return [] if self.kv_events is None else self.kv_events.take()

    def supports_swa(self) -> bool:
        return False

    def swa_retain_floor(self, req) -> int | None:
        # A match lands on a state checkpoint rather than on the tail, so a cache
        # that pairs SWA with mamba/conv checkpoints has to keep the window behind
        # the last checkpoint. Those caches override this. Everyone else has
        # nothing deeper than the tail to protect.
        return None

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
