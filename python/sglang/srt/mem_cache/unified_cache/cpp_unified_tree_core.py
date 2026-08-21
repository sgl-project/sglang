"""C++ TreeCore backend for the FULL, device-only unified radix cache.

This is intentionally a narrow first stage.  UnifiedRadixCache remains the
controller and owns allocator side effects, while the C++ radix tree owns the
trie, node values, reference counts, LRU timestamps, and eviction decisions.
Unsupported unified-cache features fail during construction instead of
silently falling back to the Python tree.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    DecLockRefResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cpp_radix_tree.radix_tree import RadixTreeCpp
from sglang.srt.mem_cache.unified_cache.cache_action import FreeDeviceKV
from sglang.srt.mem_cache.unified_cache.component_type import (
    BASE_COMPONENT_TYPE,
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    DecSwaLockOnlyResult,
    EvictDeviceNextNodeResult,
    NodeId,
)

logger = logging.getLogger(__name__)


class CppUnifiedTreeCore(UnifiedTreeCore):
    """FULL/device-only implementation backed by ``radix_tree_v2``.

    Subclassing keeps the broad TreeCore surface available while the first
    stage is deliberately restricted.  Every operation used by the supported
    path is overridden below and operates on ``self.tree``.
    """

    def __init__(self, params, components):
        from sglang.srt.mem_cache.unified_cache.tree_core_registry import (
            cpp_tree_core_unsupported_reason,
        )

        unsupported_reason = cpp_tree_core_unsupported_reason(params, components)
        if unsupported_reason is not None:
            raise NotImplementedError(
                "The C++ unified TreeCore does not support this configuration "
                f"yet: {unsupported_reason}"
            )

        # Initializes common configuration and the inert Python root facade.
        # The C++ tree below is the authoritative tree for the supported path.
        super().__init__(params, components)
        self.tree = RadixTreeCpp(
            disabled=params.disable,
            host_size=None,
            page_size=self.page_size,
            write_through_threshold=self.write_through_threshold,
        )
        self._pending_evicted_values: list[torch.Tensor] = []
        self._ongoing_cpp_insert = False
        self._reset_root_facade()
        logger.info("Using C++ unified radix TreeCore (FULL component, device-only).")

    def _reset_root_facade(self) -> None:
        # C++ reserves node id 0 for its root.  A Python root object is retained
        # only for compatibility with callers that inspect ``cache.root_node``.
        self.root_node.id = 0
        self._node_arena = {0: self.root_node}
        self._empty_match_result = MatchResult(
            device_indices=torch.empty(0, dtype=torch.int64, device=self.device),
            last_device_node=0,
            last_host_node=0,
            best_match_node=0,
            cache_actions=[],
        )

    @staticmethod
    def _validate_key(key) -> None:
        if key.extra_key is not None:
            raise ValueError(
                "The C++ unified TreeCore does not support RadixKey.extra_key yet"
            )
        if key.cache_salt is not None:
            raise ValueError("The C++ unified TreeCore does not support cache_salt yet")
        if key.is_bigram:
            raise ValueError(
                "The C++ unified TreeCore does not support bigram keys yet"
            )

    def reset(self) -> None:
        # UnifiedTreeCore.__init__ calls this before ``self.tree`` exists.
        super().reset()
        if hasattr(self, "tree"):
            self.tree.reset()
            self._pending_evicted_values = []
            self._ongoing_cpp_insert = False
            self._reset_root_facade()

    def node_by_id(self, node_id: NodeId):
        if node_id == 0:
            return self.root_node
        raise NotImplementedError(
            "Materializing non-root C++ nodes as Python objects is not supported"
        )

    def is_backuped(self, node_id: NodeId) -> bool:
        return False

    def is_root(self, node_id: NodeId) -> bool:
        return node_id == 0

    def root_node_handle(self, extra_key: Optional[str] = None) -> NodeId:
        if extra_key is not None:
            raise ValueError(
                "The C++ unified TreeCore does not support extra_key namespaces yet"
            )
        return 0

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        key = params.key.page_aligned(self.page_size)
        self._validate_key(key)
        if len(key) == 0:
            return self._empty_match_result

        chunks, host_hit_length, device_node, host_node = self.tree.match_prefix(
            key.raw_token_ids()
        )
        if chunks:
            device_indices = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
        else:
            device_indices = self._empty_match_result.device_indices
        return MatchResult(
            device_indices=device_indices,
            last_device_node=device_node,
            last_host_node=host_node,
            best_match_node=host_node,
            host_hit_length=host_hit_length,
            full_kv_hit_length=len(device_indices) + host_hit_length,
            cache_actions=[],
        )

    def begin_insert(self, params: InsertParams):
        from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
            InsertStepResult,
        )

        assert not self._ongoing_cpp_insert, "concurrent insert walks"
        key = params.key.page_aligned(self.page_size)
        self._validate_key(key)
        value = params.value
        if value is None:
            value = torch.tensor(
                key.raw_token_ids(), dtype=torch.int64, device=self.device
            )
        value = value[: len(key)]
        self._ongoing_cpp_insert = True
        if len(key) == 0:
            return InsertStepResult(
                actions=[], result=InsertResult(prefix_len=0, last_device_node=0)
            )

        try:
            pending_io, prefix_len = self.tree.writing_through(
                key.raw_token_ids(), value
            )
            assert not pending_io, "device-only C++ tree emitted an unexpected I/O"
            _, _, last_device_node, _ = self.tree.match_prefix(key.raw_token_ids())
        except BaseException:
            self._ongoing_cpp_insert = False
            raise

        # The request already owns [0, prev_prefix_len).  Any additional
        # overlap is a duplicate allocation and must be returned by the
        # unified controller; the unmatched suffix is now owned by C++.
        duplicate_start = min(max(params.prev_prefix_len, 0), prefix_len)
        actions = []
        if duplicate_start < prefix_len:
            actions.append(FreeDeviceKV([value[duplicate_start:prefix_len]]))
        return InsertStepResult(
            actions=actions,
            result=InsertResult(
                prefix_len=prefix_len, last_device_node=last_device_node
            ),
        )

    def resume_insert(self):
        raise AssertionError("The device-only C++ insert never suspends")

    def has_ongoing_insert(self) -> bool:
        return self._ongoing_cpp_insert

    def end_insert(self):
        self._ongoing_cpp_insert = False
        return []

    def inc_lock_ref(
        self,
        node_id: NodeId,
        skip_lock_components: Sequence[ComponentType] = (),
    ) -> IncLockRefResult:
        if skip_lock_components:
            raise NotImplementedError(
                "The FULL-only C++ tree cannot skip component locks"
            )
        self.tree.lock_ref(node_id, True)
        return IncLockRefResult()

    def dec_lock_ref(
        self,
        node_id: NodeId,
        params: Optional[DecLockRefParams] = None,
        skip_swa: bool = False,
    ) -> DecLockRefResult:
        if skip_swa:
            raise NotImplementedError("The C++ unified TreeCore has no SWA component")
        self.tree.lock_ref(node_id, False)
        return DecLockRefResult()

    def dec_swa_lock_only(
        self,
        node_id: NodeId,
        swa_uuid_for_lock: Optional[int],
        skip_lock_node_ids: Optional[dict] = None,
    ) -> DecSwaLockOnlyResult:
        raise NotImplementedError("The C++ unified TreeCore has no SWA component")

    def evict_device_start(
        self, component_type: ComponentType, request_cnt: int
    ) -> None:
        if component_type != BASE_COMPONENT_TYPE:
            raise NotImplementedError(
                f"Unsupported C++ eviction component: {component_type}"
            )
        assert not self._pending_evicted_values, "nested C++ eviction walk"
        self._pending_evicted_values = self.tree.evict(request_cnt)

    def evict_device_next_node(
        self, component_type: ComponentType, tracker: dict[ComponentType, int]
    ) -> EvictDeviceNextNodeResult:
        result = EvictDeviceNextNodeResult(node_id=None)
        if self._pending_evicted_values:
            values, self._pending_evicted_values = self._pending_evicted_values, []
            result.device_frees[BASE_COMPONENT_TYPE].extend(values)
            result.tracker[BASE_COMPONENT_TYPE] = sum(len(v) for v in values)
        return result

    def evict_device_end(self, component_type: ComponentType) -> None:
        self._pending_evicted_values = []

    def evictable_size(self) -> int:
        return self.tree.evictable_size()

    def protected_size(self) -> int:
        return self.tree.protected_size()

    def component_evictable_size(self, component_type: ComponentType) -> int:
        return self.evictable_size() if component_type == BASE_COMPONENT_TYPE else 0

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def full_protected_size(self) -> int:
        return self.protected_size()

    def swa_evictable_size(self) -> int:
        return 0

    def mamba_evictable_size(self) -> int:
        return 0

    def swa_protected_size(self) -> int:
        return 0

    def mamba_protected_size(self) -> int:
        return 0

    def total_size(self) -> tuple[int, int]:
        return self.tree.total_size(), 0

    def all_values_flatten(self) -> torch.Tensor:
        values = self.tree.all_values()
        if not values:
            return self._empty_match_result.device_indices
        return values[0] if len(values) == 1 else torch.cat(values)

    def all_mamba_values_flatten(self) -> torch.Tensor:
        return self._empty_match_result.device_indices

    def set_hicache_enabled(self) -> None:
        raise NotImplementedError(
            "The C++ unified TreeCore does not support HiCache yet"
        )

    def sanity_check(self, ongoing_write_through, ongoing_load_back) -> None:
        if ongoing_write_through or ongoing_load_back:
            raise AssertionError("Device-only C++ tree cannot have in-flight I/O")
        total = self.tree.total_size()
        assert total == self.tree.evictable_size() + self.tree.protected_size()

    def pretty_print(self) -> None:
        self.tree.debug_print()
