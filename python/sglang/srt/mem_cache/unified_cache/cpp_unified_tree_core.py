"""C++ TreeCore backend for device-only FULL and FULL+SWA unified caches.

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
from sglang.srt.mem_cache.unified_cache.cache_action import (
    FreeDeviceKV,
    RecoverSWAWithLockedFull,
    SWARebuild,
)
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
    """FULL or FULL+SWA device-only implementation backed by ``radix_tree_v2``.

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
        self._has_swa = ComponentType.SWA in components
        self._sliding_window_size = params.sliding_window_size or 0
        self.tree = RadixTreeCpp(
            disabled=params.disable,
            host_size=None,
            page_size=self.page_size,
            write_through_threshold=self.write_through_threshold,
            sliding_window_size=(self._sliding_window_size if self._has_swa else 0),
        )
        self._pending_evicted_values: Optional[tuple] = None
        self._ongoing_cpp_insert = False
        self._reset_root_facade()
        logger.info(
            "Using C++ unified radix TreeCore (%s, device-only).",
            "FULL+SWA" if self._has_swa else "FULL",
        )

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

    def _key_buffer(self, key):
        """Return RadixKey storage plus the page-aligned logical length.

        The backing ``array('q')`` stays alive for the complete pybind call.
        Passing the length separately honors both RadixKey.limit and page
        alignment without constructing an aligned Python slice.
        """
        self._validate_key(key)
        return key.token_ids, len(key) // self.page_size * self.page_size

    def reset(self) -> None:
        # UnifiedTreeCore.__init__ calls this before ``self.tree`` exists.
        super().reset()
        if hasattr(self, "tree"):
            self.tree.reset()
            self._pending_evicted_values = None
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
        key_buffer, key_len = self._key_buffer(params.key)
        if key_len == 0:
            return self._empty_match_result

        if self._has_swa:
            device_indices, device_node, full_kv_hit_length = (
                self.tree.match_prefix_swa_flat(key_buffer, key_len)
            )
            host_hit_length = 0
            host_node = device_node
        else:
            device_indices, host_hit_length, device_node, host_node = (
                self.tree.match_prefix_flat(key_buffer, key_len)
            )
            full_kv_hit_length = (
                len(device_indices) + host_hit_length
                if device_indices is not None
                else host_hit_length
            )
        if device_indices is None:
            device_indices = self._empty_match_result.device_indices
        return MatchResult(
            device_indices=device_indices,
            last_device_node=device_node,
            last_host_node=host_node,
            best_match_node=host_node,
            host_hit_length=host_hit_length,
            full_kv_hit_length=full_kv_hit_length,
            cache_actions=[],
        )

    def begin_insert(self, params: InsertParams):
        from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
            InsertStepResult,
        )

        assert not self._ongoing_cpp_insert, "concurrent insert walks"
        key_buffer, key_len = self._key_buffer(params.key)
        value = params.value
        if value is None:
            value = torch.tensor(
                key_buffer[:key_len], dtype=torch.int64, device=self.device
            )
        value = value[:key_len]
        self._ongoing_cpp_insert = True
        if key_len == 0:
            return InsertStepResult(
                actions=[], result=InsertResult(prefix_len=0, last_device_node=0)
            )

        try:
            if self._has_swa:
                (
                    prefix_len,
                    last_device_node,
                    duplicate_frees,
                    rebuilds,
                    recoveries,
                ) = self.tree.writing_through_swa(
                    key_buffer,
                    value,
                    max(params.prev_prefix_len, 0),
                    max(params.swa_evicted_seqlen, 0),
                    key_len,
                )
                actions = []
                if duplicate_frees:
                    actions.append(FreeDeviceKV(list(duplicate_frees)))
                actions.extend(
                    SWARebuild(node_id, source_value)
                    for node_id, source_value in rebuilds
                )
                actions.extend(
                    RecoverSWAWithLockedFull(node_id, kept_full, incoming_full)
                    for node_id, kept_full, incoming_full in recoveries
                )
            else:
                pending_io, prefix_len, last_device_node = self.tree.writing_through(
                    key_buffer, value, key_len
                )
                assert not pending_io, "device-only C++ tree emitted an unexpected I/O"
                duplicate_start = min(max(params.prev_prefix_len, 0), prefix_len)
                actions = []
                if duplicate_start < prefix_len:
                    actions.append(FreeDeviceKV([value[duplicate_start:prefix_len]]))
        except BaseException:
            self._ongoing_cpp_insert = False
            raise

        # The request already owns [0, prev_prefix_len).  Any additional
        # overlap is a duplicate allocation and must be returned by the
        # unified controller; the unmatched suffix is now owned by C++.
        return InsertStepResult(
            actions=actions,
            result=InsertResult(
                prefix_len=prefix_len, last_device_node=last_device_node
            ),
        )

    def match_inserted_prefix_and_lock(
        self,
        node_id: NodeId,
        skip_lock_components: Sequence[ComponentType] = (),
        inserted_value: Optional[torch.Tensor] = None,
        existing_prefix_len: int = 0,
        reused_prefix_len: int = 0,
    ) -> tuple[MatchResult, IncLockRefResult]:
        """Finish insert rematching and locking without sending the key again.

        ``inserted_value`` is already a private int64 copy of the request's
        full KV index vector.  Only the overlap discovered after the request's
        previous protected prefix can differ from the authoritative tree.
        Patch that small range in place and reuse the vector, avoiding a
        second full-prefix Tensor concatenation for every chunked prefill.
        """
        unsupported = set(skip_lock_components) - {
            ComponentType.FULL,
            ComponentType.SWA,
        }
        if unsupported:
            raise NotImplementedError(f"Unsupported skipped C++ locks: {unsupported}")
        if not self._has_swa and skip_lock_components:
            raise NotImplementedError(
                "The FULL-only C++ tree cannot skip component locks"
            )

        if inserted_value is None:
            device_indices, best_node, full_hit_length, swa_uuid, skipped = (
                self.tree.match_node_and_lock_flat(
                    node_id,
                    ComponentType.FULL not in skip_lock_components,
                    self._has_swa and ComponentType.SWA not in skip_lock_components,
                )
            )
            if device_indices is None:
                device_indices = self._empty_match_result.device_indices
        else:
            value_len = len(inserted_value)
            range_start = min(max(reused_prefix_len, 0), value_len)
            range_end = min(max(existing_prefix_len, range_start), value_len)
            (
                authoritative_overlap,
                best_node,
                best_prefix_len,
                full_hit_length,
                swa_uuid,
                skipped,
            ) = self.tree.match_node_range_and_lock_flat(
                node_id,
                range_start,
                range_end,
                ComponentType.FULL not in skip_lock_components,
                self._has_swa and ComponentType.SWA not in skip_lock_components,
            )
            best_prefix_len = min(best_prefix_len, value_len)
            patch_end = min(range_end, best_prefix_len)
            if patch_end > range_start:
                assert authoritative_overlap is not None
                assert len(authoritative_overlap) == patch_end - range_start
                inserted_value[range_start:patch_end].copy_(authoritative_overlap)
            device_indices = inserted_value[:best_prefix_len]
        match_result = MatchResult(
            device_indices=device_indices,
            last_device_node=best_node,
            last_host_node=best_node,
            best_match_node=best_node,
            host_hit_length=0,
            full_kv_hit_length=full_hit_length,
            cache_actions=[],
        )
        lock_result = IncLockRefResult(swa_uuid_for_lock=swa_uuid)
        if skipped:
            lock_result.skip_lock_node_ids[ComponentType.SWA] = set(skipped)
        return match_result, lock_result

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
        if not self._has_swa:
            if skip_lock_components:
                raise NotImplementedError(
                    "The FULL-only C++ tree cannot skip component locks"
                )
            self.tree.lock_ref(node_id, True)
            return IncLockRefResult()
        unsupported = set(skip_lock_components) - {
            ComponentType.FULL,
            ComponentType.SWA,
        }
        if unsupported:
            raise NotImplementedError(f"Unsupported skipped C++ locks: {unsupported}")
        swa_uuid, skipped_swa_nodes = self.tree.lock_ref_swa(
            node_id,
            ComponentType.FULL not in skip_lock_components,
            ComponentType.SWA not in skip_lock_components,
        )
        result = IncLockRefResult(swa_uuid_for_lock=swa_uuid)
        if skipped_swa_nodes:
            result.skip_lock_node_ids[ComponentType.SWA] = set(skipped_swa_nodes)
        return result

    def dec_lock_ref(
        self,
        node_id: NodeId,
        params: Optional[DecLockRefParams] = None,
        skip_swa: bool = False,
    ) -> DecLockRefResult:
        if not self._has_swa:
            if skip_swa:
                raise NotImplementedError(
                    "The C++ unified TreeCore has no SWA component"
                )
            self.tree.lock_ref(node_id, False)
            return DecLockRefResult()
        skip_ids = []
        swa_uuid = None
        if params is not None:
            skip_ids = list(params.skip_lock_node_ids.get(ComponentType.SWA, ()))
            swa_uuid = params.swa_uuid_for_lock
        self.tree.unlock_ref_swa(
            node_id,
            True,
            not skip_swa,
            swa_uuid,
            skip_ids,
        )
        return DecLockRefResult()

    def dec_swa_lock_only(
        self,
        node_id: NodeId,
        swa_uuid_for_lock: Optional[int],
        skip_lock_node_ids: Optional[dict] = None,
    ) -> DecSwaLockOnlyResult:
        if not self._has_swa:
            raise NotImplementedError("The C++ unified TreeCore has no SWA component")
        swa_frees, freed = self.tree.unlock_swa_only(node_id, swa_uuid_for_lock)
        result = DecSwaLockOnlyResult()
        if swa_frees:
            result.device_frees[ComponentType.SWA].extend(swa_frees)
            result.tracker[ComponentType.SWA] = freed
        return result

    def evict_device_start(
        self, component_type: ComponentType, request_cnt: int
    ) -> None:
        if component_type not in self.components_by_type:
            raise NotImplementedError(
                f"Unsupported C++ eviction component: {component_type}"
            )
        assert self._pending_evicted_values is None, "nested C++ eviction walk"
        if self._has_swa:
            self._pending_evicted_values = self.tree.evict_component(
                int(component_type), request_cnt
            )
        else:
            values = self.tree.evict(request_cnt)
            self._pending_evicted_values = (values, [], sum(map(len, values)), 0)

    def evict_device_next_node(
        self, component_type: ComponentType, tracker: dict[ComponentType, int]
    ) -> EvictDeviceNextNodeResult:
        result = EvictDeviceNextNodeResult(node_id=None)
        if self._pending_evicted_values is not None:
            full_values, swa_values, full_count, swa_count = (
                self._pending_evicted_values
            )
            self._pending_evicted_values = None
            if full_values:
                result.device_frees[BASE_COMPONENT_TYPE].extend(full_values)
                result.tracker[BASE_COMPONENT_TYPE] = full_count
            if swa_values:
                result.device_frees[ComponentType.SWA].extend(swa_values)
                result.tracker[ComponentType.SWA] = swa_count
        return result

    def evict_device_end(self, component_type: ComponentType) -> None:
        self._pending_evicted_values = None

    def evictable_size(self) -> int:
        return self.tree.evictable_size()

    def protected_size(self) -> int:
        return self.tree.protected_size()

    def component_evictable_size(self, component_type: ComponentType) -> int:
        return self.tree.component_evictable_size(int(component_type))

    def full_evictable_size(self) -> int:
        return self.evictable_size()

    def full_protected_size(self) -> int:
        return self.protected_size()

    def swa_evictable_size(self) -> int:
        return self.component_evictable_size(ComponentType.SWA) if self._has_swa else 0

    def mamba_evictable_size(self) -> int:
        return 0

    def swa_protected_size(self) -> int:
        return (
            self.tree.component_protected_size(int(ComponentType.SWA))
            if self._has_swa
            else 0
        )

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

    def set_component_device_value(
        self, node_id: NodeId, component_type: ComponentType, value: torch.Tensor
    ) -> None:
        if component_type != ComponentType.SWA or not self._has_swa:
            raise NotImplementedError(
                f"Unsupported C++ component value store: {component_type}"
            )
        self.tree.set_swa_value(node_id, value)

    def get_component_device_value(
        self, node_id: NodeId, component_type: ComponentType
    ) -> Optional[torch.Tensor]:
        if component_type != ComponentType.SWA or not self._has_swa:
            return None
        value = self.tree.get_swa_value(node_id)
        return value if value is not None and value.numel() else None

    def set_hicache_enabled(self) -> None:
        raise NotImplementedError(
            "The C++ unified TreeCore does not support HiCache yet"
        )

    def sanity_check(self, ongoing_write_through, ongoing_load_back) -> None:
        if ongoing_write_through or ongoing_load_back:
            raise AssertionError("Device-only C++ tree cannot have in-flight I/O")
        total = self.tree.total_size()
        assert total == self.tree.evictable_size() + self.tree.protected_size()
        if self._has_swa:
            swa_total = self.swa_evictable_size() + self.swa_protected_size()
            assert 0 <= swa_total <= total

    def pretty_print(self) -> None:
        self.tree.debug_print()
