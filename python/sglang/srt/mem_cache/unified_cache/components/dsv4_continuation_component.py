from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.unified_cache.cache_action import (
    FreeComponentDeviceSlot,
    FreeComponentHostSlot,
)
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    LRURefreshPhase,
    TreeComponent,
    get_and_increase_time_counter,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_radix_cache import (
        NodeId,
        UnifiedRadixCache,
        UnifiedTreeNode,
    )


class DSV4ContinuationComponent(TreeComponent):
    component_type = ComponentType.DSV4_CONTINUATION

    def __init__(self, cache: UnifiedRadixCache, params: CacheInitParams):
        super().__init__(cache, params)
        if params.dsv4_continuation_pool is None:
            raise ValueError("DSV4 continuation component requires its device pool")
        self.pool = params.dsv4_continuation_pool
        self._host_pool = None

    def reset_session_state(self) -> None:
        super().reset_session_state()
        self.pool.clear()

    def _inc_session_coverage(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        cd = leaf.component_data[self.component_type]
        cd.session_ref += 1
        if cd.session_ref == 1:
            self._refresh_session_partition(leaf)

    def _dec_session_coverage(self, session_id: str, leaf: UnifiedTreeNode) -> None:
        cd = leaf.component_data[self.component_type]
        assert cd.session_ref > 0
        cd.session_ref -= 1
        if cd.session_ref == 0:
            self._refresh_session_partition(leaf)

    def _advance_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        old_ancestor: Optional[UnifiedTreeNode],
    ) -> None:
        self._inc_session_coverage(session_id, leaf)
        if old_ancestor is not None:
            self._dec_session_coverage(session_id, old_ancestor)

    def _recede_session_coverage(
        self,
        session_id: str,
        leaf: UnifiedTreeNode,
        fallback: Optional[UnifiedTreeNode],
    ) -> None:
        self._dec_session_coverage(session_id, leaf)
        if fallback is not None:
            self._inc_session_coverage(session_id, fallback)

    def needs_incremental_backup(self, node: UnifiedTreeNode) -> bool:
        cd = node.component_data[self.component_type]
        return cd.value is not None and cd.host_value is None

    def refresh_lru(
        self,
        phase: LRURefreshPhase,
        node: UnifiedTreeNode,
        root_node: UnifiedTreeNode,
    ) -> None:
        if phase is LRURefreshPhase.WALKDOWN:
            return
        if phase is LRURefreshPhase.MATCH_END:
            cd = node.component_data[self.component_type]
            if cd.value is not None:
                self.tree_core.lru_lists[self.component_type].reset_node_mru(node)
            return
        if phase is LRURefreshPhase.INSERT_END:
            return
        raise ValueError(f"Unknown LRURefreshPhase: {phase}")

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        ct = self.component_type
        if match_device_only:
            return lambda node: node.component_data[ct].value is not None
        return lambda node: (
            node.component_data[ct].value is not None
            or node.component_data[ct].host_value is not None
        )

    def finalize_match_result_in_cache(
        self, params: MatchPrefixParams, result: MatchResult
    ) -> MatchResult:
        req = params.req
        if req is not None:
            node_id = result.best_match_node
            has_device = (
                self.tree_core.get_component_device_value(node_id, self.component_type)
                is not None
            )
            has_host_only = self.tree_core.component_has_host_value_only(
                node_id, self.component_type
            )
            req.dsv4_continuation_node = (
                node_id if has_device or has_host_only else None
            )
        return result

    def finalize_match_result_in_tree_core(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        if self.has_host_value_only(result.best_match_node):
            return result._replace(dsv4_continuation_host_hit=True)
        return result

    def prepare_for_caching_req(
        self,
        req: Req,
        insert_params: InsertParams,
        token_ids_len: int,
        is_finished: bool,
    ) -> Optional[int]:
        value = getattr(req, "dsv4_continuation_value", None)
        if value is not None:
            endpoint = getattr(req, "dsv4_continuation_endpoint", None)
            if endpoint is None:
                raise RuntimeError("DSV4 continuation capture is missing its endpoint")
            if not endpoint <= token_ids_len <= endpoint + 1:
                self.pool.free(value)
                req.dsv4_continuation_value = None
                req.dsv4_continuation_endpoint = None
                return None
            req.dsv4_continuation_value = None
            req.dsv4_continuation_endpoint = None
            insert_params.dsv4_continuation_value = value
            return endpoint
        return None

    def commit_insert_component_data(
        self,
        node: UnifiedTreeNode,
        is_new_leaf: bool,
        params: InsertParams,
        result: InsertResult,
        cache_actions: list[CacheAction | ComponentAction],
    ) -> None:
        value = params.dsv4_continuation_value
        if value is None:
            return
        ct = self.component_type
        cd = node.component_data[ct]
        if cd.value is None:
            cd.value = value
            host_lru = self.tree_core.host_lru_lists[ct]
            if host_lru.in_list(node):
                host_lru.remove_node(node)
            self.tree_core.lru_lists[ct].insert_mru(node)
            self.tree_core.component_evictable_size_[ct] += len(value)
            node.last_access_time = get_and_increase_time_counter()
            return
        result.dsv4_continuation_exist = True
        self.tree_core.lru_lists[ct].reset_node_mru(node)
        node.last_access_time = get_and_increase_time_counter()

    def cleanup_after_caching_req(
        self,
        req: Req,
        is_finished: bool,
        insert_result: Optional[InsertResult] = None,
        insert_params: Optional[InsertParams] = None,
    ) -> None:
        if insert_params is None:
            value = getattr(req, "dsv4_continuation_value", None)
            if value is not None:
                self.pool.free(value)
                req.dsv4_continuation_value = None
                req.dsv4_continuation_endpoint = None
            return
        value = insert_params.dsv4_continuation_value
        if value is not None and (
            insert_result is None or insert_result.dsv4_continuation_exist
        ):
            self.pool.free(value)

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ) -> None:
        ct = self.component_type
        new_cd = new_parent.component_data[ct]
        child_cd = child.component_data[ct]
        new_cd.value = None
        new_cd.lock_ref = 0
        new_cd.host_value = None
        new_cd.host_lock_ref = 0
        new_cd.session_ref = 0
        assert new_cd.session_ids is None

    def evict_component(
        self,
        node: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        ct = self.component_type
        cd = node.component_data[ct]
        freed = 0
        host_freed = 0
        if EvictLayer.DEVICE in target and cd.value is not None:
            device_frees[ct].append(cd.value)
            freed = len(cd.value)
            self.tree_core.component_evictable_size_[ct] -= freed
            cd.value = None
        host_lru = self.tree_core.host_lru_lists[ct]
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            host_frees[ct].append(cd.host_value)
            cd.host_value = None
            if host_lru.in_list(node):
                host_lru.remove_node(node)
        if (
            target is EvictLayer.DEVICE
            and cd.value is None
            and cd.host_value is not None
            and not host_lru.in_list(node)
        ):
            host_lru.insert_mru(node)
        return freed, host_freed

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else -1

    def _evict_device_start(self, request_cnt: int) -> None:
        self._evict_device_request_cnt = request_cnt
        lru = self.tree_core.lru_lists[self.component_type]
        if self.tree_core.enable_session_radix_cache:
            lru.cursor_begin()
            self._evict_device_cursor = lru.cursor_next()
        else:
            self._evict_device_cursor = lru.get_lru_no_lock()

    def _evict_device_next_node(
        self,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> Optional[NodeId]:
        ct = self.component_type
        lru = self.tree_core.lru_lists[ct]
        session_enabled = self.tree_core.enable_session_radix_cache
        if self._evict_device_cursor is not None and not lru.in_list(
            self._evict_device_cursor
        ):
            self._evict_device_cursor = (
                lru.cursor_next() if session_enabled else lru.get_lru_no_lock()
            )
        while (
            tracker[ct] < self._evict_device_request_cnt
            and self._evict_device_cursor is not None
            and lru.in_list(self._evict_device_cursor)
        ):
            node = self._evict_device_cursor
            self._evict_device_cursor = (
                lru.cursor_next() if session_enabled else lru.get_prev_no_lock(node)
            )
            cd = node.component_data[ct]
            if node in self.tree_core.evictable_device_leaves and (
                not session_enabled or self._can_evict_leaf_atomically(node)
            ):
                return node.id
            if cd.lock_ref > 0:
                continue
            if cd.host_value is None:
                return node.id
            self.tree_core._evict_component_and_detach_lru(
                node,
                self,
                target=EvictLayer.DEVICE,
                tracker=tracker,
                device_frees=device_frees,
                host_frees=host_frees,
            )
        return None

    def _evict_device_end(self) -> None:
        if self.tree_core.enable_session_radix_cache:
            self.tree_core.lru_lists[self.component_type].cursor_end()
        self._evict_device_cursor = None

    def build_hicache_transfers(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        *,
        mamba_pool_idx: Optional[torch.Tensor] = None,
        host_indices: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[int]] = None,
        prefetch_tokens: int = 0,
        last_hash: Optional[str] = None,
    ) -> Optional[list[PoolTransfer]]:
        cd = node.component_data[self.component_type]
        if phase == CacheTransferPhase.BACKUP_HOST:
            if cd.value is None:
                return None
            return [
                PoolTransfer(
                    name=PoolName.DSV4_CONTINUATION,
                    device_indices=cd.value,
                )
            ]
        if phase == CacheTransferPhase.LOAD_BACK:
            if cd.value is not None or cd.host_value is None:
                return None
            return [
                PoolTransfer(
                    name=PoolName.DSV4_CONTINUATION,
                    host_indices=cd.host_value,
                    nodes_to_load=[node.id],
                )
            ]
        return None

    def commit_hicache_transfer(
        self,
        node: UnifiedTreeNode,
        phase: CacheTransferPhase,
        transfers: list[PoolTransfer] = (),
        *,
        cache_actions: list[CacheAction | ComponentAction],
        insert_result: Optional[InsertResult] = None,
        pool_storage_result: Optional[PoolTransferResult] = None,
    ) -> None:
        if not transfers:
            return
        ct = self.component_type
        cd = node.component_data[ct]
        transfer = transfers[0]
        if phase == CacheTransferPhase.BACKUP_HOST:
            if transfer.host_indices is not None and cd.host_value is None:
                cd.host_value = transfer.host_indices.clone()
            return
        if phase != CacheTransferPhase.LOAD_BACK:
            return
        if transfer.device_indices is None or cd.value is not None:
            return
        cd.value = transfer.device_indices.clone()
        host_lru = self.tree_core.host_lru_lists[ct]
        if host_lru.in_list(node):
            host_lru.remove_node(node)
        self.tree_core.lru_lists[ct].insert_mru(node)
        self.tree_core.component_evictable_size_[ct] += len(cd.value)

    def drive_host_eviction(
        self,
        num_tokens: int,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        ct = self.component_type
        host_lru = self.tree_core.host_lru_lists[ct]
        session_enabled = self.tree_core.enable_session_radix_cache
        if session_enabled:
            host_lru.cursor_begin()
            node = host_lru.cursor_next(host_lock=True)
        else:
            node = host_lru.get_lru_no_host_lock()
        while tracker[ct] < num_tokens and node is not None and host_lru.in_list(node):
            if not session_enabled:
                next_node = host_lru.get_prev_no_host_lock(node)
            if node in self.tree_core.evictable_host_leaves and (
                not session_enabled or self._can_evict_leaf_atomically(node)
            ):
                self.tree_core._evict_host_leaf(node, tracker, device_frees, host_frees)
            else:
                self.tree_core._evict_component_and_detach_lru(
                    node,
                    self,
                    target=EvictLayer.HOST,
                    tracker=tracker,
                    device_frees=device_frees,
                    host_frees=host_frees,
                )
                self.tree_core._cascade_evict(
                    node,
                    self,
                    tracker,
                    device_frees=device_frees,
                    host_frees=host_frees,
                    target=EvictLayer.HOST,
                )
                self.tree_core._update_evictable_leaf_sets(node)
            node = (
                host_lru.cursor_next(host_lock=True) if session_enabled else next_node
            )
        if session_enabled:
            host_lru.cursor_end()

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        if node is self.tree_core.root_node:
            return result
        ct = self.component_type
        cd = node.component_data[ct]
        value = cd.host_value if lock_host else cd.value
        if value is None:
            result.skip_lock_node_ids.setdefault(ct, set()).add(node.id)
            return result
        if lock_host:
            if cd.host_lock_ref == 0:
                host_lru = self.tree_core.host_lru_lists[ct]
                if host_lru.in_list(node):
                    host_lru.remove_node(node)
            cd.host_lock_ref += 1
        else:
            if cd.lock_ref == 0:
                self.tree_core.component_evictable_size_[ct] -= len(value)
                self.tree_core.component_protected_size_[ct] += len(value)
            cd.lock_ref += 1
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: Optional[DecLockRefParams],
        lock_host: bool = False,
    ) -> None:
        if node is self.tree_core.root_node:
            return
        ct = self.component_type
        if node.id in (params.skip_lock_node_ids.get(ct, ()) if params else ()):
            return
        cd = node.component_data[ct]
        if lock_host:
            assert cd.host_lock_ref > 0
            cd.host_lock_ref -= 1
            if cd.host_lock_ref == 0 and cd.value is None and cd.host_value is not None:
                self.tree_core.host_lru_lists[ct].insert_mru(node)
            return
        assert cd.value is not None and cd.lock_ref > 0
        cd.lock_ref -= 1
        if cd.lock_ref == 0:
            self.tree_core.component_protected_size_[ct] -= len(cd.value)
            self.tree_core.component_evictable_size_[ct] += len(cd.value)

    def free_host_values(self, host_values: list[torch.Tensor]) -> None:
        if self._host_pool is None:
            return
        for host_value in host_values:
            self._host_pool.free(host_value)

    def apply_component_action(self, action: ComponentAction) -> None:
        if isinstance(action, FreeComponentDeviceSlot):
            for indices in action.indices:
                self.pool.free(indices)
            return
        if isinstance(action, FreeComponentHostSlot):
            for host_indices in action.host_indices:
                if host_indices is not None and host_indices.numel() > 0:
                    self.cache.cache_controller.append_host_mem_release(
                        extra_pools=[
                            PoolTransfer(
                                name=PoolName.DSV4_CONTINUATION,
                                host_indices=host_indices,
                            )
                        ]
                    )
            return
        raise AssertionError(
            "DSV4ContinuationComponent: unhandled action " f"{type(action).__name__}"
        )
