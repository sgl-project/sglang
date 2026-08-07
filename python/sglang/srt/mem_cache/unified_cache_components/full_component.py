from __future__ import annotations

import heapq
from typing import TYPE_CHECKING, Callable, Optional, Sequence

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    IncLockRefResult,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.unified_cache.cache_action import FreeComponentDeviceSlot
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    CacheTransferPhase,
    ComponentType,
    EvictLayer,
    TreeComponent,
)

if TYPE_CHECKING:
    from sglang.srt.mem_cache.unified_cache.cache_action import (
        CacheAction,
        ComponentAction,
    )
    from sglang.srt.mem_cache.unified_radix_cache import (
        NodeId,
        UnifiedTreeNode,
    )


class FullComponent(TreeComponent):
    component_type = ComponentType.FULL

    def __init__(self, cache, params):
        super().__init__(cache, params)
        # HiCache state: set to host KV pool when HiCache enabled
        self._full_kv_pool_host = None

    def create_match_validator(
        self, match_device_only: bool = False
    ) -> Callable[[UnifiedTreeNode], bool]:
        if match_device_only:
            return (
                lambda node: node.component_data[self.component_type].value is not None
            )

        # HiCache: evicted + backuped nodes are valid match boundaries.
        return lambda node: (
            node.component_data[self.component_type].value is not None or node.backuped
        )

    def finalize_match_result_in_tree_core(
        self,
        result: MatchResult,
        params: MatchPrefixParams,
        value_chunks: list[torch.Tensor],
        best_value_len: int,
    ) -> MatchResult:
        # Compute Full KV host hit length: walk from last_host_node up to
        # last_device_node, summing host_value lengths of evicted nodes.
        ct = self.component_type
        kv_host_hit = 0
        node = result.best_match_node
        root_node = self.tree_core.root_node
        while node is not result.last_device_node and node is not root_node:
            full_host = node.component_data[ct].host_value
            if full_host is not None:
                kv_host_hit += len(full_host)
            node = node.parent
        if kv_host_hit > 0:
            return result._replace(
                host_hit_length=max(result.host_hit_length, kv_host_hit)
            )
        return result

    def redistribute_on_node_split(
        self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    ):
        ct = self.component_type
        new_parent.component_data[ct].lock_ref = child.component_data[ct].lock_ref
        child_cd = child.component_data[ct]
        split_len = len(new_parent.key)
        if child_cd.value is not None:
            new_parent.component_data[ct].value = child_cd.value[:split_len].clone()
            child_cd.value = child_cd.value[split_len:].clone()
        if child_cd.host_value is not None:
            new_parent.component_data[ct].host_value = child_cd.host_value[
                :split_len
            ].clone()
            child_cd.host_value = child_cd.host_value[split_len:].clone()

    def evict_component(
        self,
        node: UnifiedTreeNode,
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
        target: EvictLayer = EvictLayer.DEVICE,
    ) -> tuple[int, int]:
        cd = node.component_data[self.component_type]
        freed = 0
        host_freed = 0

        # Device layer
        if EvictLayer.DEVICE in target and cd.value is not None:
            device_frees[self.component_type].append(cd.value)
            freed = len(cd.value)
            self.tree_core.component_evictable_size_[self.component_type] -= freed
            # NOTE: cd.value = None is deferred to _cascade_evict (Full as trigger)
            # because SWA's free_swa still needs to read Full.value.
            # cd.value = None

        # Host layer
        if EvictLayer.HOST in target and cd.host_value is not None:
            host_freed = len(cd.host_value)
            host_frees[self.component_type].append(cd.host_value)
            cd.host_value = None
        return freed, host_freed

    def eviction_priority(self, is_leaf: bool) -> int:
        return 0 if is_leaf else 2

    def _evict_device_start(self, request_cnt: int) -> None:
        self._evict_device_request_cnt = request_cnt
        self._evict_device_last_node = None
        self._evict_device_heap = [
            (self.tree_core.eviction_strategy.get_priority(n), n)
            for n in self.tree_core.evictable_device_leaves
        ]
        heapq.heapify(self._evict_device_heap)

    def _evict_device_next_node(
        self,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> Optional[NodeId]:
        ct = self.component_type
        lv = self._evict_device_last_node
        if (
            lv is not None
            and lv.parent is not None
            and lv.parent in self.tree_core.evictable_device_leaves
        ):
            heapq.heappush(
                self._evict_device_heap,
                (self.tree_core.eviction_strategy.get_priority(lv.parent), lv.parent),
            )
        self._evict_device_last_node = None
        while tracker[ct] < self._evict_device_request_cnt and self._evict_device_heap:
            _, x = heapq.heappop(self._evict_device_heap)
            if x not in self.tree_core.evictable_device_leaves:
                continue
            self._evict_device_last_node = x
            return x.id
        return None

    def _evict_device_end(self) -> None:
        self._evict_device_heap = []
        self._evict_device_last_node = None

    def drive_host_eviction(
        self,
        num_tokens: int,
        tracker: dict[ComponentType, int],
        device_frees: dict[ComponentType, list[torch.Tensor]],
        host_frees: dict[ComponentType, list[torch.Tensor]],
    ) -> None:
        """Evict host leaves to free KV host pool space."""
        heap = [
            (self.tree_core.eviction_strategy.get_priority(n), n)
            for n in self.tree_core.evictable_host_leaves
        ]
        heapq.heapify(heap)
        ct = self.component_type
        while tracker[ct] < num_tokens and heap:
            _, x = heapq.heappop(heap)
            if x not in self.tree_core.evictable_host_leaves:
                continue
            self.tree_core._evict_host_leaf(x, tracker, device_frees, host_frees)
            if (
                x.parent is not None
                and x.parent in self.tree_core.evictable_host_leaves
            ):
                heapq.heappush(
                    heap,
                    (self.tree_core.eviction_strategy.get_priority(x.parent), x.parent),
                )

    def acquire_component_lock(
        self,
        node: UnifiedTreeNode,
        result: IncLockRefResult,
        lock_host: bool = False,
    ) -> IncLockRefResult:
        ct = self.component_type

        # Only the last host node needs to be protected.
        if lock_host:
            cd = node.component_data[ct]
            # write_back mode: the anchor may be device-only (no host_value); pin it anyway.
            if cd.host_value is None and not self.tree_core.is_write_back:
                return result
            cd.host_lock_ref += 1
            self.tree_core._update_evictable_leaf_sets(node)
            return result

        root = self.tree_core.root_node
        cur = node

        # Skip the bottom evicted segment
        while cur is not root and cur.component_data[ct].value is None:
            result.skip_lock_node_ids.setdefault(ct, set()).add(cur.id)
            cur = cur.parent

        # Lock the device-on segment up to root
        delta = 0
        while cur is not root:
            cd = cur.component_data[ct]
            assert (
                cd.value is not None
            ), f"FULL invariant broken: evicted ancestor {cur.id} above device-on segment"
            if cd.lock_ref == 0:
                key_len = len(cd.value)
                self.tree_core.component_evictable_size_[ct] -= key_len
                self.tree_core.component_protected_size_[ct] += key_len
                delta += key_len
            cd.lock_ref += 1
            self.tree_core.evictable_device_leaves.discard(cur)
            cur = cur.parent
        result.delta = delta
        return result

    def release_component_lock(
        self,
        node: UnifiedTreeNode,
        params: Optional[DecLockRefParams],
        lock_host: bool = False,
    ) -> None:
        ct = self.component_type
        if lock_host:
            cd = node.component_data[ct]
            if cd.host_lock_ref == 0:
                return
            # Mirror of `acquire`. write_back uses a pure counter.
            if cd.host_value is None and not self.tree_core.is_write_back:
                return
            cd.host_lock_ref -= 1
            self.tree_core._update_evictable_leaf_sets(node)
            return

        root = self.tree_core.root_node
        skip_lock_node_ids = params.skip_lock_node_ids.get(ct, ()) if params else ()
        cur = node
        while cur != root:
            if cur.id in skip_lock_node_ids:
                cur = cur.parent
                continue
            cd = cur.component_data[ct]
            assert cd.value is not None
            assert cd.lock_ref > 0

            if cd.lock_ref == 1:
                key_len = len(cd.value)
                self.tree_core.component_evictable_size_[ct] += key_len
                self.tree_core.component_protected_size_[ct] -= key_len
            cd.lock_ref -= 1
            if cd.lock_ref == 0:
                self.tree_core._update_evictable_leaf_sets(cur)
            cur = cur.parent

    # ---- HiCache Hooks ----

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
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            # Full KV backup is handled by the main flow
            # (cache_controller.write on host_value directly).
            # No extra PoolTransfer needed.
            return None

        if phase == CacheTransferPhase.LOAD_BACK:
            # `node` is best_match_node. FULL device evict only from leaves,
            # so once we hit a device-on node, everything above is also device-on
            backed_up: list[torch.Tensor] = []
            nodes: list = []
            cur = node
            while cur.evicted:
                cd = cur.component_data[ct]
                assert cd.host_value is not None
                backed_up.append(cd.host_value)
                nodes.append(cur)
                cur = cur.parent
            backed_up.reverse()
            nodes.reverse()
            return [
                PoolTransfer(
                    name=PoolName.KV,
                    host_indices=(
                        torch.cat(backed_up)
                        if backed_up
                        else torch.empty((0,), dtype=torch.int64, device="cpu")
                    ),
                    device_indices=None,
                    nodes_to_load=[n.id for n in nodes],
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
        ct = self.component_type

        if phase == CacheTransferPhase.BACKUP_HOST:
            if transfers and transfers[0].host_indices is not None:
                node.component_data[ct].host_value = transfers[0].host_indices.clone()

        elif phase == CacheTransferPhase.LOAD_BACK:
            if not transfers or transfers[0].device_indices is None:
                self.tree_core._update_evictable_leaf_sets(node)
                return

            xfer = transfers[0]
            device_indices = xfer.device_indices
            offset = 0
            for nid in xfer.nodes_to_load or []:
                n = self.tree_core.node_by_id(nid)
                cd = n.component_data[ct]
                n_len = len(cd.host_value)
                cd.value = device_indices[offset : offset + n_len].clone()
                offset += n_len
                # Full uses leaf sets, not LRU
                self.tree_core.component_evictable_size_[ct] += n_len
                self.tree_core._update_evictable_leaf_sets(n)

            self.tree_core._update_evictable_leaf_sets(node)

    def free_host_values(self, host_values: list[torch.Tensor]) -> None:
        if self._full_kv_pool_host is None:
            return
        for host_value in host_values:
            self._full_kv_pool_host.free(host_value)

    def apply_component_action(self, action: ComponentAction) -> None:
        if isinstance(action, FreeComponentDeviceSlot):
            alloc = self.cache.token_to_kv_pool_allocator
            for indices in action.indices:
                if self.cache.is_swa_enabled:
                    alloc.full_attn_allocator.free(indices)
                else:
                    alloc.free(indices)
            return
        raise AssertionError(
            f"FullComponent: unhandled ComponentAction {type(action).__name__}"
        )
