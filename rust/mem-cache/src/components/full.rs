//! FULL attention component driver: overrides the methods FULL customizes and inherits
//! the rest from the `TreeComponent` defaults.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use tch::{Kind, Tensor};

use crate::components::TreeComponent;
use crate::components::{ComponentType, FULL};
use crate::node::ChildKeyType;
use crate::node::Node;
use crate::node::{NodeId, NodeIdx_, ValueSlotIdx};
use crate::unified_lru_list::PriorityKey;
use crate::unified_tree_core::{
    CacheAction, CacheTransferPhase, DecLockRefParams, EvictLayer, IncLockRefResult, InsertResult,
    MatchPrefixParams, MatchResult, PoolName, PoolTransfer, PoolTransferResult, UnifiedTreeCore,
};

/// FULL attention component driver; owns the FULL device/host value slots.
pub struct FullComponent;

impl FullComponent {
    /// The component's device value slot.
    pub const DEVICE: ValueSlotIdx = ValueSlotIdx::device(FULL);
    /// The component's host value slot.
    pub const HOST: ValueSlotIdx = ValueSlotIdx::host(FULL);
}

impl<K: ChildKeyType> TreeComponent<K> for FullComponent {
    fn component_type(&self) -> ComponentType {
        todo!()
    }

    fn create_match_validator(
        &self,
        _tree_core: &UnifiedTreeCore<K>,
        match_device_only: bool,
    ) -> Box<dyn FnMut(&UnifiedTreeCore<K>, NodeIdx_) -> bool> {
        // def create_match_validator(
        //         self, match_device_only: bool = False
        //     ) -> Callable[[UnifiedTreeNode], bool]:
        //         if match_device_only:
        //             return (
        //                 lambda node: node.component_data[self.component_type].value is not None
        //             )
        //
        //         # HiCache: evicted + backuped nodes are valid match boundaries.
        //         return lambda node: (
        //             node.component_data[self.component_type].value is not None or node.backuped
        //         )
        todo!()
    }

    fn finalize_match_result_in_tree_core(
        &self,
        tree_core: &UnifiedTreeCore<K>,
        mut result: MatchResult,
        params: &MatchPrefixParams<'_, K>,
        value_chunks: &[Tensor],
        best_value_len: usize,
    ) -> MatchResult {
        // def finalize_match_result_in_tree_core(
        //         self,
        //         result: MatchResult,
        //         params: MatchPrefixParams,
        //         value_chunks: list[torch.Tensor],
        //         best_value_len: int,
        //     ) -> MatchResult:
        //         # Compute Full KV host hit length: walk from last_host_node up to
        //         # last_device_node, summing host_value lengths of evicted nodes.
        //         ct = self.component_type
        //         kv_host_hit = 0
        //         node = result.best_match_node
        //         root_node = self.tree_core.root_node
        //         while node is not result.last_device_node and node is not root_node:
        //             full_host = node.component_data[ct].host_value
        //             if full_host is not None:
        //                 kv_host_hit += len(full_host)
        //             node = node.parent
        //         if kv_host_hit > 0:
        //             return result._replace(
        //                 host_hit_length=max(result.host_hit_length, kv_host_hit)
        //             )
        //         return result
        todo!()
    }

    fn redistribute_on_node_split(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        new_parent_id: NodeIdx_,
        child_id: NodeIdx_,
    ) {
        // def redistribute_on_node_split(
        //         self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
        //     ):
        //         ct = self.component_type
        //         new_parent.component_data[ct].lock_ref = child.component_data[ct].lock_ref
        //         child_cd = child.component_data[ct]
        //         split_len = len(new_parent.key)
        //         if child_cd.value is not None:
        //             new_parent.component_data[ct].value = child_cd.value[:split_len].clone()
        //             child_cd.value = child_cd.value[split_len:].clone()
        //         if child_cd.host_value is not None:
        //             new_parent.component_data[ct].host_value = child_cd.host_value[
        //                 :split_len
        //             ].clone()
        //             child_cd.host_value = child_cd.host_value[split_len:].clone()
        todo!()
    }

    fn evict_component(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        target: EvictLayer,
    ) -> (usize, usize) {
        // def evict_component(
        //         self,
        //         node: UnifiedTreeNode,
        //         device_frees: dict[ComponentType, list[torch.Tensor]],
        //         host_frees: dict[ComponentType, list[torch.Tensor]],
        //         target: EvictLayer = EvictLayer.DEVICE,
        //     ) -> tuple[int, int]:
        //         cd = node.component_data[self.component_type]
        //         freed = 0
        //         host_freed = 0
        //
        //         # Device layer
        //         if EvictLayer.DEVICE in target and cd.value is not None:
        //             device_frees[self.component_type].append(cd.value)
        //             freed = len(cd.value)
        //             self.tree_core.component_evictable_size_[self.component_type] -= freed
        //             # NOTE: cd.value = None is deferred to _cascade_evict (Full as trigger)
        //             # because SWA's free_swa still needs to read Full.value.
        //             # cd.value = None
        //
        //         # Host layer
        //         if EvictLayer.HOST in target and cd.host_value is not None:
        //             host_freed = len(cd.host_value)
        //             host_frees[self.component_type].append(cd.host_value)
        //             cd.host_value = None
        //         return freed, host_freed
        todo!()
    }

    fn eviction_priority(&self, is_leaf: bool) -> i64 {
        // def eviction_priority(self, is_leaf: bool) -> int:
        //         return 0 if is_leaf else 2
        todo!()
    }

    fn evict_device_start(&self, tree_core: &mut UnifiedTreeCore<K>, request_cnt: usize) {
        // def _evict_device_start(self, request_cnt: int) -> None:
        //         self._evict_device_request_cnt = request_cnt
        //         self._evict_device_last_node = None
        //         self._evict_device_heap = [
        //             (self.tree_core.eviction_strategy.get_priority(n), n)
        //             for n in self.tree_core.evictable_device_leaves
        //         ]
        //         heapq.heapify(self._evict_device_heap)
        todo!()
    }

    fn evict_device_next_node(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        tracker: &mut HashMap<ComponentType, usize>,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) -> Option<NodeIdx_> {
        // def _evict_device_next_node(
        //         self,
        //         tracker: dict[ComponentType, int],
        //         device_frees: dict[ComponentType, list[torch.Tensor]],
        //         host_frees: dict[ComponentType, list[torch.Tensor]],
        //     ) -> Optional[NodeId]:
        //         ct = self.component_type
        //         lv = self._evict_device_last_node
        //         if (
        //             lv is not None
        //             and lv.parent is not None
        //             and lv.parent in self.tree_core.evictable_device_leaves
        //         ):
        //             heapq.heappush(
        //                 self._evict_device_heap,
        //                 (self.tree_core.eviction_strategy.get_priority(lv.parent), lv.parent),
        //             )
        //         self._evict_device_last_node = None
        //         while tracker[ct] < self._evict_device_request_cnt and self._evict_device_heap:
        //             _, x = heapq.heappop(self._evict_device_heap)
        //             if x not in self.tree_core.evictable_device_leaves:
        //                 continue
        //             self._evict_device_last_node = x
        //             return x.id
        //         return None
        todo!()
    }

    fn evict_device_end(&self, tree_core: &mut UnifiedTreeCore<K>) {
        // def _evict_device_end(self) -> None:
        //         self._evict_device_heap = []
        //         self._evict_device_last_node = None
        todo!()
    }

    /// Evict host leaves to free KV host pool space.
    fn drive_host_eviction(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        num_tokens: usize,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        // def drive_host_eviction(
        //         self,
        //         num_tokens: int,
        //         tracker: dict[ComponentType, int],
        //         device_frees: dict[ComponentType, list[torch.Tensor]],
        //         host_frees: dict[ComponentType, list[torch.Tensor]],
        //     ) -> None:
        //         """Evict host leaves to free KV host pool space."""
        //         heap = [
        //             (self.tree_core.eviction_strategy.get_priority(n), n)
        //             for n in self.tree_core.evictable_host_leaves
        //         ]
        //         heapq.heapify(heap)
        //         ct = self.component_type
        //         while tracker[ct] < num_tokens and heap:
        //             _, x = heapq.heappop(heap)
        //             if x not in self.tree_core.evictable_host_leaves:
        //                 continue
        //             self.tree_core._evict_host_leaf(x, tracker, device_frees, host_frees)
        //             if (
        //                 x.parent is not None
        //                 and x.parent in self.tree_core.evictable_host_leaves
        //             ):
        //                 heapq.heappush(
        //                     heap,
        //                     (self.tree_core.eviction_strategy.get_priority(x.parent), x.parent),
        //                 )
        todo!()
    }

    fn acquire_component_lock(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        mut result: IncLockRefResult,
        lock_host: bool,
    ) -> IncLockRefResult {
        // def acquire_component_lock(
        //         self,
        //         node: UnifiedTreeNode,
        //         result: IncLockRefResult,
        //         lock_host: bool = False,
        //     ) -> IncLockRefResult:
        //         ct = self.component_type
        //
        //         # Only the last host node needs to be protected.
        //         if lock_host:
        //             cd = node.component_data[ct]
        //             # write_back mode: the anchor may be device-only (no host_value); pin it anyway.
        //             if cd.host_value is None and not self.tree_core.is_write_back:
        //                 return result
        //             cd.host_lock_ref += 1
        //             self.tree_core._update_evictable_leaf_sets(node)
        //             return result
        //
        //         root = self.tree_core.root_node
        //         cur = node
        //
        //         # Skip the bottom evicted segment
        //         while cur is not root and cur.component_data[ct].value is None:
        //             result.skip_lock_node_ids.setdefault(ct, set()).add(cur.id)
        //             cur = cur.parent
        //
        //         # Lock the device-on segment up to root
        //         delta = 0
        //         while cur is not root:
        //             cd = cur.component_data[ct]
        //             assert (
        //                 cd.value is not None
        //             ), f"FULL invariant broken: evicted ancestor {cur.id} above device-on segment"
        //             if cd.lock_ref == 0:
        //                 key_len = len(cd.value)
        //                 self.tree_core.component_evictable_size_[ct] -= key_len
        //                 self.tree_core.component_protected_size_[ct] += key_len
        //                 delta += key_len
        //             cd.lock_ref += 1
        //             self.tree_core.evictable_device_leaves.discard(cur)
        //             cur = cur.parent
        //         result.delta = delta
        //         return result
        todo!()
    }

    fn release_component_lock(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        params: Option<&DecLockRefParams>,
        lock_host: bool,
    ) {
        // def release_component_lock(
        //         self,
        //         node: UnifiedTreeNode,
        //         params: Optional[DecLockRefParams],
        //         lock_host: bool = False,
        //     ) -> None:
        //         ct = self.component_type
        //         if lock_host:
        //             cd = node.component_data[ct]
        //             if cd.host_lock_ref == 0:
        //                 return
        //             # Mirror of `acquire`. write_back uses a pure counter.
        //             if cd.host_value is None and not self.tree_core.is_write_back:
        //                 return
        //             cd.host_lock_ref -= 1
        //             self.tree_core._update_evictable_leaf_sets(node)
        //             return
        //
        //         root = self.tree_core.root_node
        //         skip_lock_node_ids = params.skip_lock_node_ids.get(ct, ()) if params else ()
        //         cur = node
        //         while cur != root:
        //             if cur.id in skip_lock_node_ids:
        //                 cur = cur.parent
        //                 continue
        //             cd = cur.component_data[ct]
        //             assert cd.value is not None
        //             assert cd.lock_ref > 0
        //
        //             if cd.lock_ref == 1:
        //                 key_len = len(cd.value)
        //                 self.tree_core.component_evictable_size_[ct] += key_len
        //                 self.tree_core.component_protected_size_[ct] -= key_len
        //             cd.lock_ref -= 1
        //             if cd.lock_ref == 0:
        //                 self.tree_core._update_evictable_leaf_sets(cur)
        //             cur = cur.parent
        todo!()
    }

    fn build_hicache_transfers(
        &self,
        tree_core: &UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        phase: CacheTransferPhase,
        _mamba_pool_idx: Option<Tensor>,
        _host_indices: Option<Tensor>,
        _token_ids: Option<&[i64]>,
        _prefetch_tokens: usize,
        _last_hash: Option<&str>,
    ) -> Option<Vec<PoolTransfer>> {
        // def build_hicache_transfers(
        //         self,
        //         node: UnifiedTreeNode,
        //         phase: CacheTransferPhase,
        //         *,
        //         mamba_pool_idx: Optional[torch.Tensor] = None,
        //         host_indices: Optional[torch.Tensor] = None,
        //         token_ids: Optional[Sequence[int]] = None,
        //         prefetch_tokens: int = 0,
        //         last_hash: Optional[str] = None,
        //     ) -> Optional[list[PoolTransfer]]:
        //         ct = self.component_type
        //
        //         if phase == CacheTransferPhase.BACKUP_HOST:
        //             # Full KV backup is handled by the main flow
        //             # (cache_controller.write on host_value directly).
        //             # No extra PoolTransfer needed.
        //             return None
        //
        //         if phase == CacheTransferPhase.LOAD_BACK:
        //             # `node` is best_match_node. FULL device evict only from leaves,
        //             # so once we hit a device-on node, everything above is also device-on
        //             backed_up: list[torch.Tensor] = []
        //             nodes: list = []
        //             cur = node
        //             while cur.evicted:
        //                 cd = cur.component_data[ct]
        //                 assert cd.host_value is not None
        //                 backed_up.append(cd.host_value)
        //                 nodes.append(cur)
        //                 cur = cur.parent
        //             backed_up.reverse()
        //             nodes.reverse()
        //             return [
        //                 PoolTransfer(
        //                     name=PoolName.KV,
        //                     host_indices=(
        //                         torch.cat(backed_up)
        //                         if backed_up
        //                         else torch.empty((0,), dtype=torch.int64, device="cpu")
        //                     ),
        //                     device_indices=None,
        //                     nodes_to_load=[n.id for n in nodes],
        //                 )
        //             ]
        //
        //         return None
        todo!()
    }

    fn commit_hicache_transfer(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        phase: CacheTransferPhase,
        transfers: Vec<PoolTransfer>,
        cache_actions: &mut Vec<CacheAction>,
        insert_result: Option<&mut InsertResult>,
        pool_storage_result: Option<&PoolTransferResult>,
    ) {
        // def commit_hicache_transfer(
        //         self,
        //         node: UnifiedTreeNode,
        //         phase: CacheTransferPhase,
        //         transfers: list[PoolTransfer] = (),
        //         *,
        //         cache_actions: list[CacheAction | ComponentAction],
        //         insert_result: Optional[InsertResult] = None,
        //         pool_storage_result: Optional[PoolTransferResult] = None,
        //     ) -> None:
        //         ct = self.component_type
        //
        //         if phase == CacheTransferPhase.BACKUP_HOST:
        //             if transfers and transfers[0].host_indices is not None:
        //                 node.component_data[ct].host_value = transfers[0].host_indices.clone()
        //
        //         elif phase == CacheTransferPhase.LOAD_BACK:
        //             if not transfers or transfers[0].device_indices is None:
        //                 self.tree_core._update_evictable_leaf_sets(node)
        //                 return
        //
        //             xfer = transfers[0]
        //             device_indices = xfer.device_indices
        //             offset = 0
        //             for nid in xfer.nodes_to_load or []:
        //                 n = self.tree_core.node_by_id(nid)
        //                 cd = n.component_data[ct]
        //                 n_len = len(cd.host_value)
        //                 cd.value = device_indices[offset : offset + n_len].clone()
        //                 offset += n_len
        //                 # Full uses leaf sets, not LRU
        //                 self.tree_core.component_evictable_size_[ct] += n_len
        //                 self.tree_core._update_evictable_leaf_sets(n)
        //
        //             self.tree_core._update_evictable_leaf_sets(node)
        todo!()
    }
}
