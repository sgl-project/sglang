//! FULL attention component driver: overrides the methods FULL customizes and inherits
//! the rest from the `TreeComponent` defaults.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

use tch::{Kind, Tensor};

use crate::components::TreeComponent;
use crate::components::{ComponentType, FULL};
use crate::node::ChildKeyType;
use crate::node::Node;
use crate::node::{NodeId, NodeIdx_, TreeCoreRuntimeError, ValueSlotIdx};
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
        FULL
    }

    fn create_match_validator(
        &self,
        _tree_core: &UnifiedTreeCore<K>,
        match_device_only: bool,
    ) -> Box<dyn FnMut(&UnifiedTreeCore<K>, NodeIdx_) -> bool> {
        // Device value present -> always a boundary; otherwise a backuped (host-resident)
        // node also matches, unless the match is restricted to device.
        Box::new(move |tree_core: &UnifiedTreeCore<K>, node_id: NodeIdx_| {
            let node = tree_core.arena.node(node_id);
            node.has_device_value(FULL) || (!match_device_only && node.has_host_value(FULL))
        })
    }

    fn finalize_match_result_in_tree_core(
        &self,
        tree_core: &UnifiedTreeCore<K>,
        mut result: MatchResult,
        params: &MatchPrefixParams<'_, K>,
        value_chunks: &[Tensor],
        best_value_len: usize,
    ) -> MatchResult {
        // Compute Full KV host hit length: walk from last_host_node up to
        // last_device_node, summing host_value lengths of evicted nodes.
        let mut kv_host_hit = 0;
        let mut node_idx = tree_core.arena.resolve(result.best_match_node_id);
        let last_device_idx = tree_core.arena.resolve(result.last_device_node_id);
        while node_idx != last_device_idx {
            let node = tree_core.arena.node(node_idx);
            let parent = node.try_parent().unwrap_or_else(|| {
                panic!(
                    "finalize walk from best_match_node {} hit root {} before \
                     last_device_node {}",
                    result.best_match_node_id, node.id, result.last_device_node_id
                )
            });
            kv_host_hit += node.host_value_len(FULL);
            node_idx = parent;
        }
        if kv_host_hit > 0 {
            result.host_hit_length = result.host_hit_length.max(kv_host_hit);
        }
        result
    }

    fn redistribute_on_node_split(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        new_parent_id: NodeIdx_,
        child_id: NodeIdx_,
    ) {
        let (new_parent, child) = tree_core.arena.node_pair_mut(new_parent_id, child_id);
        let split_len = new_parent.key.atom_len() as i64;
        new_parent.copy_device_lock_ref(FULL, child);
        if child.has_device_value(FULL) {
            Node::redistribute_child_device_value(new_parent, child, FULL, split_len);
        }
        if child.has_host_value(FULL) {
            Node::redistribute_child_host_value(new_parent, child, FULL, split_len);
        }
    }

    fn evict_component(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        target: EvictLayer,
    ) -> (usize, usize) {
        let node = tree_core.arena.node_mut(node_id);
        let mut freed = 0;
        let mut host_freed = 0;
        if target.contains(EvictLayer::Device) && node.has_device_value(FULL) {
            let value = node.device_value(FULL);
            freed = node.device_value_len(FULL);
            device_frees
                .entry(FULL)
                .or_default()
                .push(value.shallow_clone());
            // NOTE: cd.value = None is deferred to _cascade_evict (Full as trigger)
            // because SWA's free_swa still needs to read Full.value.
        }
        if target.contains(EvictLayer::Host) && node.has_host_value(FULL) {
            host_freed = node.host_value_len(FULL);
            host_frees
                .entry(FULL)
                .or_default()
                .push(node.take_host_value(FULL));
        }
        if freed > 0 {
            tree_core.dec_evictable_size(FULL, freed);
        }
        (freed, host_freed)
    }

    fn eviction_priority(&self, is_leaf: bool) -> i64 {
        if is_leaf { 0 } else { 2 }
    }

    fn evict_device_start(&self, tree_core: &mut UnifiedTreeCore<K>, request_cnt: usize) {
        tree_core.set_evict_device_start(FULL, request_cnt);
        tree_core.full_evict_device_heap.clear();
        let arena = &tree_core.arena;
        let strategy = &tree_core.eviction_strategy;
        tree_core.full_evict_device_heap.extend(
            tree_core
                .evictable_device_leaves
                .iter()
                .map(|id| Reverse((strategy.get_priority(arena.node(id)), id))),
        );
    }

    fn evict_device_next_node(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        tracker: &mut HashMap<ComponentType, usize>,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) -> Option<NodeIdx_> {
        let ct = FULL;
        assert!(
            tree_core.component_state(FULL).is_evict_device_ongoing,
            "Full device eviction not started"
        );
        // Re-admit the previously returned leaf's parent once it became a
        // D-leaf; the parent id was captured at return time because the leaf
        // itself may have been freed by the eviction in between.
        if let Some(last_node_parent) = tree_core.component_state(FULL).evict_device_cursor
            && tree_core.evictable_device_leaves.contains(last_node_parent)
        {
            let key = tree_core
                .eviction_strategy
                .get_priority(tree_core.arena.node(last_node_parent));
            tree_core
                .full_evict_device_heap
                .push(Reverse((key, last_node_parent)));
        }
        tree_core.component_state_mut(FULL).evict_device_cursor = None;
        // The budget only advances between calls (the driver's evictions fill
        // the tracker), so it gates the walk once up front.
        if tracker[&ct] >= tree_core.component_state(FULL).evict_device_request_cnt {
            return None;
        }
        while let Some(Reverse((_, x))) = tree_core.full_evict_device_heap.pop() {
            if !tree_core.evictable_device_leaves.contains(x) {
                continue;
            }
            let last_node_parent = tree_core.arena.node(x).try_parent();
            tree_core.component_state_mut(FULL).evict_device_cursor = last_node_parent;
            return Some(x);
        }
        None
    }

    fn evict_device_end(&self, tree_core: &mut UnifiedTreeCore<K>) {
        tree_core.set_evict_device_end(FULL);
        tree_core.full_evict_device_heap.clear();
    }

    fn reclaim_coexisting_host_values(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        num_tokens: usize,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        for spare_imminent_demotes in [true, false] {
            if tracker[&FULL] >= num_tokens {
                break;
            }
            let candidates: Vec<NodeIdx_> = tree_core.full_coexisting_host_nodes.iter().collect();
            for node_id in candidates {
                if tracker[&FULL] >= num_tokens {
                    break;
                }
                let node = tree_core.arena.node(node_id);
                if !node.has_device_value(FULL) || !node.has_host_value(FULL) {
                    tree_core.full_coexisting_host_nodes.discard(node_id);
                    continue;
                }
                if spare_imminent_demotes && tree_core.evictable_device_leaves.contains(node_id) {
                    continue;
                }
                if !tree_core.can_reclaim_coexisting_host_value_(node_id, FULL) {
                    continue;
                }
                tree_core.release_coexisting_host_value_(
                    node_id,
                    FULL,
                    tracker,
                    device_frees,
                    host_frees,
                );
                tree_core.full_coexisting_host_nodes.discard(node_id);
            }
        }
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
        let ct = FULL;
        let arena = &tree_core.arena;
        let strategy = &tree_core.eviction_strategy;
        let mut heap: BinaryHeap<Reverse<(PriorityKey, NodeIdx_)>> = tree_core
            .evictable_host_leaves
            .iter()
            .map(|id| Reverse((strategy.get_priority(arena.node(id)), id)))
            .collect();
        while tracker[&ct] < num_tokens {
            let Some(Reverse((_, x))) = heap.pop() else {
                break;
            };
            if !tree_core.evictable_host_leaves.contains(x) {
                continue;
            }
            // The parent id is captured before the eviction frees the leaf.
            let parent = tree_core.arena.node(x).try_parent();
            tree_core.evict_host_leaf_(x, tracker, device_frees, host_frees);
            if let Some(parent) = parent
                && tree_core.evictable_host_leaves.contains(parent)
            {
                let key = tree_core
                    .eviction_strategy
                    .get_priority(tree_core.arena.node(parent));
                heap.push(Reverse((key, parent)));
            }
        }
    }

    fn acquire_component_lock(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        mut result: IncLockRefResult,
        lock_host: bool,
    ) -> IncLockRefResult {
        let ct = FULL;

        // Only the last host node needs to be protected.
        if lock_host {
            let node = tree_core.arena.node_mut(node_id);
            // write_back mode: the anchor may be device-only (no host_value); pin it anyway.
            if !node.has_host_value(FULL) && !tree_core.is_write_back {
                return result;
            }
            node.inc_host_lock_ref(FULL);
            tree_core.update_evictable_leaf_sets_(node_id);
            return result;
        }

        // Skip the bottom evicted segment, recording it for the matching release.
        let on_boundary = |node: &Node<K>| node.is_root() || node.has_device_value(FULL);
        let mut cur = node_id;
        let mut node = tree_core.arena.node(cur);
        if !on_boundary(node) {
            let skip_lock_node_ids = result.skip_lock_node_ids.entry(ct).or_default();
            loop {
                skip_lock_node_ids.insert(node.id);
                cur = node.parent();
                node = tree_core.arena.node(cur);
                if on_boundary(node) {
                    break;
                }
            }
        }

        // Lock the device-on segment up to the root.
        let mut delta = 0;
        loop {
            let node = tree_core.arena.node_mut(cur);
            if node.is_root() {
                break;
            }
            assert!(
                node.has_device_value(FULL),
                "FULL invariant broken: evicted ancestor {cur} above device-on segment"
            );
            let parent = node.parent();
            let newly_locked_len = if node.device_lock_ref(FULL) == 0 {
                Some(node.device_value_len(FULL))
            } else {
                None
            };
            node.inc_device_lock_ref(FULL);
            if let Some(key_len) = newly_locked_len {
                tree_core.dec_evictable_size(FULL, key_len);
                tree_core.inc_protected_size(FULL, key_len);
                delta += key_len;
            }
            tree_core.evictable_device_leaves.discard(cur);
            cur = parent;
        }
        result.delta = Some(delta);
        result
    }

    fn release_component_lock(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        params: Option<&DecLockRefParams>,
        lock_host: bool,
    ) {
        let ct = FULL;

        if lock_host {
            let node = tree_core.arena.node_mut(node_id);
            if node.host_lock_ref(FULL) == 0 {
                return;
            }
            // Mirror of `acquire`. write_back uses a pure counter.
            if !node.has_host_value(FULL) && !tree_core.is_write_back {
                return;
            }
            node.dec_host_lock_ref(FULL);
            tree_core.update_evictable_leaf_sets_(node_id);
            return;
        }

        let empty = HashSet::new();
        let skip_lock_node_ids = params
            .and_then(|p| p.skip_lock_node_ids.get(&ct))
            .unwrap_or(&empty);
        let mut cur = node_id;
        loop {
            let node = tree_core.arena.node_mut(cur);
            if node.is_root() {
                break;
            }
            let parent = node.parent();
            if skip_lock_node_ids.contains(&node.id) {
                cur = parent;
                continue;
            }
            assert!(
                node.has_device_value(FULL),
                "release_component_lock: node {cur} has no FULL device value"
            );
            let old_lock_ref = node.device_lock_ref(FULL);
            assert!(
                old_lock_ref > 0,
                "release_component_lock: node {cur} is not locked"
            );
            let newly_unlocked_len = if old_lock_ref == 1 {
                Some(node.device_value_len(FULL))
            } else {
                None
            };
            node.dec_device_lock_ref(FULL);
            if let Some(key_len) = newly_unlocked_len {
                tree_core.dec_protected_size(FULL, key_len);
                tree_core.inc_evictable_size(FULL, key_len);
                tree_core.update_evictable_leaf_sets_(cur);
            }
            cur = parent;
        }
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
    ) -> Result<Option<Vec<PoolTransfer>>, TreeCoreRuntimeError> {
        Ok(match phase {
            // Full KV backup is handled by the main flow
            // (cache_controller.write on host_value directly).
            // No extra PoolTransfer needed.
            CacheTransferPhase::BackupHost => None,
            CacheTransferPhase::LoadBack => {
                // `node` is best_match_node. FULL device evict only from leaves,
                // so once we hit a device-on node, everything above is also device-on.
                let mut backed_up: Vec<Tensor> = Vec::new();
                let mut nodes_to_load: Vec<NodeId> = Vec::new();
                let mut cur = tree_core.arena.node(node_id);
                while cur.evicted() {
                    backed_up.push(cur.host_value(FULL).shallow_clone());
                    nodes_to_load.push(cur.id);
                    cur = tree_core.arena.node(cur.parent());
                }
                backed_up.reverse();
                nodes_to_load.reverse();
                let host_indices = if backed_up.is_empty() {
                    Tensor::empty([0], (Kind::Int64, tch::Device::Cpu))
                } else {
                    Tensor::cat(&backed_up, 0)
                };
                Some(vec![PoolTransfer {
                    name: PoolName::Kv,
                    host_indices: Some(host_indices),
                    nodes_to_load: Some(nodes_to_load),
                    ..Default::default()
                }])
            }
            CacheTransferPhase::BackupStorage | CacheTransferPhase::Prefetch => None,
        })
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
        match phase {
            CacheTransferPhase::BackupHost => {
                if let Some(transfer) = transfers.first()
                    && let Some(host_indices) = &transfer.host_indices
                {
                    tree_core
                        .arena
                        .set_host_value(node_id, FULL, host_indices.copy());
                }
            }
            CacheTransferPhase::LoadBack => {
                if let Some(transfer) = transfers.first()
                    && let Some(device_indices) = &transfer.device_indices
                {
                    let mut offset = 0i64;
                    for &loaded_id in transfer.nodes_to_load.iter().flatten() {
                        let loaded_idx = tree_core.arena.resolve(loaded_id);
                        let loaded = tree_core.arena.node_mut(loaded_idx);
                        let n_len = loaded.host_value_len(FULL) as i64;
                        loaded
                            .set_device_value(FULL, device_indices.narrow(0, offset, n_len).copy());
                        offset += n_len;
                        // Full uses leaf sets, not LRU.
                        tree_core.inc_evictable_size(FULL, n_len as usize);
                        tree_core.update_evictable_leaf_sets_(loaded_idx);
                    }
                }
                tree_core.update_evictable_leaf_sets_(node_id);
            }
            // The Full component has no storage-phase commits.
            CacheTransferPhase::BackupStorage | CacheTransferPhase::Prefetch => {}
        }
    }
}

#[cfg(test)]
#[path = "../tests/components/full.rs"]
mod tests;
