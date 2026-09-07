//! Mamba (SSM state) component driver: overrides the methods Mamba customizes
//! and inherits the rest from the `TreeComponent` defaults.
//! Mamba data is per-leaf single-slot state; sizes count slots, not tokens.

use std::collections::HashMap;

use tch::Tensor;

use crate::components::TreeComponent;
use crate::components::{ComponentType, MAMBA};
use crate::node::ChildKeyType;
use crate::node::Node;
use crate::node::{NodeId, NodeIdx_, TreeCoreRuntimeError, ValueSlotIdx};
use crate::unified_tree_core::{
    CacheAction, CacheInitParams, CacheTransferPhase, DecLockRefParams, EvictLayer,
    IncLockRefResult, InsertParams, InsertResult, LRURefreshPhase, MatchPrefixParams, MatchResult,
    PoolHitPolicy, PoolName, PoolTransfer, PoolTransferResult, UnifiedTreeCore,
};

/// Mamba component driver; owns the Mamba device/host value slots.
pub struct MambaComponent {
    /// Joint chunk/tree-page alignment for the mamba branching seqlen.
    mamba_checkpoint_grid: usize,
    /// Per-root-path cap on cached Mamba states; None means unlimited.
    mamba_max_states_per_path: Option<usize>,
}

impl MambaComponent {
    /// The component's device value slot.
    pub const DEVICE: ValueSlotIdx = ValueSlotIdx::device(MAMBA);
    /// The component's host value slot.
    pub const HOST: ValueSlotIdx = ValueSlotIdx::host(MAMBA);
}

impl MambaComponent {
    /// Build the driver from the tree's init params.
    pub fn new(params: &CacheInitParams) -> Self {
        let mamba_cache_chunk_size = params
            .mamba_cache_chunk_size
            .expect("the Mamba component requires mamba_cache_chunk_size");
        MambaComponent {
            // A donated checkpoint must land on both the model's chunk grid and
            // a radix-node boundary. `params.page_size` is already widened by DCP.
            mamba_checkpoint_grid: least_common_multiple(mamba_cache_chunk_size, params.page_size),
            mamba_max_states_per_path: params.mamba_max_states_per_path,
        }
    }
}

fn least_common_multiple(lhs: usize, rhs: usize) -> usize {
    let mut a = lhs;
    let mut b = rhs;
    while b != 0 {
        (a, b) = (b, a % b);
    }
    lhs / a * rhs
}

impl MambaComponent {
    // Tier-selected mamba slot read for the lock paths; `host` picks the host slot.
    fn has_value<K: ChildKeyType>(node: &Node<K>, host: bool) -> bool {
        if host {
            node.has_host_value(MAMBA)
        } else {
            node.has_device_value(MAMBA)
        }
    }

    /// Defer the path-cap eviction so it runs after the insert's BackupKV.
    fn emit_excess_path_states_eviction_(
        &self,
        tail_node_id: NodeId,
        cache_actions: &mut Vec<CacheAction>,
    ) {
        if self.mamba_max_states_per_path.is_none() {
            return;
        }
        cache_actions.push(CacheAction::MambaEvictExcessPathStates { tail_node_id });
    }
}

impl<K: ChildKeyType> TreeComponent<K> for MambaComponent {
    fn component_type(&self) -> ComponentType {
        MAMBA
    }

    fn needs_incremental_backup(&self, tree_core: &UnifiedTreeCore<K>, node_id: NodeIdx_) -> bool {
        let node = tree_core.arena.node(node_id);
        node.has_device_value(MAMBA) && !node.has_host_value(MAMBA)
    }

    /// A match consumes only the best-match node's mamba state, so MATCH_END
    /// touches just that node; new-leaf states enter the LRU at insert commit,
    /// so WALKDOWN and INSERT_END are no-ops.
    fn refresh_lru(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        phase: LRURefreshPhase,
        node_id: NodeIdx_,
    ) {
        match phase {
            LRURefreshPhase::Walkdown => {}
            LRURefreshPhase::MatchEnd => {
                if tree_core.arena.has_device_value(node_id, MAMBA) {
                    tree_core.device_lru_list_mut(MAMBA).reset_node_mru(node_id);
                }
            }
            LRURefreshPhase::InsertEnd => {}
        }
    }

    /// A per-match predicate accepting nodes that hold mamba data.
    fn create_match_validator(
        &self,
        _tree_core: &UnifiedTreeCore<K>,
        match_device_only: bool,
    ) -> Box<dyn FnMut(&UnifiedTreeCore<K>, NodeIdx_) -> bool> {
        // HiCache: evicted + backuped (host_value present) is also a valid match.
        Box::new(move |tree_core: &UnifiedTreeCore<K>, node_id: NodeIdx_| {
            let node = tree_core.arena.node(node_id);
            node.has_device_value(MAMBA) || (!match_device_only && node.has_host_value(MAMBA))
        })
    }

    /// The mamba branching seqlen and the host-only hit bump.
    fn finalize_match_result_in_tree_core(
        &self,
        tree_core: &UnifiedTreeCore<K>,
        mut result: MatchResult,
        _params: &MatchPrefixParams<'_, K>,
        _value_chunks: &[Tensor],
        _best_value_len: usize,
    ) -> MatchResult {
        let mamba_boundary_len = result.device_indices.size()[0] as usize + result.host_hit_length;

        // Full KV may extend beyond the latest reusable Mamba state. The branching
        // point is the last checkpoint-grid-aligned position within the Full-KV hit
        // that lies beyond the current Mamba boundary.
        let aligned_seqlen =
            result.full_kv_hit_length / self.mamba_checkpoint_grid * self.mamba_checkpoint_grid;
        result.mamba_branching_seqlen =
            (aligned_seqlen > mamba_boundary_len).then_some(aligned_seqlen);

        // HiCache: if mamba was evicted from device but has host backup,
        // ensure mamba_host_hit_length >= 1 so load_back is triggered.
        let last_node = tree_core
            .arena
            .node(tree_core.arena.resolve(result.best_match_node_id));
        if !last_node.has_device_value(MAMBA) && last_node.has_host_value(MAMBA) {
            result.mamba_host_hit_length = result.mamba_host_hit_length.max(1);
        }
        result
    }

    /// Attach the donated mamba slot to the insert target leaf.
    fn commit_insert_component_data(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        is_new_leaf: bool,
        params: &InsertParams<'_, K>,
        result: &mut InsertResult,
        cache_actions: &mut Vec<CacheAction>,
    ) {
        let mamba_value = params
            .mamba_value
            .as_ref()
            .expect("mamba insert requires a donated mamba_value");
        let slot_len = mamba_value.size()[0] as usize;

        if is_new_leaf {
            tree_core
                .arena
                .set_device_value(node_id, MAMBA, mamba_value.shallow_clone());
            tree_core.device_lru_list_mut(MAMBA).insert_mru(node_id);
            tree_core.inc_evictable_size(MAMBA, slot_len);
            self.emit_excess_path_states_eviction_(tree_core.arena.node(node_id).id, cache_actions);
            return;
        }
        if !tree_core.arena.has_device_value(node_id, MAMBA) {
            // Tombstone refill: the node moves from the host LRU to the device LRU.
            tree_core
                .arena
                .set_device_value(node_id, MAMBA, mamba_value.shallow_clone());
            let host_lru = tree_core.host_lru_list_mut(MAMBA);
            if host_lru.in_list(Some(node_id)) {
                host_lru.remove_node(node_id);
            }
            tree_core.device_lru_list_mut(MAMBA).insert_mru(node_id);
            tree_core.inc_evictable_size(MAMBA, slot_len);
            let tick = tree_core.arena.get_and_bump_access_counter();
            tree_core.arena.node_mut(node_id).last_access_counter = tick;
            self.emit_excess_path_states_eviction_(tree_core.arena.node(node_id).id, cache_actions);
            return;
        }
        tree_core.device_lru_list_mut(MAMBA).reset_node_mru(node_id);
        let tick = tree_core.arena.get_and_bump_access_counter();
        tree_core.arena.node_mut(node_id).last_access_counter = tick;
        result.mamba_exist = true;
    }

    /// Mamba data stays on the original leaf; the new prefix node gets none.
    /// Evict shallow Mamba device checkpoints beyond the per-path cap on the
    /// tail's root path; Full KV, host backups, the tail, forks, locked nodes,
    /// and device leaves are preserved (a best-effort soft cap).
    fn evict_excess_path_states(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        tail_node_id: NodeIdx_,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        let Some(cap) = self.mamba_max_states_per_path else {
            return;
        };
        // Mamba-value holders on the root path, tail-first.
        let mut holders: Vec<NodeIdx_> = Vec::new();
        let mut cursor = Some(tail_node_id);
        while let Some(node_id) = cursor {
            let node = tree_core.arena.node(node_id);
            if node.is_root() {
                break;
            }
            if node.has_device_value(MAMBA) {
                holders.push(node_id);
            }
            cursor = node.parent;
        }
        let mut excess = holders.len().saturating_sub(cap);
        if excess == 0 {
            return;
        }
        // Cache-level apply: the counts are not reported, only the frees.
        let mut tracker: HashMap<ComponentType, usize> = HashMap::new();
        for &node_id in holders.iter().rev() {
            if excess == 0 || node_id == tail_node_id {
                break;
            }
            let node = tree_core.arena.node(node_id);
            if node.device_lock_ref(MAMBA) > 0 || node.children.len() != 1 {
                continue;
            }
            if tree_core.evictable_device_leaves.contains(node_id) {
                continue;
            }
            tree_core.evict_component_and_detach_lru_(
                node_id,
                MAMBA,
                device_frees,
                host_frees,
                EvictLayer::Device,
                Some(&mut tracker),
            );
            tree_core.cascade_evict_(
                node_id,
                MAMBA,
                &mut tracker,
                device_frees,
                host_frees,
                EvictLayer::Device,
            );
            excess -= 1;
        }
    }

    fn redistribute_on_node_split(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        new_parent_id: NodeIdx_,
        _child_id: NodeIdx_,
    ) {
        let new_parent = tree_core.arena.node_mut(new_parent_id);
        if new_parent.has_device_value(MAMBA) {
            let _ = new_parent.take_device_value(MAMBA);
        }
        new_parent.set_lock_ref_(ValueSlotIdx::device(MAMBA), 0);
        // HiCache: mamba host_value stays on child (mamba = leaf-only data).
        if new_parent.has_host_value(MAMBA) {
            let _ = new_parent.take_host_value(MAMBA);
        }
        new_parent.set_lock_ref_(ValueSlotIdx::host(MAMBA), 0);
    }

    /// Free the node's mamba slot on the targeted layer(s).
    fn evict_component(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        target: EvictLayer,
    ) -> (usize, usize) {
        let ct = MAMBA;
        let node = tree_core.arena.node_mut(node_id);
        let mut freed = 0;
        let mut host_freed = 0;

        // Device layer
        if target.contains(EvictLayer::Device) && node.has_device_value(MAMBA) {
            freed = node.device_value_len(MAMBA);
            device_frees
                .entry(ct)
                .or_default()
                .push(node.take_device_value(MAMBA));
            tree_core.dec_evictable_size(MAMBA, freed);
        }

        // Host layer
        let node = tree_core.arena.node_mut(node_id);
        if target.contains(EvictLayer::Host) && node.has_host_value(MAMBA) {
            host_freed = node.host_value_len(MAMBA);
            host_frees
                .entry(ct)
                .or_default()
                .push(node.take_host_value(MAMBA));
            let host_lru = tree_core.host_lru_list_mut(MAMBA);
            if host_lru.in_list(Some(node_id)) {
                host_lru.remove_node(node_id);
            }
        }

        // After device tombstone: if only host_value remains, insert into host LRU
        let node = tree_core.arena.node(node_id);
        if target == EvictLayer::Device
            && !node.has_device_value(MAMBA)
            && node.has_host_value(MAMBA)
        {
            let host_lru = tree_core.host_lru_list_mut(MAMBA);
            if !host_lru.in_list(Some(node_id)) {
                host_lru.insert_mru(node_id);
            }
        }

        (freed, host_freed)
    }

    /// Begin the device-eviction walk from this component's LRU cursor.
    fn evict_device_start(&self, tree_core: &mut UnifiedTreeCore<K>, request_cnt: usize) {
        tree_core.set_evict_device_start(MAMBA, request_cnt);
        let cursor = tree_core
            .device_lru_list(MAMBA)
            .get_lru_no_lock(&tree_core.arena);
        tree_core.component_state_mut(MAMBA).evict_device_cursor = cursor;
    }

    /// Advance one device-eviction step and return a leaf, if selected.
    ///
    /// An internal tombstone is one complete step so the caller can apply its
    /// pending frees and recheck allocator capacity before the next mutation.
    fn evict_device_next_node(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) -> Option<NodeIdx_> {
        let ct = MAMBA;
        assert!(
            tree_core.component_state(MAMBA).is_evict_device_ongoing,
            "Mamba device eviction not started"
        );
        let mut cursor = tree_core.component_state(MAMBA).evict_device_cursor;
        // The cursor is re-validated (reset to LRU head) if the previous
        // node's eviction removed it.
        if cursor.is_some_and(|c| !tree_core.device_lru_list(MAMBA).in_list(Some(c))) {
            cursor = tree_core
                .device_lru_list(MAMBA)
                .get_lru_no_lock(&tree_core.arena);
        }
        let next = loop {
            if tracker[&ct] >= tree_core.component_state(MAMBA).evict_device_request_cnt {
                break None;
            }
            let Some(x) = cursor else {
                break None;
            };
            if !tree_core.device_lru_list(MAMBA).in_list(Some(x)) {
                break None;
            }
            assert!(
                tree_core.arena.has_device_value(x, MAMBA),
                "Mamba eviction cursor on a valueless node {x}"
            );
            cursor = tree_core
                .device_lru_list(MAMBA)
                .get_prev_no_lock(x, &tree_core.arena);
            // A load-back pin means an in-flight DMA targets this node's slices.
            if tree_core.arena.node(x).is_load_back_pending() {
                continue;
            }
            if tree_core.evictable_device_leaves.contains(x) {
                break Some(x);
            }
            // Internal nodes are tombstoned inline (no IO).
            tree_core.evict_component_and_detach_lru_(
                x,
                ct,
                device_frees,
                host_frees,
                EvictLayer::Device,
                Some(tracker),
            );
            tree_core.cascade_evict_(x, ct, tracker, device_frees, host_frees, EvictLayer::Device);
            break None;
        };
        tree_core.component_state_mut(MAMBA).evict_device_cursor = cursor;
        next
    }

    /// Clear the device-eviction walk cursor state.
    fn evict_device_end(&self, tree_core: &mut UnifiedTreeCore<K>) {
        tree_core.set_evict_device_end(MAMBA);
    }

    /// Single-node mamba lock; host locks also detach from the host LRU.
    fn acquire_component_lock(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        mut result: IncLockRefResult,
        lock_host: bool,
    ) -> IncLockRefResult {
        let node = tree_core.arena.node(node_id);
        if node.is_root() {
            return result;
        }
        // A node in skip_lock_node_ids was a tombstone when this lock was acquired.
        if !Self::has_value(node, lock_host) {
            result
                .skip_lock_node_ids
                .entry(MAMBA)
                .or_default()
                .insert(node.id);
            return result;
        }
        if lock_host {
            if node.host_lock_ref(MAMBA) == 0 {
                let host_lru = tree_core.host_lru_list_mut(MAMBA);
                if host_lru.in_list(Some(node_id)) {
                    host_lru.remove_node(node_id);
                }
            }
            tree_core.arena.inc_host_lock_ref(node_id, MAMBA);
        } else {
            let value_len = node.device_value_len(MAMBA);
            if node.device_lock_ref(MAMBA) == 0 {
                tree_core.dec_evictable_size(MAMBA, value_len);
                tree_core.inc_protected_size(MAMBA, value_len);
            }
            tree_core.arena.inc_device_lock_ref(node_id, MAMBA);
        }
        result
    }

    /// Single-node mamba unlock; host unlocks reinsert into the host LRU.
    fn release_component_lock(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        params: Option<&DecLockRefParams>,
        lock_host: bool,
    ) {
        if tree_core.arena.node(node_id).is_root() {
            return;
        }
        if let Some(params) = params
            && params
                .skip_lock_node_ids
                .get(&MAMBA)
                .is_some_and(|ids| ids.contains(&tree_core.arena.node(node_id).id))
        {
            return;
        }
        if lock_host {
            let node = tree_core.arena.node_mut(node_id);
            node.dec_host_lock_ref(MAMBA);
            if node.host_lock_ref(MAMBA) == 0
                && !node.has_device_value(MAMBA)
                && node.has_host_value(MAMBA)
            {
                let host_lru = tree_core.host_lru_list_mut(MAMBA);
                if !host_lru.in_list(Some(node_id)) {
                    host_lru.insert_mru(node_id);
                }
            }
            return;
        }
        let node = tree_core.arena.node(node_id);
        let device_lock_ref = node.device_lock_ref(MAMBA);
        if device_lock_ref > 0 {
            if device_lock_ref == 1 {
                let value_len = node.device_value_len(MAMBA);
                tree_core.inc_evictable_size(MAMBA, value_len);
                tree_core.dec_protected_size(MAMBA, value_len);
            }
            tree_core.arena.dec_device_lock_ref(node_id, MAMBA);
        }
    }

    /// Build the mamba transfer descriptors for the given phase.
    fn build_hicache_transfers(
        &self,
        tree_core: &UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        phase: CacheTransferPhase,
        mamba_pool_idx: Option<Tensor>,
        host_indices: Option<Tensor>,
        _token_ids: Option<&[i64]>,
        _prefetch_tokens: usize,
        _last_hash: Option<&str>,
    ) -> Result<Option<Vec<PoolTransfer>>, TreeCoreRuntimeError> {
        Ok(match phase {
            CacheTransferPhase::BackupHost => {
                let node = tree_core.arena.node(node_id);
                if node.has_host_value(MAMBA) {
                    return Ok(None);
                }
                node.try_device_value(MAMBA).map(|value| {
                    vec![PoolTransfer {
                        name: PoolName::Mamba,
                        device_indices: Some(value.shallow_clone()),
                        ..Default::default()
                    }]
                })
            }
            CacheTransferPhase::LoadBack => {
                let node = tree_core.arena.node(node_id);
                if node.has_device_value(MAMBA) {
                    return Ok(None);
                }
                let mut transfers = Vec::new();
                // restore single node if host_value exists
                if let Some(host_value) = node.try_host_value(MAMBA) {
                    transfers.push(PoolTransfer {
                        name: PoolName::Mamba,
                        host_indices: Some(host_value.shallow_clone()),
                        nodes_to_load: Some(vec![node.id]),
                        ..Default::default()
                    });
                }
                // Per-request mamba CoW (H->D copy into the request's device slot,
                // pre-allocated on caller side).
                if let (Some(mamba_pool_idx), Some(host_value)) =
                    (mamba_pool_idx, node.try_host_value(MAMBA))
                {
                    transfers.push(PoolTransfer {
                        name: PoolName::Mamba,
                        host_indices: Some(host_value.shallow_clone()),
                        device_indices: Some(mamba_pool_idx.unsqueeze(0)),
                        ..Default::default()
                    });
                }
                if transfers.is_empty() {
                    None
                } else {
                    Some(transfers)
                }
            }
            CacheTransferPhase::BackupStorage => {
                let node = tree_core.arena.node(node_id);
                let Some(host_value) = node.try_host_value(MAMBA) else {
                    return Ok(None);
                };
                let Some(hash_value) = node.hash_value.as_ref().filter(|h| !h.is_empty()) else {
                    return Ok(None);
                };
                Some(vec![PoolTransfer {
                    name: PoolName::Mamba,
                    host_indices: Some(host_value.shallow_clone()),
                    keys: Some(vec![hash_value[hash_value.len() - 1].clone()]),
                    hit_policy: PoolHitPolicy::TrailingPages,
                    ..Default::default()
                }])
            }
            CacheTransferPhase::Prefetch => {
                let host_indices =
                    host_indices.expect("Mamba PREFETCH build requires host indices");
                Some(vec![PoolTransfer {
                    name: PoolName::Mamba,
                    host_indices: Some(host_indices),
                    keys: Some(vec!["__placeholder__".to_string()]),
                    hit_policy: PoolHitPolicy::TrailingPages,
                    ..Default::default()
                }])
            }
        })
    }

    /// Post-transfer mamba bookkeeping for the given phase.
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
                    let node = tree_core.arena.node_mut(node_id);
                    if !node.has_host_value(MAMBA) {
                        node.set_host_value(MAMBA, host_indices.copy());
                    }
                }
            }
            CacheTransferPhase::LoadBack => {
                let Some(transfer) = transfers.first() else {
                    return;
                };
                if let Some(device_indices) = &transfer.device_indices {
                    let node = tree_core.arena.node_mut(node_id);
                    node.set_device_value(MAMBA, device_indices.copy());
                    let count = node.device_value_len(MAMBA);
                    // Move from host LRU to device LRU
                    let host_lru = tree_core.host_lru_list_mut(MAMBA);
                    if host_lru.in_list(Some(node_id)) {
                        host_lru.remove_node(node_id);
                    }
                    tree_core.device_lru_list_mut(MAMBA).insert_mru(node_id);
                    tree_core.inc_evictable_size(MAMBA, count);
                }
            }
            // The python elif chain has no BACKUP_STORAGE arm.
            CacheTransferPhase::BackupStorage => {}
            CacheTransferPhase::Prefetch => {
                let Some(transfer) = transfers.first() else {
                    return;
                };
                let host_indices = transfer.host_indices.as_ref();
                let loaded = pool_storage_result.is_some_and(|result| {
                    result
                        .extra_pool_hit_pages
                        .get(&PoolName::Mamba)
                        .copied()
                        .unwrap_or(0)
                        >= 1
                });
                let target_node_id = insert_result
                    .as_deref()
                    .and_then(|result| result.inserted_host_node)
                    .map(|id| tree_core.arena.resolve(id));
                let attach_target = match (host_indices, target_node_id) {
                    (Some(_), Some(target))
                        if loaded && !tree_core.arena.has_host_value(target, MAMBA) =>
                    {
                        Some(target)
                    }
                    _ => None,
                };
                let Some(target) = attach_target else {
                    // The buffer cannot attach: free it and let the caller keep
                    // its own donated slot bookkeeping.
                    cache_actions.push(CacheAction::FreeComponentHostSlot {
                        component_type: MAMBA,
                        host_indices: host_indices
                            .map(|host| vec![host.shallow_clone()])
                            .unwrap_or_default(),
                    });
                    if let Some(insert_result) = insert_result {
                        insert_result.mamba_exist = true;
                    }
                    return;
                };
                let host_indices = host_indices.expect("an attach target implies host indices");
                tree_core
                    .arena
                    .set_host_value(target, MAMBA, host_indices.copy());
                if !tree_core.arena.has_device_value(target, MAMBA) {
                    let host_lru = tree_core.host_lru_list_mut(MAMBA);
                    if !host_lru.in_list(Some(target)) {
                        host_lru.insert_mru(target);
                    }
                }
                if let Some(insert_result) = insert_result {
                    insert_result.mamba_exist = false;
                }
            }
        }
    }

    /// Evict mamba host resources: internal nodes tombstone privately, host
    /// leaves evict atomically.
    fn drive_host_eviction(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        num_tokens: usize,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        let ct = MAMBA;
        let mut x = tree_core
            .host_lru_list(MAMBA)
            .get_lru_no_lock(&tree_core.arena);
        loop {
            if tracker[&ct] >= num_tokens {
                break;
            }
            let Some(cur) = x else {
                break;
            };
            if !tree_core.host_lru_list(MAMBA).in_list(Some(cur)) {
                break;
            }
            let x_next = tree_core
                .host_lru_list(MAMBA)
                .get_prev_no_lock(cur, &tree_core.arena);
            // A load-back pin means an in-flight DMA reads this node's host slices.
            if tree_core.arena.node(cur).is_load_back_pending() {
                x = x_next;
                continue;
            }
            if tree_core.evictable_host_leaves.contains(cur) {
                // Host leaf: atomic eviction (all components host + delete)
                tree_core.evict_host_leaf_(cur, tracker, device_frees, host_frees);
            } else {
                // Internal: tombstone Mamba + cascade
                assert!(
                    tree_core.arena.has_host_value(cur, MAMBA),
                    "Mamba host LRU member {cur} has no host value"
                );
                tree_core.evict_component_and_detach_lru_(
                    cur,
                    ct,
                    device_frees,
                    host_frees,
                    EvictLayer::Host,
                    Some(tracker),
                );
                tree_core.cascade_evict_(
                    cur,
                    ct,
                    tracker,
                    device_frees,
                    host_frees,
                    EvictLayer::Host,
                );
            }
            x = x_next;
        }
    }
}

#[cfg(test)]
#[path = "../tests/components/mamba.rs"]
mod tests;
