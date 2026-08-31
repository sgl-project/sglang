//! SWA (sliding-window attention) component driver: overrides the methods SWA
//! customizes and inherits the rest from the `TreeComponent` defaults.
//! SWA values arrive pool-resolved; the full->SWA index translation happens at
//! the cache boundary.

use std::collections::{HashMap, HashSet};

use tch::{Kind, Tensor};

use crate::components::TreeComponent;
use crate::components::{ComponentType, FULL, SWA};
use crate::node::ChildKeyType;
use crate::node::Node;
use crate::node::{NodeId, NodeIdx_, TreeCoreRuntimeError, ValueSlotIdx};
use crate::unified_tree_core::{
    CacheAction, CacheInitParams, CacheTransferPhase, DecLockRefParams, EvictLayer,
    IncLockRefResult, InsertParams, InsertResult, LRURefreshPhase, MatchPrefixParams, MatchResult,
    PoolHitPolicy, PoolName, PoolTransfer, PoolTransferResult, UnifiedTreeCore,
};

/// SWA component driver; owns the SWA device/host value slots.
pub struct SwaComponent {
    /// Sliding window size in tokens.
    sliding_window_size: usize,
}

impl SwaComponent {
    /// The component's device value slot.
    pub const DEVICE: ValueSlotIdx = ValueSlotIdx::device(SWA);
    /// The component's host value slot.
    pub const HOST: ValueSlotIdx = ValueSlotIdx::host(SWA);
}

impl SwaComponent {
    /// Build the driver from the tree's init params.
    pub fn new(params: &CacheInitParams) -> Self {
        SwaComponent {
            sliding_window_size: params
                .swa_sliding_window_size
                .expect("the SWA component requires swa_sliding_window_size"),
        }
    }

    /// Cap a fresh in-window SWA leaf at one page-aligned window so locking it pins
    /// only one window of SWA pool, not the whole (long chunked-prefill) leaf; return
    /// the split-off parent (older window) or None. The SWA value is stamped later, so
    /// this runs on the tombstone leaf.
    fn maybe_split_leaf_for_swa_lock_<K: ChildKeyType>(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        leaf_id: NodeIdx_,
    ) -> Option<NodeIdx_> {
        let leaf = tree_core.arena.node(leaf_id);
        if leaf.is_root() || leaf.device_lock_ref(SWA) > 0 {
            return None;
        }

        let page_size = tree_core.page_size;
        // Smallest page-aligned size that still covers the sliding window.
        let tail_size = self.sliding_window_size.div_ceil(page_size) * page_size;
        let leaf_len = leaf.key.atom_len();
        if leaf_len <= tail_size {
            return None;
        }
        let split_at = leaf_len - tail_size;
        if page_size > 1
            && (!split_at.is_multiple_of(page_size) || !leaf_len.is_multiple_of(page_size))
        {
            return None;
        }

        let (new_parent, action) = tree_core.split_node_(leaf_id, split_at);
        assert!(
            action.is_none(),
            "fresh SWA leaf cannot be write-through-pending"
        );
        Some(new_parent)
    }

    // Tier-selected SWA slot reads for the lock walks; `host` picks the host slot.
    fn has_value<K: ChildKeyType>(node: &Node<K>, host: bool) -> bool {
        if host {
            node.has_host_value(SWA)
        } else {
            node.has_device_value(SWA)
        }
    }

    fn lock_ref<K: ChildKeyType>(node: &Node<K>, host: bool) -> u32 {
        if host {
            node.host_lock_ref(SWA)
        } else {
            node.device_lock_ref(SWA)
        }
    }

    fn inc_lock_ref<K: ChildKeyType>(node: &mut Node<K>, host: bool) {
        if host {
            node.inc_host_lock_ref(SWA);
        } else {
            node.inc_device_lock_ref(SWA);
        }
    }

    fn dec_lock_ref<K: ChildKeyType>(node: &mut Node<K>, host: bool) {
        if host {
            node.dec_host_lock_ref(SWA);
        } else {
            node.dec_device_lock_ref(SWA);
        }
    }

    fn value_len<K: ChildKeyType>(node: &Node<K>, host: bool) -> usize {
        if host {
            node.host_value_len(SWA)
        } else {
            node.device_value_len(SWA)
        }
    }

    fn swa_uuid<K: ChildKeyType>(node: &Node<K>, host: bool) -> Option<i64> {
        if host {
            node.swa_host_uuid
        } else {
            node.swa_uuid
        }
    }

    /// The node's SWA lock-window uuid for the tier, stamping a fresh one if absent.
    fn ensure_swa_uuid<K: ChildKeyType>(
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        host: bool,
    ) -> i64 {
        match Self::swa_uuid(tree_core.arena.node(node_id), host) {
            Some(uuid) => uuid,
            None => {
                let minted = tree_core.next_swa_uuid_();
                let node = tree_core.arena.node_mut(node_id);
                if host {
                    node.swa_host_uuid = Some(minted);
                } else {
                    node.swa_uuid = Some(minted);
                }
                minted
            }
        }
    }

    fn next_host_unlocked_device_lru_node<K: ChildKeyType>(
        tree_core: &UnifiedTreeCore<K>,
        from: Option<NodeIdx_>,
    ) -> Option<NodeIdx_> {
        let lru = tree_core.device_lru_list(SWA);
        let unlocked = |id: NodeIdx_| tree_core.arena.node(id).host_lock_ref(SWA) == 0;
        match from {
            Some(node_id) => lru.get_prev_where(node_id, unlocked),
            None => lru.get_lru_where(unlocked),
        }
    }
}

impl SwaComponent {
    /// Queue a free of the given SWA host slots; empty tensors are dropped.
    fn release_swa_host_(&self, host_indices: Tensor, cache_actions: &mut Vec<CacheAction>) {
        if host_indices.numel() > 0 {
            cache_actions.push(CacheAction::FreeComponentHostSlot {
                component_type: SWA,
                host_indices: vec![host_indices],
            });
        }
    }

    /// Write host_indices into node's SWA host_value and refresh tree state.
    fn attach_swa_host_value_<K: ChildKeyType>(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        host_indices: Tensor,
    ) {
        let node = tree_core.arena.node_mut(node_id);
        let device_on = node.has_device_value(SWA);
        node.set_host_value(SWA, host_indices.copy());
        let host_lru = tree_core.host_lru_list_mut(SWA);
        if !device_on && !host_lru.in_list(Some(node_id)) {
            host_lru.insert_mru(node_id);
        }
        tree_core.update_evictable_leaf_sets_(node_id);
        if let Some(parent) = tree_core.arena.node(node_id).try_parent() {
            tree_core.update_evictable_leaf_sets_(parent);
        }
    }

    /// Fill the prefetched SWA window onto the leaf→anchor path.
    ///
    /// All-or-nothing over one full window: `loaded_pages` is the cross-rank
    /// MIN, so `loaded_pages < window_pages` drops the whole window (keeps the
    /// tree identical across TP ranks). Otherwise map the buffer to token range
    /// `[loaded_start, total_len)` and walk leaf→anchor, filling SWA
    /// tombstones and releasing slices that already have host_value.
    fn commit_prefetch_<K: ChildKeyType>(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        transfers: Vec<PoolTransfer>,
        cache_actions: &mut Vec<CacheAction>,
        insert_result: Option<&InsertResult>,
        pool_storage_result: Option<&PoolTransferResult>,
    ) {
        if transfers.is_empty() {
            return;
        }
        let page_size = tree_core.page_size;
        let transfer = &transfers[0];
        let window_require_pages = transfer
            .host_indices
            .as_ref()
            .map_or(0, |host| host.numel() / page_size);
        let loaded_pages = pool_storage_result.map_or(0, |result| {
            result
                .extra_pool_hit_pages
                .get(&PoolName::Swa)
                .copied()
                .unwrap_or(0)
        });
        let target = insert_result
            .and_then(|result| result.inserted_host_node)
            .map(|id| tree_core.arena.resolve(id));

        let (Some(target), Some(host_indices)) = (target, transfer.host_indices.as_ref()) else {
            if let Some(host_indices) = &transfer.host_indices {
                self.release_swa_host_(host_indices.shallow_clone(), cache_actions);
            }
            return;
        };
        if window_require_pages == 0 || loaded_pages < window_require_pages {
            self.release_swa_host_(host_indices.shallow_clone(), cache_actions);
            return;
        }
        let insert_result = insert_result.expect("target implies an insert result");

        // The buffer covers token range [loaded_start, total_len).
        let loaded_start = insert_result.total_len - window_require_pages * page_size;

        // Walk leaf -> anchor; pos is the right edge of cur in tokens.
        let mut pos = insert_result.total_len;
        let mut cur = target;
        while cur != node_id && pos > loaded_start {
            let cur_node = tree_core.arena.node(cur);
            let node_start = pos - cur_node.key.atom_len();
            // Intersection of cur's range and the buffer.
            let fill_start = node_start.max(loaded_start);
            let fill_len = pos - fill_start;
            let buf_off = fill_start - loaded_start;
            let slice = host_indices.narrow(0, buf_off as i64, fill_len as i64);
            let parent = cur_node
                .try_parent()
                .expect("prefetch walk reached a root before the anchor");

            if !cur_node.has_host_value(SWA) && fill_len > 0 {
                // Tombstone: split off the in-buffer tail if needed, then fill.
                if fill_start > node_start {
                    let (_, action) = tree_core.split_node_(cur, fill_start - node_start);
                    if let Some(action) = action {
                        cache_actions.push(action);
                    }
                }
                self.attach_swa_host_value_(tree_core, cur, slice);
            } else {
                // Already has SWA (or empty overlap): drop this slice.
                self.release_swa_host_(slice, cache_actions);
            }

            pos = node_start;
            cur = parent;
        }

        // Buffer prefix that fell outside the anchor->leaf path.
        if pos > loaded_start {
            self.release_swa_host_(
                host_indices.narrow(0, 0, (pos - loaded_start) as i64),
                cache_actions,
            );
        }
    }
}

impl<K: ChildKeyType> TreeComponent<K> for SwaComponent {
    fn component_type(&self) -> ComponentType {
        SWA
    }

    fn refresh_lru(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        phase: LRURefreshPhase,
        node_id: NodeIdx_,
    ) {
        match phase {
            // Walk-down would refresh every visited ancestor to MRU, but most
            // are outside the active sliding window and must stay evictable.
            // Window-bounded refresh runs at MATCH_END / INSERT_END instead.
            LRURefreshPhase::Walkdown => {}
            LRURefreshPhase::MatchEnd | LRURefreshPhase::InsertEnd => {
                let window = self.sliding_window_size + tree_core.page_size;
                let (lru, arena) = tree_core.device_lru_list_mut_with_arena(SWA);
                lru.reset_node_and_window_ancestors_mru(node_id, window, arena, |node| {
                    node.has_device_value(SWA)
                });
            }
        }
    }

    fn create_match_validator(
        &self,
        tree_core: &UnifiedTreeCore<K>,
        match_device_only: bool,
    ) -> Box<dyn FnMut(&UnifiedTreeCore<K>, NodeIdx_) -> bool> {
        let sliding_window_size = self.sliding_window_size;
        // unified_kv never caches the SWA ring (per-request, not
        // content-stable), so SWA bookkeeping must not gate the match here.
        let swa_device_only_hicache = !tree_core.has_swa_host_pool && tree_core.enable_hicache;
        let mut contiguous_len = usize::MAX;
        Box::new(move |tree_core: &UnifiedTreeCore<K>, node_id: NodeIdx_| {
            let node = tree_core.arena.node(node_id);
            // HiCache: a host-only tombstone is a valid match boundary too
            // — load_back will restore SWA from host before use.
            if !node.has_device_value(SWA) && (match_device_only || !node.has_host_value(SWA)) {
                contiguous_len = 0;
                return swa_device_only_hicache && (node.backuped() || !node.evicted());
            }
            contiguous_len = contiguous_len.saturating_add(node.key.atom_len());
            contiguous_len >= sliding_window_size
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
        // Sum the SWA tokens backing the match, walking up from the best match
        // until one sliding window is covered; host-resident chunks count
        // toward the SWA host hit.
        let mut n_swa = 0;
        let mut swa_host_hit = 0;
        let mut node = tree_core
            .arena
            .node(tree_core.arena.resolve(result.best_match_node_id));
        while !node.is_root() && n_swa < self.sliding_window_size {
            if node.has_device_value(SWA) {
                n_swa += node.device_value_len(SWA);
            } else if node.has_host_value(SWA) {
                // TODO(hzh): once load_back is constrained to fetch only one
                // sliding window worth of pages, cap swa_host_hit at
                // sliding_window_size so the scheduler budget matches the
                // actual device-pool consumption.
                let host_len = node.host_value_len(SWA);
                swa_host_hit += host_len;
                n_swa += host_len;
            } else {
                break;
            }
            node = tree_core.arena.node(node.parent());
        }
        if swa_host_hit > 0 {
            result.swa_host_hit_length = result.swa_host_hit_length.max(swa_host_hit);
        }
        result
    }

    fn update_component_on_insert_overlap(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        prefix_len: usize,
        total_prefix_len: usize,
        value_slice: Tensor,
        params: &InsertParams<'_, K>,
        result: &mut InsertResult,
        cache_actions: &mut Vec<CacheAction>,
    ) -> usize {
        if params.prev_prefix_len >= total_prefix_len + prefix_len {
            return prefix_len;
        }

        let node = tree_core.arena.node_mut(node_id);
        let is_tombstone = !node.has_device_value(SWA);
        if !is_tombstone {
            return prefix_len;
        }

        let swa_evicted_seqlen = params.swa_evicted_seqlen;
        assert_eq!(
            node.device_lock_ref(SWA),
            0,
            "tombstone Swa lock_ref should be 0, node {node_id}"
        );
        assert_eq!(
            swa_evicted_seqlen % tree_core.page_size,
            0,
            "Swa: swa_evicted_seqlen must be page-aligned, swa_evicted_seqlen={swa_evicted_seqlen}"
        );

        if swa_evicted_seqlen <= total_prefix_len {
            // Branch 1: entire value_slice is within SWA window — recover
            result.record_adopted_range(SWA, total_prefix_len, total_prefix_len + prefix_len);
            if node.device_lock_ref(FULL) > 0 {
                cache_actions.push(CacheAction::RecoverSwaWithLockedFull {
                    node_id: node.id,
                    kept_full: node.device_value(FULL).shallow_clone(),
                    incoming_full: value_slice,
                });
                return 0;
            }
            result.record_adopted_range(FULL, total_prefix_len, total_prefix_len + prefix_len);
            let old_full = node.take_device_value(FULL);
            node.set_device_value(FULL, value_slice.copy());
            cache_actions.push(CacheAction::FreeDeviceKVFullOnly(vec![old_full]));
            cache_actions.push(CacheAction::SwaRebuild {
                node_id: node.id,
                source_value: value_slice,
            });
            0
        } else if swa_evicted_seqlen < total_prefix_len + prefix_len {
            // Branch 2: value_slice[start_idx:] is within SWA window — partial recover
            let start_idx = swa_evicted_seqlen - total_prefix_len;
            result.record_adopted_range(SWA, swa_evicted_seqlen, total_prefix_len + prefix_len);
            let node_ext_id = node.id;
            let is_locked = node.device_lock_ref(FULL) > 0;
            let full_len = node.device_value_len(FULL);
            let old_full =
                node.device_value(FULL)
                    .narrow(0, start_idx as i64, (full_len - start_idx) as i64);
            let (_, action) = tree_core.split_node_(node_id, start_idx);
            if let Some(action) = action {
                cache_actions.push(action);
            }
            let new_full = value_slice.narrow(0, start_idx as i64, (prefix_len - start_idx) as i64);
            if is_locked {
                cache_actions.push(CacheAction::RecoverSwaWithLockedFull {
                    node_id: node_ext_id,
                    kept_full: old_full,
                    incoming_full: new_full,
                });
                return start_idx;
            }
            result.record_adopted_range(FULL, swa_evicted_seqlen, total_prefix_len + prefix_len);
            let node = tree_core.arena.node_mut(node_id);
            let _ = node.take_device_value(FULL);
            node.set_device_value(FULL, new_full.copy());
            cache_actions.push(CacheAction::FreeDeviceKVFullOnly(vec![old_full]));
            cache_actions.push(CacheAction::SwaRebuild {
                node_id: node_ext_id,
                source_value: new_full,
            });
            start_idx
        } else {
            // Branch 3: entire value_slice is outside SWA window — not consumed
            prefix_len
        }
    }

    fn recover_after_unevict(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        prefix_len: usize,
        total_prefix_len: usize,
        params: &InsertParams<'_, K>,
        result: &mut InsertResult,
        cache_actions: &mut Vec<CacheAction>,
    ) {
        // _unevict_node_on_insert already wrote the request's fresh KV slice
        // into the base value. We just need to rebuild SWA from that slice for
        // the in-window portion. There is no old SWA slot to free here.
        let node = tree_core.arena.node(node_id);
        if node.has_device_value(SWA) {
            return;
        }
        assert_eq!(
            node.device_lock_ref(SWA),
            0,
            "tombstone Swa lock_ref should be 0 on unevict, node {node_id}"
        );
        let swa_evicted_seqlen = params.swa_evicted_seqlen;
        assert_eq!(
            swa_evicted_seqlen % tree_core.page_size,
            0,
            "Swa: swa_evicted_seqlen must be page-aligned, swa_evicted_seqlen={swa_evicted_seqlen}"
        );

        if swa_evicted_seqlen <= total_prefix_len {
            // entire node is within the SWA window
        } else if swa_evicted_seqlen < total_prefix_len + prefix_len {
            let start_idx = swa_evicted_seqlen - total_prefix_len;
            let (_, action) = tree_core.split_node_(node_id, start_idx);
            if let Some(action) = action {
                cache_actions.push(action);
            }
        } else {
            return;
        }
        result.record_adopted_range(
            SWA,
            total_prefix_len.max(swa_evicted_seqlen),
            total_prefix_len + prefix_len,
        );
        cache_actions.push(CacheAction::SwaRebuild {
            node_id: tree_core.arena.node(node_id).id,
            source_value: tree_core.arena.device_value(node_id, FULL).shallow_clone(),
        });
    }

    fn commit_insert_component_data(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        is_new_leaf: bool,
        params: &InsertParams<'_, K>,
        result: &mut InsertResult,
        cache_actions: &mut Vec<CacheAction>,
    ) {
        if !is_new_leaf {
            return;
        }

        let node_start = result.prefix_len;
        let node_end = node_start + tree_core.arena.node(node_id).key.atom_len();
        // A boundary above the leaf skips the split (Python's negative split_pos).
        if params.swa_evicted_seqlen >= node_start {
            let split_pos = params.swa_evicted_seqlen - node_start;
            if split_pos >= tree_core.arena.node(node_id).key.atom_len() {
                // Entire leaf is outside the SWA window — left as a tombstone.
                return;
            }
            if split_pos > 0 {
                // Node straddles the boundary: split into an out-of-window parent
                // (tombstone) and an in-window child; `node` becomes the child.
                let (_, action) = tree_core.split_node_(node_id, split_pos);
                assert!(action.is_none(), "new leaf cannot be write-through-pending");
            }
        }
        result.record_adopted_range(SWA, node_start.max(params.swa_evicted_seqlen), node_end);
        // Cap the in-window leaf at one window for lock granularity, then rebuild SWA
        // onto the in-window node(s) at apply time; rebuild the older prefix first so
        // the in-window tail lands more-MRU.
        let capped_parent = self.maybe_split_leaf_for_swa_lock_(tree_core, node_id);
        if let Some(capped_parent) = capped_parent {
            cache_actions.push(CacheAction::SwaRebuild {
                node_id: tree_core.arena.node(capped_parent).id,
                source_value: tree_core
                    .arena
                    .device_value(capped_parent, FULL)
                    .shallow_clone(),
            });
        }
        cache_actions.push(CacheAction::SwaRebuild {
            node_id: tree_core.arena.node(node_id).id,
            source_value: tree_core.arena.device_value(node_id, FULL).shallow_clone(),
        });
    }

    fn redistribute_on_node_split(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        new_parent_id: NodeIdx_,
        child_id: NodeIdx_,
    ) {
        let (new_parent, child) = tree_core.arena.node_pair_mut(new_parent_id, child_id);
        let split_len = new_parent.key.atom_len() as i64;
        new_parent.copy_device_lock_ref(SWA, child);
        if child.has_device_value(SWA) {
            Node::redistribute_child_device_value(new_parent, child, SWA, split_len);
        }
        if child.has_host_value(SWA) {
            Node::redistribute_child_host_value(new_parent, child, SWA, split_len);
            // Device-tombstoned sides park in the host LRU.
            let parent_is_tombstone = !new_parent.has_device_value(SWA);
            let child_is_tombstone = !child.has_device_value(SWA);
            let host_lru = tree_core.host_lru_list_mut(SWA);
            if parent_is_tombstone {
                host_lru.insert_mru(new_parent_id);
            }
            if child_is_tombstone && !host_lru.in_list(Some(child_id)) {
                host_lru.insert_mru(child_id);
            }
        }

        // parent inherits the swa_uuid from child for swa lock ref
        let swa_uuid = tree_core.arena.node_mut(child_id).swa_uuid.take();
        tree_core.arena.node_mut(new_parent_id).swa_uuid = swa_uuid;
    }

    fn evict_component(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        target: EvictLayer,
    ) -> (usize, usize) {
        let ct = SWA;
        let node = tree_core.arena.node_mut(node_id);
        let mut freed = 0;
        let mut host_freed = 0;

        // Device layer
        if target.contains(EvictLayer::Device) && node.has_device_value(SWA) {
            // Pass full indices to free_swa so slots with no SWA pair are
            // skipped. Freeing swa_value directly would double free those
            // entries since they all map to the same sentinel slot.
            device_frees
                .entry(ct)
                .or_default()
                .push(node.device_value(FULL).shallow_clone());
            freed = node.device_value_len(SWA);
            let _ = node.take_device_value(SWA);
            tree_core.dec_evictable_size(SWA, freed);
        }

        // Host layer
        let node = tree_core.arena.node_mut(node_id);
        if target.contains(EvictLayer::Host) && node.has_host_value(SWA) {
            host_freed = node.host_value_len(SWA);
            host_frees
                .entry(ct)
                .or_default()
                .push(node.take_host_value(SWA));
            let host_lru = tree_core.host_lru_list_mut(SWA);
            if host_lru.in_list(Some(node_id)) {
                host_lru.remove_node(node_id);
            }
        }

        // After device tombstone: if host_value remains, move into host LRU
        let node = tree_core.arena.node(node_id);
        if target == EvictLayer::Device && !node.has_device_value(SWA) && node.has_host_value(SWA) {
            let host_lru = tree_core.host_lru_list_mut(SWA);
            if !host_lru.in_list(Some(node_id)) {
                host_lru.insert_mru(node_id);
            }
        }

        (freed, host_freed)
    }

    fn eviction_priority(&self, is_leaf: bool) -> i64 {
        if is_leaf { 0 } else { 1 }
    }

    fn evict_device_start(&self, tree_core: &mut UnifiedTreeCore<K>, request_cnt: usize) {
        tree_core.set_evict_device_start(SWA, request_cnt);
        let cursor = tree_core
            .device_lru_list(SWA)
            .get_lru_no_lock(&tree_core.arena);
        tree_core.component_state_mut(SWA).evict_device_cursor = cursor;
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
        let ct = SWA;
        assert!(
            tree_core.component_state(SWA).is_evict_device_ongoing,
            "Swa device eviction not started"
        );
        let mut cursor = tree_core.component_state(SWA).evict_device_cursor;
        // The cursor is re-validated (reset to LRU head) if the previous
        // node's eviction removed it.
        if cursor.is_some_and(|c| !tree_core.device_lru_list(SWA).in_list(Some(c))) {
            cursor = tree_core
                .device_lru_list(SWA)
                .get_lru_no_lock(&tree_core.arena);
        }
        let next = loop {
            if tracker[&ct] >= tree_core.component_state(SWA).evict_device_request_cnt {
                break None;
            }
            let Some(x) = cursor else {
                break None;
            };
            if !tree_core.device_lru_list(SWA).in_list(Some(x)) {
                break None;
            }
            assert!(
                tree_core.arena.has_device_value(x, SWA),
                "Swa eviction cursor on a valueless node {x}"
            );
            cursor = tree_core
                .device_lru_list(SWA)
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
        tree_core.component_state_mut(SWA).evict_device_cursor = cursor;
        next
    }

    fn evict_device_end(&self, tree_core: &mut UnifiedTreeCore<K>) {
        tree_core.set_evict_device_end(SWA);
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
            if tracker[&SWA] >= num_tokens {
                break;
            }
            let mut next = Self::next_host_unlocked_device_lru_node(tree_core, None);
            while let Some(node_id) = next {
                if tracker[&SWA] >= num_tokens {
                    break;
                }
                next = Self::next_host_unlocked_device_lru_node(tree_core, Some(node_id));
                if spare_imminent_demotes && tree_core.evictable_device_leaves.contains(node_id) {
                    continue;
                }
                if !tree_core.can_reclaim_coexisting_host_value_(node_id, SWA) {
                    continue;
                }
                tree_core.release_coexisting_host_value_(
                    node_id,
                    SWA,
                    tracker,
                    device_frees,
                    host_frees,
                );
            }
        }
    }

    /// Evict SWA host resources.
    /// Internal nodes: private tombstone (free SWA host only).
    /// Host leaves: atomic eviction via _evict_host_leaf.
    fn drive_host_eviction(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        num_tokens: usize,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        let ct = SWA;
        let mut x = tree_core
            .host_lru_list(SWA)
            .get_lru_no_lock(&tree_core.arena);
        loop {
            if tracker[&ct] >= num_tokens {
                break;
            }
            let Some(cur) = x else {
                break;
            };
            if !tree_core.host_lru_list(SWA).in_list(Some(cur)) {
                break;
            }
            let x_next = tree_core
                .host_lru_list(SWA)
                .get_prev_no_lock(cur, &tree_core.arena);
            // A load-back pin means an in-flight DMA reads this node's host slices.
            if tree_core.arena.node(cur).is_load_back_pending() {
                x = x_next;
                continue;
            }
            if tree_core.evictable_host_leaves.contains(cur) {
                tree_core.evict_host_leaf_(cur, tracker, device_frees, host_frees);
            } else {
                assert!(
                    tree_core.arena.has_host_value(cur, SWA),
                    "SWA host LRU member {cur} has no host value"
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

    fn build_hicache_transfers(
        &self,
        tree_core: &UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        phase: CacheTransferPhase,
        _mamba_pool_idx: Option<Tensor>,
        host_indices: Option<Tensor>,
        _token_ids: Option<&[i64]>,
        _prefetch_tokens: usize,
        _last_hash: Option<&str>,
    ) -> Result<Option<Vec<PoolTransfer>>, TreeCoreRuntimeError> {
        // unified_kv keeps SWA as a device-only ring.
        if !tree_core.has_swa_host_pool && tree_core.enable_hicache {
            return Ok(None);
        }
        Ok(match phase {
            CacheTransferPhase::BackupHost => {
                let node = tree_core.arena.node(node_id);
                if node.has_host_value(SWA) {
                    return Ok(None);
                }
                // cd.value already holds SWA-pool indices (translated at insert time).
                // Host pool indexing wants int64.
                node.try_device_value(SWA).map(|value| {
                    vec![PoolTransfer {
                        name: PoolName::Swa,
                        device_indices: Some(value.to_kind(Kind::Int64)),
                        ..Default::default()
                    }]
                })
            }
            CacheTransferPhase::LoadBack => {
                // `node` is best_match_node; the SWA validator guarantees every
                // ancestor within `sliding_window_size` has value or host_value.
                let mut n_swa = 0;
                let mut backed_up: Vec<Tensor> = Vec::new();
                let mut nodes_to_load: Vec<NodeId> = Vec::new();
                let mut cur = tree_core.arena.node(node_id);
                while !cur.is_root() && n_swa < self.sliding_window_size {
                    if let Some(value) = cur.try_device_value(SWA) {
                        // Device exists, skip it.
                        n_swa += value.size()[0] as usize;
                    } else if let Some(host_value) = cur.try_host_value(SWA) {
                        // Host only, collect it.
                        backed_up.push(host_value.shallow_clone());
                        nodes_to_load.push(cur.id);
                        n_swa += host_value.size()[0] as usize;
                    } else {
                        return Err(TreeCoreRuntimeError::SwaLoadBackMissingValue {
                            node_id: cur.id,
                        });
                    }
                    cur = tree_core.arena.node(cur.parent());
                }
                if backed_up.is_empty() {
                    return Ok(None);
                }
                backed_up.reverse();
                nodes_to_load.reverse();
                Some(vec![PoolTransfer {
                    name: PoolName::Swa,
                    host_indices: Some(Tensor::cat(&backed_up, 0)),
                    nodes_to_load: Some(nodes_to_load),
                    ..Default::default()
                }])
            }
            CacheTransferPhase::BackupStorage => {
                let node = tree_core.arena.node(node_id);
                let Some(host_value) = node.try_host_value(SWA) else {
                    return Ok(None);
                };
                let Some(hash_value) = node.hash_value.as_ref().filter(|h| !h.is_empty()) else {
                    return Ok(None);
                };
                let page_size = tree_core.page_size as i64;
                let num_pages = host_value.size()[0] / page_size;
                if num_pages == 0 {
                    return Ok(None);
                }
                let host_len = host_value.size()[0];
                Some(vec![PoolTransfer {
                    name: PoolName::Swa,
                    host_indices: Some(host_value.narrow(
                        0,
                        host_len - num_pages * page_size,
                        num_pages * page_size,
                    )),
                    keys: Some(
                        hash_value[hash_value.len().saturating_sub(num_pages as usize)..].to_vec(),
                    ),
                    hit_policy: PoolHitPolicy::TrailingPages,
                    ..Default::default()
                }])
            }
            CacheTransferPhase::Prefetch => {
                let host_indices = host_indices.expect("SWA PREFETCH build requires host indices");
                let sw_pages = host_indices.numel() / tree_core.page_size;
                Some(vec![PoolTransfer {
                    name: PoolName::Swa,
                    host_indices: Some(host_indices),
                    keys: Some(vec!["__placeholder__".to_string(); sw_pages]),
                    hit_policy: PoolHitPolicy::TrailingPages,
                    ..Default::default()
                }])
            }
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
                    let node = tree_core.arena.node_mut(node_id);
                    if !node.has_host_value(SWA) {
                        node.set_host_value(SWA, host_indices.copy());
                    }
                }
            }
            CacheTransferPhase::LoadBack => {
                let transfer = transfers
                    .first()
                    .expect("SWA LOAD_BACK commit requires a transfer");
                let device_indices = transfer
                    .device_indices
                    .as_ref()
                    .expect("SWA LOAD_BACK commit requires device indices");
                let mut full_chunks: Vec<Tensor> = Vec::new();
                let mut swa_chunks: Vec<Tensor> = Vec::new();
                let mut offset = 0i64;
                for &loaded_id in transfer.nodes_to_load.iter().flatten() {
                    let loaded_idx = tree_core.arena.resolve(loaded_id);
                    let n_tokens = tree_core.arena.host_value_len(loaded_idx, SWA) as i64;
                    let swa_chunk = device_indices.narrow(0, offset, n_tokens).copy();
                    tree_core.set_component_device_value_(
                        loaded_idx,
                        SWA,
                        swa_chunk.shallow_clone(),
                    );
                    let full_value = tree_core
                        .arena
                        .device_value(loaded_idx, FULL)
                        .shallow_clone();
                    assert_eq!(full_value.size()[0], n_tokens);
                    full_chunks.push(full_value);
                    swa_chunks.push(swa_chunk);
                    offset += n_tokens;
                }
                let host_indices = transfer
                    .host_indices
                    .as_ref()
                    .expect("SWA LOAD_BACK commit requires host indices");
                assert_eq!(offset, host_indices.size()[0]);
                // Rebuild the full->swa mapping for the loaded chunks at the orchestration layer.
                if !full_chunks.is_empty() {
                    cache_actions.push(CacheAction::RebuildFullToSwaMapping {
                        full_indices: full_chunks,
                        swa_indices: swa_chunks,
                    });
                }
            }
            CacheTransferPhase::Prefetch => {
                self.commit_prefetch_(
                    tree_core,
                    node_id,
                    transfers,
                    cache_actions,
                    insert_result.as_deref(),
                    pool_storage_result,
                );
            }
            CacheTransferPhase::BackupStorage => {}
        }
    }

    fn acquire_component_lock(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        mut result: IncLockRefResult,
        lock_host: bool,
    ) -> IncLockRefResult {
        let ct = SWA;
        let sliding_window_size = self.sliding_window_size;
        let mut swa_lock_size = 0;
        let mut swa_uuid = None;

        // Tombstoned nodes (cd.value is None) have no SWA chunk to protect
        // skip them and keep walking up. This path is hit when HiCache
        // backs up a FULL present internal node whose SWA was already evicted.
        let mut cur = node_id;
        loop {
            let node = tree_core.arena.node_mut(cur);
            if node.is_root() || swa_lock_size >= sliding_window_size {
                break;
            }
            let parent = node.parent();
            if !Self::has_value(node, lock_host) {
                result
                    .skip_lock_node_ids
                    .entry(ct)
                    .or_default()
                    .insert(node.id);
                cur = parent;
                continue;
            }
            let key_len = node.key.atom_len();
            let newly_locked = Self::lock_ref(node, lock_host) == 0;
            Self::inc_lock_ref(node, lock_host);
            swa_lock_size += Self::value_len(node, lock_host);
            if newly_locked {
                if lock_host {
                    let host_lru = tree_core.host_lru_list_mut(SWA);
                    if host_lru.in_list(Some(cur)) {
                        host_lru.remove_node(cur);
                    }
                } else {
                    tree_core.dec_evictable_size(SWA, key_len);
                    tree_core.inc_protected_size(SWA, key_len);
                }
            }
            if swa_lock_size >= sliding_window_size {
                swa_uuid = Some(Self::ensure_swa_uuid(tree_core, cur, lock_host));
            }
            cur = parent;
        }

        if lock_host {
            result.swa_uuid_for_host_lock = swa_uuid;
        } else {
            result.swa_uuid_for_lock = swa_uuid;
        }
        result
    }

    fn release_component_lock(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        params: Option<&DecLockRefParams>,
        lock_host: bool,
    ) {
        let ct = SWA;
        let swa_uuid_for_lock = params.and_then(|p| {
            if lock_host {
                p.swa_uuid_for_host_lock
            } else {
                p.swa_uuid_for_lock
            }
        });
        let empty = HashSet::new();
        let skip_lock_node_ids = params
            .and_then(|p| p.skip_lock_node_ids.get(&ct))
            .unwrap_or(&empty);

        // A node in skip_lock_node_ids was a tombstone when this lock was acquired.
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
            let lock_ref = Self::lock_ref(node, lock_host);
            if lock_ref == 0 {
                cur = parent;
                continue;
            }
            if lock_ref == 1 {
                if lock_host {
                    if !node.has_device_value(SWA) && node.has_host_value(SWA) {
                        let host_lru = tree_core.host_lru_list_mut(SWA);
                        if !host_lru.in_list(Some(cur)) {
                            host_lru.insert_mru(cur);
                        }
                    }
                } else {
                    let key_len = node.device_value_len(SWA);
                    tree_core.inc_evictable_size(SWA, key_len);
                    tree_core.dec_protected_size(SWA, key_len);
                }
            }
            Self::dec_lock_ref(tree_core.arena.node_mut(cur), lock_host);
            if swa_uuid_for_lock.is_some()
                && Self::swa_uuid(tree_core.arena.node(cur), lock_host) == swa_uuid_for_lock
            {
                break;
            }
            cur = parent;
        }
    }

    /// Early-release the SWA lock along [node, swa_uuid_for_lock] while
    /// leaving Full and Mamba locks intact.
    ///
    /// Called when a request's decode position has advanced past the sliding
    /// window — the SWA portion of the tree lock is no longer needed but the
    /// Full lock must stay so the request's prefix is protected.
    ///
    /// Caller (UnifiedRadixCache.dec_swa_lock_only) must ensure this is
    /// invoked at most once per (node, swa_uuid_for_lock) pair.
    fn release_window_lock(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        swa_uuid_for_lock: Option<i64>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        let ct = SWA;
        let mut cur = node_id;
        loop {
            let node = tree_core.arena.node_mut(cur);
            if node.is_root() {
                break;
            }
            let parent = node.parent();
            // Acquire skips tombstoned nodes; release must skip them too. Same
            // for nodes with lock_ref == 0 — acquire never credited them.
            if !node.has_device_value(SWA) || node.device_lock_ref(SWA) == 0 {
                if swa_uuid_for_lock.is_some() && node.swa_uuid == swa_uuid_for_lock {
                    break;
                }
                cur = parent;
                continue;
            }

            node.dec_device_lock_ref(SWA);
            if node.device_lock_ref(SWA) == 0 {
                let key_len = node.key.atom_len();
                tree_core.dec_protected_size(SWA, key_len);
                tree_core.inc_evictable_size(SWA, key_len);
                if tree_core.is_evictable_device_leaf_(tree_core.arena.node(cur)) {
                    tree_core.evict_component_and_detach_lru_(
                        cur,
                        ct,
                        device_frees,
                        host_frees,
                        EvictLayer::Device,
                        /* tracker = */ None,
                    );
                }
            }

            if swa_uuid_for_lock.is_some()
                && tree_core.arena.node(cur).swa_uuid == swa_uuid_for_lock
            {
                break;
            }
            cur = parent;
        }
    }
}

#[cfg(test)]
#[path = "../tests/components/swa.rs"]
mod tests;
