//! Per-component drivers; each receives the whole `UnifiedTreeCore` for backward access.
#![allow(unused_variables)]

use std::collections::HashMap;

use tch::Tensor;

use crate::node::{ChildKeyType, NodeArena, NodeIdx_, TreeCoreRuntimeError};
use crate::unified_tree_core::{
    CacheAction, CacheTransferPhase, DecLockRefParams, EvictLayer, IncLockRefResult, InsertParams,
    InsertResult, LRURefreshPhase, MatchPrefixParams, MatchResult, PoolTransfer,
    PoolTransferResult, UnifiedTreeCore,
};

mod full;
mod mamba;
mod swa;

pub use full::FullComponent;
pub use mamba::MambaComponent;
pub use swa::SwaComponent;

/// Whether `node_id` holds the component's data on `target`, checking its
/// device or host slot.
pub(crate) fn node_has_component_data<K: ChildKeyType>(
    arena: &NodeArena<K>,
    node_id: NodeIdx_,
    component_type: ComponentType,
    target: EvictLayer,
) -> bool {
    match target {
        EvictLayer::Device => arena.has_device_value(node_id, component_type),
        EvictLayer::Host => arena.has_host_value(node_id, component_type),
        EvictLayer::All => panic!("node_has_component_data: EvictLayer::All is not a single layer"),
    }
}

/// Every device value of the component across all roots, concatenated.
pub(crate) fn all_values_flatten<K: ChildKeyType>(
    tree_core: &UnifiedTreeCore<K>,
    component_type: ComponentType,
) -> Tensor {
    let mut values: Vec<Tensor> = Vec::new();
    let mut stack: Vec<NodeIdx_> = vec![tree_core.arena.root()];
    while let Some(node_id) = stack.pop() {
        let node = tree_core.arena.node(node_id);
        if let Some(value) = node.try_device_value(component_type) {
            values.push(value.shallow_clone());
        }
        stack.extend(node.children.values().copied());
    }
    if values.is_empty() {
        return tree_core.empty_device_indices.shallow_clone();
    }
    Tensor::cat(&values, 0)
}

/// A per-component lock/value/eviction driver over the shared `UnifiedTreeCore`.
pub trait TreeComponent<K: ChildKeyType> {
    /// The component this driver serves.
    fn component_type(&self) -> ComponentType;

    /// Whether this component has device data that still needs a host backup.
    fn needs_incremental_backup(
        &self,
        _tree_core: &UnifiedTreeCore<K>,
        _node_id: NodeIdx_,
    ) -> bool {
        false
    }

    /// Refresh this component's LRU position for `node_id` at the given walk phase.
    fn refresh_lru(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        phase: LRURefreshPhase,
        node_id: NodeIdx_,
    ) {
        // Python reference — tree_component.py::TreeComponent.refresh_lru:
        //     def refresh_lru(
        //         self,
        //         phase: LRURefreshPhase,
        //         node: UnifiedTreeNode,
        //         root_node: UnifiedTreeNode,
        //     ) -> None:
        //         ct = self.component_type
        //         match phase:
        //             case LRURefreshPhase.WALKDOWN:
        //                 if node.component_data[ct].value is None:
        //                     return
        //                 self.tree_core.lru_lists[ct].reset_node_mru(node)
        //             case LRURefreshPhase.MATCH_END:
        //                 self.tree_core.lru_lists[ct].reset_node_and_parents_mru(
        //                     node, root_node, self.node_has_component_data
        //                 )
        //             case LRURefreshPhase.INSERT_END:
        //                 # WALKDOWN already refreshed every node on the insert path
        //                 # (including the new leaf), so there is nothing more to do.
        //                 return
        //             case _:
        //                 raise ValueError(f"Unknown LRURefreshPhase: {phase}")
        unimplemented!("TreeComponent.refresh_lru")
    }

    /// Return a per-match stateful predicate deciding whether a node is a valid
    /// match boundary for this component.
    // Python reference — tree_component.py::TreeComponent.create_match_validator:
    //     @abstractmethod
    //     def create_match_validator(
    //         self, match_device_only: bool = False
    //     ) -> Callable[[UnifiedTreeNode], bool]:
    //         """Return a per-match stateful predicate that decides whether a node
    //         is a valid match boundary for this component.
    //         Called once per match_prefix; the returned closure may carry state.
    //         When match_device_only is true, host-backed nodes must not be accepted
    //         as valid match boundaries.
    //         - Full: returns True if the node has full component data.
    //         - SWA: tracks accumulated length since last gap; returns True only
    //           when the contiguous window reaches swa_sliding_window_size.
    //         - Mamba: returns True iff the node has mamba component data."""
    //         ...
    fn create_match_validator(
        &self,
        tree_core: &UnifiedTreeCore<K>,
        match_device_only: bool,
    ) -> Box<dyn FnMut(&UnifiedTreeCore<K>, NodeIdx_) -> bool>;

    /// Tree-side post-processing inside the match walk (no cache access).
    fn finalize_match_result_in_tree_core(
        &self,
        tree_core: &UnifiedTreeCore<K>,
        result: MatchResult,
        params: &MatchPrefixParams<'_, K>,
        value_chunks: &[Tensor],
        best_value_len: usize,
    ) -> MatchResult {
        result
    }

    /// Called per-node when an insert's key overlaps an existing node.
    /// Returns the index within `value_slice` from which this component
    /// consumed (took ownership of) the underlying KV pool slots.
    /// Returns `prefix_len` if nothing was consumed (default).
    /// The insert walk uses this to free only the non-consumed duplicate
    /// portion: `value_slice[dup_start..consumed_from]`.
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
        prefix_len
    }

    /// Called after `unevict_node_on_insert_` restores the base (Full) value
    /// on an evicted node. Aux components (e.g. SWA) override this to rebuild
    /// their own data from the freshly assigned base value when their entry
    /// is still tombstoned. Default no-op.
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
    }

    /// Finalize component data on the target (leaf) node after the insert
    /// walk completes. Called once per insert.
    /// - Full: no-op (full data is handled by `add_new_node_`).
    /// - SWA: for new leaves, checks whether the node straddles the SWA
    ///   eviction boundary (`swa_evicted_seqlen`). If so, splits the node
    ///   via `split_node_` — the parent becomes a tombstone (no SWA) and the
    ///   child (the deeper portion) receives SWA data. If the entire node
    ///   is within the window, sets SWA directly. If entirely outside,
    ///   leaves SWA as None (tombstone).
    /// - Mamba: sets the mamba component value from params, inserts into the
    ///   mamba LRU list, and increments evictable size. If the node already
    ///   has mamba data, resets its LRU position instead.
    fn commit_insert_component_data(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        is_new_leaf: bool,
        params: &InsertParams<'_, K>,
        result: &mut InsertResult,
        cache_actions: &mut Vec<CacheAction>,
    ) {
    }

    /// Evict shallow device checkpoints beyond the per-path state cap on the
    /// tail's root path; only the Mamba component caps its states.
    fn evict_excess_path_states(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        tail_node_id: NodeIdx_,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
    }

    /// Redistribute component data between `new_parent` and `child` when a node is
    /// split; `new_parent` is the newly created prefix node.
    // Python reference — tree_component.py::TreeComponent.redistribute_on_node_split:
    //     @abstractmethod
    //     def redistribute_on_node_split(
    //         self, new_parent: UnifiedTreeNode, child: UnifiedTreeNode
    //     ):
    //         """Redistribute component data between new_parent and child when a
    //         node is split. new_parent is the newly created prefix node.
    //         - Full: copies child's lock_ref to new_parent.
    //         - SWA: slices (or clones) the swa value for new_parent, copies
    //           lock_ref and component_uuid metadata, then syncs child's swa
    //           value with its (now-trimmed) full_value.
    //         - Mamba: sets new_parent's mamba value to None and lock_ref to 0
    //           (mamba data stays on the original leaf, not on prefix nodes)."""
    //         ...
    fn redistribute_on_node_split(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        new_parent_id: NodeIdx_,
        child_id: NodeIdx_,
    );

    /// Free this component's KV resources on a node being evicted; returns
    /// (device_freed, host_freed) token counts.
    // Python reference — tree_component.py::TreeComponent.evict_component:
    //     @abstractmethod
    //     def evict_component(
    //         self,
    //         node: UnifiedTreeNode,
    //         device_frees: dict[ComponentType, list[torch.Tensor]],
    //         host_frees: dict[ComponentType, list[torch.Tensor]],
    //         target: EvictLayer = EvictLayer.DEVICE,
    //     ) -> tuple[int, int]:
    //         """Free this component's KV resources on a node being evicted.
    //
    //         *target* controls which layer(s) to evict:
    //           - DEVICE: free device memory and tombstone (value = None).
    //                     Host data is untouched.
    //           - HOST:   free host memory (host_value = None).
    //                     Device data is untouched.
    //           - ALL:    free both device and host memory.
    //                     No tombstone — caller will delete the node.
    //
    //         Returns (device_freed, host_freed) token counts."""
    //         ...
    fn evict_component(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        target: EvictLayer,
    ) -> (usize, usize);

    /// Eviction priority on this node type; higher = evicted later, and evicting a
    /// component cascade-evicts every component of equal or lower priority.
    fn eviction_priority(&self, is_leaf: bool) -> i64 {
        0
    }

    /// Begin this component's device-eviction walk (build its cursor/heap).
    fn evict_device_start(&self, tree_core: &mut UnifiedTreeCore<K>, request_cnt: usize);

    /// Advance one eviction step and return a device leaf, if selected.
    ///
    /// Implementations must return after one allocator-relevant internal
    /// mutation so the caller can drain pending frees before continuing.
    fn evict_device_next_node(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        tracker: &mut HashMap<ComponentType, usize>,
        device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) -> Option<NodeIdx_>;

    /// Clear this component's device-eviction walk state.
    fn evict_device_end(&self, tree_core: &mut UnifiedTreeCore<K>);

    /// Increment component lock refs, protecting nodes from eviction.
    // Python reference — tree_component.py::TreeComponent.acquire_component_lock:
    //     @abstractmethod
    //     def acquire_component_lock(
    //         self,
    //         node: UnifiedTreeNode,
    //         result: IncLockRefResult,
    //         lock_host: bool = False,
    //     ) -> IncLockRefResult:
    //         """Increment component lock refs, protecting nodes from
    //         eviction. Updates evictable → protected size on first lock.
    //         - Full: path-lock — walks from node up to root, incrementing
    //           lock_ref on every ancestor.
    //         - SWA: path-lock — walks upward collecting swa values until the
    //           sliding window is filled; records a component_uuid at the
    //           boundary for release_component_lock to know where to stop.
    //         - Mamba: single-node lock — only increments lock_ref on the
    //           node itself (mamba state is per-leaf, not per-path).
    //
    //         When ``lock_host`` is True, the lock applies to host-side state:
    //         - Full: single-node host lock.
    //         - SWA: host window-lock with a dedicated host UUID boundary.
    //         - Mamba: single-node host lock with host LRU detach."""
    //         ...
    fn acquire_component_lock(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        result: IncLockRefResult,
        lock_host: bool,
    ) -> IncLockRefResult;

    /// Decrement component lock refs, un-protecting nodes.
    // Python reference — tree_component.py::TreeComponent.release_component_lock:
    //     @abstractmethod
    //     def release_component_lock(
    //         self,
    //         node: UnifiedTreeNode,
    //         params: Optional[DecLockRefParams],
    //         lock_host: bool = False,
    //     ) -> None:
    //         """Decrement component lock refs, un-protecting nodes.
    //         Updates protected → evictable size when lock_ref drops to 0.
    //         - Full: path-unlock — walks from node up to root, decrementing
    //           lock_ref on every ancestor.
    //         - SWA: path-unlock — walks upward, stopping at the node whose
    //           component_uuid matches the one recorded during acquire.
    //         - Mamba: single-node unlock — only decrements lock_ref on the
    //           node itself.
    //
    //         When ``lock_host`` is True, the inverse host-side semantics apply."""
    //         ...
    fn release_component_lock(
        &self,
        tree_core: &mut UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        params: Option<&DecLockRefParams>,
        lock_host: bool,
    );

    /// Early-release the SWA lock along [node, swa_uuid_for_lock] while leaving
    /// the other components' locks intact; only the SWA component supports it.
    fn release_window_lock(
        &self,
        _tree_core: &mut UnifiedTreeCore<K>,
        _node_id: NodeIdx_,
        _swa_uuid_for_lock: Option<i64>,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
        unimplemented!("release_window_lock is SWA-only")
    }

    /// Build transfer descriptors for this component in the given phase; None when
    /// the component has nothing to transfer.
    fn build_hicache_transfers(
        &self,
        tree_core: &UnifiedTreeCore<K>,
        node_id: NodeIdx_,
        phase: CacheTransferPhase,
        mamba_pool_idx: Option<Tensor>,
        host_indices: Option<Tensor>,
        token_ids: Option<&[i64]>,
        prefetch_tokens: usize,
        last_hash: Option<&str>,
    ) -> Result<Option<Vec<PoolTransfer>>, TreeCoreRuntimeError> {
        // Python reference — tree_component.py::TreeComponent.build_hicache_transfers:
        //     def build_hicache_transfers(
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
        //         """Build transfer descriptors for this component in the given phase.
        //         Returns None if the component has nothing to transfer."""
        //         return None
        unimplemented!("TreeComponent.build_hicache_transfers")
    }

    /// Post-transfer bookkeeping: store host indices, update LRU, etc.
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
        // Python reference — tree_component.py::TreeComponent.commit_hicache_transfer:
        //     def commit_hicache_transfer(
        //         self,
        //         node: UnifiedTreeNode,
        //         phase: CacheTransferPhase,
        //         transfers: list[PoolTransfer] = (),
        //         *,
        //         cache_actions: list[CacheAction | ComponentAction],
        //         insert_result: Optional[InsertResult] = None,
        //         pool_storage_result: Optional[PoolTransferResult] = None,
        //     ) -> None:
        //         """Post-transfer bookkeeping: store host indices, update LRU, etc."""
        //         pass
        unimplemented!("TreeComponent.commit_hicache_transfer")
    }

    /// Reclaim host values that coexist with device values before ordinary
    /// host eviction. Called only under the write-back policy.
    fn reclaim_coexisting_host_values(
        &self,
        _tree_core: &mut UnifiedTreeCore<K>,
        _num_tokens: usize,
        _tracker: &mut HashMap<ComponentType, usize>,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
    }

    /// Evict from this component's host-side resources.
    /// Called by HostPoolGroup when the host pool is full.
    /// Default no-op for components without host storage.
    fn drive_host_eviction(
        &self,
        _tree_core: &mut UnifiedTreeCore<K>,
        _num_tokens: usize,
        _tracker: &mut HashMap<ComponentType, usize>,
        _device_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
        _host_frees: &mut HashMap<ComponentType, Vec<Tensor>>,
    ) {
    }
}

// Tree component types.

/// The tree components; discriminants define the per-component array indexes.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum ComponentType {
    Full = 0,
    Swa = 1,
    Mamba = 2,
}

/// Short call-site aliases for the component types.
pub const FULL: ComponentType = ComponentType::Full;
pub const SWA: ComponentType = ComponentType::Swa;
pub const MAMBA: ComponentType = ComponentType::Mamba;

/// The base component every tree runs; the others are auxiliary.
pub const BASE_COMPONENT_TYPE: ComponentType = ComponentType::Full;

/// Slots per tier — the arrays are sized to this, not the enabled subset.
pub const NUM_COMPONENT_TYPES: usize = ComponentType::Mamba as usize + 1;

impl ComponentType {
    /// Index into a per-component array.
    pub const fn idx(self) -> usize {
        self as usize
    }

    /// Whether the component stores one state slot per node (Mamba) instead of
    /// one row per key atom.
    pub fn single_value_per_node(self) -> bool {
        matches!(self, ComponentType::Mamba)
    }

    /// The component at a per-component array index; panics out of range.
    pub fn from_idx(idx: usize) -> ComponentType {
        match idx {
            0 => ComponentType::Full,
            1 => ComponentType::Swa,
            2 => ComponentType::Mamba,
            _ => panic!("from_idx: {idx} is not a component index"),
        }
    }
}
#[cfg(test)]
#[path = "../tests/components/base.rs"]
mod tests;
