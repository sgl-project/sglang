//! Tree node component which maintains device FULL values.

use super::{Component, IncLockRefResult, MatchValidator};
use crate::component_type::ComponentType;
use crate::tree_node_lru::{EvictRequest, EvictResult, FullLRUSlot, LRUSlot, evict_full};
use crate::tree_node_pool::{ChildKeyType, NodeIdx, TreeNodePool};

#[derive(Default)]
pub struct FullComponent;

impl FullComponent {
    pub fn new() -> Self {
        Self
    }
}

impl<K: ChildKeyType> Component<K> for FullComponent {
    fn create_match_validator(&self) -> Option<Box<dyn MatchValidator<K>>> {
        // FULL has no boundary-gating; `None` lets the orchestrator skip the
        // per-node validate() call and its allocation.
        None
    }

    /// Walk from `node_idx` up to root (but not including) to bump each node's
    /// lock_ref counter.
    fn inc_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
    ) -> Option<IncLockRefResult> {
        let mut delta: i64 = 0;
        let mut current = node_idx;
        while let Some(parent) = pool.get(current).parent() {
            delta += FullLRUSlot::inc_lock_ref(pool, current);
            current = parent;
        }
        Some(IncLockRefResult {
            delta,
            swa_uuid_for_lock: None,
        })
    }

    /// Walk from `node_idx` up to root (but not including) to decrease each
    /// node's lock_ref counter.
    fn dec_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
        _swa_uuid_for_lock: Option<u64>,
    ) -> Option<i64> {
        let mut delta: i64 = 0;
        let mut current = node_idx;
        while let Some(parent) = pool.get(current).parent() {
            delta += FullLRUSlot::dec_lock_ref(pool, current);
            current = parent;
        }
        Some(delta)
    }

    /// Evict FULL values and remove unreferenced leaves.
    fn evict(&self, pool: &mut TreeNodePool<K>, request: &EvictRequest, result: &mut EvictResult) {
        let ct = ComponentType::Full as usize;
        let target = request.num_tokens[ct];
        let already = result.evicted[ct];
        if already < target {
            evict_full(pool, target - already, result);
        }
    }

    /// Bump device FULL LRU; a node-level bump is skipped when its value is
    /// absent (not-in-list).
    fn bump_mru_walk(&self, pool: &mut TreeNodePool<K>, node_idx: NodeIdx) {
        FullLRUSlot::bump_mru_walk(pool, node_idx);
    }

    /// Redistribute node values from `child_idx` to `new_parent_idx`
    /// (`child_idx` was just structurally split into
    /// `new_parent_idx -> child_idx` at `split_len`).
    fn redistribute_on_node_split(
        &self,
        pool: &mut TreeNodePool<K>,
        new_parent_idx: NodeIdx,
        child_idx: NodeIdx,
        split_len: usize,
    ) {
        // A device-only cache always has the value present, so require it.
        FullLRUSlot::redistribute_on_split(pool, new_parent_idx, child_idx, split_len, true);
    }
}
