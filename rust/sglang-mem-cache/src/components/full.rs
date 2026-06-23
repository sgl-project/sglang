//! Tree node component for device FULL values.

use super::{Component, IncLockRefResult, MatchValidator};
use crate::component_type::ComponentType;
use super::{EvictRequest, EvictResult, Slot, evict_full};
use crate::tree_node_pool::{ChildKeyType, NodeIdx, TreeNode, TreeNodePool};

#[derive(Default)]
pub struct FullComponent;

impl FullComponent {
    pub fn new() -> Self {
        Self
    }
}

impl<K: ChildKeyType> Component<K> for FullComponent {
    fn create_match_validator(&self) -> Option<Box<dyn MatchValidator<K>>> {
        None
    }

    /// Bump each node's lock_ref counter from `node_idx` up to root.
    fn inc_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
    ) -> Option<IncLockRefResult> {
        let mut delta: i64 = 0;
        let mut current = node_idx;
        while let Some(parent) = pool.get(current).parent() {
            delta += FullSlot::inc_lock_ref(pool, current);
            current = parent;
        }
        Some(IncLockRefResult {
            delta,
            swa_uuid_for_lock: None,
        })
    }

    /// Decrease each node's lock_ref counter from `node_idx` up to root.
    fn dec_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
        _swa_uuid_for_lock: Option<u64>,
    ) -> Option<i64> {
        let mut delta: i64 = 0;
        let mut current = node_idx;
        while let Some(parent) = pool.get(current).parent() {
            delta += FullSlot::dec_lock_ref(pool, current);
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

    /// Bump device FULL LRU.
    fn bump_mru_walk(&self, pool: &mut TreeNodePool<K>, node_idx: NodeIdx) {
        FullSlot::bump_mru_walk(pool, node_idx);
    }

    /// Redistribute node values from `child_idx` to `new_parent_idx` after a split.
    fn redistribute_on_node_split(
        &self,
        pool: &mut TreeNodePool<K>,
        new_parent_idx: NodeIdx,
        child_idx: NodeIdx,
        split_len: usize,
    ) {
        // Device-only cache always has the value present.
        FullSlot::redistribute_on_split(pool, new_parent_idx, child_idx, split_len, true);
    }
}

/// LRU for FULL values.
pub struct FullSlot;

impl Slot for FullSlot {
    const COMPONENT: ComponentType = ComponentType::Full;
    const NAME: &'static str = "Full";

    /// Evictable when unreferenced and no child holds a FULL device value.
    fn is_evictable<K: ChildKeyType>(n: &TreeNode<K>) -> bool {
        Self::lock_ref(n) == 0 && n.num_children_with_device_full == 0
    }

    /// Credit the parent's `num_children_with_device_full`.
    fn postprocess_set_value<K: ChildKeyType>(pool: &mut TreeNodePool<K>, parent_idx: NodeIdx) {
        pool.get_mut(parent_idx).num_children_with_device_full += 1;
    }

    /// Debit the parent's `num_children_with_device_full`.
    fn postprocess_take_value<K: ChildKeyType>(pool: &mut TreeNodePool<K>, parent_idx: NodeIdx) {
        let count = &mut pool.get_mut(parent_idx).num_children_with_device_full;
        #[allow(clippy::expect_used, reason = "underflow = dec without matching inc")]
        let new = count
            .checked_sub(1)
            .expect("num_children_with_device_full underflow");
        *count = new;
    }

    fn inc_lock_ref<K: ChildKeyType>(pool: &mut TreeNodePool<K>, node_idx: NodeIdx) -> i64 {
        let node = pool.get_mut(node_idx);
        #[allow(
            clippy::expect_used,
            reason = "u32 lock_ref overflow effectively impossible"
        )]
        let new = Self::lock_ref(node)
            .checked_add(1)
            .expect("FullSlot::inc_lock_ref: u32 overflow");
        Self::set_lock_ref(node, new);
        if new != 1 {
            return 0;
        }
        let key_len = node.key().len();
        let state = Self::pool_state_mut(pool);
        state.unlocked_size -= key_len;
        state.locked_size += key_len;
        -(key_len as i64)
    }

    fn dec_lock_ref<K: ChildKeyType>(pool: &mut TreeNodePool<K>, node_idx: NodeIdx) -> i64 {
        let node = pool.get_mut(node_idx);
        let component_idx = Self::COMPONENT as usize;
        #[allow(clippy::expect_used, reason = "underflow = dec without matching inc")]
        let new = Self::lock_ref(node)
            .checked_sub(1)
            .expect("FullSlot::dec_lock_ref: underflow");
        // FULL's lock_ref must stay >= every other component's.
        assert!(
            node.components
                .iter()
                .enumerate()
                .all(|(i, c)| i == component_idx || c.lock_ref <= new),
            "FullSlot::dec_lock_ref: prospective full_lock_ref {new} not >= \
             every other component's lock_ref ({:?}) — caller must dec non-FULL \
             components on this node before FULL",
            node.components
                .iter()
                .map(|c| c.lock_ref)
                .collect::<Vec<_>>(),
        );
        Self::set_lock_ref(node, new);
        if new != 0 {
            return 0;
        }
        let key_len = node.key().len();
        let state = Self::pool_state_mut(pool);
        state.locked_size -= key_len;
        state.unlocked_size += key_len;
        key_len as i64
    }
}
