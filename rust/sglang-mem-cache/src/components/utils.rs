//! Eviction helpers shared across components.

use tch::Tensor;

use super::{EvictResult, FullSlot, MambaSlot, Slot, SwaSlot};
use crate::tree_node_pool::{ChildKeyType, NodeIdx, TreeNodePool};

/// Evict a node's FULL device value.
#[must_use = "free the returned value to avoid KV leakage"]
pub(crate) fn evict_full_value<K: ChildKeyType>(
    pool: &mut TreeNodePool<K>,
    idx: NodeIdx,
    result: &mut EvictResult,
) -> Option<Tensor> {
    let value_len = FullSlot::value_len(pool.get(idx));
    let value = FullSlot::node_state_mut(pool.get_mut(idx)).value.take();
    if value.is_some() {
        result.evicted[FullSlot::COMPONENT as usize] += value_len;
        #[allow(clippy::expect_used, reason = "eviction never selects the root")]
        let parent_idx = pool
            .get(idx)
            .parent()
            .expect("evict_full_value never runs on the root");
        FullSlot::postprocess_take_value(pool, parent_idx);
    }
    FullSlot::lru_remove(pool, idx);
    FullSlot::pool_state_mut(pool).unlocked_size -= value_len;
    value
}

/// Best-effort eviction of up to `num_tokens` FULL values in LRU order.
pub(crate) fn evict_full<K: ChildKeyType>(
    pool: &mut TreeNodePool<K>,
    num_tokens: usize,
    result: &mut EvictResult,
) {
    if num_tokens == 0 {
        return;
    }
    let ct = FullSlot::COMPONENT as usize;
    let target = result.evicted[ct] + num_tokens;
    let mut victim = FullSlot::next_evictable(pool);

    while result.evicted[ct] < target {
        let Some(idx) = victim else { break };

        let next_after = FullSlot::next_evictable_after(pool, idx);

        #[allow(clippy::expect_used, reason = "FULL LRU never includes root")]
        let parent_idx = pool
            .get(idx)
            .parent()
            .expect("evict: LRU returned a node without a parent (root spliced into LRU?)");

        pool.evict_leaf(idx, result);

        // Capture before the cascade frees `parent_idx`.
        let parent_was_evictable = FullSlot::is_evictable(pool.get(parent_idx));

        let pre_cascade_evicted = result.evicted[ct];
        if pool.has_swa_component() {
            iteratively_delete_tombstone_leaf::<K, SwaSlot>(pool, parent_idx, result);
        }
        if pool.has_mamba_component() {
            iteratively_delete_tombstone_leaf::<K, MambaSlot>(pool, parent_idx, result);
        }
        let cascade_freed_anything = result.evicted[ct] > pre_cascade_evicted;

        victim = if parent_was_evictable || cascade_freed_anything {
            // Restart from the tail: a newly-evictable node may sit ahead of `next_after`.
            FullSlot::next_evictable(pool)
        } else {
            next_after
        };
    }
}

/// Best-effort eviction of up to `num_tokens` of component `S`'s values in LRU order.
pub(crate) fn evict_non_full<K: ChildKeyType, S: Slot>(
    pool: &mut TreeNodePool<K>,
    num_tokens: usize,
    result: &mut EvictResult,
) {
    if num_tokens == 0 {
        return;
    }
    let component = S::COMPONENT;
    let ct = component as usize;
    let target = result.evicted[ct] + num_tokens;
    let mut victim = S::next_evictable(pool);

    while result.evicted[ct] < target {
        let Some(idx) = victim else { break };
        let next_after = S::next_evictable_after(pool, idx);

        let node = pool.get(idx);
        assert!(
            S::has_value(node),
            "{component:?} LRU invariant violated: node {idx} has no value",
        );
        let delta = S::value_len(node);
        let is_leaf = node.is_leaf();

        S::take_value(pool.get_mut(idx), result);
        S::lru_remove(pool, idx);
        S::pool_state_mut(pool).unlocked_size -= delta;

        if is_leaf {
            iteratively_delete_tombstone_leaf::<K, S>(pool, idx, result);
        }
        victim = next_after;
    }
}

/// Iteratively clean up a tombstone leaf chain from `node_idx`.
pub(crate) fn iteratively_delete_tombstone_leaf<K: ChildKeyType, S: Slot>(
    pool: &mut TreeNodePool<K>,
    mut node_idx: NodeIdx,
    result: &mut EvictResult,
) {
    loop {
        let node = pool.get(node_idx);

        if S::has_value(node) || !node.is_leaf() || node.is_root() || FullSlot::lock_ref(node) > 0 {
            break;
        }
        assert_eq!(
            S::lock_ref(node),
            0,
            "tombstone must have this component's lock_ref == 0",
        );

        #[allow(
            clippy::expect_used,
            reason = "is_root() check above guarantees parent"
        )]
        let parent = node.parent().expect("just checked node.is_root() above");

        pool.evict_leaf(node_idx, result);
        node_idx = parent;
    }
}

/// Lock-ref increment shared by non-FULL components.
pub(crate) fn inc_lock_ref_non_full<K: ChildKeyType, S: Slot>(
    pool: &mut TreeNodePool<K>,
    node_idx: NodeIdx,
    enforce_full_cap: bool,
) -> i64 {
    let component = S::COMPONENT;
    let node = pool.get_mut(node_idx);
    let new = S::lock_ref(node) + 1;
    if enforce_full_cap {
        let full_ref = FullSlot::lock_ref(node);
        assert!(
            new <= full_ref,
            "{component:?}Slot::inc_lock_ref: prospective lock_ref {new} exceeds \
             full_lock_ref {full_ref} — caller must inc FULL on this node first",
        );
    }
    if new == 1 {
        assert!(
            S::has_value(node),
            "{component:?}Slot::inc_lock_ref called on a node without value \
             populated (node_idx={node_idx})",
        );
    }
    S::set_lock_ref(node, new);
    if new != 1 {
        return 0;
    }
    let delta = S::value_len(node);
    let state = S::pool_state_mut(pool);
    state.unlocked_size -= delta;
    state.locked_size += delta;
    -(delta as i64)
}

/// Lock-ref decrement shared by non-FULL components.
pub(crate) fn dec_lock_ref_non_full<K: ChildKeyType, S: Slot>(
    pool: &mut TreeNodePool<K>,
    node_idx: NodeIdx,
) -> i64 {
    let component = S::COMPONENT;
    let node = pool.get_mut(node_idx);
    let new = S::lock_ref(node) - 1;
    if new == 0 {
        assert!(
            S::has_value(node),
            "{component:?}Slot::dec_lock_ref called on a node without value \
             populated (node_idx={node_idx})",
        );
    }
    S::set_lock_ref(node, new);
    if new != 0 {
        return 0;
    }
    let delta = S::value_len(node);
    let state = S::pool_state_mut(pool);
    state.unlocked_size += delta;
    state.locked_size -= delta;
    delta as i64
}
