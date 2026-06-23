//! Per-slot intrusive LRU machinery layered on top of `TreeNodePool`.
//!
//! Each `LRUSlot` owns a doubly-linked list over one component's per-node link
//! fields, bounded by a permanent `(head_sentinel, tail_sentinel)` pair. The
//! list algorithm lives as default methods on the trait; each slot only routes
//! accessors to its component's fields.

use tch::Tensor;

use crate::component_type::{ComponentType, NUM_COMPONENT_TYPES};
use crate::deferred_action::DeferredAction;
use crate::error::RadixCacheRuntimeError;
use crate::tree_node_pool::{ChildKeyType, ComponentNodeState, NodeIdx, TreeNode, TreeNodePool};

/// Per-component eviction budget, indexed by `ct as usize`. 0 means no
/// eviction requested for that component.
#[derive(Default, Clone, Copy)]
pub struct EvictRequest {
    pub num_tokens: [usize; NUM_COMPONENT_TYPES],
}

/// Result of node eviction.
#[derive(Default)]
pub struct EvictResult {
    /// Tensors to free per component.
    pub freed: [Vec<Tensor>; NUM_COMPONENT_TYPES],
    /// Evict units per component: token count for FULL/SWA, state-slot
    /// count for Mamba (at most one per tree node).
    pub evicted: [usize; NUM_COMPONENT_TYPES],
    /// Actions that need to be coordinated after Rust radix tree eviction.
    pub deferred_actions: Vec<DeferredAction>,
}

/// Per-component pool-level metadata.
#[derive(Clone, Copy)]
pub struct ComponentPoolState {
    /// MRU fake end.
    pub head_sentinel: NodeIdx,
    /// LRU fake end.
    pub tail_sentinel: NodeIdx,
    /// Number of unlocked units (lock_ref == 0).
    pub unlocked_size: usize,
    /// Number of locked units (lock_ref > 0).
    pub locked_size: usize,
}

impl Default for ComponentPoolState {
    fn default() -> Self {
        // Fake default that errors out on access before `init` runs.
        Self {
            head_sentinel: NodeIdx::MAX,
            tail_sentinel: NodeIdx::MAX,
            unlocked_size: 0,
            locked_size: 0,
        }
    }
}

/// Per-node intrusive LRU state for one LRU slot.
#[derive(Default, Clone, Copy)]
pub struct LRUData {
    pub prev: NodeIdx,
    pub next: NodeIdx,
    pub in_list: bool,
}

/// Marker trait selecting the right LRU slot to use.
pub trait LRUSlot: Sized {
    /// Component this slot belongs to. Used as the index to access
    /// pool-level and node-level metadata.
    const COMPONENT: ComponentType;

    /// Slot identifier (e.g. "Full", "Swa", "Mamba"), used in the
    /// duplicate-value error.
    const NAME: &'static str;

    /// Whether the node value is evictable. Default: this slot is unlocked;
    /// FullLRUSlot additionally requires the node to be a leaf.
    fn is_evictable<K: ChildKeyType>(node: &TreeNode<K>) -> bool {
        Self::lock_ref(node) == 0
    }

    /// Bookkeeping post-processing on the value's parent after `set_value`
    /// (e.g. the parent's full-device child count). Default no-op.
    fn postprocess_set_value<K: ChildKeyType>(_pool: &mut TreeNodePool<K>, _parent_idx: NodeIdx) {}

    /// Like `postprocess_set_value`, but after `take_value`. Default no-op.
    fn postprocess_take_value<K: ChildKeyType>(_pool: &mut TreeNodePool<K>, _parent_idx: NodeIdx) {}

    fn node_components<K: ChildKeyType>(
        node: &TreeNode<K>,
    ) -> &[ComponentNodeState; NUM_COMPONENT_TYPES] {
        &node.components
    }
    fn node_components_mut<K: ChildKeyType>(
        node: &mut TreeNode<K>,
    ) -> &mut [ComponentNodeState; NUM_COMPONENT_TYPES] {
        &mut node.components
    }

    /// This slot's component state.
    fn node_state<K: ChildKeyType>(node: &TreeNode<K>) -> &ComponentNodeState {
        &Self::node_components(node)[Self::COMPONENT as usize]
    }
    fn node_state_mut<K: ChildKeyType>(node: &mut TreeNode<K>) -> &mut ComponentNodeState {
        &mut Self::node_components_mut(node)[Self::COMPONENT as usize]
    }

    fn data<K: ChildKeyType>(node: &TreeNode<K>) -> &LRUData {
        &Self::node_state(node).lru_data
    }
    fn data_mut<K: ChildKeyType>(node: &mut TreeNode<K>) -> &mut LRUData {
        &mut Self::node_state_mut(node).lru_data
    }

    fn value<K: ChildKeyType>(node: &TreeNode<K>) -> Option<&Tensor> {
        Self::node_state(node).value.as_ref()
    }

    fn has_value<K: ChildKeyType>(node: &TreeNode<K>) -> bool {
        Self::value(node).is_some()
    }

    fn set_value<K: ChildKeyType>(
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
        value: Tensor,
    ) -> Result<(), RadixCacheRuntimeError> {
        if Self::has_value(pool.get(node_idx)) {
            return Err(RadixCacheRuntimeError::DuplicateValueSet {
                node_idx,
                slot: Self::NAME,
            });
        }
        let parent_idx = pool
            .get(node_idx)
            .parent()
            .ok_or(RadixCacheRuntimeError::ValueSetOnRoot { slot: Self::NAME })?;
        Self::node_state_mut(pool.get_mut(node_idx)).value = Some(value);
        Self::postprocess_set_value(pool, parent_idx);
        Ok(())
    }

    fn replace_value<K: ChildKeyType>(
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
        value: Tensor,
    ) {
        Self::node_state_mut(pool.get_mut(node_idx)).value = Some(value);
    }

    /// Size unit per node — token count by default; Mamba overrides to 1.
    fn value_len<K: ChildKeyType>(node: &TreeNode<K>) -> usize {
        node.key().len()
    }

    fn lock_ref<K: ChildKeyType>(node: &TreeNode<K>) -> u32 {
        Self::node_state(node).lock_ref
    }

    fn set_lock_ref<K: ChildKeyType>(node: &mut TreeNode<K>, value: u32) {
        Self::node_state_mut(node).lock_ref = value;
    }

    fn pool_components<K: ChildKeyType>(
        pool: &TreeNodePool<K>,
    ) -> &[ComponentPoolState; NUM_COMPONENT_TYPES] {
        &pool.components
    }
    fn pool_components_mut<K: ChildKeyType>(
        pool: &mut TreeNodePool<K>,
    ) -> &mut [ComponentPoolState; NUM_COMPONENT_TYPES] {
        &mut pool.components
    }

    /// This slot's pool-level state.
    fn pool_state<K: ChildKeyType>(pool: &TreeNodePool<K>) -> &ComponentPoolState {
        &Self::pool_components(pool)[Self::COMPONENT as usize]
    }
    fn pool_state_mut<K: ChildKeyType>(pool: &mut TreeNodePool<K>) -> &mut ComponentPoolState {
        &mut Self::pool_components_mut(pool)[Self::COMPONENT as usize]
    }

    fn head_sentinel<K: ChildKeyType>(pool: &TreeNodePool<K>) -> NodeIdx {
        Self::pool_state(pool).head_sentinel
    }
    fn tail_sentinel<K: ChildKeyType>(pool: &TreeNodePool<K>) -> NodeIdx {
        Self::pool_state(pool).tail_sentinel
    }

    fn unlocked_size<K: ChildKeyType>(pool: &TreeNodePool<K>) -> usize {
        Self::pool_state(pool).unlocked_size
    }
    fn locked_size<K: ChildKeyType>(pool: &TreeNodePool<K>) -> usize {
        Self::pool_state(pool).locked_size
    }
    /// Total size (unlocked + locked).
    fn total_size<K: ChildKeyType>(pool: &TreeNodePool<K>) -> usize {
        let s = Self::pool_state(pool);
        s.unlocked_size + s.locked_size
    }

    // ---- Provided: algorithm, written once ----

    /// Splice `left` and `right` together as neighbors. The single mutation
    /// primitive — every other list op composes from this.
    fn connect<K: ChildKeyType>(pool: &mut TreeNodePool<K>, left: NodeIdx, right: NodeIdx) {
        Self::data_mut(pool.get_mut(left)).next = right;
        Self::data_mut(pool.get_mut(right)).prev = left;
    }

    /// Allocate the sentinel pair as an empty list and store it on the
    /// pool's per-component state. Called once per slot from `TreeNodePool::new`.
    fn init<K: ChildKeyType>(pool: &mut TreeNodePool<K>) {
        let head = pool.alloc_sentinel();
        let tail = pool.alloc_sentinel();
        // Self-loop the unread directions (head.prev, tail.next) so debug
        // prints don't show the placeholder zeros from `new_sentinel`.
        Self::data_mut(pool.get_mut(head)).prev = head;
        Self::data_mut(pool.get_mut(tail)).next = tail;
        Self::connect::<K>(pool, head, tail);
        let state = Self::pool_state_mut(pool);
        state.head_sentinel = head;
        state.tail_sentinel = tail;
    }

    /// Move `idx` to the MRU end, inserting it if not already in the list.
    /// Handles both first-time insert and bump-on-access. Caller must
    /// guarantee `idx` is not a sentinel.
    fn bump_mru<K: ChildKeyType>(pool: &mut TreeNodePool<K>, idx: NodeIdx) {
        if Self::data(pool.get(idx)).in_list {
            Self::remove(pool, idx);
        }
        let head = Self::head_sentinel(pool);
        let first_real = Self::data(pool.get(head)).next;
        Self::connect::<K>(pool, head, idx);
        Self::connect::<K>(pool, idx, first_real);
        Self::data_mut(pool.get_mut(idx)).in_list = true;
    }

    /// Unlink `idx` from the list, marking it not-in-list. Caller must
    /// guarantee `idx` is currently in the list and is not a sentinel.
    fn remove<K: ChildKeyType>(pool: &mut TreeNodePool<K>, idx: NodeIdx) {
        let data = Self::data_mut(pool.get_mut(idx));
        let prev = data.prev;
        let next = data.next;
        data.in_list = false;
        Self::connect::<K>(pool, prev, next);
    }

    /// Re-splice both halves of a `split_node` into the LRU at MRU. Bumps
    /// `parent` first, then `child`, so `child` ends up more recent than
    /// `parent` (leaf-MRU), matching `bump_mru_walk`'s layout.
    fn bump_mru_split<K: ChildKeyType>(
        pool: &mut TreeNodePool<K>,
        parent: NodeIdx,
        child: NodeIdx,
    ) {
        Self::bump_mru(pool, parent);
        Self::bump_mru(pool, child);
    }

    /// Redistribute lock_ref, values and update LRUs after a structural node split.
    fn redistribute_on_split<K: ChildKeyType>(
        pool: &mut TreeNodePool<K>,
        new_parent: NodeIdx,
        child: NodeIdx,
        split_len: usize,
        require_value: bool,
    ) {
        if let Some(value) = Self::value(pool.get(child)) {
            let lock_ref = Self::lock_ref(pool.get(child));
            let total_len = value.size()[0];
            let prefix = value.narrow(0, 0, split_len as i64);
            let suffix = value.narrow(0, split_len as i64, total_len - split_len as i64);
            Self::set_lock_ref(pool.get_mut(new_parent), lock_ref);
            Self::replace_value(pool, new_parent, prefix);
            Self::replace_value(pool, child, suffix);
            Self::bump_mru_split(pool, new_parent, child);
        } else {
            assert!(
                !require_value,
                "redistribute_on_split: value required but absent — a device FULL \
                 value must exist on a node split",
            );
        }
    }

    /// Walk from `leaf` up its parents (excluding root), bumping each
    /// in-list node into a contiguous MRU run that keeps the leaf more
    /// recent than its ancestors. Out-of-list nodes (e.g. SWA tombstones)
    /// are skipped, not revived. The access-bump primitive after a match.
    fn bump_mru_walk<K: ChildKeyType>(pool: &mut TreeNodePool<K>, leaf: NodeIdx) {
        let mut chain_first: Option<NodeIdx> = None;
        let mut chain_last: Option<NodeIdx> = None;
        let mut cur = leaf;

        while let Some(parent) = pool.get(cur).parent() {
            // Don't revive nodes intentionally kept out of the list
            // (e.g. SWA tombstones).
            if !Self::data(pool.get(cur)).in_list {
                cur = parent;
                continue;
            }
            Self::remove(pool, cur);
            match chain_last {
                Some(prev_last) => Self::connect::<K>(pool, prev_last, cur),
                None => chain_first = Some(cur),
            }
            Self::data_mut(pool.get_mut(cur)).in_list = true;
            chain_last = Some(cur);
            cur = parent;
        }

        // Both are `Some` iff at least one node was bumped.
        if let (Some(first), Some(last)) = (chain_first, chain_last) {
            // Read head.next only now: it still points at the first
            // non-bumped node since the loop never wrote it.
            let head = Self::head_sentinel(pool);
            let original_first = Self::data(pool.get(head)).next;
            Self::connect::<K>(pool, head, first);
            Self::connect::<K>(pool, last, original_first);
        }
    }

    /// Walk from `from.prev` toward MRU, returning the first node passing
    /// `is_evictable`. Stops at `head_sentinel`, so sentinels are never
    /// returned. `from` is the `tail_sentinel` to seed an eviction loop or
    /// a prior victim to step it incrementally.
    fn next_evictable_after<K: ChildKeyType>(
        pool: &TreeNodePool<K>,
        from: NodeIdx,
    ) -> Option<NodeIdx> {
        let head_sentinel = Self::head_sentinel(pool);
        let mut cur = Self::data(pool.get(from)).prev;
        while cur != head_sentinel {
            let node = pool.get(cur);
            if Self::is_evictable::<K>(node) {
                return Some(cur);
            }
            cur = Self::data(node).prev;
        }
        None
    }

    /// LRU evictable target — seeds an eviction loop by walking from the
    /// tail sentinel.
    fn next_evictable<K: ChildKeyType>(pool: &TreeNodePool<K>) -> Option<NodeIdx> {
        Self::next_evictable_after::<K>(pool, Self::tail_sentinel(pool))
    }

    /// Take this slot's freed handle out of `node` and record it into
    /// `result`. No-op when this slot has no value at the node. Default
    /// works when the freed handle is the slot's own value (FULL, Mamba);
    /// slots whose freed handle differs (SWA) override.
    fn take_value<K: ChildKeyType>(node: &mut TreeNode<K>, result: &mut EvictResult) {
        let ct = Self::COMPONENT as usize;
        if let Some(t) = node.components[ct].value.take() {
            result.evicted[ct] += t.size()[0] as usize;
            result.freed[ct].push(t);
        }
    }

    /// Increase `lock_ref` of the given slot, and return the resulting
    /// delta to `unlocked_size`. Per-slot impls choose the unit
    /// (tokens vs slot count) via `value_len`.
    fn inc_lock_ref<K: ChildKeyType>(pool: &mut TreeNodePool<K>, node_idx: NodeIdx) -> i64;

    /// Decrease `lock_ref` of the given slot, and return the resulting
    /// delta to `unlocked_size`. Panics on underflow.
    fn dec_lock_ref<K: ChildKeyType>(pool: &mut TreeNodePool<K>, node_idx: NodeIdx) -> i64;

    /// Lock-ref increment shared by non-FULL components. When
    /// `enforce_full_cap` is true, asserts `lock_ref + 1 <= full_lock_ref`
    /// to catch an unexpected lock-ref state (non-FULL exceeding FULL).
    fn inc_lock_ref_non_full<K: ChildKeyType>(
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
        enforce_full_cap: bool,
    ) -> i64 {
        let component = Self::COMPONENT;
        let node = pool.get_mut(node_idx);
        #[allow(clippy::panic, reason = "u32 lock_ref overflow effectively impossible")]
        let new = Self::lock_ref(node)
            .checked_add(1)
            .unwrap_or_else(|| panic!("{component:?}LRUSlot::inc_lock_ref: u32 overflow"));
        if enforce_full_cap {
            let full_ref = FullLRUSlot::lock_ref(node);
            assert!(
                new <= full_ref,
                "{component:?}LRUSlot::inc_lock_ref: prospective lock_ref {new} exceeds \
                 full_lock_ref {full_ref} — caller must inc FULL on this node first",
            );
        }
        if new == 1 {
            assert!(
                Self::has_value(node),
                "{component:?}LRUSlot::inc_lock_ref called on a node without value \
                 populated (node_idx={node_idx})",
            );
        }
        Self::set_lock_ref(node, new);
        if new != 1 {
            return 0;
        }
        let delta = Self::value_len(node);
        let state = Self::pool_state_mut(pool);
        state.unlocked_size -= delta;
        state.locked_size += delta;
        -(delta as i64)
    }

    /// Shared body for non-FULL components: maintain `lock_ref`, panic
    /// on underflow, shift `value_len` from `locked_size` back to
    /// `unlocked_size` on the 1→0 transition.
    fn dec_lock_ref_non_full<K: ChildKeyType>(
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
    ) -> i64 {
        let component = Self::COMPONENT;
        let node = pool.get_mut(node_idx);
        #[allow(clippy::panic, reason = "underflow = dec without matching inc")]
        let new = Self::lock_ref(node)
            .checked_sub(1)
            .unwrap_or_else(|| panic!("{component:?}LRUSlot::dec_lock_ref: underflow"));
        if new == 0 {
            assert!(
                Self::has_value(node),
                "{component:?}LRUSlot::dec_lock_ref called on a node without value \
                 populated (node_idx={node_idx})",
            );
        }
        Self::set_lock_ref(node, new);
        if new != 0 {
            return 0;
        }
        let delta = Self::value_len(node);
        let state = Self::pool_state_mut(pool);
        state.unlocked_size += delta;
        state.locked_size -= delta;
        delta as i64
    }
}

/// LRU for FULL values.
pub struct FullLRUSlot;

impl LRUSlot for FullLRUSlot {
    const COMPONENT: ComponentType = ComponentType::Full;
    const NAME: &'static str = "Full";

    /// Device full value is evictable when unreferenced and no child holds a
    /// FULL device value.
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
            .expect("FullLRUSlot::inc_lock_ref: u32 overflow");
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
            .expect("FullLRUSlot::dec_lock_ref: underflow");
        // Invariant: FULL's lock_ref must stay >= every other component's.
        // Check the prospective `new` against the others before writing.
        assert!(
            node.components
                .iter()
                .enumerate()
                .all(|(i, c)| i == component_idx || c.lock_ref <= new),
            "FullLRUSlot::dec_lock_ref: prospective full_lock_ref {new} not >= \
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

/// LRU for SWA values.
pub struct SwaLRUSlot;

impl LRUSlot for SwaLRUSlot {
    const COMPONENT: ComponentType = ComponentType::Swa;
    const NAME: &'static str = "Swa";

    /// SWA quirk: the freed handle is a clone of FULL's value (the cookie
    /// `free_swa(full_value)` consumes), not SWA's own translated value,
    /// which is dropped here. Gates on SWA's value being `Some` so
    /// tombstoned nodes are no-ops.
    fn take_value<K: ChildKeyType>(node: &mut TreeNode<K>, result: &mut EvictResult) {
        let ct = Self::COMPONENT as usize;
        if node.components[ct].value.take().is_some() {
            if let Some(full) = node.components[ComponentType::Full as usize].value.as_ref() {
                let cloned = full.shallow_clone();
                result.evicted[ct] += cloned.size()[0] as usize;
                result.freed[ct].push(cloned);
            }
        }
    }

    fn inc_lock_ref<K: ChildKeyType>(pool: &mut TreeNodePool<K>, node_idx: NodeIdx) -> i64 {
        Self::inc_lock_ref_non_full(pool, node_idx, /* enforce_full_cap */ true)
    }

    fn dec_lock_ref<K: ChildKeyType>(pool: &mut TreeNodePool<K>, node_idx: NodeIdx) -> i64 {
        Self::dec_lock_ref_non_full(pool, node_idx)
    }
}

/// LRU for Mamba values.
pub struct MambaLRUSlot;

impl LRUSlot for MambaLRUSlot {
    const COMPONENT: ComponentType = ComponentType::Mamba;
    const NAME: &'static str = "Mamba";

    /// 1 SSM state per TreeNode regardless of the length of key.
    fn value_len<K: ChildKeyType>(_node: &TreeNode<K>) -> usize {
        1
    }

    fn inc_lock_ref<K: ChildKeyType>(pool: &mut TreeNodePool<K>, node_idx: NodeIdx) -> i64 {
        Self::inc_lock_ref_non_full(pool, node_idx, /* enforce_full_cap */ true)
    }

    fn dec_lock_ref<K: ChildKeyType>(pool: &mut TreeNodePool<K>, node_idx: NodeIdx) -> i64 {
        Self::dec_lock_ref_non_full(pool, node_idx)
    }
}

/// Evict a node's FULL device value; the caller decides how to handle the
/// returned value (freed on delete, deferred action on demote).
#[must_use = "free the returned value to avoid KV leakage"]
pub(crate) fn evict_full_value<K: ChildKeyType>(
    pool: &mut TreeNodePool<K>,
    idx: NodeIdx,
    result: &mut EvictResult,
) -> Option<Tensor> {
    let value_len = FullLRUSlot::value_len(pool.get(idx));
    let value = FullLRUSlot::node_state_mut(pool.get_mut(idx)).value.take();
    if value.is_some() {
        result.evicted[FullLRUSlot::COMPONENT as usize] += value_len;
        #[allow(clippy::expect_used, reason = "eviction never selects the root")]
        let parent_idx = pool
            .get(idx)
            .parent()
            .expect("evict_full_value never runs on the root");
        FullLRUSlot::postprocess_take_value(pool, parent_idx);
    }
    FullLRUSlot::remove(pool, idx);
    FullLRUSlot::pool_state_mut(pool).unlocked_size -= value_len;
    value
}

/// Best-effort eviction of up to `num_tokens` FULL values in LRU
/// order; clean up unreferenced leaves as needed.
pub(crate) fn evict_full<K: ChildKeyType>(
    pool: &mut TreeNodePool<K>,
    num_tokens: usize,
    result: &mut EvictResult,
) {
    if num_tokens == 0 {
        return;
    }
    let ct = FullLRUSlot::COMPONENT as usize;
    let target = result.evicted[ct] + num_tokens;
    let mut victim = FullLRUSlot::next_evictable(pool);

    while result.evicted[ct] < target {
        let Some(idx) = victim else { break };

        let next_after = FullLRUSlot::next_evictable_after(pool, idx);

        #[allow(clippy::expect_used, reason = "FULL LRU never includes root")]
        let parent_idx = pool
            .get(idx)
            .parent()
            .expect("evict: LRU returned a node without a parent (root spliced into LRU?)");

        #[allow(clippy::expect_used, reason = "non-root leaf has child key in parent")]
        pool.evict_leaf(idx, result)
            .expect("evict: leaf must have valid child key for parent removal");

        // Capture before the cascade frees `parent_idx`. A newly-evictable
        // parent can sit deeper toward the tail than `next_after`.
        let parent_was_evictable = FullLRUSlot::is_evictable(pool.get(parent_idx));

        let pre_cascade_evicted = result.evicted[ct];
        if pool.has_swa_component() {
            iteratively_delete_tombstone_leaf::<K, SwaLRUSlot>(pool, parent_idx, result);
        }
        if pool.has_mamba_component() {
            iteratively_delete_tombstone_leaf::<K, MambaLRUSlot>(pool, parent_idx, result);
        }
        let cascade_freed_anything = result.evicted[ct] > pre_cascade_evicted;

        victim = if parent_was_evictable || cascade_freed_anything {
            // Restart from the LRU tail: a newly-evictable node may sit ahead of `next_after`.
            FullLRUSlot::next_evictable(pool)
        } else {
            next_after
        };
    }
}

/// Best-effort eviction of up to `num_tokens` of component `S`'s
/// values in LRU order; clean up unreferenced leaves as needed.
pub(crate) fn evict_non_full<K: ChildKeyType, S: LRUSlot>(
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
        S::remove(pool, idx);
        S::pool_state_mut(pool).unlocked_size -= delta;

        if is_leaf {
            iteratively_delete_tombstone_leaf::<K, S>(pool, idx, result);
        }
        victim = next_after;
    }
}

/// Iteratively clean up a tombstone leaf chain starting from
/// `node_idx`.
pub(crate) fn iteratively_delete_tombstone_leaf<K: ChildKeyType, S: LRUSlot>(
    pool: &mut TreeNodePool<K>,
    mut node_idx: NodeIdx,
    result: &mut EvictResult,
) {
    loop {
        let node = pool.get(node_idx);

        if S::has_value(node)                     // not a tombstone for this component
            || !node.is_leaf()                    // still has children
            || node.is_root()                     // hit namespace root
            || FullLRUSlot::lock_ref(node) > 0
        // FULL-locked
        {
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

        #[allow(
            clippy::expect_used,
            reason = "non-root tombstone leaf has child key in parent"
        )]
        pool.evict_leaf(node_idx, result)
            .expect("just-validated tombstone leaf");
        node_idx = parent;
    }
}
