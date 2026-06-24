//! Two layers behind `RadixCache`: `Component` is the dynamically-dispatched
//! orchestration layer that drives per-component tree walks; `Slot` is the
//! static layer that manages a value and its LRU.

mod full;
mod mamba;
mod swa;
mod utils;

use tch::Tensor;

use crate::component_type::{ComponentType, NUM_COMPONENT_TYPES};
use crate::deferred_action::DeferredAction;
use crate::error::RadixCacheRuntimeError;
use crate::radix_cache::{InsertResult, MatchResult};
use crate::tree_node_pool::{ChildKeyType, ComponentNodeState, NodeIdx, TreeNode, TreeNodePool};

/// Per-walk predicate gating the match-prefix boundary advance.
pub trait MatchValidator<K: ChildKeyType> {
    fn validate(&mut self, n: &TreeNode<K>) -> bool;
}

/// Per-component contribution to one `inc_lock_ref` call.
#[derive(Debug, Clone, Copy, Default)]
pub struct IncLockRefResult {
    /// Signed delta to `evictable_token_size`.
    pub delta: i64,
    /// SWA's lock-walk boundary marker, passed back to `dec_lock_ref`.
    pub swa_uuid_for_lock: Option<u64>,
}

/// Dynamically-dispatched orchestration layer for one component's tree walks.
pub trait Component<K: ChildKeyType>: Send {
    /// Stateful validator for one prefix-match walk; `None` if no gating.
    fn create_match_validator(&self) -> Option<Box<dyn MatchValidator<K>>>;

    /// Post-match hook: fill this component's `MatchResult` fields (e.g. Mamba
    /// branching seqlen + value). Default no-op, keeping the driver agnostic.
    fn finalize_match_result(
        &self,
        _pool: &TreeNodePool<K>,
        _last_matched_node_idx: NodeIdx,
        _values: &[Tensor],
        _last_device_value_len: usize,
        _result: &mut MatchResult,
    ) {
    }

    /// Per-component inc-lock walk from `node_idx` up the tree.
    fn inc_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
    ) -> Option<IncLockRefResult>;

    /// Per-component dec-lock walk, inverse of `inc_lock_ref`.
    fn dec_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
        swa_uuid_for_lock: Option<u64>,
    ) -> Option<i64>;

    /// Per-component evict step.
    fn evict(&self, pool: &mut TreeNodePool<K>, request: &EvictRequest, result: &mut EvictResult);

    /// Per-overlap-node insert hook. Returns `consumed_from`: slots in
    /// `value_slice[consumed_from..]` are claimed by this component.
    #[allow(clippy::too_many_arguments)]
    fn update_component_on_insert_overlap(
        &self,
        _pool: &mut TreeNodePool<K>,
        _components: &[Box<dyn Component<K>>],
        _child_idx: NodeIdx,
        node_key_len: usize,
        _total_prefix_len: usize,
        _prev_prefix_len: usize,
        _value_slice: &Tensor,
        _swa_evicted_seqlen: usize,
        _deferred: &mut Vec<DeferredAction>,
    ) -> Result<usize, RadixCacheRuntimeError> {
        Ok(node_key_len)
    }

    /// Pre-leaf-creation veto hook; `true` aborts leaf creation. Default `false`.
    fn should_skip_leaf_creation(
        &self,
        _total_prefix_len: usize,
        _key_len: usize,
        _swa_evicted_seqlen: usize,
    ) -> bool {
        false
    }

    /// Post-leaf-creation hook. Default is a no-op.
    fn commit_insert_data_on_new_leaf(
        &self,
        _pool: &mut TreeNodePool<K>,
        _components: &[Box<dyn Component<K>>],
        _leaf_idx: NodeIdx,
        _consumed: usize,
        _swa_evicted_seqlen: usize,
        _deferred: &mut Vec<DeferredAction>,
    ) {
    }

    /// Commit this component's insert-time aux value (e.g. Mamba SSM state) onto
    /// the final inserted node, recording outcome on `result` (e.g. set
    /// `mamba_value_exists` when the value was not consumed). Default: no-op.
    fn commit_insert_value(
        &self,
        _pool: &mut TreeNodePool<K>,
        _node_idx: NodeIdx,
        _new_leaf: bool,
        _value: Option<&Tensor>,
        _result: &mut InsertResult,
    ) -> Result<(), RadixCacheRuntimeError> {
        Ok(())
    }

    /// Per-component LRU recency bump from `node_idx` up to MRU.
    fn bump_mru_walk(&self, pool: &mut TreeNodePool<K>, node_idx: NodeIdx);

    /// Per-component value redistribution after a node split.
    fn redistribute_on_node_split(
        &self,
        pool: &mut TreeNodePool<K>,
        new_parent_idx: NodeIdx,
        child_idx: NodeIdx,
        split_len: usize,
    );
}

pub use full::{FullComponent, FullSlot};
pub use mamba::{MambaComponent, MambaSlot};
pub use swa::{SwaComponent, SwaSlot};
pub(crate) use utils::{
    dec_lock_ref_non_full, evict_full, evict_full_value, evict_non_full, inc_lock_ref_non_full,
};

/// Per-component eviction budget, indexed by `ct as usize`.
#[derive(Default, Clone, Copy)]
pub struct EvictRequest {
    pub num_tokens: [usize; NUM_COMPONENT_TYPES],
}

/// Result of node eviction.
#[derive(Default)]
pub struct EvictResult {
    /// Tensors to free per component.
    pub freed: [Vec<Tensor>; NUM_COMPONENT_TYPES],
    /// Evict units per component.
    pub evicted: [usize; NUM_COMPONENT_TYPES],
    /// Actions to coordinate after eviction.
    pub deferred_actions: Vec<DeferredAction>,
}

/// Per-component pool-level metadata.
#[derive(Clone, Copy)]
pub struct ComponentPoolState {
    /// MRU fake end.
    pub head_sentinel: NodeIdx,
    /// LRU fake end.
    pub tail_sentinel: NodeIdx,
    pub unlocked_size: usize,
    pub locked_size: usize,
}

impl Default for ComponentPoolState {
    fn default() -> Self {
        Self {
            head_sentinel: NodeIdx::MAX,
            tail_sentinel: NodeIdx::MAX,
            unlocked_size: 0,
            locked_size: 0,
        }
    }
}

/// Per-node intrusive LRU state for one slot.
#[derive(Default, Clone, Copy)]
pub struct LRUData {
    pub prev: NodeIdx,
    pub next: NodeIdx,
    pub in_list: bool,
}

/// Static layer that manages a value and its LRU (lock_ref, recency, eviction).
pub trait Slot: Sized {
    /// Component this slot belongs to.
    const COMPONENT: ComponentType;

    /// Slot identifier.
    const NAME: &'static str;

    /// Whether the node value is evictable.
    fn is_evictable<K: ChildKeyType>(node: &TreeNode<K>) -> bool {
        Self::lock_ref(node) == 0
    }

    /// Bookkeeping after `set_value`. Default no-op.
    fn postprocess_set_value<K: ChildKeyType>(_pool: &mut TreeNodePool<K>, _parent_idx: NodeIdx) {}

    /// Bookkeeping after `take_value`. Default no-op.
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

    /// This slot's component node state.
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

    /// Size unit per node.
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
    fn total_size<K: ChildKeyType>(pool: &TreeNodePool<K>) -> usize {
        let s = Self::pool_state(pool);
        s.unlocked_size + s.locked_size
    }

    /// Splice `left` and `right` together as neighbors.
    fn connect<K: ChildKeyType>(pool: &mut TreeNodePool<K>, left: NodeIdx, right: NodeIdx) {
        Self::data_mut(pool.get_mut(left)).next = right;
        Self::data_mut(pool.get_mut(right)).prev = left;
    }

    /// Allocate the sentinel pair as an empty list.
    fn init<K: ChildKeyType>(pool: &mut TreeNodePool<K>) {
        let head = pool.alloc_sentinel();
        let tail = pool.alloc_sentinel();
        // Self-loop the unread directions so debug prints don't show zeros.
        Self::data_mut(pool.get_mut(head)).prev = head;
        Self::data_mut(pool.get_mut(tail)).next = tail;
        Self::connect::<K>(pool, head, tail);
        let state = Self::pool_state_mut(pool);
        state.head_sentinel = head;
        state.tail_sentinel = tail;
    }

    /// Move `idx` to the MRU end, inserting it if not already in the list.
    fn bump_mru<K: ChildKeyType>(pool: &mut TreeNodePool<K>, idx: NodeIdx) {
        if Self::data(pool.get(idx)).in_list {
            Self::lru_remove(pool, idx);
        }
        let head = Self::head_sentinel(pool);
        let first_real = Self::data(pool.get(head)).next;
        Self::connect::<K>(pool, head, idx);
        Self::connect::<K>(pool, idx, first_real);
        Self::data_mut(pool.get_mut(idx)).in_list = true;
    }

    /// Unlink `idx` from the list, marking it not-in-list.
    fn lru_remove<K: ChildKeyType>(pool: &mut TreeNodePool<K>, idx: NodeIdx) {
        let data = Self::data_mut(pool.get_mut(idx));
        let prev = data.prev;
        let next = data.next;
        data.in_list = false;
        Self::connect::<K>(pool, prev, next);
    }

    /// Re-splice both halves of a `split_node` into the LRU at MRU, leaving
    /// `child` more recent than `parent`.
    fn bump_mru_split<K: ChildKeyType>(
        pool: &mut TreeNodePool<K>,
        parent: NodeIdx,
        child: NodeIdx,
    ) {
        Self::bump_mru(pool, parent);
        Self::bump_mru(pool, child);
    }

    /// Redistribute lock_ref, values and LRUs after a structural node split.
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
                "redistribute_on_split: require_value set but this slot has no \
                 value on the node split",
            );
        }
    }

    /// Walk from `leaf` up its parents (excluding root), bumping each in-list
    /// node into a contiguous MRU run with the leaf most recent.
    fn bump_mru_walk<K: ChildKeyType>(pool: &mut TreeNodePool<K>, leaf: NodeIdx) {
        let mut chain_first: Option<NodeIdx> = None;
        let mut chain_last: Option<NodeIdx> = None;
        let mut cur = leaf;

        while let Some(parent) = pool.get(cur).parent() {
            // Don't revive nodes kept out of the list.
            if !Self::data(pool.get(cur)).in_list {
                cur = parent;
                continue;
            }
            Self::lru_remove(pool, cur);
            match chain_last {
                Some(prev_last) => Self::connect::<K>(pool, prev_last, cur),
                None => chain_first = Some(cur),
            }
            Self::data_mut(pool.get_mut(cur)).in_list = true;
            chain_last = Some(cur);
            cur = parent;
        }

        if let (Some(first), Some(last)) = (chain_first, chain_last) {
            // head.next still points at the first non-bumped node.
            let head = Self::head_sentinel(pool);
            let original_first = Self::data(pool.get(head)).next;
            Self::connect::<K>(pool, head, first);
            Self::connect::<K>(pool, last, original_first);
        }
    }

    /// Walk from `from.prev` toward MRU, returning the first `is_evictable` node.
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

    /// LRU evictable target, walking from the tail sentinel.
    fn next_evictable<K: ChildKeyType>(pool: &TreeNodePool<K>) -> Option<NodeIdx> {
        Self::next_evictable_after::<K>(pool, Self::tail_sentinel(pool))
    }

    /// Take this slot's freed handle out of `node` and record it into `result`.
    fn take_value<K: ChildKeyType>(node: &mut TreeNode<K>, result: &mut EvictResult) {
        let ct = Self::COMPONENT as usize;
        if let Some(t) = node.components[ct].value.take() {
            result.evicted[ct] += t.size()[0] as usize;
            result.freed[ct].push(t);
        }
    }

    /// Increase this slot's `lock_ref`; returns the delta to `unlocked_size`.
    fn inc_lock_ref<K: ChildKeyType>(pool: &mut TreeNodePool<K>, node_idx: NodeIdx) -> i64;

    /// Decrease this slot's `lock_ref`; returns the delta to `unlocked_size`.
    fn dec_lock_ref<K: ChildKeyType>(pool: &mut TreeNodePool<K>, node_idx: NodeIdx) -> i64;
}
