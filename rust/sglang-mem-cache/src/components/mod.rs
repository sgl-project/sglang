//! Per-component walk/validation logic for `RadixCache`.

mod full;
mod mamba;
mod swa;

use tch::Tensor;

use crate::deferred_action::DeferredAction;
use crate::error::RadixCacheRuntimeError;
use crate::tree_node_lru::{EvictRequest, EvictResult};
use crate::tree_node_pool::{ChildKeyType, NodeIdx, TreeNode, TreeNodePool};

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

/// Orchestrator-facing per-component trait.
pub trait Component<K: ChildKeyType>: Send {
    /// Fresh per-walk validator, or `None` if this component has no gating.
    fn create_match_validator(&self) -> Option<Box<dyn MatchValidator<K>>>;

    /// Per-component inc-lock walk from `node_idx` up the tree.
    ///
    /// Dispatched forward (FULL then SWA) so FULL's `lock_ref` is bumped
    /// before SWA's `swa_lock_ref <= full_lock_ref` assert runs.
    fn inc_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
    ) -> Option<IncLockRefResult>;

    /// Per-component dec-lock walk, inverse of `inc_lock_ref`.
    ///
    /// Dispatched in REVERSE (SWA then FULL) so FULL's
    /// `swa_lock_ref <= full_lock_ref` assert doesn't fire transiently.
    fn dec_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
        swa_uuid_for_lock: Option<u64>,
    ) -> Option<i64>;

    /// Per-component evict step.
    ///
    /// Dispatched forward (FULL then SWA): FULL's evict cross-bumps
    /// `result.evicted[Swa]` per leaf, so SWA won't over-evict.
    fn evict(&self, pool: &mut TreeNodePool<K>, request: &EvictRequest, result: &mut EvictResult);

    /// Per-overlap-node insert hook. Returns `consumed_from`: slots in
    /// `value_slice[consumed_from..]` are claimed by this component.
    ///
    /// Default claims nothing. `insert_helper` frees up to `min` across
    /// components.
    #[allow(clippy::too_many_arguments)]
    fn consume_value(
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

    /// Pre-leaf-creation veto hook; `true` aborts leaf creation. Default
    /// `false`. `insert_helper` takes `any()` across components.
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

    /// Per-component LRU recency bump from `node_idx` up to MRU. No default
    /// so a new LRU-bearing component can't forget to override.
    fn bump_mru_walk(&self, pool: &mut TreeNodePool<K>, node_idx: NodeIdx);

    /// Per-component value redistribution after a node split. No default so
    /// a new per-node-state component can't forget to override.
    fn redistribute_on_node_split(
        &self,
        pool: &mut TreeNodePool<K>,
        new_parent_idx: NodeIdx,
        child_idx: NodeIdx,
        split_len: usize,
    );
}

pub use full::FullComponent;
pub use mamba::MambaComponent;
pub use swa::SwaComponent;
