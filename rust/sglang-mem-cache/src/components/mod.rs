//! Per-component walk/validation logic for `RadixCache`. Each component
//! owns a shell type (`FullComponent`, `SwaComponent`, ...) implementing
//! the `Component` trait, dispatched dynamically but called sparsely.
//! Per-node LRU mutation stays on `tree_node_lru.rs::LRUSlot` (static
//! dispatch); shells delegate to it.
//!
//! Trivial-true validators are represented as `None`, not a wrapper that
//! always returns true, so components that don't gate pay nothing.

mod full;
mod mamba;
mod swa;

use tch::Tensor;

use crate::deferred_action::DeferredAction;
use crate::error::RadixCacheRuntimeError;
use crate::tree_node_lru::{EvictRequest, EvictResult};
use crate::tree_node_pool::{ChildKeyType, NodeIdx, TreeNode, TreeNodePool};

/// Stateful per-walk predicate that gates the match-prefix boundary
/// advance. Called once per visited node; `&mut self` lets the validator
/// carry state (e.g. SWA's run-length counter). The walk advances its
/// boundary only when ALL configured components' validators approve.
pub trait MatchValidator<K: ChildKeyType> {
    fn validate(&mut self, n: &TreeNode<K>) -> bool;
}

/// Per-component contribution to one `inc_lock_ref` call. Used both
/// per-component (each returns its own, `None` to skip) and aggregated by
/// `RadixCache` (`delta`s summed; `swa_uuid_for_lock` taken from the lone
/// component that produces it).
#[derive(Debug, Clone, Copy, Default)]
pub struct IncLockRefResult {
    /// Signed delta to `evictable_token_size`: negative on acquire (tokens
    /// shifted from evictable to protected), positive on release.
    pub delta: i64,
    /// SWA's lock-walk boundary marker, stamped at the boundary node on
    /// acquire (`None` for FULL). The caller passes it back to
    /// `dec_lock_ref` so SWA's release stops at the right boundary.
    pub swa_uuid_for_lock: Option<u64>,
}

/// Orchestrator-facing per-component trait. Each configured component
/// (FULL always; SWA when `sliding_window_size` is set; Mamba) is stored
/// on `RadixCache` as `Box<dyn Component<K>>` and dyn-dispatched, but
/// called sparsely. LRU mutation (lock_ref, is_evictable) is NOT on this
/// trait; it stays on `LRUSlot` for static dispatch.
pub trait Component<K: ChildKeyType>: Send {
    /// Construct a fresh per-walk validator, or `None` if this component
    /// has no boundary-gating logic. `None` skips the per-node `validate()`
    /// call and the allocation entirely. FULL returns `None` because every
    /// node is a valid FULL boundary; SWA gates so returns `Some`.
    fn create_match_validator(&self) -> Option<Box<dyn MatchValidator<K>>>;

    /// Per-component inc-lock walk from `node_idx` up the tree per the
    /// component's policy (FULL: to root excl.; SWA: until accumulated
    /// `len(node.key)` reaches `sliding_window_size`), bumping `lock_ref`
    /// and per-component pool-state aggregates. `None` = no inc walk.
    ///
    /// `RadixCache::inc_lock_ref` dispatches `self.components` forward
    /// (FULL then SWA); forward order is required so FULL's `lock_ref` is
    /// bumped before SWA's `SwaLRUSlot::inc_lock_ref` mutator-assert checks
    /// `prospective swa_lock_ref <= full_lock_ref` on the same node.
    fn inc_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
    ) -> Option<IncLockRefResult>;

    /// Per-component dec-lock walk, inverse of `inc_lock_ref`.
    /// `swa_uuid_for_lock` is threaded through every component (FULL
    /// ignores; SWA stops at the boundary node's matching stamp). Returns
    /// the signed `evictable_token_size` delta (positive on release), or
    /// `None` when the component has no dec walk.
    ///
    /// `RadixCache::dec_lock_ref` dispatches `self.components` in REVERSE
    /// (SWA then FULL), required so FULL's `dec_lock_ref` mutator-assert
    /// (`swa_lock_ref <= prospective full_lock_ref`) doesn't fire
    /// transiently when `full_lock_ref == swa_lock_ref` before the dec.
    fn dec_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
        swa_uuid_for_lock: Option<u64>,
    ) -> Option<i64>;

    /// Per-component evict step. Each impl checks its own budget in
    /// `request.num_tokens[ct]` against what's already drained in
    /// `result.evicted[ct]`, then runs its own `LRUSlot::evict` to make up
    /// the shortfall.
    ///
    /// `RadixCache::evict` dispatches forward (FULL then SWA): FULL's evict
    /// cross-bumps `result.evicted[Swa]` per leaf (the `free_swa(full_value)`
    /// quirk), so SWA sees a smaller residual budget and won't over-evict.
    fn evict(&self, pool: &mut TreeNodePool<K>, request: &EvictRequest, result: &mut EvictResult);

    /// Per-overlap-node insert hook, called for every node whose key
    /// matches a portion of the incoming insert key.
    ///
    /// Returns `consumed_from`: slots in `value_slice[consumed_from..]` are
    /// claimed by this component (Python must NOT free them as duplicates);
    /// `value_slice[0..consumed_from]` is unclaimed.
    ///
    /// Default returns `node_key_len` — claim nothing. FULL inherits the
    /// default; SWA overrides for tombstone recovery (replaces FULL value,
    /// may split node, emits `SwaRecover`).
    ///
    /// `RadixCache::insert_helper` takes `min(consumed_from)` across all
    /// components, then frees `value_slice[0..min]` as duplicate (a single
    /// component claiming any slot vetoes its freeing).
    ///
    /// TODO: rename to a clearer name (e.g. `claim_value_prefix`) — the
    /// real semantic is claiming ownership of a prefix-slice and returning
    /// the unclaimed-from offset.
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

    /// Pre-leaf-creation veto hook. Returning `true` aborts leaf creation
    /// for the unmatched suffix — its value slots are freed as duplicates.
    ///
    /// Default returns `false`. FULL inherits the default; SWA overrides to
    /// skip when the entire suffix lies outside the SWA window
    /// (`swa_evicted_seqlen >= total_prefix_len + key_len`).
    ///
    /// `RadixCache::insert_helper` takes `any()` across all components — a
    /// single component vetoing blocks leaf creation.
    ///
    /// TODO: confirm the veto path is reachable; if the scheduler never
    /// enqueues work past the SWA boundary, replace it with an assert.
    fn should_skip_leaf_creation(
        &self,
        _total_prefix_len: usize,
        _key_len: usize,
        _swa_evicted_seqlen: usize,
    ) -> bool {
        false
    }

    /// Post-leaf-creation hook, called after a brand-new leaf is created
    /// (the `should_skip_leaf_creation` veto did not fire). Components may
    /// inspect/split the leaf and emit deferred actions.
    ///
    /// Default is a no-op. FULL inherits the default; SWA overrides to split
    /// the leaf at the SWA window boundary if it straddles (tombstone parent
    /// + in-window child) and to emit `SwaStamp` for the in-window portion.
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

    /// Per-component LRU recency bump on a path walk, called after
    /// `match_prefix` (validator-approved boundary) and after `insert`
    /// (deepest touched node) — bumps the path from `node_idx` up through
    /// ancestors to MRU in this component's LRU.
    ///
    /// Required (no default) so a new LRU-bearing component can't forget to
    /// override and silently drift recency tracking; LRU-less components
    /// implement a no-op. SWA's impl gates on the per-node `in_list` flag
    /// inside `SwaLRUSlot::bump_mru_walk`, so tombstones aren't revived.
    fn bump_mru_walk(&self, pool: &mut TreeNodePool<K>, node_idx: NodeIdx);

    /// Per-component value redistribution after a node split, called by
    /// `pool.split_node` immediately after the structural split + FULL
    /// baseline (value slice + lock_ref copy + LRU bump). Components slice
    /// their value, copy their lock_ref, and update their LRU.
    ///
    /// Required (no default) so a new per-node-state component can't forget
    /// to override and silently leak/corrupt at every split. Components
    /// without per-node state (FullComponent — handled in the pool baseline)
    /// implement an explicit no-op.
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
