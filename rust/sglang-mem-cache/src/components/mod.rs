//! Per-component WALK / VALIDATION logic for `RadixCache`. Each
//! component owns a "shell" type (`FullComponent`, `SwaComponent`,
//! ...) that implements the `Component` trait. Future SWA PRs grow
//! the trait with per-component dispatchers for SWA-specific walks
//! (acquire / release / cascade evict / insert hooks).
//!
//! Mirrors OSS `unified_cache_components/*.py`'s component pattern.
//!
//! NOT to be confused with `tree_node_lru.rs::LRUSlot` — orthogonal
//! axis covering per-NODE intrusive-LRU mutation (`inc/dec_lock_ref`,
//! `is_evictable`, sentinels). LRUSlot stays statically dispatched;
//! component shells delegate to it via static calls when they need
//! LRU work.
//!
//! ```text
//! RadixCache.components: Vec<Box<dyn Component<K>>>
//!                          (dyn dispatch — once per match_prefix walk
//!                           via `create_match_validator()`. Components
//!                           with no gating return `None` and skip the
//!                           per-node loop entirely.)
//!                          ↓
//! Vec<Box<dyn MatchValidator<K>>> with ONLY non-trivial validators
//!                          (Some-returning components only — FULL
//!                           contributes nothing today, so FULL-only
//!                           callers pay zero allocs and zero per-node
//!                           vtable hits. SWA-enabled adds 1 Box and
//!                           1 vtable hit per visited node for the
//!                           SwaMatchValidator.)
//!                          ↓ component shells delegate to:
//! tree_node_lru.rs::LRUSlot impls (FullLRUSlot, etc.)
//!                          (static dispatch; per-node hot path for
//!                           LRU mutation — `inc/dec_lock_ref`,
//!                           `is_evictable`)
//! ```
//!
//! Skip-unnecessary-work pattern: trivial-true validators are
//! represented as `None`, not as a wrapper that always returns true.
//! Generalize this when adding other per-component APIs that have a
//! "no-op" shape — pay nothing for components that don't participate.

mod full;
mod mamba;
mod swa;

use tch::Tensor;

use crate::deferred_action::DeferredAction;
use crate::error::RadixCacheRuntimeError;
use crate::tree_node_lru::{EvictRequest, EvictResult};
use crate::tree_node_pool::{ChildKeyType, NodeIdx, TreeNode, TreeNodePool};

/// Per-walk match-prefix validator. Mirrors OSS
/// `unified_cache_components/*.py::create_match_validator()` — a
/// stateful per-walk predicate that gates the boundary advance.
/// Called once per visited node during the `match_prefix` walk;
/// `&mut self` lets the validator carry state (e.g. SWA's run-length
/// counter). The walk advances its boundary only when ALL configured
/// components' validators approve.
pub trait MatchValidator<K: ChildKeyType> {
    fn validate(&mut self, n: &TreeNode<K>) -> bool;
}

/// Per-component contribution to one `inc_lock_ref` call. Used at two
/// layers with the same shape:
///   * Per-component (Component trait): each component returns its own
///     contribution (`Option<IncLockRefResult>` — `None` skips
///     unnecessary work for components with no inc walk).
///   * Aggregated (RadixCache): `delta`s summed across components,
///     `swa_uuid_for_lock` taken from the unique component that
///     produces it (SWA today; `None` for FULL-only configs).
///
/// Mirrors OSS `sglang.srt.mem_cache.base_prefix_cache.IncLockRefResult`
/// — same field names, same semantics. PyO3 surface returns this as
/// `(i64, Option<u64>)` to keep the wrapper boundary tuple-shaped (no
/// new pyclass needed for a 2-field POD).
#[derive(Debug, Clone, Copy, Default)]
pub struct IncLockRefResult {
    /// Signed delta to `evictable_token_size` produced by this walk:
    /// negative on acquire (tokens shifted from evictable to
    /// protected), positive on release. SWA contributes `0` today —
    /// SWA's per-pool size aggregates aren't wired up yet (the
    /// `node.components[Swa].value` write path doesn't exist, so
    /// `SwaLRUSlot`'s pool-state aggregates are 0 and updating them
    /// here would underflow). See the TODO at the SWA size update
    /// site in `SwaComponent::inc_lock_ref` / `dec_lock_ref`.
    pub delta: i64,
    /// SWA's lock-walk boundary marker. `Some(uuid)` when this acquire
    /// walked SWA and stamped (or reused) a uuid at the boundary node;
    /// `None` for FULL-only configs and for FULL's contribution. The
    /// caller must pass this back to `dec_lock_ref` so SWA's release
    /// stops decrementing at the right boundary.
    pub swa_uuid_for_lock: Option<u64>,
}

/// Orchestrator-facing per-component trait. Each configured component
/// (FULL always; SWA when `sliding_window_size` is set; future Mamba)
/// is stored on `RadixCache` as `Box<dyn Component<K>>`. Methods on
/// this trait are dyn-dispatched but called sparsely (e.g.
/// `create_match_validator` is once per `match_prefix` walk). The
/// returned `Box<dyn MatchValidator<K>>` then dispatches per-node —
/// see the module-level diagram for the full hot-path picture. LRU
/// mutation (lock_ref, is_evictable) is NOT on this trait; it stays
/// on `LRUSlot` for static dispatch.
pub trait Component<K: ChildKeyType>: Send {
    /// Construct a fresh per-walk validator, or `None` if this
    /// component has no boundary-gating logic. Returning `None` lets
    /// the orchestrator skip the per-node `validate()` call AND the
    /// allocation entirely — pay nothing for trivial-true validators.
    /// Pattern: components that gate (SWA, future Mamba) return
    /// `Some(...)` only when they actually have work to do; FULL
    /// returns `None` because every node is a valid FULL boundary.
    /// Generalize this "skip unnecessary work" pattern to other
    /// per-component APIs as they grow.
    fn create_match_validator(&self) -> Option<Box<dyn MatchValidator<K>>>;

    /// Per-component inc-lock walk. Walks from `node_idx` up the tree
    /// per the component's policy (FULL: to root excl.; SWA: until
    /// accumulated `len(node.key)` reaches `sliding_window_size`),
    /// bumping the component's `lock_ref` and updating any per-
    /// component pool-state aggregates inline. `None` = component has
    /// no inc walk to perform — same skip-unnecessary-work pattern as
    /// `create_match_validator`. All current components return `Some`;
    /// the `Option` is preserved for future metadata-only components
    /// that carry no LRU state.
    ///
    /// Dispatched by `RadixCache::inc_lock_ref`, which iterates
    /// `self.components` forward (FULL first, then SWA — matching OSS
    /// `unified_radix_cache.py`'s `inc_lock_ref`) and aggregates
    /// per-component `IncLockRefResult`s field-by-field. Forward order
    /// is required so FULL's `lock_ref` is bumped before SWA's per-slot
    /// `SwaLRUSlot::inc_lock_ref` mutator-assert checks
    /// `prospective new swa_lock_ref <= full_lock_ref` on the same
    /// node.
    fn inc_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
    ) -> Option<IncLockRefResult>;

    /// Per-component dec-lock walk. Inverse of `inc_lock_ref` — walks
    /// from `node_idx` up, decrementing this component's `lock_ref`
    /// and updating per-component pool-state aggregates.
    /// `swa_uuid_for_lock` is threaded through every component (FULL
    /// ignores; SWA matches against the boundary node's stamp to know
    /// when to stop). Returns the signed delta to
    /// `evictable_token_size` (positive on release), or `None` when
    /// the component has no dec walk.
    ///
    /// Dispatched by `RadixCache::dec_lock_ref`, which iterates
    /// `self.components` in **reverse** (SWA first, then FULL). This
    /// is a Rust-side deviation from OSS `unified_radix_cache.py`'s
    /// `dec_lock_ref` forward iteration; the reverse order is required
    /// so FULL's per-slot `dec_lock_ref` mutator-assert (which checks
    /// `swa_lock_ref <= prospective new full_lock_ref`) doesn't fire
    /// transiently when `full_lock_ref == swa_lock_ref` before the
    /// dec.
    ///
    /// **Why OSS works in either order**: OSS's per-component `lock_ref`
    /// fields are independent counters that don't observe each other
    /// at mutation time — each component's dec only touches its own
    /// slot's `lock_ref`, so the order in which components are
    /// dispatched is irrelevant to per-component correctness. (SWA's
    /// boundary-uuid / window-fill semantics are also unaffected:
    /// SWA's release walk independently stops at the matching uuid
    /// regardless of when FULL ran.) Our Rust mutator-asserts ADD a
    /// cross-component invariant check at the per-mutation point,
    /// which is why the order matters here. Reverse-iter keeps the
    /// assert load-bearing as a regression net without paying for a
    /// separate post-dispatch validation walk.
    fn dec_lock_ref(
        &self,
        pool: &mut TreeNodePool<K>,
        node_idx: NodeIdx,
        swa_uuid_for_lock: Option<u64>,
    ) -> Option<i64>;

    /// Per-component evict step. Each impl checks its own budget in
    /// `request.num_tokens[ct]` against what's already drained in
    /// `result.evicted[ct]` (cross-bumps from earlier components in
    /// the dispatch order), and runs its own `LRUSlot::evict` to
    /// make up the shortfall.
    ///
    /// Dispatched by `RadixCache::evict`, which iterates
    /// `self.components` forward (FULL first, then SWA — matching
    /// the existing `inc_lock_ref` order). Forward iteration matters
    /// here because FULL's evict body cross-bumps `result.evicted[Swa]`
    /// for each leaf evicted (the `free_swa(full_value)` quirk), so
    /// SWA's pass sees a smaller residual budget by the time it runs
    /// — without forward iter, SWA would over-evict.
    fn evict(&self, pool: &mut TreeNodePool<K>, request: &EvictRequest, result: &mut EvictResult);

    /// Per-overlap-node insert hook. Called by `RadixCache::insert_helper`
    /// for every node whose key matches a portion of the incoming insert
    /// key (both `MatchChildResult::FullMatch` and the new parent
    /// produced by `MatchChildResult::PartialMatch`).
    ///
    /// Returns `consumed_from`: the start index within `value_slice` of
    /// the portion this component took ownership of. Slots in
    /// `value_slice[consumed_from..]` are claimed by the component
    /// (mutated into the node, recovered, etc. — Python must NOT free
    /// them as duplicates). Slots in `value_slice[0..consumed_from]`
    /// are unclaimed by this component.
    ///
    /// Default returns `node_key_len` — claim nothing, treat the entire
    /// slice as a duplicate. FULL inherits the default; SWA overrides
    /// for tombstone recovery (replaces FULL value, may split node,
    /// emits `SwaRecover`).
    ///
    /// Aggregation: `RadixCache::insert_helper` takes
    /// `min(consumed_from)` across all components, then frees
    /// `value_slice[0..min]` as duplicate (a single component claiming
    /// any slot vetoes its freeing). Mirrors OSS
    /// `unified_radix_cache._insert_helper`'s
    /// `consumed_from = min(consumed_from, comp_consumed_from)' pattern.
    ///
    /// TODO: rename `consume_value` to something more explicit. The
    /// "consume" verb is ambiguous (does it free? mutate? read?) — the
    /// real semantic is "claim ownership of a prefix-slice of incoming
    /// value slots, returning the unclaimed-from offset". Candidates:
    /// `claim_value_prefix`, `compute_unclaimed_offset`,
    /// `report_owned_prefix_len`.
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

    /// Pre-leaf-creation veto hook. Called by `RadixCache::insert_helper`
    /// before creating a new leaf for the unmatched suffix. Returning
    /// `true` aborts leaf creation — the suffix's value slots are freed
    /// as duplicates instead.
    ///
    /// Default returns `false` (no veto). FULL inherits the default;
    /// SWA overrides to skip when the entire suffix lies outside the
    /// SWA window (`swa_evicted_seqlen >= total_prefix_len + key_len`).
    ///
    /// Aggregation: `RadixCache::insert_helper` takes `any()` across
    /// all components — a single component vetoing blocks leaf
    /// creation. Mirrors OSS `unified_radix_cache._insert_helper`'s
    /// `any(comp.should_skip_leaf_creation(...))` pattern.
    ///
    /// TODO: revisit whether the veto path is reachable in practice.
    /// The SWA implementation only vetoes when the entire suffix is
    /// outside the SWA window — but the scheduler upstream is supposed
    /// to never enqueue work past the SWA boundary. If we can prove
    /// it's unreachable, replace `should_skip_leaf_creation` with a
    /// validation-style assert (or remove it entirely from the trait)
    /// rather than carrying a real veto path that's never hit.
    fn should_skip_leaf_creation(
        &self,
        _total_prefix_len: usize,
        _key_len: usize,
        _swa_evicted_seqlen: usize,
    ) -> bool {
        false
    }

    /// Post-leaf-creation hook. Called by `RadixCache::insert_helper`
    /// after a brand-new leaf is created (the `should_skip_leaf_creation`
    /// veto did not fire). Components may inspect / split the leaf and
    /// emit deferred actions for the Python orchestrator to process.
    ///
    /// Default is a no-op. FULL inherits the default; SWA overrides
    /// to split the leaf at the SWA window boundary if it straddles
    /// (tombstone parent + in-window child) and to emit `SwaStamp`
    /// for the in-window portion.
    ///
    /// Aggregation: `RadixCache::insert_helper` iterates
    /// `self.components` in order. Mirrors OSS
    /// `unified_radix_cache._insert_helper`'s
    /// `for comp in components: comp.commit_insert_component_data(...)`
    /// loop.
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

    /// Per-component LRU recency bump on a path walk. Called after
    /// `match_prefix` (with the validator-approved boundary node) and
    /// after `insert` (with the deepest touched node) — bumps the path
    /// from `node_idx` up through ancestors to MRU position in this
    /// component's LRU.
    ///
    /// **Required (no default)**: every component with an LRU MUST
    /// decide. A silent default would let a new component with LRU
    /// state forget to override and silently let recency tracking
    /// drift. Components without an LRU implement a no-op.
    ///
    /// SwaComponent's impl naturally gates on the per-node `in_list`
    /// flag inside `SwaLRUSlot::bump_mru_walk`, so SWA tombstones
    /// (no SWA value) are skipped — they're not revived by an access
    /// bump. Mirrors OSS `unified_radix_cache._for_each_component_lru`'s
    /// per-component dispatch.
    fn bump_mru_walk(&self, pool: &mut TreeNodePool<K>, node_idx: NodeIdx);

    /// Per-component value redistribution after a node split. Called
    /// by `pool.split_node` immediately after the structural split +
    /// FULL baseline (FULL value slice + FULL lock_ref copy + FULL LRU
    /// bump). Components slice their value, copy their lock_ref, and
    /// update their LRU.
    ///
    /// **Required (no default)**: every component MUST decide what to
    /// do at split time. A silent no-op default would let a new
    /// component with per-node state forget to override and silently
    /// leak / corrupt at every split. Components without per-node
    /// state (e.g. FullComponent — FULL is handled in pool baseline)
    /// implement an explicit no-op body documenting the delegation.
    ///
    /// Mirrors OSS `unified_radix_cache._split_node`'s per-component
    /// `comp.redistribute_on_node_split` loop.
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
