//! Per-slot intrusive LRU machinery layered on top of `TreeNodePool`.
//!
//! TODO(Jialin): [Performance][Worst-case] The current LRU walk is fast in
//! the typical case (O(1) per insert/remove, O(K) per `evict` for K freed
//! leaves) but the predicate-skip iteration in `next_evictable_from` can
//! degrade to O(N) per step when the LRU tail accumulates many locked or
//! internal nodes that the predicate skips at every step. Worst-case
//! behavior is O(N²) for an `evict` that frees N nodes in a pool whose
//! tail region is dominated by skipped entries (e.g. a long-lived shared
//! prefix that's continuously re-locked across many concurrent requests).
//!
//! For a stronger asymptotic guarantee, replace the doubly-linked list
//! with a `SortedSet<(InsertionTick, NodeIdx)>` keyed by a monotonic
//! insertion counter, with lazy invalidation: lock/unlock leaves the
//! entry in the set but marks it stale; `evict` pops the lowest tick,
//! re-evaluates the predicate, and skips stale entries. This gives
//! O(log N) per insert/remove and O(log N) amortized per eviction step
//! regardless of how locked the tail is — at the cost of a log-N
//! constant on every hot-path mutation versus today's O(1).
//!
//! Deferred until profiling actually flags the LRU walk as hot. The
//! current design's strength is its O(1) constants on the dominant
//! mutation paths (insert / inc/dec_lock_ref); the SortedSet alternative
//! trades that for worst-case bounds.
//!
//! ## Design overview
//!
//! Recency tracking is modeled as one or more LRU "slots". Each slot owns:
//!   * A pair of `(prev, next)` link fields on every `TreeNode<K>` — picked
//!     by the slot's accessor methods (e.g. `FullLRUSlot` walks
//!     `node.lru_data` for the full slot, `node.swa_lru_data` for SWA, etc.).
//!   * A pair of sentinel `NodeIdx`s on `TreeNodePool<K>` (head + tail) —
//!     picked by the slot's pool-level accessors.
//!   * A predicate `is_evictable` that defines which nodes are valid
//!     eviction targets for this slot.
//!
//! Algorithmic primitives — `connect`, `bump_mru`, `remove`,
//! `bump_mru_walk`, `next_evictable[_after]`, `init` — are provided as
//! default methods on the `LRUSlot` trait so each slot only implements
//! field routing. Static dispatch via monomorphization compiles slot calls
//! down to direct field reads/writes — zero runtime cost relative to
//! inlining the algorithm into `TreeNodePool` directly.
//!
//! ## Sentinel pattern
//!
//! Each LRU instance allocates a permanent `(head_sentinel, tail_sentinel)`
//! pair at pool construction. Real nodes are spliced strictly between the
//! two sentinels. This eliminates empty-list and single-element edge cases
//! from `bump_mru` and `remove` — every operation sees at least two
//! neighbors and writes are unconditional. Walks terminate by comparing
//! against `head_sentinel` rather than `Option::is_none`, so sentinels are
//! never returned to callers as eviction candidates.
//!
//! ## Adding a new slot
//!
//! When SWA / Mamba land, each adds a marker struct (e.g. `SwaLRUSlot`) and
//! an `impl LRUSlot for SwaLRUSlot` block routing accessors to the
//! component's per-node link fields and per-pool sentinels. The trait's
//! default methods then provide the full LRU API for the new slot
//! unchanged.

use tch::Tensor;

use crate::component_type::{ComponentType, NUM_COMPONENT_TYPES};
use crate::deferred_action::DeferredAction;
use crate::error::RadixCacheRuntimeError;
use crate::tree_node_pool::{ChildKeyType, ComponentNodeState, NodeIdx, TreeNode, TreeNodePool};

/// Per-component eviction request — one budget per `ComponentType`,
/// indexed by `ct as usize`. Missing slots default to 0 (no eviction
/// requested for that component). Mirrors OSS
/// `unified_radix_cache.py`'s `EvictParams` shape with the per-
/// component dict flattened into an array.
#[derive(Default, Clone, Copy)]
pub struct EvictRequest {
    pub num_tokens: [usize; NUM_COMPONENT_TYPES],
}

/// Result of node eviction.
///
/// TODO(Jialin): [Optimization][Major] At large M (~10K leaves) the
/// per-component lists trigger Python GC pressure — N freshly-wrapped
/// `PyTensor` objects trip the young-gen GC threshold mid-loop,
/// producing occasional ~2× outliers in the bench's tail. The whole
/// cost class (per-leaf PyO3 wrap + Python ref accounting + GC
/// interaction) disappears once the kv_pool_allocator is itself a
/// Rust component: `evict` can call `allocator.free(value)` directly
/// inside the inner loop without ever crossing the PyO3 boundary, and
/// `EvictResult` can shrink to just `evicted` per-component counts
/// (no tensors returned). Defer until the allocator migration lands.
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

    /// Slot identifier (component + tier; e.g. "Full" device vs "HostFull"),
    /// used in the duplicate-value error.
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

    /// Get the per-node components.
    fn node_components<K: ChildKeyType>(
        node: &TreeNode<K>,
    ) -> &[ComponentNodeState; NUM_COMPONENT_TYPES] {
        &node.components
    }
    /// Get the mutable per-node components.
    fn node_components_mut<K: ChildKeyType>(
        node: &mut TreeNode<K>,
    ) -> &mut [ComponentNodeState; NUM_COMPONENT_TYPES] {
        &mut node.components
    }

    /// Get this slot's component state.
    fn node_state<K: ChildKeyType>(node: &TreeNode<K>) -> &ComponentNodeState {
        &Self::node_components(node)[Self::COMPONENT as usize]
    }
    /// Get the mutable component state.
    fn node_state_mut<K: ChildKeyType>(node: &mut TreeNode<K>) -> &mut ComponentNodeState {
        &mut Self::node_components_mut(node)[Self::COMPONENT as usize]
    }

    /// Get LRU data.
    fn data<K: ChildKeyType>(node: &TreeNode<K>) -> &LRUData {
        &Self::node_state(node).lru_data
    }
    /// Get mutable LRU data.
    fn data_mut<K: ChildKeyType>(node: &mut TreeNode<K>) -> &mut LRUData {
        &mut Self::node_state_mut(node).lru_data
    }

    /// Get value.
    fn value<K: ChildKeyType>(node: &TreeNode<K>) -> Option<&Tensor> {
        Self::node_state(node).value.as_ref()
    }

    /// Whether the value is populated.
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

    /// Get lock ref.
    fn lock_ref<K: ChildKeyType>(node: &TreeNode<K>) -> u32 {
        Self::node_state(node).lock_ref
    }

    /// Set lock ref.
    fn set_lock_ref<K: ChildKeyType>(node: &mut TreeNode<K>, value: u32) {
        Self::node_state_mut(node).lock_ref = value;
    }

    /// Get the per-pool component array (device tier; host slots override).
    fn pool_components<K: ChildKeyType>(
        pool: &TreeNodePool<K>,
    ) -> &[ComponentPoolState; NUM_COMPONENT_TYPES] {
        &pool.components
    }
    /// Get the mutable per-pool component array.
    fn pool_components_mut<K: ChildKeyType>(
        pool: &mut TreeNodePool<K>,
    ) -> &mut [ComponentPoolState; NUM_COMPONENT_TYPES] {
        &mut pool.components
    }

    /// Get the pool state.
    fn pool_state<K: ChildKeyType>(pool: &TreeNodePool<K>) -> &ComponentPoolState {
        &Self::pool_components(pool)[Self::COMPONENT as usize]
    }
    /// Get the mutable pool state.
    fn pool_state_mut<K: ChildKeyType>(pool: &mut TreeNodePool<K>) -> &mut ComponentPoolState {
        &mut Self::pool_components_mut(pool)[Self::COMPONENT as usize]
    }

    /// Get MRU fake end.
    fn head_sentinel<K: ChildKeyType>(pool: &TreeNodePool<K>) -> NodeIdx {
        Self::pool_state(pool).head_sentinel
    }
    /// Get LRU fake end.
    fn tail_sentinel<K: ChildKeyType>(pool: &TreeNodePool<K>) -> NodeIdx {
        Self::pool_state(pool).tail_sentinel
    }

    /// Get unlocked size.
    fn unlocked_size<K: ChildKeyType>(pool: &TreeNodePool<K>) -> usize {
        Self::pool_state(pool).unlocked_size
    }
    /// Get locked size.
    fn locked_size<K: ChildKeyType>(pool: &TreeNodePool<K>) -> usize {
        Self::pool_state(pool).locked_size
    }
    /// Get total size (unlocked + locked).
    fn total_size<K: ChildKeyType>(pool: &TreeNodePool<K>) -> usize {
        let s = Self::pool_state(pool);
        s.unlocked_size + s.locked_size
    }

    // ---- Provided: algorithm, written once ----

    /// Splice `left` and `right` together as neighbors:
    ///   left.next = right
    ///   right.prev = left
    /// The single mutation primitive — every other op composes from this.
    fn connect<K: ChildKeyType>(pool: &mut TreeNodePool<K>, left: NodeIdx, right: NodeIdx) {
        Self::data_mut(pool.get_mut(left)).next = right;
        Self::data_mut(pool.get_mut(right)).prev = left;
    }

    /// Allocate two sentinel slots, cross-link them as an empty list, and
    /// store the pair on the pool's per-component sentinel slot. Called
    /// once per pool lifecycle from `TreeNodePool::new` (per slot type).
    /// `alloc_sentinel` bumps the pool's `sentinel_count` so
    /// `active_node_count` excludes these slots — adding a new LRU instance
    /// (e.g. SWA) just adds another `init` call; no constants need updating.
    fn init<K: ChildKeyType>(pool: &mut TreeNodePool<K>) {
        let head = pool.alloc_sentinel();
        let tail = pool.alloc_sentinel();
        // Defensive self-loops on the unread directions (head.prev,
        // tail.next) — never followed by walks but keeps debug-print state
        // sane and avoids leaving the placeholder zeros from
        // `new_sentinel`.
        Self::data_mut(pool.get_mut(head)).prev = head;
        Self::data_mut(pool.get_mut(tail)).next = tail;
        // Empty-list state: head <-> tail, no real nodes between.
        Self::connect::<K>(pool, head, tail);
        let state = Self::pool_state_mut(pool);
        state.head_sentinel = head;
        state.tail_sentinel = tail;
    }

    /// Ensure `idx` sits at MRU. If `idx` is already in the list, removes
    /// it first; otherwise just splices it in. Single primitive for both
    /// "first-time insert" (e.g. new leaf from `insert_leaf`) and
    /// "bump on access" (e.g. re-touched node during `match_prefix`).
    /// Caller must guarantee `idx` is not a sentinel.
    fn bump_mru<K: ChildKeyType>(pool: &mut TreeNodePool<K>, idx: NodeIdx) {
        if Self::data(pool.get(idx)).in_list {
            Self::remove(pool, idx);
        }
        // Splice between head_sentinel and the current first real node.
        let head = Self::head_sentinel(pool);
        let first_real = Self::data(pool.get(head)).next;
        Self::connect::<K>(pool, head, idx);
        Self::connect::<K>(pool, idx, first_real);
        Self::data_mut(pool.get_mut(idx)).in_list = true;
    }

    /// Unlink `idx` from the list, marking it not-in-list. Caller must
    /// guarantee `idx` is currently in the list and is not a sentinel.
    /// Used for permanent removal (eviction); for "move to MRU" use
    /// `bump_mru` which handles both states.
    fn remove<K: ChildKeyType>(pool: &mut TreeNodePool<K>, idx: NodeIdx) {
        // One mut borrow: read prev/next, clear in_list. The mut borrow
        // drops after the last field write (NLL), so `connect` re-borrows
        // the pool cleanly below.
        let data = Self::data_mut(pool.get_mut(idx));
        let prev = data.prev;
        let next = data.next;
        data.in_list = false;
        Self::connect::<K>(pool, prev, next);
    }

    /// Re-splice both halves of a `split_node` operation into the LRU at
    /// MRU. After a split the original child has been re-parented under a
    /// newly-allocated intermediate; both need to be in the LRU at the
    /// MRU end. The order encoded here decides which one ends up most
    /// recently used — the slot's path-ordering policy.
    ///
    /// Currently leaf-MRU: `parent` is bumped first, then `child`, so the
    /// final layout is `head_sentinel -> child -> parent -> [rest]`
    /// (child more recent than parent). Mirrors Python `_split_node`'s
    /// `insert_mru(new_node); insert_mru(child)` and matches
    /// `bump_mru_walk`'s leaf-at-MRU layout. The TODO on `bump_mru_walk`
    /// covers the eventual per-slot ordering optimization that would
    /// flip both methods together for FULL/MAMBA.
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
                 value must exist on a node split when HiCache is off",
            );
        }
    }

    /// Walk from `leaf` up via parent links (excluding the root), bumping
    /// each in-list node into a contiguous MRU run that preserves
    /// child-more-recent-than-parent ordering. After this call the LRU
    /// layout near the head is:
    ///   `head_sentinel -> leaf -> leaf.parent -> leaf.grandparent -> ... -> root_child -> [other nodes]`
    ///
    /// Mirrors Python `unified_radix_cache.UnifiedLRUList.reset_node_and_parents_mru`,
    /// which is the access-bump primitive used by `_match_post_processor`
    /// after `match_prefix` walks the tree. `RadixCache::match_prefix_helper`
    /// and `insert_helper` call this to give access-order LRU semantics.
    ///
    /// Implementation: builds a chain inline (1 `connect` per node beyond
    /// the first) and only writes `head.next` once at the end. Crucially,
    /// `head.next` is never overwritten during the loop — `remove` keeps
    /// it pointing at the first non-bumped node, so the closing splice
    /// reads it correctly even when `leaf` was originally at MRU.
    ///
    /// `in_list` gating means already-not-in-list nodes (e.g. SWA
    /// tombstones in step 6) are skipped — we don't revive them on
    /// access. Stops at the root (no parent); passing root is a no-op.
    ///
    /// TODO(Jialin): [Performance][Per-slot ordering] The leaf-MRU
    /// in-path order chosen here matches Python `unified_radix_cache.py`
    /// uniformly across components, but it's only *required* for SWA
    /// (where eviction must tombstone root-side / out-of-window nodes
    /// before leaf-side / in-window nodes — leaf-MRU naturally puts the
    /// older root-side at the LRU end of the path). For FULL and MAMBA,
    /// the opposite ordering ("root_child-MRU within path") would let
    /// eviction find the leaf immediately when walking into the path,
    /// instead of skipping past N-1 ancestors via the `is_evictable`
    /// non-leaf predicate. Per-slot ordering policy (e.g. an associated
    /// `const PATH_BUMP_ORDER` on `LRUSlot`) would let FullLRUSlot /
    /// MambaLRUSlot pick root_child-MRU for faster eviction walks while
    /// SwaLRUSlot keeps leaf-MRU for sliding-window correctness.
    /// Deferred — eviction walk cost is bounded by tree depth and only
    /// pays during cache pressure; uniformity with Python wins for now.
    fn bump_mru_walk<K: ChildKeyType>(pool: &mut TreeNodePool<K>, leaf: NodeIdx) {
        // Track the chain we're building as it grows from the leaf upward.
        // `Option` makes the "no nodes bumped yet" state explicit; using
        // `head_sentinel` as a "no chain yet" marker would overload its
        // meaning.
        let mut chain_first: Option<NodeIdx> = None;
        let mut chain_last: Option<NodeIdx> = None;
        let mut cur = leaf;

        while let Some(parent) = pool.get(cur).parent() {
            // Gate on `in_list`: nodes intentionally kept out of the list
            // (e.g. SWA tombstones — value freed, list slot vacated) must
            // NOT be revived by an access bump. Skip them and continue
            // walking ancestors.
            if !Self::data(pool.get(cur)).in_list {
                cur = parent;
                continue;
            }
            // In-list node: splice out of its current position and append
            // to the chain we're building. `in_list` stays true (we re-add
            // it to the list as part of the closing splice).
            Self::remove(pool, cur);
            match chain_last {
                Some(prev_last) => Self::connect::<K>(pool, prev_last, cur),
                None => chain_first = Some(cur),
            }
            Self::data_mut(pool.get_mut(cur)).in_list = true;
            chain_last = Some(cur);
            cur = parent;
        }

        // Splice chain into MRU position. Both options are `Some` iff at
        // least one node was bumped — destructure both to make that
        // invariant visible at the call site.
        if let (Some(first), Some(last)) = (chain_first, chain_last) {
            // Capture original_first AFTER all removes — `head.next` still
            // points at the correct first non-bumped node because we never
            // wrote `head.next` during the loop.
            let head = Self::head_sentinel(pool);
            let original_first = Self::data(pool.get(head)).next;
            Self::connect::<K>(pool, head, first);
            Self::connect::<K>(pool, last, original_first);
        }
    }

    /// Walk from `from.prev` toward MRU, returning the first node passing
    /// `is_evictable`. Stops at `head_sentinel`, so sentinels are never
    /// returned. `from` may be the `tail_sentinel` to seed the walk at the
    /// LRU end of real nodes (used by `next_evictable`), or a previously
    /// returned victim to step the eviction loop incrementally.
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

    /// Take this slot's freed-handle contribution out of `node` and
    /// record it into `result.freed[Self::COMPONENT]` /
    /// `result.evicted[Self::COMPONENT]`. No-op when this slot has
    /// no contribution at this node (e.g., SWA on a node whose SWA
    /// value is `None` — either never SWA-tracked or already
    /// tombstoned).
    ///
    /// Default impl works for components whose freed handle IS their
    /// own per-component value (FULL, future Mamba). Slots whose
    /// freed handle is SOMETHING ELSE (e.g., SWA's
    /// `free_swa(full_value)` quirk where the freed list carries FULL
    /// handles, not SWA handles) override this method.
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
        // Validate against the PROSPECTIVE new full_lock_ref before
        // writing. Skip FULL's own slot in the comparison (`new <= new`
        // is trivially true); compare every other component's current
        // lock_ref against the prospective new value.
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

/// LRU slot for CPU/L2 FULL values.
pub struct HostFullLRUSlot;

impl LRUSlot for HostFullLRUSlot {
    const COMPONENT: ComponentType = ComponentType::Full;
    const NAME: &'static str = "HostFull";

    // Route to the right host tier components.
    fn node_components<K: ChildKeyType>(
        node: &TreeNode<K>,
    ) -> &[ComponentNodeState; NUM_COMPONENT_TYPES] {
        &node.host_components
    }
    fn node_components_mut<K: ChildKeyType>(
        node: &mut TreeNode<K>,
    ) -> &mut [ComponentNodeState; NUM_COMPONENT_TYPES] {
        &mut node.host_components
    }
    fn pool_components<K: ChildKeyType>(
        pool: &TreeNodePool<K>,
    ) -> &[ComponentPoolState; NUM_COMPONENT_TYPES] {
        &pool.host_components
    }
    fn pool_components_mut<K: ChildKeyType>(
        pool: &mut TreeNodePool<K>,
    ) -> &mut [ComponentPoolState; NUM_COMPONENT_TYPES] {
        &mut pool.host_components
    }

    fn inc_lock_ref<K: ChildKeyType>(pool: &mut TreeNodePool<K>, node_idx: NodeIdx) -> i64 {
        // Skip the device-FULL cap: host_full_lock_ref can exceed
        // device_full_lock_ref when a preempted request's value is evicted
        // from device but kept (pinned) on the host backup.
        Self::inc_lock_ref_non_full(pool, node_idx, /* enforce_full_cap */ false)
    }

    fn dec_lock_ref<K: ChildKeyType>(pool: &mut TreeNodePool<K>, node_idx: NodeIdx) -> i64 {
        Self::dec_lock_ref_non_full(pool, node_idx)
    }
}

/// LRU for SWA values.
pub struct SwaLRUSlot;

impl LRUSlot for SwaLRUSlot {
    const COMPONENT: ComponentType = ComponentType::Swa;
    const NAME: &'static str = "Swa";

    /// SWA quirk: the freed handle pushed to `result.freed[Swa]` is a
    /// CLONE of FULL's value (the cookie Python's `free_swa(full_value)`
    /// API consumes), NOT SWA's own translated value. SWA's own value
    /// is dropped here — its memory is reclaimed by `free_swa`'s
    /// internals on the Python side. Gates on SWA's own value being
    /// `Some` (= not a tombstone); already-tombstoned nodes are
    /// no-ops.
    fn take_value<K: ChildKeyType>(node: &mut TreeNode<K>, result: &mut EvictResult) {
        let ct = Self::COMPONENT as usize;
        if node.components[ct].value.take().is_some() {
            // SWA's own translated value is dropped above.
            // Push a clone of FULL's value as the free_swa cookie.
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
        // Device-leaf bookkeeping: debit the parent's full-device child count.
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
    hicache_write_back: bool,
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

        // Potential next node for eviction.
        let next_after = FullLRUSlot::next_evictable_after(pool, idx);

        // Host-backed victim: FULL value is already backed up on host; free
        // only the device value without removing the tree node.
        if HostFullLRUSlot::has_value(pool.get(idx)) {
            // TODO(Jialin): Revisit when SWA/Mamba gain their own host tiers.
            #[allow(
                clippy::expect_used,
                reason = "a host-backed device victim still holds its device value"
            )]
            let device_value =
                evict_full_value(pool, idx, result).expect("demote victim has a device value");
            result
                .deferred_actions
                .push(DeferredAction::FullDeviceEvictOnBackedUp {
                    node_idx: idx,
                    device_value,
                });
            victim = next_after;
            continue;
        }

        // HiCache write-back: back up to host on eviction without deleting the node.
        if hicache_write_back {
            #[allow(clippy::expect_used, reason = "device-only victim has a device value")]
            let device_value =
                evict_full_value(pool, idx, result).expect("write-back victim has a device value");
            result
                .deferred_actions
                .push(DeferredAction::FullWriteBackOnEvict {
                    node_idx: idx,
                    value: device_value,
                });
            victim = next_after;
            continue;
        }

        // Not host-backed: DELETE the leaf.
        #[allow(clippy::expect_used, reason = "FULL LRU never includes root")]
        let parent_idx = pool
            .get(idx)
            .parent()
            .expect("evict: LRU returned a node without a parent (root spliced into LRU?)");

        #[allow(clippy::expect_used, reason = "non-root leaf has child key in parent")]
        pool.evict_leaf(idx, result)
            .expect("evict: leaf must have valid child key for parent removal");

        // Capture parent evictability BEFORE the cascade (which may free
        // `parent_idx`). A freshly-exposed evictable parent can sit deeper
        // toward the LRU tail than `next_after`.
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

/// Best-effort eviction of at least `num_tokens` of host FULL values in LRU order.
pub(crate) fn evict_host_full<K: ChildKeyType>(
    pool: &mut TreeNodePool<K>,
    num_tokens: usize,
    result: &mut EvictResult,
) {
    if num_tokens == 0 {
        return;
    }
    let ct = HostFullLRUSlot::COMPONENT as usize;
    let target = result.evicted[ct] + num_tokens;
    let mut victim = HostFullLRUSlot::next_evictable(pool);

    while result.evicted[ct] < target {
        let Some(idx) = victim else { break };
        let next_after = HostFullLRUSlot::next_evictable_after(pool, idx);

        // TODO(Jialin): host-evicting a node with a host-backed descendant would
        // strand that descendant's load-back chain (PrepareLoadBackMissingHostValue).
        // Holds implicitly today (the root-ward host-LRU bump keeps the tail
        // leaf-first); the structural fix is H-leaf-only host eviction — gate
        // `is_evictable` on "no host-backed child" (needs a per-node
        // host-backed-child counter).
        let value_len = HostFullLRUSlot::value_len(pool.get(idx));
        #[allow(
            clippy::expect_used,
            reason = "host FULL LRU victim always has a host value"
        )]
        let host_value = HostFullLRUSlot::node_state_mut(pool.get_mut(idx))
            .value
            .take()
            .expect("host FULL LRU victim has a host value");
        result.evicted[ct] += value_len;
        result.deferred_actions.push(DeferredAction::FullHostEvict {
            node_idx: idx,
            host_value,
        });
        HostFullLRUSlot::remove(pool, idx);
        HostFullLRUSlot::pool_state_mut(pool).unlocked_size -= value_len;

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
